"""Model-free tests for multiplexed stream isolation and backpressure."""
import os
import sys
import threading
import time

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "scripts"))

import multi_stream
from multi_stream import (FLUSH, AsrScheduler, MultiStreamManager, ScopedEventSink,
                          StreamBuffer)
from ws_ingest_v2 import AudioFrame, ProtocolError, StreamInfo


class EventHub:
    def __init__(self):
        self.events = []

    def publish(self, event):
        self.events.append(event)


class FakeSession:
    pass


class FakeSharedASR:
    def new_session(self):
        return FakeSession()


def info(stream_id=1, speaker_id="u1", speaker="Alice"):
    return StreamInfo(stream_id, speaker_id, speaker, "discord", 1000,
                      {"guild_id": "g1"})


def test_stream_buffer_overflow_keeps_newest_audio_after_flush():
    buf = StreamBuffer(2)
    first = np.array([1], dtype=np.float32)
    second = np.array([2], dtype=np.float32)
    newest = np.array([3], dtype=np.float32)
    assert not buf.put(first)
    assert not buf.put(second)
    assert buf.put(newest)
    items = buf.snapshot()
    assert items[0] is FLUSH
    np.testing.assert_array_equal(items[1], newest)
    assert buf.dropped_items == 2


def test_scoped_sink_attaches_identity_and_distinct_utterance_ids():
    hub = EventHub()
    current = info()
    sink = ScopedEventSink(hub, lambda: current)
    sink.partial("draft")
    first_id = sink.final("hello", "en", latency_ms=12.0, tier="v3")
    second_id = sink.final("again", "en", latency_ms=10.0, tier="v3")

    assert hub.events[0]["speaker_id"] == "u1"
    assert hub.events[0]["utterance_id"] == "1-1"
    assert hub.events[1]["utterance_id"] == "1-1"
    assert hub.events[2]["utterance_id"] == "1-2"
    assert hub.events[1]["metadata"] == {"guild_id": "g1"}
    assert (first_id, second_id) == ("1-1", "1-2")


def test_manager_keeps_sequences_and_identity_separate_without_workers():
    hub = EventHub()
    manager = MultiStreamManager(FakeSharedASR(), hub, queue_seconds=0.1,
                                 start_workers=False)
    manager.open_stream(info(1, "alice", "Alice"))
    manager.open_stream(info(2, "bob", "Bob"))
    pcm_a = np.array([100, 200], dtype="<i2").tobytes()
    pcm_b = np.array([-100, -200], dtype="<i2").tobytes()

    manager.audio(AudioFrame(1, 10, 1000, pcm_a))
    manager.audio(AudioFrame(2, 50, 1000, pcm_b))
    manager.audio(AudioFrame(1, 12, 1020, pcm_a))  # gap: 11 was lost

    alice = manager.get_stream_for_test(1)
    bob = manager.get_stream_for_test(2)
    assert alice.sequence_gaps == 1
    assert bob.sequence_gaps == 0
    assert alice.expected_sequence == 13
    assert bob.expected_sequence == 51
    assert any(item is FLUSH for item in alice.buffer.snapshot())
    assert all(item is not FLUSH for item in bob.buffer.snapshot())


def test_manager_identity_update_and_end_affect_only_target_stream():
    hub = EventHub()
    manager = MultiStreamManager(FakeSharedASR(), hub, start_workers=False)
    manager.open_stream(info(1, "alice", "Alice"))
    manager.open_stream(info(2, "bob", "Bob"))

    manager.update_identity(1, "Alice New", {"channel_id": "c1"})
    manager.end_stream(1, "left")

    with pytest.raises(ProtocolError, match="unknown"):
        manager.get_stream_for_test(1)
    bob = manager.get_stream_for_test(2)
    assert bob.info.speaker == "Bob"
    assert manager.status()["active_streams"] == 1


def test_unknown_audio_is_rejected_not_misattributed():
    manager = MultiStreamManager(FakeSharedASR(), EventHub(), start_workers=False)
    with pytest.raises(ProtocolError, match="unknown stream"):
        manager.audio(AudioFrame(99, 0, 0, b"\x00\x00"))
    assert manager.status()["unknown_frames"] == 1
    manager.scheduler.close()


def test_stream_worker_reports_asr_error_and_continues(monkeypatch):
    hub = EventHub()
    manager = MultiStreamManager(FakeSharedASR(), hub, start_workers=False,
                                 refine=False)
    manager.open_stream(info())
    managed = manager.get_stream_for_test(1)
    manager.end_stream(1, "test_complete")
    calls = 0

    def fail_once_then_drain(chunks, *args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RuntimeError("temporary inference failure")
        list(chunks)

    monkeypatch.setattr(multi_stream, "run_stream", fail_once_then_drain)
    manager._run_stream(managed)
    manager.scheduler.close()

    assert calls == 2
    assert managed.errors == 1
    assert managed.last_error == "RuntimeError: temporary inference failure"
    assert any(event["type"] == "stream_error" for event in hub.events)
    assert hub.events[-1]["type"] == "stream_end"


def test_scheduler_runs_waiting_final_before_partial_and_refine():
    scheduler = AsrScheduler(max_partial_backlog=10)
    blocker_started = threading.Event()
    release_blocker = threading.Event()
    order = []

    def blocking_final():
        blocker_started.set()
        release_blocker.wait(timeout=2)
        order.append("first-final")

    first = threading.Thread(target=lambda: scheduler.call("final", blocking_final))
    first.start()
    assert blocker_started.wait(timeout=1)

    refine = threading.Thread(target=lambda: scheduler.call("refine",
                                                             lambda: order.append("refine")))
    partial = threading.Thread(target=lambda: scheduler.call("partial",
                                                              lambda: order.append("partial")))
    final = threading.Thread(target=lambda: scheduler.call("final",
                                                            lambda: order.append("final")))
    refine.start()
    partial.start()
    final.start()
    time.sleep(0.02)
    release_blocker.set()
    for thread in (first, refine, partial, final):
        thread.join(timeout=2)
    scheduler.close()

    assert order == ["first-final", "final", "partial", "refine"]

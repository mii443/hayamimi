"""Per-speaker realtime pipelines for multiplexed network audio."""
from __future__ import annotations

import sys
import threading
import time
from collections import deque
from dataclasses import dataclass, replace
from typing import Any, Callable

import numpy as np

from audio_utils import decode_pcm16
from realtime_transcribe import (AudioHistory, PartialPrinter, Refiner, SAMPLE_RATE,
                                 SessionStats, TranslationWorker, WINDOW_SIZE,
                                 build_vad, run_stream)
from ws_ingest_v2 import AudioFrame, ProtocolError, StreamInfo

FLUSH = object()
STOP = object()


@dataclass
class _ScheduledJob:
    work: Callable[[], Any]
    done: threading.Event
    result: Any = None
    error: BaseException | None = None


class AsrScheduler:
    """One priority worker for all access to shared recognizer objects."""

    FINAL = 0
    PARTIAL = 1
    REFINE = 2

    def __init__(self, max_partial_backlog: int = 4):
        import queue

        self._queue = queue.PriorityQueue()
        self._counter = 0
        self._lock = threading.Lock()
        self._stopped = False
        self.max_partial_backlog = max_partial_backlog
        self.skipped_partials = 0
        self.completed = {"final": 0, "partial": 0, "refine": 0}
        self._thread = threading.Thread(target=self._worker, daemon=True,
                                        name="asr-priority-worker")
        self._thread.start()

    def call(self, kind: str, work: Callable[[], Any], skipped_value=None):
        priority = {"final": self.FINAL, "partial": self.PARTIAL,
                    "refine": self.REFINE}[kind]
        with self._lock:
            if self._stopped:
                raise RuntimeError("ASR scheduler is stopped")
            # Under overload, a stale draft is less useful than leaving CPU
            # available for a final. A stream worker waits for each call, so
            # there can already be at most one queued partial per stream.
            if kind == "partial" and self._queue.qsize() >= self.max_partial_backlog:
                self.skipped_partials += 1
                return skipped_value
            self._counter += 1
            order = self._counter
        job = _ScheduledJob(work, threading.Event())
        self._queue.put((priority, order, kind, job))
        job.done.wait()
        if job.error is not None:
            raise job.error
        return job.result

    def status(self) -> dict[str, Any]:
        return {"queue_depth": self._queue.qsize(),
                "skipped_partials": self.skipped_partials,
                "completed": dict(self.completed)}

    def close(self):
        with self._lock:
            if self._stopped:
                return
            self._stopped = True
            self._counter += 1
            order = self._counter
        done = threading.Event()
        self._queue.put((99, order, "stop", _ScheduledJob(lambda: None, done)))
        self._thread.join(timeout=5)

    def _worker(self):
        while True:
            _priority, _order, kind, job = self._queue.get()
            try:
                if kind == "stop":
                    return
                job.result = job.work()
                self.completed[kind] += 1
            except BaseException as exc:
                job.error = exc
            finally:
                job.done.set()
                self._queue.task_done()


class ScheduledAsrSession:
    """Synchronous facade which routes ASR calls through AsrScheduler."""

    def __init__(self, inner, scheduler: AsrScheduler):
        self.inner = inner
        self.scheduler = scheduler

    @property
    def forced_lang(self):
        return self.inner.forced_lang

    @property
    def ko_spacer(self):
        return self.scheduler.call("refine", lambda: self.inner.ko_spacer)

    @property
    def _ko_provider(self):
        return self.inner._ko_provider

    def identify(self, samples, sample_rate):
        return self.scheduler.call("partial",
                                   lambda: self.inner.identify(samples, sample_rate),
                                   skipped_value=None)

    def partial(self, samples, sample_rate, lang_hint=None):
        return self.scheduler.call(
            "partial", lambda: self.inner.partial(samples, sample_rate, lang_hint=lang_hint),
            skipped_value="")

    def transcribe(self, samples, sample_rate, known_lang=None, speech_s=None, live=True):
        kind = "final" if live else "refine"
        return self.scheduler.call(
            kind, lambda: self.inner.transcribe(samples, sample_rate,
                                                 known_lang=known_lang,
                                                 speech_s=speech_s, live=live))

    def _identify_lang(self, samples, sample_rate):
        return self.scheduler.call("refine",
                                   lambda: self.inner._identify_lang(samples, sample_rate))

    def _get(self, name):
        return self.scheduler.call("refine", lambda: self.inner._get(name))

    def _decode_full(self, rec, samples, sample_rate):
        return self.scheduler.call("refine",
                                   lambda: self.inner._decode_full(rec, samples, sample_rate))

    def _replace(self, text):
        return self.inner._replace(text)


class StreamBuffer:
    """Bounded blocking buffer which turns overflow into an ASR boundary."""

    def __init__(self, capacity: int):
        if capacity < 2:
            raise ValueError("stream buffer capacity must be at least 2")
        self.capacity = capacity
        self._items = deque()
        self._cv = threading.Condition()
        self.dropped_items = 0

    def put(self, item) -> bool:
        """Append an item; clear stale audio and add FLUSH on overflow.

        Returns True when an overflow occurred. Keeping the newest audio is
        preferable for realtime subtitles, but audio on opposite sides of a
        drop must never be joined into one utterance.
        """
        overflowed = False
        with self._cv:
            if len(self._items) >= self.capacity:
                self.dropped_items += len(self._items)
                self._items.clear()
                self._items.append(FLUSH)
                overflowed = True
            self._items.append(item)
            self._cv.notify()
        return overflowed

    def get(self):
        with self._cv:
            while not self._items:
                self._cv.wait()
            return self._items.popleft()

    def snapshot(self) -> list:
        with self._cv:
            return list(self._items)


class ScopedEventSink:
    """Attach stable stream identity to every hayamimi subtitle event."""

    def __init__(self, event_hub, info_getter: Callable[[], StreamInfo]):
        self.event_hub = event_hub
        self.info_getter = info_getter
        self._lock = threading.Lock()
        self._utterance = 0

    def _identity(self) -> dict[str, Any]:
        info = self.info_getter()
        return {
            "stream_id": info.stream_id,
            "source": info.source,
            "speaker_id": info.speaker_id,
            "speaker": info.speaker,
            "metadata": dict(info.metadata),
        }

    def publish(self, event: dict):
        value = {**event, **self._identity(), "emitted_at_ms": int(time.time() * 1000)}
        self.event_hub.publish(value)

    def partial(self, text: str):
        with self._lock:
            utterance_id = f"{self.info_getter().stream_id}-{self._utterance + 1}"
        self.publish({"type": "partial", "text": text, "utterance_id": utterance_id})

    def final(self, text: str, lang: str = "", speaker: str = "",
              latency_ms: float | None = None, tier: str = ""):
        with self._lock:
            self._utterance += 1
            utterance_id = f"{self.info_getter().stream_id}-{self._utterance}"
        self.publish({"type": "final", "text": text, "lang": lang,
                      "utterance_id": utterance_id, "latency_ms": latency_ms,
                      "tier": tier})
        return utterance_id


@dataclass
class ManagedStream:
    info: StreamInfo
    buffer: StreamBuffer
    asr: Any
    expected_sequence: int | None = None
    last_capture_ms: int = 0
    sequence_gaps: int = 0
    overflows: int = 0
    errors: int = 0
    last_error: str = ""
    end_reason: str = ""
    thread: threading.Thread | None = None


def stream_chunks(buffer: StreamBuffer):
    """Re-chunk arbitrary network payloads to the 512-sample VAD window."""
    leftover = np.zeros(0, dtype=np.float32)
    while True:
        item = buffer.get()
        if item is STOP:
            if len(leftover):
                yield np.pad(leftover, (0, WINDOW_SIZE - len(leftover)))
            yield FLUSH
            return
        if item is FLUSH:
            if len(leftover):
                yield np.pad(leftover, (0, WINDOW_SIZE - len(leftover)))
                leftover = np.zeros(0, dtype=np.float32)
            yield FLUSH
            continue
        leftover = np.concatenate([leftover, item])
        while len(leftover) >= WINDOW_SIZE:
            yield leftover[:WINDOW_SIZE]
            leftover = leftover[WINDOW_SIZE:]


class MultiStreamManager:
    """Own one VAD/history pipeline per input stream and one shared ASR pool."""

    def __init__(self, shared_asr, event_hub, max_streams: int = 32,
                 queue_seconds: float = 2.0, min_silence: float = 0.35,
                 max_speech: float = 12.0, partial: bool = True,
                 refine: bool = True, start_workers: bool = True,
                 translation_backend=None, translation_workers: int = 2):
        self.shared_asr = shared_asr
        self.event_hub = event_hub
        self.max_streams = max_streams
        # v2 normally sends 20ms packets; retain at least two items and add
        # headroom for clients that use smaller frames.
        self.queue_capacity = max(2, int(queue_seconds / 0.02))
        self.min_silence = min_silence
        self.max_speech = max_speech
        self.partial = partial
        self.refine = refine
        self.start_workers = start_workers
        self.translation_backend = translation_backend
        self.translation_worker = (
            TranslationWorker(translation_backend, workers=translation_workers)
            if translation_backend is not None else None)
        self.scheduler = AsrScheduler()
        self._streams: dict[int, ManagedStream] = {}
        self._retired_threads: list[threading.Thread] = []
        self._lock = threading.RLock()
        self.unknown_frames = 0
        self.total_sequence_gaps = 0
        self.total_overflows = 0

    def open_stream(self, info: StreamInfo) -> None:
        with self._lock:
            if info.stream_id in self._streams:
                raise ProtocolError(f"stream {info.stream_id} is already open")
            if len(self._streams) >= self.max_streams:
                raise ProtocolError("maximum active stream count reached")
            session = ScheduledAsrSession(self.shared_asr.new_session(), self.scheduler)
            managed = ManagedStream(info, StreamBuffer(self.queue_capacity), session)
            self._streams[info.stream_id] = managed
            if self.start_workers:
                managed.thread = threading.Thread(
                    target=self._run_stream, args=(managed,), daemon=True,
                    name=f"asr-stream-{info.stream_id}")
                managed.thread.start()
        self.event_hub.publish({"type": "stream_open", **self._identity(info),
                                "emitted_at_ms": int(time.time() * 1000)})

    def audio(self, frame: AudioFrame) -> None:
        managed = self._find(frame.stream_id)
        if managed.expected_sequence is not None and frame.sequence != managed.expected_sequence:
            managed.sequence_gaps += 1
            self.total_sequence_gaps += 1
            managed.buffer.put(FLUSH)
        managed.expected_sequence = (frame.sequence + 1) & 0xFFFFFFFF
        managed.last_capture_ms = frame.captured_at_ms
        samples = decode_pcm16(frame.pcm, channels=1)
        if len(samples) and managed.buffer.put(samples):
            managed.overflows += 1
            self.total_overflows += 1

    def stream_idle(self, stream_id: int) -> None:
        self._find(stream_id).buffer.put(FLUSH)

    def gap(self, stream_id: int, reason: str) -> None:
        managed = self._find(stream_id)
        managed.sequence_gaps += 1
        self.total_sequence_gaps += 1
        managed.buffer.put(FLUSH)

    def update_identity(self, stream_id: int, speaker: str,
                        metadata: dict[str, Any]) -> None:
        managed = self._find(stream_id)
        with self._lock:
            merged = {**managed.info.metadata, **metadata}
            managed.info = replace(managed.info,
                                   speaker=speaker or managed.info.speaker,
                                   metadata=merged)

    def end_stream(self, stream_id: int, reason: str) -> None:
        with self._lock:
            managed = self._streams.pop(stream_id, None)
        if managed is None:
            raise ProtocolError(f"unknown stream {stream_id}")
        managed.end_reason = reason
        managed.buffer.put(STOP)
        if managed.thread is not None:
            with self._lock:
                self._retired_threads.append(managed.thread)

    def bridge_disconnected(self, epoch: str) -> None:
        with self._lock:
            ids = list(self._streams)
        for stream_id in ids:
            try:
                self.end_stream(stream_id, "bridge_disconnected")
            except ProtocolError:
                pass

    def status(self) -> dict[str, Any]:
        with self._lock:
            streams = [{"stream_id": stream.info.stream_id,
                        "speaker_id": stream.info.speaker_id,
                        "speaker": stream.info.speaker,
                        "last_capture_ms": stream.last_capture_ms,
                        "sequence_gaps": stream.sequence_gaps,
                        "overflows": stream.overflows,
                        "errors": stream.errors,
                        "last_error": stream.last_error}
                       for stream in self._streams.values()]
        return {"active_streams": len(streams), "streams": streams,
                "unknown_frames": self.unknown_frames,
                "sequence_gaps": self.total_sequence_gaps,
                "overflows": self.total_overflows,
                "scheduler": self.scheduler.status()}

    def close(self):
        self.bridge_disconnected("shutdown")
        with self._lock:
            threads = list(self._retired_threads)
        for thread in threads:
            thread.join(timeout=10)
        if self.translation_worker is not None:
            self.translation_worker.close(wait=True)
        self.scheduler.close()

    def get_stream_for_test(self, stream_id: int) -> ManagedStream:
        """Read-only test hook; production callers should use status()."""
        return self._find(stream_id)

    def _find(self, stream_id: int) -> ManagedStream:
        with self._lock:
            managed = self._streams.get(stream_id)
        if managed is None:
            self.unknown_frames += 1
            raise ProtocolError(f"unknown stream {stream_id}")
        return managed

    def _run_stream(self, managed: ManagedStream):
        sink = ScopedEventSink(self.event_hub, lambda: managed.info)
        printer = PartialPrinter(enabled=self.partial, server=sink)
        history = AudioHistory(SAMPLE_RATE)
        stats = SessionStats()
        vad = build_vad(self.min_silence, self.max_speech)
        refiner = (Refiner(managed.asr, history, SAMPLE_RATE, printer, stats=stats,
                           translators=self.translation_backend)
                   if self.refine else None)
        chunks = stream_chunks(managed.buffer)
        try:
            while True:
                try:
                    run_stream(chunks, vad, SAMPLE_RATE, managed.asr, stats, printer,
                               refiner=refiner, history=history,
                               translator_worker=self.translation_worker)
                    break
                except Exception as exc:
                    self._report_stream_error(managed, sink, exc)
        finally:
            if refiner is not None:
                try:
                    refiner.maybe_refine(0, force=True)
                except Exception as exc:
                    self._report_stream_error(managed, sink, exc)
                finally:
                    refiner.close(wait=True)
            sink.publish({"type": "stream_end", "reason": managed.end_reason or "ended",
                          "summary": stats.summary()})

    @staticmethod
    def _report_stream_error(managed: ManagedStream, sink: ScopedEventSink,
                             exc: Exception) -> None:
        managed.errors += 1
        managed.last_error = f"{type(exc).__name__}: {exc}"
        print(f"[hayamimi] stream {managed.info.stream_id} ASR error; "
              f"continuing: {managed.last_error}", file=sys.stderr)
        sink.publish({"type": "stream_error", "error": type(exc).__name__,
                      "message": str(exc)})

    @staticmethod
    def _identity(info: StreamInfo) -> dict[str, Any]:
        return {"stream_id": info.stream_id, "source": info.source,
                "speaker_id": info.speaker_id, "speaker": info.speaker,
                "metadata": dict(info.metadata)}

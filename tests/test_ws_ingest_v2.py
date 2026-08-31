"""Wire-contract and routing-session tests for multiplexed ingest v2."""
import os
import sys
import threading

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "scripts"))

from asr_engine import RoutedASRSession
from ws_ingest_v2 import (AUDIO_HEADER_SIZE, AudioFrame, ProtocolError,
                          decode_audio_frame, encode_audio_frame, parse_stream_open)


def test_audio_frame_roundtrip_matches_fixed_header_size():
    pcm = np.array([0, 1, -1, 32767, -32768], dtype="<i2").tobytes()
    encoded = encode_audio_frame(AudioFrame(17, 42, 1_788_100_002_510, pcm))
    assert len(encoded) == AUDIO_HEADER_SIZE + len(pcm)
    assert encoded.hex() == "0201000000110000002a000001a05310c2ce00000100ffffff7f0080"
    assert decode_audio_frame(encoded) == AudioFrame(17, 42, 1_788_100_002_510, pcm)


@pytest.mark.parametrize("payload", [b"", b"short", b"\x02\x01" + b"\x00" * 16])
def test_audio_frame_rejects_missing_or_empty_pcm(payload):
    with pytest.raises(ProtocolError):
        decode_audio_frame(payload)


def test_audio_frame_rejects_wrong_version_and_odd_pcm():
    good = bytearray(encode_audio_frame(AudioFrame(1, 0, 0, b"\x00\x00")))
    good[0] = 3
    with pytest.raises(ProtocolError, match="version"):
        decode_audio_frame(bytes(good))
    with pytest.raises(ProtocolError, match="whole s16"):
        encode_audio_frame(AudioFrame(1, 0, 0, b"\x00"))


def test_parse_stream_open_keeps_generic_source_metadata():
    info = parse_stream_open({
        "type": "stream_open", "stream_id": 7, "speaker_id": "789",
        "speaker": "Alice", "source": "discord", "started_at_ms": 1234,
        "metadata": {"guild_id": "123", "channel_id": "456"},
    })
    assert info.stream_id == 7
    assert info.speaker_id == "789"
    assert info.speaker == "Alice"
    assert info.source == "discord"
    assert info.metadata["guild_id"] == "123"


def test_parse_stream_open_rejects_non_object_metadata():
    with pytest.raises(ProtocolError, match="metadata"):
        parse_stream_open({"type": "stream_open", "stream_id": 1,
                           "speaker_id": "u1", "metadata": []})


class FakeRoutedASR:
    forced_lang = None
    min_switch_s = 2.0
    ko_spacer = None

    def __init__(self):
        self._session_lock = threading.RLock()
        self.last_lang = None
        self._pending_lang = None
        self._pending_count = 0

    def transcribe(self, samples, sample_rate, known_lang=None, speech_s=None, live=True):
        lang = known_lang or str(samples[0])
        previous = self.last_lang
        if live:
            self.last_lang = lang
            self._pending_lang = f"pending-{lang}"
            self._pending_count += 1
        return {"text": f"{previous}->{lang}", "lang": lang}

    def partial(self, samples, sample_rate, lang_hint=None):
        return f"{self.last_lang}:{lang_hint or ''}"

    def identify(self, samples, sample_rate):
        return str(samples[0])


def test_routed_asr_sessions_do_not_leak_sticky_language_state():
    owner = FakeRoutedASR()
    alice = RoutedASRSession(owner)
    bob = RoutedASRSession(owner)
    samples = np.array([1.0], dtype=np.float32)

    assert alice.transcribe(samples, 16000, known_lang="ja")["text"] == "None->ja"
    assert bob.transcribe(samples, 16000, known_lang="en")["text"] == "None->en"
    assert alice.partial(samples, 16000) == "ja:"
    assert bob.partial(samples, 16000) == "en:"
    assert alice.state.pending_count == 1
    assert bob.state.pending_count == 1
    assert owner.last_lang is None


def test_resetting_one_routed_session_does_not_reset_another():
    owner = FakeRoutedASR()
    alice = RoutedASRSession(owner)
    bob = RoutedASRSession(owner)
    samples = np.array([1.0], dtype=np.float32)
    alice.transcribe(samples, 16000, known_lang="ja")
    bob.transcribe(samples, 16000, known_lang="en")

    alice.reset_session()

    assert alice.state.last_lang is None
    assert bob.state.last_lang == "en"

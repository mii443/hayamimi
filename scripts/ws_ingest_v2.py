"""Multiplexed WebSocket ingest protocol for trusted audio bridges.

Version 1 accepts one microphone connection. Version 2 carries many
independent PCM streams over one socket; it is intended for bridges such as
a Discord voice receiver where the source already provides one lane per
user. Wire parsing stays independent of ASR models for deterministic tests.
"""
from __future__ import annotations

import asyncio
import hmac
import json
import struct
import threading
from dataclasses import dataclass, field
from typing import Any, Protocol

from ws_protocol import (OP_BINARY, OP_CLOSE, OP_PING, OP_PONG, OP_TEXT,
                         build_handshake_response, decode_frame, encode_frame,
                         parse_handshake_request)

PROTOCOL_VERSION = 2
INGEST_V2_PATH = "/ingest/v2"
AUDIO_KIND = 1
AUDIO_HEADER = struct.Struct(">BBIIQ")
AUDIO_HEADER_SIZE = AUDIO_HEADER.size
MAX_HEADER_BYTES = 16_384
MAX_WS_BUFFER_BYTES = 2 * 1024 * 1024
MAX_PCM_BYTES = 256 * 1024


class ProtocolError(ValueError):
    """A peer sent a syntactically invalid protocol-v2 message."""


@dataclass(frozen=True)
class AudioFrame:
    stream_id: int
    sequence: int
    captured_at_ms: int
    pcm: bytes


@dataclass(frozen=True)
class StreamInfo:
    stream_id: int
    speaker_id: str
    speaker: str
    source: str = "unknown"
    started_at_ms: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)


def encode_audio_frame(frame: AudioFrame) -> bytes:
    """Encode one protocol-v2 binary audio payload (without WS framing)."""
    if not 0 < frame.stream_id <= 0xFFFFFFFF:
        raise ProtocolError("stream_id must be between 1 and u32::MAX")
    if not 0 <= frame.sequence <= 0xFFFFFFFF:
        raise ProtocolError("sequence must fit in u32")
    if not 0 <= frame.captured_at_ms <= 0xFFFFFFFFFFFFFFFF:
        raise ProtocolError("captured_at_ms must fit in u64")
    if not frame.pcm or len(frame.pcm) % 2:
        raise ProtocolError("PCM payload must contain whole s16 samples")
    if len(frame.pcm) > MAX_PCM_BYTES:
        raise ProtocolError("PCM payload is too large")
    return (AUDIO_HEADER.pack(PROTOCOL_VERSION, AUDIO_KIND, frame.stream_id,
                              frame.sequence, frame.captured_at_ms) + frame.pcm)


def decode_audio_frame(payload: bytes) -> AudioFrame:
    """Decode and validate one protocol-v2 binary audio payload."""
    if len(payload) < AUDIO_HEADER_SIZE:
        raise ProtocolError("audio frame is shorter than its header")
    version, kind, stream_id, sequence, captured_at_ms = AUDIO_HEADER.unpack_from(payload)
    if version != PROTOCOL_VERSION:
        raise ProtocolError(f"unsupported protocol version {version}")
    if kind != AUDIO_KIND:
        raise ProtocolError(f"unsupported binary frame kind {kind}")
    pcm = payload[AUDIO_HEADER_SIZE:]
    if stream_id == 0:
        raise ProtocolError("stream_id 0 is reserved")
    if not pcm or len(pcm) % 2:
        raise ProtocolError("PCM payload must contain whole s16 samples")
    if len(pcm) > MAX_PCM_BYTES:
        raise ProtocolError("PCM payload is too large")
    return AudioFrame(stream_id, sequence, captured_at_ms, pcm)


def parse_stream_open(message: dict[str, Any]) -> StreamInfo:
    """Validate a ``stream_open`` control message."""
    if message.get("type") != "stream_open":
        raise ProtocolError("expected stream_open")
    try:
        stream_id = int(message["stream_id"])
        speaker_id = str(message["speaker_id"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ProtocolError("stream_open requires stream_id and speaker_id") from exc
    if not 0 < stream_id <= 0xFFFFFFFF:
        raise ProtocolError("stream_id must be between 1 and u32::MAX")
    if not speaker_id or len(speaker_id) > 128:
        raise ProtocolError("speaker_id must contain 1..128 characters")
    speaker = str(message.get("speaker") or speaker_id)
    if len(speaker) > 256:
        raise ProtocolError("speaker is too long")
    source = str(message.get("source") or "unknown")
    metadata = message.get("metadata", {})
    if metadata is None:
        metadata = {}
    if not isinstance(metadata, dict):
        raise ProtocolError("metadata must be a JSON object")
    started_at_ms = int(message.get("started_at_ms") or 0)
    return StreamInfo(stream_id, speaker_id, speaker, source, started_at_ms,
                      dict(metadata))


class MultiplexHandler(Protocol):
    def open_stream(self, info: StreamInfo) -> None: ...
    def audio(self, frame: AudioFrame) -> None: ...
    def stream_idle(self, stream_id: int) -> None: ...
    def gap(self, stream_id: int, reason: str) -> None: ...
    def update_identity(self, stream_id: int, speaker: str,
                        metadata: dict[str, Any]) -> None: ...
    def end_stream(self, stream_id: int, reason: str) -> None: ...
    def bridge_disconnected(self, epoch: str) -> None: ...
    def status(self) -> dict[str, Any]: ...


class MultiplexIngestServer:
    """Small RFC6455 server implementing hayamimi ingest protocol v2.

    Only one bridge is active at a time. Handler calls must be quick queue
    operations; model work belongs behind the handler. Result events are
    mirrored from ``event_hub`` when it exposes subscribe/unsubscribe.
    """

    def __init__(self, host: str, port: int, handler: MultiplexHandler,
                 secret: str, event_hub=None, max_streams: int = 32):
        if not secret:
            raise ValueError("a non-empty bridge secret is required")
        self.host = host
        self.port = port
        self.handler = handler
        self.secret = secret
        self.event_hub = event_hub
        self.max_streams = max_streams
        self._active = False
        self._lock = threading.Lock()
        self._loop: asyncio.AbstractEventLoop | None = None

    def start(self) -> "MultiplexIngestServer":
        ready = threading.Event()
        errors: list[Exception] = []

        def run():
            try:
                self._loop = asyncio.new_event_loop()
                asyncio.set_event_loop(self._loop)
                self._loop.create_task(self._serve(ready))
                self._loop.run_forever()
            except Exception as exc:
                errors.append(exc)
                ready.set()

        threading.Thread(target=run, daemon=True, name="ws-ingest-v2").start()
        ready.wait(timeout=5)
        if errors:
            raise errors[0]
        if not ready.is_set():
            raise TimeoutError("timed out starting ws ingest v2")
        return self

    async def _serve(self, ready: threading.Event):
        server = await asyncio.start_server(self._handle, self.host, self.port,
                                            limit=MAX_WS_BUFFER_BYTES)
        ready.set()
        async with server:
            await server.serve_forever()

    async def _upgrade(self, reader: asyncio.StreamReader,
                       writer: asyncio.StreamWriter) -> bool:
        try:
            head = await reader.readuntil(b"\r\n\r\n")
        except (asyncio.IncompleteReadError, asyncio.LimitOverrunError):
            return False
        if len(head) > MAX_HEADER_BYTES:
            return False
        req = parse_handshake_request(head[:-4])
        if req is None or req["path"] != INGEST_V2_PATH:
            writer.write(b"HTTP/1.1 404 Not Found\r\nConnection: close\r\n\r\n")
            await writer.drain()
            return False
        writer.write(build_handshake_response(req["key"]))
        await writer.drain()
        return True

    async def _handle(self, reader: asyncio.StreamReader,
                      writer: asyncio.StreamWriter):
        epoch = ""
        forward_task = None
        owns_active_slot = False
        try:
            if not await self._upgrade(reader, writer):
                writer.close()
                return
            with self._lock:
                if self._active:
                    await self._send_json(writer, {"type": "error",
                                                    "message": "bridge already connected"})
                    writer.close()
                    return
                self._active = True
                owns_active_slot = True

            buf = bytearray()
            authenticated = False

            while True:
                data = await reader.read(4096)
                if not data:
                    break
                buf += data
                if len(buf) > MAX_WS_BUFFER_BYTES:
                    raise ProtocolError("WebSocket receive buffer exceeded limit")
                while True:
                    decoded = decode_frame(bytes(buf))
                    if decoded is None:
                        break
                    opcode, payload, consumed = decoded
                    del buf[:consumed]
                    if opcode == OP_CLOSE:
                        return
                    if opcode == OP_PING:
                        writer.write(encode_frame(payload, opcode=OP_PONG))
                        await writer.drain()
                        continue
                    if opcode == OP_PONG:
                        continue
                    if not authenticated:
                        if opcode != OP_TEXT:
                            raise ProtocolError("hello must be the first WebSocket message")
                        hello = self._parse_json(payload)
                        if hello.get("type") != "hello" or hello.get("protocol") != 2:
                            raise ProtocolError("expected protocol-v2 hello")
                        if not hmac.compare_digest(str(hello.get("auth", "")), self.secret):
                            await self._send_json(writer, {"type": "error",
                                                           "message": "authentication failed"})
                            return
                        if (hello.get("format") != "pcm_s16le"
                                or int(hello.get("sr", 0)) != 16000
                                or int(hello.get("channels", 0)) != 1):
                            raise ProtocolError("only 16kHz mono pcm_s16le is supported")
                        epoch = str(hello.get("epoch") or "")
                        if not epoch or len(epoch) > 128:
                            raise ProtocolError("hello requires a bounded non-empty epoch")
                        authenticated = True
                        await self._send_json(writer, {"type": "ready", "protocol": 2,
                                                       "sr": 16000,
                                                       "max_streams": self.max_streams})
                        if self.event_hub is not None:
                            forward_task = asyncio.create_task(self._forward_events(writer))
                        continue
                    if opcode == OP_BINARY:
                        self.handler.audio(decode_audio_frame(payload))
                    elif opcode == OP_TEXT:
                        response = self._handle_control(self._parse_json(payload))
                        if response is not None:
                            await self._send_json(writer, response)
        except (ProtocolError, TypeError, ValueError) as exc:
            try:
                await self._send_json(writer, {"type": "error", "message": str(exc)})
            except (ConnectionError, OSError):
                pass
        except (ConnectionError, asyncio.IncompleteReadError, OSError):
            pass
        finally:
            if forward_task is not None:
                forward_task.cancel()
            if epoch:
                self.handler.bridge_disconnected(epoch)
            if owns_active_slot:
                with self._lock:
                    self._active = False
            writer.close()

    @staticmethod
    def _parse_json(payload: bytes) -> dict[str, Any]:
        try:
            value = json.loads(payload.decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise ProtocolError("invalid JSON control message") from exc
        if not isinstance(value, dict):
            raise ProtocolError("control message must be a JSON object")
        return value

    def _handle_control(self, message: dict[str, Any]) -> dict[str, Any] | None:
        kind = message.get("type")
        if kind == "stream_open":
            self.handler.open_stream(parse_stream_open(message))
        elif kind == "stream_idle":
            self.handler.stream_idle(self._stream_id(message))
        elif kind == "gap":
            self.handler.gap(self._stream_id(message), str(message.get("reason") or "gap"))
        elif kind == "identity_update":
            metadata = message.get("metadata", {})
            if metadata is None:
                metadata = {}
            if not isinstance(metadata, dict):
                raise ProtocolError("metadata must be a JSON object")
            self.handler.update_identity(self._stream_id(message),
                                         str(message.get("speaker") or ""), metadata)
        elif kind == "stream_end":
            self.handler.end_stream(self._stream_id(message),
                                    str(message.get("reason") or "ended"))
        elif kind == "status":
            return {"type": "status", **self.handler.status()}
        else:
            raise ProtocolError(f"unsupported control message {kind!r}")
        return None

    @staticmethod
    def _stream_id(message: dict[str, Any]) -> int:
        try:
            stream_id = int(message["stream_id"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ProtocolError("control message requires stream_id") from exc
        if not 0 < stream_id <= 0xFFFFFFFF:
            raise ProtocolError("invalid stream_id")
        return stream_id

    @staticmethod
    async def _send_json(writer: asyncio.StreamWriter, value: dict[str, Any]):
        payload = json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        writer.write(encode_frame(payload, opcode=OP_TEXT))
        await writer.drain()

    async def _forward_events(self, writer: asyncio.StreamWriter):
        q = self.event_hub.subscribe()
        loop = asyncio.get_running_loop()
        try:
            while True:
                data = await loop.run_in_executor(None, q.get)
                writer.write(encode_frame(data.encode("utf-8"), opcode=OP_TEXT))
                await writer.drain()
        except (asyncio.CancelledError, ConnectionError, OSError):
            pass
        finally:
            self.event_hub.unsubscribe(q)

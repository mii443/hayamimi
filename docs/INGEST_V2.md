# Multiplexed WebSocket ingest protocol v2

Protocol v2 carries independent 16 kHz mono PCM streams over one WebSocket.
It is intended for trusted bridges which already know speaker identity, such
as a Discord voice receiver. The original single-microphone `/ingest`
endpoint remains available and unchanged.

## Start the service

Set a random shared secret and start the multiplexed entrypoint:

```bash
export HAYAMIMI_BRIDGE_SECRET='replace-with-a-random-secret'
uv run python scripts/multi_realtime_transcribe.py --serve
```

The defaults are:

- ingest: `ws://127.0.0.1:8766/ingest/v2`
- dashboard: `http://127.0.0.1:8833/dashboard`
- audio: signed 16-bit little-endian PCM, mono, 16 kHz

The endpoint binds to loopback by default. If the bridge runs on another
host, place the endpoint behind TLS and authenticated network controls in
addition to the protocol secret.

## WebSocket messages

All client-to-server WebSocket frames must follow RFC 6455 client masking.
Text messages are UTF-8 JSON objects. Audio messages are binary.

### 1. Hello

The first message after the HTTP upgrade must be:

```json
{
  "type": "hello",
  "protocol": 2,
  "client": "rstt",
  "format": "pcm_s16le",
  "sr": 16000,
  "channels": 1,
  "auth": "shared secret",
  "epoch": "unique connection incarnation"
}
```

The server replies:

```json
{"type":"ready","protocol":2,"sr":16000,"max_streams":32}
```

A bridge must not send streams before `ready`. A new `epoch` invalidates all
streams from the previous bridge connection.

### 2. Open a stream

```json
{
  "type": "stream_open",
  "stream_id": 17,
  "source": "discord",
  "speaker_id": "789",
  "speaker": "Alice",
  "started_at_ms": 1788100000000,
  "metadata": {
    "guild_id": "123",
    "channel_id": "456"
  }
}
```

`stream_id` is a non-zero unsigned 32-bit integer unique within one epoch.
`speaker_id` is an opaque stable identifier. hayamimi does not interpret
source-specific metadata.

### 3. Send audio

Each binary WebSocket message has an 18-byte header followed by PCM:

```text
offset  size  encoding  field
0       1     u8        protocol version = 2
1       1     u8        kind = 1 (audio)
2       4     u32 BE    stream_id
6       4     u32 BE    sequence
10      8     u64 BE    captured_at_ms
18      N     s16 LE    mono 16 kHz PCM
```

The recommended payload is 20 ms: 320 samples or 640 bytes. Sequence numbers
are per-stream and wrap from `u32::MAX` to zero. A missing or out-of-order
sequence flushes that stream's current VAD segment so audio on opposite sides
of a gap is never joined into one utterance.

### 4. Stream controls

End an utterance while retaining its language-routing state:

```json
{"type":"stream_idle","stream_id":17}
```

Report dropped/discontinuous audio:

```json
{"type":"gap","stream_id":17,"reason":"bridge_queue_overflow"}
```

Update display identity without reopening audio:

```json
{
  "type":"identity_update",
  "stream_id":17,
  "speaker":"Alice New",
  "metadata":{"channel_id":"999"}
}
```

Close a stream:

```json
{"type":"stream_end","stream_id":17,"reason":"left"}
```

Request service status:

```json
{"type":"status"}
```

## Result events

The bridge receives the same partial/final/refine events as the dashboard,
with source identity attached:

```json
{
  "type": "final",
  "stream_id": 17,
  "utterance_id": "17-42",
  "source": "discord",
  "speaker_id": "789",
  "speaker": "Alice",
  "metadata": {"guild_id":"123","channel_id":"456"},
  "lang": "ja",
  "text": "こんにちは。",
  "latency_ms": 110,
  "tier": "rz",
  "emitted_at_ms": 1788100002510
}
```

Partial results use the next final's `utterance_id`. ASR jobs are globally
scheduled with `final > partial > refine` priority. When overloaded, stale
partials may be skipped; finals are not intentionally dropped.

A recoverable inference failure emits a `stream_error` event with the same
stream identity plus `error` and `message`. The affected speaker worker stays
alive and continues with subsequent audio instead of silently disappearing.

## Backpressure and failure behavior

- Each stream has a bounded realtime input buffer (two seconds by default).
- On overflow, queued stale audio is discarded, a VAD flush is inserted, and
  the newest audio is retained.
- Unknown or closed stream IDs are rejected; they are never guessed from the
  most recent speaker.
- Disconnecting the bridge ends and flushes every stream in that epoch.
- Missing optional fallback models do not close a stream. The router tries a
  language-compatible installed model and otherwise skips only that result.
- PCM and credentials are not written to logs by the v2 service.

# Japanese / English / Korean live translation (Hy-MT2)

`--translate` enables full three-language translation for finalized ASR lines:

| Detected source | Translation outputs |
|---|---|
| `ja` | `en`, `ko` |
| `en` | `ja`, `ko` |
| `ko` | `ja`, `en` |

The implementation uses Tencent `Hy-MT2-1.8B` Q4_K_M behind an
OpenAI-compatible `llama-server`. One server is shared by every Discord
speaker stream; the ASR process sends the two target languages as independent
jobs so llama.cpp can batch them on the GPU. Languages outside `ja/en/ko` are
left untranslated.

## Setup

Download the optional 1.13 GB GGUF model:

```bash
uv run python scripts/download_models.py --hymt
```

Install a current CUDA-enabled llama.cpp build and ensure `llama-server` is on
`PATH`. The llama.cpp build must recognize the `hunyuan_v1_dense` GGUF
architecture; old builds fail while loading Hy-MT2.

When `--translate` is supplied, hayamimi first probes
`http://127.0.0.1:18081/health`. If no server is present, it starts one from
`models/Hy-MT2-1.8B-Q4_K_M.gguf`, offloads all layers to the GPU, and stops
that child process during shutdown. An already-running local or remote server
is reused and is not stopped by hayamimi.

Single-stream input:

```bash
uv run python scripts/realtime_transcribe.py --serve --translate
```

Multiplexed Discord bridge input:

```bash
uv run --with-requirements requirements-gpu.txt \
  python scripts/multi_realtime_transcribe.py \
  --serve --ja-provider cuda --translate
```

Use an existing server or a different binary/model with:

```bash
uv run python scripts/multi_realtime_transcribe.py --translate \
  --translation-url http://127.0.0.1:18081 \
  --translation-model /path/to/Hy-MT2-1.8B-Q4_K_M.gguf \
  --llama-server /path/to/llama-server
```

`--translation-workers 2` is the default: one request can generate each of the
two target languages concurrently. `--translation-timeout 10` limits each
request. If the service fails, times out, returns an empty result, or changes
an ASCII digit sequence, hayamimi keeps the source line and emits no incorrect
translation for that target.

## Events and ordering

Each translation SSE/WebSocket event carries the source language, target
language, and the final's `utterance_id`:

```json
{
  "type": "translation",
  "source_lang": "ko",
  "lang": "en",
  "text": "The deployment must be completed by Friday at 5 p.m.",
  "utterance_id": "12-48",
  "stream_id": 12,
  "speaker_id": "discord-user-id"
}
```

The dashboard keys translation rows by stream and utterance rather than
attaching them to the latest global final. This prevents a slow translation
from one speaker appearing under another speaker's simultaneous line.

## Local smoke result

The official Q4_K_M model produced natural outputs for `ja->en`, `en->ja`,
`ko->ja`, `ja->ko`, and `en->ko` spot checks and preserved `500万円`,
`午後3時`, and `500만 엔`. With the model resident on this machine, CPU-only
generation took about 1.35-1.80 seconds per short line. Production use should
therefore use GPU offload; GPU latency and corpus-level ja/en/ko quality must
be recorded separately on the deployed CUDA llama.cpp build.

Model source and license: [tencent/Hy-MT2-1.8B-GGUF](https://huggingface.co/tencent/Hy-MT2-1.8B-GGUF), Apache-2.0.

The old explicit target-list form (for example `--translate zh,es`) remains a
Japanese-only compatibility path using FuguMT/M2M-100. Bare `--translate`,
`--translate tri`, and `--translate ja,en,ko` select Hy-MT2.

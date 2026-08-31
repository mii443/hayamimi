# High-accuracy Japanese ASR on NVIDIA GPU

hayamimi can run the full-FP32 Japanese-only
[`reazon-research/reazonspeech-k2-v2`](https://huggingface.co/reazon-research/reazonspeech-k2-v2)
model through sherpa-onnx's CUDA provider. The CPU path uses the publisher's
accuracy-preserving INT8-encoder + FP32-decoder variant of the same model.

## Setup

Use an NVIDIA driver compatible with CUDA 12, then install the optional GPU
runtime into hayamimi's virtual environment:

```bash
uv pip install --python .venv/bin/python -r requirements-gpu.txt
uv run python scripts/download_models.py
```

No manual `LD_LIBRARY_PATH` is required. hayamimi preloads the CUDA libraries
installed below the virtual environment before constructing the recognizer.

Start the Discord multiplex service with GPU Japanese ASR:

```bash
export HAYAMIMI_BRIDGE_SECRET='the same random secret used by rstt'
uv run python scripts/multi_realtime_transcribe.py --serve --ja-provider cuda
```

The first startup performs CUDA graph/kernel optimization during model warmup.
On the tested RTX 5080 this took about one minute. It is paid before the ingest
server starts, so the first Discord utterance does not incur the delay.

## Measured trade-off

Fifteen human-spoken Japanese TV clips (85.6 seconds total) were decoded with
language fixed to Japanese. CER is micro-averaged after punctuation and
whitespace normalization.

| Japanese model | Provider | CER | Mean decode latency after warmup |
|---|---:|---:|---:|
| Previous bilingual ReazonSpeech all-INT8 | CPU | 6.87% | 75 ms |
| ReazonSpeech-k2-v2 INT8-encoder/FP32-decoder | CPU | 5.15% | 120 ms |
| ReazonSpeech-k2-v2 full FP32 | CUDA (RTX 5080) | 5.15% | 175 ms |
| Whisper large-v3 FP16, beam 1 | CUDA (RTX 5080) | 25.43% | 494 ms |

The k2-v2 paths reduce errors by about 25% relative to the previous Japanese
route. GPU is not faster than CPU for these short utterances, but it moves the
largest Japanese model off the CPU and remains comfortably realtime. Whisper
large-v3 was rejected for live Japanese: the measured set regressed sharply
and short clips produced hallucinated phrases.

For the lowest latency with the same measured CER, omit `--ja-provider cuda`
and use the default CPU path.

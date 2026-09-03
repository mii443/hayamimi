# High-accuracy Korean ASR on NVIDIA GPU

`--ko-provider cuda` uses the FP16 CTranslate2 conversion of
[`openai/whisper-large-v3`](https://huggingface.co/openai/whisper-large-v3)
for finalized Korean utterances. Refined subtitles merge those high-accuracy
finals without a second decode. SenseVoice remains active for language
identification, low-latency partial text, and fallback if a Korean decode is
empty. This avoids running the 1.55B model every 500 ms while a speaker is
still talking.

## Setup

Download the optional 2.9 GB model and launch through the GPU dependency
overlay:

```bash
uv run python scripts/download_models.py --ko-whisper

export HAYAMIMI_BRIDGE_SECRET='the same random secret used by rstt'
uv run --with-requirements requirements-gpu.txt \
  python scripts/multi_realtime_transcribe.py \
  --host 0.0.0.0 --port 8766 --serve \
  --ja-provider cuda --ko-provider cuda --translate
```

The download script pins the verified conversion revision and checks its core
files. The model is loaded from `models/faster-whisper-large-v3`; production
does not depend on a mutable Hugging Face cache or download weights at
startup. The CUDA overlay supplies faster-whisper, cuBLAS, cuDNN, and NVRTC.
The existing library preload makes these pip-installed shared libraries
visible without a manual `LD_LIBRARY_PATH`.

## Measured result

The comparison used the existing 12 native-speaker Korean FLEURS clips in
`testdata/eval_real_zhko` (about 3-9 seconds each). Both systems received the
same 16 kHz audio and known `ko` language. CER is micro-averaged after NFKC
normalization and removing punctuation and whitespace.

Reproduce the GPU row with the pinned model revision downloaded above:

```bash
uv run --group dev python scripts/make_realset_zhko.py --skip-eval
uv run --with-requirements requirements-gpu.txt python scripts/eval_ko_gpu.py
```

The first command downloads the 12 Korean and 12 Chinese FLEURS clips into
the git-ignored `testdata/eval_real_zhko` directory. The evaluator refuses to
report a score unless all 12 Korean clips use the `wk` GPU tier.

| Korean final model | Provider | CER | Mean RTF |
|---|---:|---:|---:|
| SenseVoice small INT8 | CPU | 9.26% | 0.027 |
| **Whisper large-v3 FP16, beam 5** | **RTX 5080 CUDA** | **6.81%** | **0.091** |

The CUDA route reduces character errors by about **26% relative** while
remaining about 11x faster than real time after startup warmup. It correctly recovered several
phrases SenseVoice changed or dropped, and also handled the two quiet clips
that made the dedicated Korean Zipformer return empty output.

The test set is small and read speech rather than noisy Discord conversation,
so the old SenseVoice route remains the default for CPU-only installations.
Use `--ko-provider cuda` when accuracy is preferred and NVIDIA GPU capacity is
available.

Model loading and the first CUDA kernels take several seconds on the tested
machine. When `--ko-provider cuda` is selected, hayamimi pays that cost during
startup before opening the ingest service rather than on the first Korean
utterance. The CUDA recognizer is pinned outside the CPU `--max-resident` LRU
budget, so switching through other languages does not discard that warmup.
If CUDA inference later fails (for example, a driver reset or out-of-memory
condition), that utterance is retried with SenseVoice and the CUDA route stays
disabled until the process is restarted.

Refinement merges the already-high-accuracy Korean finals without running a
second long large-v3 decode. This keeps one speaker's 25-second refinement
from blocking final subtitles queued for every other Discord speaker.

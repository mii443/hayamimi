#!/usr/bin/env python
"""Reproduce Korean CUDA route CER/RTF on the local FLEURS evaluation set.

Run after downloading the optional model:

  uv run --with-requirements requirements-gpu.txt \
    python scripts/eval_ko_gpu.py
"""
from __future__ import annotations

import json
import os
import time

import numpy as np
import soundfile as sf

from asr_engine import RoutedASR
from eval_accuracy import levenshtein, normalize_ja


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "testdata", "eval_real_zhko")


def main() -> None:
    manifest_path = os.path.join(DATA_DIR, "manifest.json")
    if not os.path.isfile(manifest_path):
        raise SystemExit(
            "Korean evaluation data is missing. Build it first with: "
            "uv run --group dev python scripts/make_realset_zhko.py --skip-eval"
        )
    started = time.perf_counter()
    asr = RoutedASR(
        warmup=True,
        preload=False,
        punctuate=False,
        forced_lang="ko",
        ko_provider="cuda",
    )
    print(f"startup_s={time.perf_counter() - started:.3f}")

    with open(manifest_path, encoding="utf-8") as source:
        entries = [entry for entry in json.load(source) if entry["lang"] == "ko"]
    if len(entries) != 12:
        raise SystemExit(f"expected 12 Korean evaluation clips, found {len(entries)}")

    total_distance = 0
    total_chars = 0
    rtfs = []
    for entry in entries:
        path = os.path.join(DATA_DIR, entry["wav"])
        samples, sample_rate = sf.read(path, dtype="float32", always_2d=False)
        if samples.ndim > 1:
            samples = np.mean(samples, axis=1, dtype=np.float32)
        duration = len(samples) / sample_rate
        started = time.perf_counter()
        result = asr.transcribe(samples, sample_rate, known_lang="ko", live=False)
        elapsed = time.perf_counter() - started
        if result["tier"] != "wk":
            raise RuntimeError(
                f"GPU route fell back to {result['tier']!r}; refusing to report it as wk"
            )

        reference = normalize_ja(entry["ref"])
        hypothesis = normalize_ja(result["text"])
        distance = levenshtein(reference, hypothesis)
        total_distance += distance
        total_chars += len(reference)
        rtf = elapsed / duration
        rtfs.append(rtf)
        print(
            f"{entry['wav']} CER={distance / len(reference):.4f} "
            f"RTF={rtf:.4f} tier={result['tier']} hyp={result['text']}"
        )

    print(
        f"aggregate CER={total_distance / total_chars:.4f} "
        f"mean_RTF={sum(rtfs) / len(rtfs):.4f} n={len(entries)}"
    )


if __name__ == "__main__":
    main()

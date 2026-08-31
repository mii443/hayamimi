"""Run hayamimi as a multiplexed realtime ASR service for audio bridges."""
import argparse
import os
import sys
import threading

from asr_engine import RoutedASR
from multi_stream import MultiStreamManager
from realtime_transcribe import DEFAULT_LLAMA_SERVER, MODELS_DIR, build_translation_backend
from subtitle_server import SubtitleServer
from ws_ingest_v2 import INGEST_V2_PATH, MultiplexIngestServer


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8766)
    ap.add_argument("--serve", type=int, nargs="?", const=8833, default=None,
                    metavar="PORT")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--ja-provider", choices=["cpu", "cuda"], default="cpu",
                    help="Japanese ReazonSpeech-k2-v2 execution provider. cuda uses the "
                         "full FP32 model and requires requirements-gpu.txt; cpu uses the "
                         "accuracy-preserving int8-encoder/FP32-decoder model.")
    ap.add_argument("--max-streams", type=int, default=32)
    ap.add_argument("--max-resident", type=int, default=3)
    ap.add_argument("--min-silence", type=float, default=0.35)
    ap.add_argument("--max-speech", type=float, default=12.0)
    ap.add_argument("--no-partial", action="store_true")
    ap.add_argument("--no-refine", action="store_true")
    ap.add_argument("--mode", choices=["single", "balanced", "fast"], default="balanced")
    ap.add_argument("--lang", default=None)
    ap.add_argument("--hotwords", default="")
    ap.add_argument("--replace", default="")
    ap.add_argument("--translate", action="store_true",
                    help="translate ja/en/ko finals into the other two languages with Hy-MT2")
    ap.add_argument("--translation-url", default="http://127.0.0.1:18081")
    ap.add_argument("--translation-timeout", type=float, default=10.0)
    ap.add_argument("--translation-model",
                    default=os.path.join(MODELS_DIR, "Hy-MT2-1.8B-Q4_K_M.gguf"))
    ap.add_argument("--llama-server", default=DEFAULT_LLAMA_SERVER)
    ap.add_argument("--translation-workers", type=int, default=2)
    ap.add_argument("--bridge-secret-env", default="HAYAMIMI_BRIDGE_SECRET")
    args = ap.parse_args()

    if args.mode == "single" and not args.lang:
        ap.error("--mode single requires --lang CODE")
    secret = os.environ.get(args.bridge_secret_env, "")
    if not secret:
        ap.error(f"environment variable {args.bridge_secret_env} must contain a bridge secret")

    event_hub = SubtitleServer(port=args.serve or 8833)
    if args.serve:
        event_hub.start()
        print(f"dashboard: http://127.0.0.1:{args.serve}/dashboard", file=sys.stderr)

    translation_backend = None
    managed_translation_server = None
    if args.translate:
        print("loading ja/en/ko translation backend...", file=sys.stderr)
        try:
            translation_backend, managed_translation_server = build_translation_backend(
                "tri", args.translation_url, args.translation_timeout,
                args.translation_model, args.llama_server,
                parallel=max(args.translation_workers, 1))
        except Exception as exc:
            ap.error(f"translation startup failed: {exc}")

    print("loading shared ASR models...", file=sys.stderr)
    asr = RoutedASR(threads=args.threads,
                    max_resident=args.max_resident if args.max_resident > 0 else None,
                    hotwords_file=args.hotwords, replace_file=args.replace,
                    lid_switch_confirm=1 if args.mode == "fast" else 2,
                    dual_confirm=args.mode != "fast",
                    forced_lang=args.lang if args.mode == "single" else None,
                    ja_provider=args.ja_provider)
    asr.min_switch_s = 0.0 if args.mode == "fast" else 2.0
    manager = MultiStreamManager(
        asr, event_hub, max_streams=args.max_streams,
        min_silence=args.min_silence, max_speech=args.max_speech,
        partial=not args.no_partial, refine=not args.no_refine,
        translation_backend=translation_backend,
        translation_workers=max(args.translation_workers, 1))
    server = MultiplexIngestServer(args.host, args.port, manager, secret,
                                   event_hub=event_hub,
                                   max_streams=args.max_streams).start()
    print(f"multiplex ingest: ws://{args.host}:{args.port}{INGEST_V2_PATH}",
          file=sys.stderr)

    stop = threading.Event()
    try:
        stop.wait()
    except KeyboardInterrupt:
        pass
    finally:
        manager.close()
        if managed_translation_server is not None:
            managed_translation_server.stop()


if __name__ == "__main__":
    main()

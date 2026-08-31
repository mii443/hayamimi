"""Run hayamimi as a multiplexed realtime ASR service for audio bridges."""
import argparse
import os
import sys
import threading

from asr_engine import RoutedASR
from multi_stream import MultiStreamManager
from subtitle_server import SubtitleServer
from ws_ingest_v2 import INGEST_V2_PATH, MultiplexIngestServer


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8766)
    ap.add_argument("--serve", type=int, nargs="?", const=8833, default=None,
                    metavar="PORT")
    ap.add_argument("--threads", type=int, default=4)
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

    print("loading shared ASR models...", file=sys.stderr)
    asr = RoutedASR(threads=args.threads,
                    max_resident=args.max_resident if args.max_resident > 0 else None,
                    hotwords_file=args.hotwords, replace_file=args.replace,
                    lid_switch_confirm=1 if args.mode == "fast" else 2,
                    dual_confirm=args.mode != "fast",
                    forced_lang=args.lang if args.mode == "single" else None)
    asr.min_switch_s = 0.0 if args.mode == "fast" else 2.0
    manager = MultiStreamManager(
        asr, event_hub, max_streams=args.max_streams,
        min_silence=args.min_silence, max_speech=args.max_speech,
        partial=not args.no_partial, refine=not args.no_refine)
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


if __name__ == "__main__":
    main()

"""gputaskman - an htop for what's *inside* your GPU."""

from __future__ import annotations

import argparse
import math
import sys
import time


def main() -> None:
    ap = argparse.ArgumentParser(prog="gputaskman", description="What's inside your GPU: per-process VRAM, engines, and LLM weights/KV-cache breakdown.")
    ap.add_argument("--once", action="store_true", help="print one snapshot and exit (no TUI)")
    ap.add_argument("--json", action="store_true", help="print one JSON snapshot and exit (implies --once)")
    ap.add_argument("--gpu", type=int, default=0, help="GPU index (default 0)")
    ap.add_argument("-i", "--interval", type=float, default=1.0, help="refresh seconds (default 1)")
    args = ap.parse_args()
    if args.gpu < 0:
        ap.error("--gpu must be a non-negative index")
    if not math.isfinite(args.interval) or args.interval <= 0:
        ap.error("--interval must be a finite number greater than zero")

    if sys.platform != "win32":
        sys.exit("gputaskman currently supports Windows (WDDM) only.")

    if args.gpu:
        from .collector import enumerate_adapters, select_adapter

        try:
            select_adapter(enumerate_adapters(), args.gpu)
        except ValueError as error:
            ap.error(str(error))

    if args.once or args.json:
        from .collector import Collector
        from .report import dump

        try:
            c = Collector(args.gpu, scan_budget=None)  # one-shot: identify everything in this sample
        except ValueError as error:
            ap.error(str(error))
        try:
            time.sleep(1.0)  # engine utilization is a rate; needs two samples
            print(dump(c.sample(), args.json))
        finally:
            c.close()
        return

    from .app import GpuTaskManager

    GpuTaskManager(gpu_index=args.gpu, interval=args.interval).run()

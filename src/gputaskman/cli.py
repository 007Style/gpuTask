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
    ap.add_argument("--no-host", action="store_true",
                    help="WSL: don't read Windows host GPU counters through interop (WSL VM total, Windows apps)")
    ap.add_argument("-i", "--interval", type=float, default=1.0, help="refresh seconds (default 1)")
    args = ap.parse_args()
    if args.gpu < 0:
        ap.error("--gpu must be a non-negative index")
    if not math.isfinite(args.interval) or args.interval <= 0:
        ap.error("--interval must be a finite number greater than zero")

    if sys.platform not in ("win32", "darwin") and not sys.platform.startswith("linux"):
        sys.exit("gputaskman supports Windows (WDDM), Linux/WSL (NVML) and macOS (Apple Silicon) only.")

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
            c = Collector(args.gpu, scan_budget=None, host=not args.no_host)  # one-shot: identify everything now
        except ValueError as error:
            ap.error(str(error))
        try:
            if c.host is not None:
                c.host.wait(15.0)  # WSL: the first Windows counter sample takes several seconds
            if hasattr(c.counters, "warm_up"):
                c.counters.warm_up(15.0)  # macOS: per-process GPU memory comes from a background footprint pass
            time.sleep(1.0)  # engine utilization is a rate; needs two samples
            print(dump(c.sample(), args.json))
        finally:
            c.close()
        return

    from .app import GpuTaskManager

    GpuTaskManager(gpu_index=args.gpu, interval=args.interval, host=not args.no_host).run()

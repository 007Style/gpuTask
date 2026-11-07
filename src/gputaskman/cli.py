"""gputaskman - an htop for what's *inside* your GPU."""

from __future__ import annotations

import argparse
import math
import sys
import time


def diff_main(argv: list[str]) -> None:
    import json

    ap = argparse.ArgumentParser(prog="gputaskman diff",
                                 description="Compare two --json snapshots, for example before and after loading a model.")
    ap.add_argument("before", help="earlier --json snapshot")
    ap.add_argument("after", help="later --json snapshot")
    ap.add_argument("--json", action="store_true", help="print the comparison as JSON")
    args = ap.parse_args(argv)

    from .diff import diff, load, to_text

    try:
        result = diff(load(args.before), load(args.after))
    except (OSError, ValueError) as error:
        ap.error(str(error))
    print(json.dumps(result, indent=2) if args.json else to_text(result))


def main() -> None:
    if len(sys.argv) > 1 and sys.argv[1] == "diff":
        return diff_main(sys.argv[2:])
    ap = argparse.ArgumentParser(prog="gputaskman", description="What's inside your GPU: per-process VRAM, engines, and LLM weights/KV-cache breakdown.",
                                 epilog="gputaskman diff BEFORE.json AFTER.json compares two --json snapshots.")
    ap.add_argument("--once", action="store_true", help="print one snapshot and exit (no TUI)")
    ap.add_argument("--json", action="store_true", help="print one JSON snapshot and exit (implies --once)")
    ap.add_argument("--gpu", type=int, help="show only this GPU (TUI default: all; one-shot default: 0)")
    ap.add_argument("--list-gpus", action="store_true", help="list selectable GPU adapters and exit")
    ap.add_argument("--no-host", action="store_true",
                    help="WSL: don't read Windows host GPU counters through interop (WSL VM total, Windows apps)")
    ap.add_argument("-i", "--interval", type=float, default=1.0, help="refresh seconds (default 1)")
    args = ap.parse_args()
    if args.gpu is not None and args.gpu < 0:
        ap.error("--gpu must be a non-negative index")
    if not math.isfinite(args.interval) or args.interval <= 0:
        ap.error("--interval must be a finite number greater than zero")

    if sys.platform not in ("win32", "darwin") and not sys.platform.startswith("linux"):
        sys.exit("gputaskman supports Windows (WDDM), Linux/WSL (NVML, or amdgpu DRM) and macOS (Apple Silicon) only.")

    if args.list_gpus:
        from .collector import enumerate_adapters, ordered_adapters

        for i, adapter in enumerate(ordered_adapters(enumerate_adapters())):
            print(f"{i}: {adapter.name} ({adapter.vendor}, LUID 0x{adapter.luid & 0xffffffffffffffff:016x})")
        return

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
            c = Collector(args.gpu or 0, scan_budget=None, host=not args.no_host)  # one-shot: identify everything now
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

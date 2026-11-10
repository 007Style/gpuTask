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


def _check_platform() -> None:
    if sys.platform not in ("win32", "darwin") and not sys.platform.startswith("linux"):
        sys.exit("gputaskman supports Windows (WDDM), Linux/WSL (NVML, or amdgpu/Intel DRM) and macOS (Apple Silicon) only.")


def _check_gpu(ap: argparse.ArgumentParser, index: int | None) -> None:
    if index:
        from .collector import enumerate_adapters, select_adapter

        try:
            select_adapter(enumerate_adapters(), index)
        except ValueError as error:
            ap.error(str(error))


def record_main(argv: list[str]) -> None:
    ap = argparse.ArgumentParser(prog="gputaskman record",
                                 description="Append one --json snapshot per interval to a JSON Lines file, "
                                             "for example to keep what a long run did to the GPU.")
    ap.add_argument("-o", "--output", required=True, help="file to append to, one --json snapshot per line")
    ap.add_argument("--gpu", type=int, default=0, help="GPU to record (default 0; gputaskman --list-gpus shows the indexes)")
    ap.add_argument("-i", "--interval", type=float, default=1.0, help="seconds between samples (default 1)")
    ap.add_argument("--duration", type=float, help="stop after this many seconds (default: until Ctrl+C)")
    ap.add_argument("--no-host", action="store_true",
                    help="WSL: don't read Windows host GPU counters through interop (WSL VM total, Windows apps)")
    ap.add_argument("--no-system", action="store_true", help="don't sample system RAM/CPU: no system section in the records")
    args = ap.parse_args(argv)
    if args.gpu < 0:
        ap.error("--gpu must be a non-negative index")
    if not math.isfinite(args.interval) or args.interval <= 0:
        ap.error("--interval must be a finite number greater than zero")
    if args.duration is not None and (not math.isfinite(args.duration) or args.duration <= 0):
        ap.error("--duration must be a finite number greater than zero")
    _check_platform()
    _check_gpu(ap, args.gpu)

    from .record import append_to, run, source

    try:
        out = append_to(args.output)
    except OSError as error:
        ap.error(f"cannot write {args.output}: {error}")
    count = 0
    try:
        with out, source(args.gpu, host=not args.no_host, system=not args.no_system) as sample:
            span = f" for {args.duration:g} s" if args.duration is not None else ""
            print(f"Recording GPU {args.gpu} every {args.interval:g} s{span} to {args.output}; "
                  "press Ctrl+C to stop.", file=sys.stderr)
            count = run(sample, out, interval=args.interval, duration=args.duration)
    except KeyboardInterrupt:
        pass  # Ctrl+C while starting up: nothing was recorded yet
    print(f"Recorded {count} sample(s) to {args.output}", file=sys.stderr)


def main() -> None:
    if len(sys.argv) > 1 and sys.argv[1] == "diff":
        return diff_main(sys.argv[2:])
    if len(sys.argv) > 1 and sys.argv[1] == "record":
        return record_main(sys.argv[2:])
    ap = argparse.ArgumentParser(prog="gputaskman", description="What's inside your GPU: per-process VRAM, engines, and LLM weights/KV-cache breakdown.",
                                 epilog="gputaskman diff BEFORE.json AFTER.json compares two --json snapshots; "
                                        "gputaskman record -o FILE appends one --json snapshot per interval to FILE.")
    ap.add_argument("--once", action="store_true", help="print one snapshot and exit (no TUI)")
    ap.add_argument("--json", action="store_true", help="print one JSON snapshot and exit (implies --once)")
    ap.add_argument("--gpu", type=int, help="show only this GPU (TUI default: all; one-shot default: 0)")
    ap.add_argument("--list-gpus", action="store_true", help="list selectable GPU adapters and exit")
    ap.add_argument("--no-host", action="store_true",
                    help="WSL: don't read Windows host GPU counters through interop (WSL VM total, Windows apps)")
    ap.add_argument("--no-system", action="store_true",
                    help="don't sample system RAM/CPU: no TUI panel or RAM/CPU%% columns, no system section in --once/--json")
    ap.add_argument("-i", "--interval", type=float, default=1.0, help="refresh seconds (default 1)")
    args = ap.parse_args()
    if args.gpu is not None and args.gpu < 0:
        ap.error("--gpu must be a non-negative index")
    if not math.isfinite(args.interval) or args.interval <= 0:
        ap.error("--interval must be a finite number greater than zero")

    _check_platform()

    if args.list_gpus:
        from .collector import enumerate_adapters, ordered_adapters

        for i, adapter in enumerate(ordered_adapters(enumerate_adapters())):
            print(f"{i}: {adapter.name} ({adapter.vendor}, LUID 0x{adapter.luid & 0xffffffffffffffff:016x})")
            if adapter.identity_status:
                print(f"   {adapter.identity_status}")
        return

    _check_gpu(ap, args.gpu)

    if args.once or args.json:
        from .collector import Collector
        from .host import HostSampler
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
            sampler, system, system_error = None, None, None
            if not args.no_system:
                try:
                    sampler = HostSampler()
                    sampler.sample()  # per-process CPU% is a rate too; this primes it
                except Exception as error:  # system RAM/CPU is extra: its failure must not cost the GPU report
                    sampler, system_error = None, f"{type(error).__name__}: {error}"
            time.sleep(1.0)  # engine utilization is a rate; needs two samples
            snapshot = c.sample()
            if sampler:
                try:
                    system = sampler.sample()
                except Exception as error:
                    system_error = f"{type(error).__name__}: {error}"
            print(dump(snapshot, args.json, system, system_error))
        finally:
            c.close()
        return

    from .app import GpuTaskManager

    GpuTaskManager(gpu_index=args.gpu, interval=args.interval, host=not args.no_host, system=not args.no_system).run()

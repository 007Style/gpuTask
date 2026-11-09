"""gputaskman record: append one --json snapshot per interval to a JSON Lines file."""

from __future__ import annotations

import json
import os
import time
from contextlib import ExitStack, contextmanager

from .report import to_json_dict


def make_sample(collect, system=None):
    """A callable returning one record: exactly the object --json prints, so every saved line is a normal snapshot."""

    def sample() -> dict:
        snapshot = collect()
        host, error = None, None
        if system is not None:
            try:
                host = system()
            except Exception as exc:  # system RAM/CPU is extra: its failure must not cost the GPU record
                error = f"{type(exc).__name__}: {exc}"
        return to_json_dict(snapshot, host, error)

    return sample


def run(sample, out, *, interval: float, duration: float | None = None,
        clock=time.monotonic, sleep=time.sleep) -> int:
    """Write one compact JSON line per interval, flushed as it goes; return how many were written.

    Stops after duration seconds, or on Ctrl+C. An interrupted sample writes nothing, so every line is complete.
    A slow sample delays the next one instead of triggering a burst to catch up.
    """
    count = 0
    start = due = clock()
    try:
        while duration is None or due - start < duration:
            if (wait := due - clock()) > 0:
                sleep(wait)
            out.write(json.dumps(sample(), default=str, separators=(",", ":")) + "\n")
            out.flush()
            count += 1
            due = max(due + interval, clock())
    except KeyboardInterrupt:
        pass
    return count


def append_to(path: str):
    """Open path for appending (never overwriting); a record never glues onto a last line that was cut short."""
    needs_newline = False
    try:
        with open(path, "rb") as existing:
            existing.seek(-1, os.SEEK_END)
            needs_newline = existing.read(1) != b"\n"
    except OSError:
        pass  # missing, empty, or not seekable: nothing to protect
    out = open(path, "a", encoding="utf-8", newline="\n")
    if needs_newline:
        out.write("\n")
    return out


@contextmanager
def source(gpu: int, *, host: bool = True, system: bool = True):
    """Sample the way the TUI does and yield a sample() callable.

    Scanning is budgeted like the TUI's, and on Windows a GPU that is disabled and re-enabled is rediscovered:
    the records in between carry null measurements and the reason, never zeros.
    """
    from .collector import BACKEND, Collector, enumerate_adapters
    from .host import HostSampler

    sampler = None

    def read_system():
        nonlocal sampler
        if sampler is None:
            sampler = HostSampler()
        return sampler.sample()

    with ExitStack() as stack:
        if BACKEND == "wddm":
            from .recovery import LiveCollectors

            live = LiveCollectors(gpu, collector_factory=Collector, discover=enumerate_adapters)
            stack.callback(live.close)

            def collect():
                return live.sample()[0]
        else:
            collector = Collector(gpu, host=host)
            stack.callback(collector.close)
            if collector.host is not None:
                collector.host.wait(15.0)  # WSL: the first Windows counter sample takes several seconds
            if hasattr(collector.counters, "warm_up"):
                collector.counters.warm_up(15.0)  # macOS: per-process GPU memory comes from a background pass
            collect = collector.sample
        if system:
            try:
                read_system()  # per-process CPU% is a rate too; this primes it
            except Exception:
                pass  # each record carries the reason if it still fails
        time.sleep(1.0)  # engine utilization is a rate; needs two samples
        yield make_sample(collect, read_system if system else None)

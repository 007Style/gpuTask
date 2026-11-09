"""System RAM and CPU, alongside the GPU view.

Every number here is reported by the OS (psutil, or NtQuerySystemInformation for processes on Windows):
measured, not inferred. Unknown stays None: a process we may not inspect has ``rss=None`` /
``cpu_pct=None``, never 0. These are system-memory figures: never GPU bytes, never part of
``Snapshot.attributed``.
"""

from __future__ import annotations

import struct
import sys
import time
from dataclasses import dataclass, field

import psutil


SAME_PROCESS_SECS = 2.0  # psutil and the NT list give one creation time to ~1 microsecond; this absorbs clock steps


@dataclass
class HostProc:
    pid: int
    name: str
    rss: int | None          # resident / working set bytes
    private: int | None      # Windows private bytes (commit charged to the process); None elsewhere
    cpu_pct: float | None    # share of the WHOLE machine (0-100), like Task Manager; None until a second reading
    created: float | None = None  # unix seconds; tells one process from the next that reuses its pid


@dataclass
class HostSample:
    cpu_pct: float                  # whole machine, 0-100
    per_core: list[float]
    logical: int
    physical: int | None
    ram_total: int
    ram_used: int                   # total - available (what Task Manager calls "In use", approximately)
    ram_available: int
    swap_total: int
    swap_used: int
    procs: list[HostProc] = field(default_factory=list)
    denied: int = 0                 # processes whose memory we were not allowed to read
    cost_ms: float = 0.0

    def proc(self, pid: int | None, created: float = 0.0) -> HostProc | None:
        """This pid's reading. A caller that knows when its process started (0.0: it doesn't) gets only that
        process: a pid can be reused between the GPU sample and this one."""
        p = self._by_pid.get(pid) if pid is not None else None
        if p and created and p.created is not None and abs(p.created - created) > SAME_PROCESS_SECS:
            return None
        return p

    def __post_init__(self) -> None:
        self._by_pid = {p.pid: p for p in self.procs}

    def top(self, key: str, n: int = 5) -> list[HostProc]:
        return sorted((p for p in self.procs if getattr(p, key) is not None), key=lambda p: -getattr(p, key))[:n]


class HostSampler:
    """Per-process CPU% is a rate: the previous reading is kept per (pid, creation time), because pids are reused."""

    def __init__(self) -> None:
        self._logical = psutil.cpu_count() or 1
        self._physical = psutil.cpu_count(logical=False)
        self._rated: set[tuple[int, float]] = set()  # psutil path: processes with a previous CPU reading
        self._last_cpu: dict[tuple[int, float], int] = {}  # NT path: CPU time (100 ns) at the previous sample
        self._last_wall = 0.0
        if _nt_query is not None:
            _check_nt_layout(_nt_query())  # raises rather than show numbers parsed from a layout we don't recognise
        psutil.cpu_percent(percpu=True)  # prime the system-wide rate

    def sample(self) -> HostSample:
        t0 = time.perf_counter()
        per_core = psutil.cpu_percent(percpu=True)
        vm, sw = psutil.virtual_memory(), psutil.swap_memory()
        if _nt_query is not None:
            return self._finish(t0, per_core, vm, sw, self._sample_nt(), 0)
        procs, denied, seen = [], 0, set()
        for p in psutil.process_iter():  # psutil keeps these objects between calls and replaces one whose pid was reused
            try:
                with p.oneshot():
                    name = p.name()
                    mi = p.memory_info()
                    cpu = p.cpu_percent(None)
                    ident = (p.pid, p.create_time())
            except psutil.NoSuchProcess:
                continue
            except psutil.AccessDenied:
                denied += 1
                procs.append(HostProc(p.pid, _safe_name(p), None, None, None))
                continue
            seen.add(ident)
            procs.append(HostProc(p.pid, name, mi.rss, getattr(mi, "private", None),
                                  cpu / self._logical if ident in self._rated else None, ident[1]))
        self._rated = seen
        return self._finish(t0, per_core, vm, sw, procs, denied)

    def _sample_nt(self) -> list[HostProc]:
        now = time.perf_counter()
        rows = _nt_query()
        wall = now - self._last_wall if self._last_wall else None
        procs, last = [], {}
        for pid, (name, ws, private, cpu100ns, created) in rows.items():
            if pid == 0:  # System Idle Process: its "CPU" is idle time
                continue
            ident = (pid, created)
            last[ident] = cpu100ns
            prev = self._last_cpu.get(ident)  # a reused pid has a new creation time: no previous reading
            cpu = None
            if wall and prev is not None and cpu100ns >= prev:
                cpu = (cpu100ns - prev) / 1e7 / wall / self._logical * 100
            procs.append(HostProc(pid, name, ws, private, cpu, created))
        self._last_cpu, self._last_wall = last, now
        return procs

    def _finish(self, t0, per_core, vm, sw, procs, denied) -> HostSample:
        return HostSample(
            cpu_pct=sum(per_core) / len(per_core) if per_core else 0.0, per_core=per_core,
            logical=self._logical, physical=self._physical,
            ram_total=vm.total, ram_used=vm.total - vm.available, ram_available=vm.available,
            swap_total=sw.total, swap_used=sw.used, procs=procs, denied=denied,
            cost_ms=(time.perf_counter() - t0) * 1000)


# Windows: one NtQuerySystemInformation(SystemProcessInformation) call returns every process's working set,
# private bytes, creation time and CPU time. psutil opens each process instead: measured 2026-11-08 on a
# 995-process Windows 11 box, ~6 s per pass versus ~28 ms here. Same numbers: WorkingSetSize = psutil rss,
# PagefileUsage = psutil private. No process handle needed, so no AccessDenied. The structure is read by
# byte offset (winternl.h, x64): HostSampler checks those offsets against our own process before trusting a row.
_FILETIME_UNIX_EPOCH = 11644473600  # seconds from 1601-01-01 (FILETIME) to 1970-01-01


def _nt_unavailable() -> dict:
    # psutil's per-process path takes seconds per pass on Windows, which would stall every GPU refresh.
    raise OSError("system RAM/CPU needs 64-bit Python on Windows")


_nt_query = _nt_unavailable if sys.platform == "win32" else None
if sys.platform == "win32" and struct.calcsize("P") == 8:
    import ctypes

    _ntdll = ctypes.WinDLL("ntdll")
    _ntdll.NtQuerySystemInformation.restype = ctypes.c_long

    def _nt_query_impl() -> dict[int, tuple[str, int, int, int, float]]:
        """pid -> (image name, working set, private bytes, CPU time in 100 ns, creation time in unix seconds)."""
        size = 1 << 20
        while True:
            buf = ctypes.create_string_buffer(size)
            need = ctypes.c_ulong()
            status = _ntdll.NtQuerySystemInformation(5, buf, size, ctypes.byref(need))  # SystemProcessInformation
            if status == -1073741820:  # STATUS_INFO_LENGTH_MISMATCH: processes appeared; grow and retry
                size = max(size * 2, need.value + (64 << 10))
                continue
            if status < 0:
                raise OSError(f"NtQuerySystemInformation failed: 0x{status & 0xffffffff:08x}")
            break
        raw, base, out, off = buf.raw, ctypes.addressof(buf), {}, 0
        while True:
            nxt = struct.unpack_from("<I", raw, off)[0]
            created, user, kernel = struct.unpack_from("<qqq", raw, off + 32)
            name_len, name_ptr = struct.unpack_from("<H6xQ", raw, off + 56)
            pid = struct.unpack_from("<Q", raw, off + 80)[0]
            ws, = struct.unpack_from("<Q", raw, off + 144)
            private, = struct.unpack_from("<Q", raw, off + 184)
            start = name_ptr - base
            if name_ptr and 0 <= start and start + name_len <= len(raw):
                name = raw[start:start + name_len].decode("utf-16-le", "replace")
            else:
                name = "System Idle Process" if pid == 0 else "?"
            out[pid] = (name, ws, private, user + kernel, created / 1e7 - _FILETIME_UNIX_EPOCH)
            if not nxt:
                return out
            off += nxt

    _nt_query = _nt_query_impl


def _check_nt_layout(rows: dict) -> None:
    """Our own process, as the NT process list reports it, must match what psutil says about it."""
    me = psutil.Process()
    row = rows.get(me.pid)
    if row is not None:
        name, ws, _private, _cpu, created = row
        rss = me.memory_info().rss
        if (name.lower() == me.name().lower() and abs(created - me.create_time()) < SAME_PROCESS_SECS
                and rss / 2 <= ws <= rss * 2):
            return
    raise OSError("NtQuerySystemInformation process layout not recognised on this Windows build")


def _safe_name(p: psutil.Process) -> str:
    try:
        return p.name()
    except psutil.Error:
        return "?"


def _proc_dict(p: HostProc) -> dict:
    return {"pid": p.pid, "name": p.name, "rss": p.rss, "private": p.private,
            "cpu_pct": None if p.cpu_pct is None else round(p.cpu_pct, 1)}


def to_dict(h: HostSample) -> dict:
    return {
        "source": "OS-reported (psutil; NtQuerySystemInformation for the process list on Windows)",
        "cpu": {"pct": round(h.cpu_pct, 1), "per_core": h.per_core,
                "logical": h.logical, "physical": h.physical},
        "ram": {"total": h.ram_total, "used": h.ram_used, "available": h.ram_available,
                "swap_total": h.swap_total, "swap_used": h.swap_used},
        "top_ram": [_proc_dict(p) for p in h.top("rss", 10)],
        "top_cpu": [_proc_dict(p) for p in h.top("cpu_pct", 10)],
        "denied": h.denied,
        "cost_ms": round(h.cost_ms, 1),
    }


def to_text(h: HostSample) -> str:
    from .llm import gb

    lines = ["System RAM and CPU (OS-reported; not GPU memory)",
             f"CPU {h.cpu_pct:.0f}%  {h.logical} threads" + (f" / {h.physical} cores" if h.physical else ""),
             f"RAM {gb(h.ram_used)} / {gb(h.ram_total)}  available {gb(h.ram_available)}"
             f"   swap/pagefile {gb(h.swap_used)} / {gb(h.swap_total)}",
             "", "Top processes by RAM, whole machine:", f"{'PID':>7}  {'PROCESS':<28} {'RAM':>9} {'PRIVATE':>9} {'CPU%':>5}"]
    for p in h.top("rss", 10):
        lines.append(f"{p.pid:>7}  {p.name[:28]:<28} {gb(p.rss):>9} {gb(p.private) if p.private is not None else 'n/a':>9} "
                     f"{f'{p.cpu_pct:.0f}' if p.cpu_pct is not None else 'n/a':>5}")
    if h.denied:
        lines.append(f"({h.denied} processes not readable without admin)")
    return "\n".join(lines)

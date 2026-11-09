"""System RAM and CPU, alongside the GPU view.

Every number here is reported by the OS through psutil (measured, not inferred). Unknown stays None:
a process we may not inspect has ``rss=None`` / ``cpu_pct=None``, never 0. Fork addition, not upstream.
"""

from __future__ import annotations

import struct
import sys
import time
from dataclasses import dataclass, field

import psutil


@dataclass
class HostProc:
    pid: int
    name: str
    rss: int | None          # resident / working set bytes
    private: int | None      # Windows private bytes (commit charged to the process); None elsewhere
    cpu_pct: float | None    # share of the WHOLE machine (0-100), like Task Manager; None on the first sample


@dataclass
class HostSample:
    cpu_pct: float                  # whole machine, 0-100
    per_core: list[float]
    cpu_mhz: float | None
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

    def proc(self, pid: int | None) -> HostProc | None:
        return self._by_pid.get(pid) if pid is not None else None

    def __post_init__(self) -> None:
        self._by_pid = {p.pid: p for p in self.procs}

    def top(self, key: str, n: int = 5) -> list[HostProc]:
        return sorted((p for p in self.procs if getattr(p, key) is not None), key=lambda p: -getattr(p, key))[:n]


class HostSampler:
    """Keeps psutil.Process objects between samples: per-process CPU% is a rate and needs the previous reading."""

    def __init__(self) -> None:
        self._procs: dict[int, psutil.Process] = {}
        self._rated: set[int] = set()  # pids with a previous CPU reading
        self._logical = psutil.cpu_count() or 1
        self._physical = psutil.cpu_count(logical=False)
        self._last_cpu: dict[int, int] = {}
        self._last_wall = 0.0
        psutil.cpu_percent(percpu=True)  # prime the system-wide rate

    def sample(self) -> HostSample:
        t0 = time.perf_counter()
        per_core = psutil.cpu_percent(percpu=True)
        vm, sw = psutil.virtual_memory(), psutil.swap_memory()
        try:
            freq = psutil.cpu_freq()
            mhz = freq.current if freq and freq.current else None
        except (OSError, NotImplementedError):  # unavailable on some platforms/containers
            mhz = None
        if _nt_query is not None:
            procs, denied = self._sample_nt(), 0
            return self._finish(t0, per_core, mhz, vm, sw, procs, denied)
        procs, denied, seen = [], 0, set()
        for p in psutil.process_iter():
            pid = p.pid
            seen.add(pid)
            p = self._procs.setdefault(pid, p)
            try:
                with p.oneshot():
                    name = p.name()
                    mi = p.memory_info()
                    cpu = p.cpu_percent(None)
            except psutil.NoSuchProcess:
                continue
            except psutil.AccessDenied:
                denied += 1
                procs.append(HostProc(pid, _safe_name(p), None, None, None))
                continue
            first = pid not in self._rated
            self._rated.add(pid)
            procs.append(HostProc(pid, name, mi.rss, getattr(mi, "private", None),
                                  None if first else cpu / self._logical))
        for gone in self._procs.keys() - seen:
            del self._procs[gone]
        self._rated &= seen
        return self._finish(t0, per_core, mhz, vm, sw, procs, denied)

    def _sample_nt(self) -> list[HostProc]:
        now = time.perf_counter()
        rows = _nt_query()
        wall = now - self._last_wall if self._last_wall else None
        procs = []
        for pid, (name, ws, private, cpu100ns) in rows.items():
            prev = self._last_cpu.get(pid)
            cpu = None
            if wall and prev is not None and pid != 0:  # pid 0 = System Idle Process: its "CPU" is idle time
                cpu = max(0.0, (cpu100ns - prev) / 1e7 / wall / self._logical * 100)
            procs.append(HostProc(pid, name, ws, private, cpu))
        self._last_cpu = {pid: r[3] for pid, r in rows.items()}
        self._last_wall = now
        return [p for p in procs if p.pid != 0]

    def _finish(self, t0, per_core, mhz, vm, sw, procs, denied) -> HostSample:
        return HostSample(
            cpu_pct=sum(per_core) / len(per_core) if per_core else 0.0, per_core=per_core, cpu_mhz=mhz,
            logical=self._logical, physical=self._physical,
            ram_total=vm.total, ram_used=vm.total - vm.available, ram_available=vm.available,
            swap_total=sw.total, swap_used=sw.used, procs=procs, denied=denied,
            cost_ms=(time.perf_counter() - t0) * 1000)


# Windows: one NtQuerySystemInformation(SystemProcessInformation) call returns every process's working set,
# private bytes and CPU time. psutil opens each process instead: measured 2026-11-08 on a 483-process
# Windows 11 box, memory_info ~1.1 s and cpu_times ~1.0 s per pass, versus a few ms here. Same numbers:
# WorkingSetSize = psutil rss, PagefileUsage = psutil private. No process handle needed, so no AccessDenied.
_nt_query = None
if sys.platform == "win32" and struct.calcsize("P") == 8:
    import ctypes

    _ntdll = ctypes.WinDLL("ntdll")
    _ntdll.NtQuerySystemInformation.restype = ctypes.c_long

    def _nt_query_impl() -> dict[int, tuple[str, int, int, int]]:
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
            user, kernel = struct.unpack_from("<qq", raw, off + 40)
            name_len, name_ptr = struct.unpack_from("<H6xQ", raw, off + 56)
            pid = struct.unpack_from("<Q", raw, off + 80)[0]
            ws, = struct.unpack_from("<Q", raw, off + 144)
            private, = struct.unpack_from("<Q", raw, off + 184)
            if name_ptr:
                name = raw[name_ptr - base:name_ptr - base + name_len].decode("utf-16-le", "replace")
            else:
                name = "System Idle Process" if pid == 0 else "?"
            out[pid] = (name, ws, private, user + kernel)
            if not nxt:
                return out
            off += nxt

    _nt_query = _nt_query_impl


def _safe_name(p: psutil.Process) -> str:
    try:
        return p.name()
    except psutil.Error:
        return "?"


def to_dict(h: HostSample) -> dict:
    return {
        "source": "psutil (OS-reported)",
        "cpu": {"pct": round(h.cpu_pct, 1), "per_core": h.per_core, "mhz": h.cpu_mhz,
                "logical": h.logical, "physical": h.physical},
        "ram": {"total": h.ram_total, "used": h.ram_used, "available": h.ram_available,
                "swap_total": h.swap_total, "swap_used": h.swap_used},
        "top_ram": [vars(p) for p in h.top("rss", 10)],
        "top_cpu": [vars(p) for p in h.top("cpu_pct", 10)],
        "denied": h.denied,
        "cost_ms": round(h.cost_ms, 1),
    }


def to_text(h: HostSample) -> str:
    from .llm import gb

    lines = [f"CPU {h.cpu_pct:.0f}%  {h.logical} threads" + (f" / {h.physical} cores" if h.physical else "")
             + (f"  {h.cpu_mhz:.0f} MHz" if h.cpu_mhz else ""),
             f"RAM {gb(h.ram_used)} / {gb(h.ram_total)}  available {gb(h.ram_available)}"
             f"   swap/pagefile {gb(h.swap_used)} / {gb(h.swap_total)}",
             "", f"{'PID':>7}  {'PROCESS':<28} {'RAM':>9} {'PRIVATE':>9} {'CPU%':>5}"]
    for p in h.top("rss", 10):
        lines.append(f"{p.pid:>7}  {p.name[:28]:<28} {gb(p.rss):>9} {gb(p.private) if p.private is not None else 'n/a':>9} "
                     f"{f'{p.cpu_pct:.0f}' if p.cpu_pct is not None else 'n/a':>5}")
    if h.denied:
        lines.append(f"({h.denied} processes not readable without admin)")
    return "\n".join(lines)

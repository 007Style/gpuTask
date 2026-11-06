"""Windows GPU performance counters (the same data Task Manager uses).

These come from the WDDM graphics kernel (dxgkrnl), so they work for every API that
goes through it -- D3D, Vulkan, OpenGL, *and* CUDA on GeForce/WDDM -- even when
nvidia-smi/NVML report per-process memory as "N/A".
"""

from __future__ import annotations

import ctypes
import re
from collections import defaultdict
from ctypes import wintypes
from dataclasses import dataclass, field

PDH_FMT_DOUBLE = 0x00000200
PDH_FMT_NOCAP100 = 0x00008000
PDH_MORE_DATA = 0x800007D2


class _Value(ctypes.Union):
    _fields_ = [
        ("longValue", ctypes.c_long),
        ("doubleValue", ctypes.c_double),
        ("largeValue", ctypes.c_longlong),
        ("str", ctypes.c_wchar_p),
    ]


class _FmtValue(ctypes.Structure):
    _fields_ = [("CStatus", wintypes.DWORD), ("u", _Value)]


class _Item(ctypes.Structure):
    _fields_ = [("szName", ctypes.c_wchar_p), ("FmtValue", _FmtValue)]


_pdh = None


def _load_pdh():
    """pdh.dll is loaded on first use so this module imports cleanly on Linux."""
    global _pdh
    if _pdh is None:
        lib = ctypes.WinDLL("pdh")
        lib.PdhOpenQueryW.argtypes = [wintypes.LPCWSTR, ctypes.c_size_t, ctypes.POINTER(wintypes.HANDLE)]
        lib.PdhAddEnglishCounterW.argtypes = [wintypes.HANDLE, wintypes.LPCWSTR, ctypes.c_size_t, ctypes.POINTER(wintypes.HANDLE)]
        lib.PdhCollectQueryData.argtypes = [wintypes.HANDLE]
        lib.PdhGetFormattedCounterArrayW.argtypes = [
            wintypes.HANDLE, wintypes.DWORD, ctypes.POINTER(wintypes.DWORD), ctypes.POINTER(wintypes.DWORD), ctypes.c_void_p,
        ]
        lib.PdhCloseQuery.argtypes = [wintypes.HANDLE]
        for _fn in ("PdhOpenQueryW", "PdhAddEnglishCounterW", "PdhCollectQueryData", "PdhGetFormattedCounterArrayW", "PdhCloseQuery"):
            getattr(lib, _fn).restype = wintypes.DWORD
        _pdh = lib
    return _pdh


_LUID_RE = r"luid_0x([0-9a-fA-F]+)_0x([0-9a-fA-F]+)"
_PROC_RE = re.compile(rf"pid_(\d+)_{_LUID_RE}_phys_(\d+)")
_ENGINE_RE = re.compile(rf"pid_(\d+)_{_LUID_RE}_phys_(\d+)_eng_(\d+)_engtype_(.*)$")
_ADAPTER_RE = re.compile(rf"^{_LUID_RE}_phys_(\d+)$")


def luid_key(hi: str, lo: str) -> int:
    return (int(hi, 16) << 32) | int(lo, 16)


@dataclass
class ProcGpu:
    dedicated: float = 0.0  # bytes resident in VRAM charged to this process ("Local Usage")
    shared: float = 0.0  # bytes resident in shared (system RAM) GPU memory -- "spill" ("Non Local Usage")
    committed: float = 0.0  # all GPU memory this process allocated, resident or not ("Total Committed")
    engines: dict[str, float] = field(default_factory=lambda: defaultdict(float))  # engtype -> util %
    available: set[str] = field(default_factory=lambda: {"dedicated", "shared", "committed", "engine"})

    @property
    def paged_available(self) -> bool:
        return {"dedicated", "shared", "committed"} <= self.available
    # False when per-process VRAM is unavailable. Native NVML may still report engines independently;
    # WSL charges every Linux process's VRAM and engine activity to the Windows VM worker.
    attributed: bool = True
    # WSL: VRAM matched from the VM's GPU contexts (see wslhost.ContextTracker). An exact counter value whose
    # owner is inferred; kept apart from `dedicated` so it isn't counted twice next to the WSL VM row.
    inferred: float | None = None
    inferred_by: str = ""  # "consistent" (candidates agree on bytes) | "order" (heuristic owner)
    # amdgpu (DRM fdinfo): resident bytes in every buffer this process can reach, shared ones included. A reference
    # for display only; the same buffer can appear under several processes, so these are never summed. None: unknown.
    resident_vram: float | None = None
    resident_gtt: float | None = None
    drm_reason: str = ""  # amdgpu: why dedicated/shared are n/a (drm.REASON_*); empty when they are measured
    kfd: bool = False  # amdgpu: holds /dev/kfd (ROCm/HIP), whose compute isn't timed per process

    @property
    def shown_vram(self) -> float | None:
        """Measured or inferred bytes for display and priority, not for summing adapter totals."""
        return (self.dedicated if "dedicated" in self.available else None) if self.attributed else self.inferred

    @property
    def paged_out(self) -> float:
        """Commit minus local/non-local usage: a non-residency estimate, not an eviction trace."""
        if not self.paged_available:
            return 0.0  # internal sentinel; callers must check paged_available before display
        return max(self.committed - self.dedicated - self.shared, 0.0)

    @property
    def gpu_pct(self) -> float:
        # Task Manager semantics: busiest engine wins.
        return max(self.engines.values(), default=0.0)


@dataclass
class AdapterGpu:
    dedicated: float | None = None
    shared: float | None = None


class GpuCounters:
    # Per-process "Dedicated/Shared Usage" count every allocation a process can *see*, including surfaces
    # shared from other processes: dwm.exe read 15+ GB on a 16 GB card and the per-process sum was 2x the
    # adapter total. "Local/Non Local Usage" is what's charged to each process and adds up to the adapter
    # total. The older counters are only a fallback.
    _COUNTERS = {
        "dedicated": (r"\GPU Process Memory(*)\Local Usage", r"\GPU Process Memory(*)\Dedicated Usage"),
        "shared": (r"\GPU Process Memory(*)\Non Local Usage", r"\GPU Process Memory(*)\Shared Usage"),
        "committed": (r"\GPU Process Memory(*)\Total Committed",),
        "engine": (r"\GPU Engine(*)\Utilization Percentage",),
        "a_dedicated": (r"\GPU Adapter Memory(*)\Dedicated Usage",),
        "a_shared": (r"\GPU Adapter Memory(*)\Shared Usage",),
    }

    def __init__(self) -> None:
        _load_pdh()
        self._query = wintypes.HANDLE()
        self._check(_pdh.PdhOpenQueryW(None, 0, ctypes.byref(self._query)), "PdhOpenQuery")
        self._handles: dict[str, wintypes.HANDLE] = {}
        for key, paths in self._COUNTERS.items():
            for path in paths:
                h = wintypes.HANDLE()
                if _pdh.PdhAddEnglishCounterW(self._query, path, 0, ctypes.byref(h)) == 0:
                    self._handles[key] = h
                    break
        _pdh.PdhCollectQueryData(self._query)  # rate counters need a baseline sample

    @staticmethod
    def _check(status: int, what: str) -> None:
        if status != 0:
            raise OSError(f"{what} failed: 0x{status:08X}")

    def _read(self, key: str) -> list[tuple[str, float]]:
        h = self._handles.get(key)
        if h is None:
            return []
        size = wintypes.DWORD(0)
        count = wintypes.DWORD(0)
        fmt = PDH_FMT_DOUBLE | PDH_FMT_NOCAP100
        status = _pdh.PdhGetFormattedCounterArrayW(h, fmt, ctypes.byref(size), ctypes.byref(count), None)
        if status != PDH_MORE_DATA:
            return []
        buf = ctypes.create_string_buffer(size.value)
        status = _pdh.PdhGetFormattedCounterArrayW(h, fmt, ctypes.byref(size), ctypes.byref(count), buf)
        if status != 0:
            return []
        items = ctypes.cast(buf, ctypes.POINTER(_Item))
        out = []
        for i in range(count.value):
            it = items[i]
            if it.FmtValue.CStatus in (0, 1):  # PDH_CSTATUS_VALID_DATA / NEW_DATA
                out.append((it.szName, it.FmtValue.u.doubleValue))
        return out

    def sample(self) -> tuple[dict[int, dict[int, ProcGpu]], dict[int, AdapterGpu]]:
        """Returns ({luid: {pid: ProcGpu}}, {luid: AdapterGpu})."""
        if _pdh.PdhCollectQueryData(self._query) != 0:
            return {}, {}
        procs: dict[int, dict[int, ProcGpu]] = defaultdict(dict)

        def proc(luid: int, pid: int) -> ProcGpu:
            p = procs[luid].get(pid)
            if p is None:
                p = procs[luid][pid] = ProcGpu(available=set())
            return p

        for attr in ("dedicated", "shared", "committed"):
            for name, val in self._read(attr):
                m = _PROC_RE.match(name)
                if m:
                    p = proc(luid_key(m[2], m[3]), int(m[1]))
                    setattr(p, attr, getattr(p, attr) + val)
                    p.available.add(attr)
        for name, val in self._read("engine"):
            m = _ENGINE_RE.match(name)
            if m:
                p = proc(luid_key(m[2], m[3]), int(m[1]))
                p.engines[_norm_engine(m[6])] += val
                p.available.add("engine")

        adapters: dict[int, AdapterGpu] = defaultdict(AdapterGpu)
        for attr, key in (("dedicated", "a_dedicated"), ("shared", "a_shared")):
            for name, val in self._read(key):
                m = _ADAPTER_RE.match(name)
                if m:
                    a = adapters[luid_key(m[1], m[2])]
                    setattr(a, attr, (getattr(a, attr) or 0.0) + val)
        return procs, adapters

    def close(self) -> None:
        if self._query:
            _pdh.PdhCloseQuery(self._query)
            self._query = wintypes.HANDLE()


def _norm_engine(t: str) -> str:
    t = t.lower()
    if t.startswith("3d"):
        return "3d"
    if t.startswith("compute") or t.startswith("cuda"):
        return "compute"
    if t.startswith("copy"):
        return "copy"
    if t.startswith("videodecode"):
        return "decode"
    if t.startswith("videoencode"):
        return "encode"
    return "other"

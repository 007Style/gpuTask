"""Assembles one Snapshot from Windows counters, NVML, process inspection and LLM introspection."""

from __future__ import annotations

import os
import sys
import time
from dataclasses import dataclass, field

import psutil

from .dxgi import VENDOR_NVIDIA, Adapter
from .handy import HandyEvidence, inspect_handy
from .llm import LlmView, ServerProber, build_view, gb, ollama_log_path
from .nvml import Nvml, NvDevice
from .pdh import ProcGpu
from .procinfo import AI_KINDS, ProcDetail, ProcInspector, classify

if sys.platform == "win32":
    from .dxgi import enumerate_adapters
    from .pdh import GpuCounters

    BACKEND = "wddm"
else:
    from .linux import LinuxCounters as GpuCounters
    from .linux import enumerate_adapters, is_wsl

    BACKEND = "wsl" if is_wsl() else "nvml"

from .wslhost import ContextTracker, HostStream, interop_enabled, match_adapter

MiB = 1 << 20


@dataclass
class ProcRow:
    pid: int
    name: str
    kind: str
    gpu: ProcGpu
    detail: ProcDetail
    llm: LlmView | None = None
    handy: HandyEvidence | None = None
    host: bool = False  # pid is from another PID namespace, never a local one (see origin)
    origin: str = ""  # with host: "windows" (a Windows process seen from WSL) | "outside" (outside this container)

    @property
    def key(self) -> str:
        """Unique row id: PIDs from other namespaces and local PIDs can collide."""
        return f"{self.origin[0]}{self.pid}" if self.host else str(self.pid)

    @property
    def pid_label(self) -> str:
        return f"{'W' if self.origin == 'windows' else 'H'}:{self.pid}" if self.host else str(self.pid)

    @property
    def runtime(self) -> str:
        return "+".join(self.detail.runtimes[:3])

    @property
    def inside(self) -> str:
        """One-line 'what's in there' summary for the process table."""
        if self.handy:
            if self.handy.last_model:
                return f"Handy last log ({self.handy.last_event}): {self.handy.last_model.rsplit('/', 1)[-1]}"
            if self.handy.selected_model:
                return f"Handy selected (not load proof): {self.handy.selected_model.rsplit('/', 1)[-1]}"
        if self.llm and self.llm.models:
            parts = []
            for i, g in enumerate(self.llm.models):
                if g.kind in ("vision projector", "LoRA adapter"):
                    parts.append(f"+{g.kind}")
                else:
                    parts.append(f"{self.llm.label(i)} {g.params_label} {g.quant}")
            return ", ".join(parts)
        if self.detail.models:
            return ", ".join((m.store_name or os.path.basename(m.path)) +
                             (f" ({m.kind_label})" if m.kind_hint else "") for m in self.detail.models[:2])
        if self.detail.candidates:
            return "installed (unproven): " + ", ".join(
                f"{m.store_name or os.path.basename(m.path)} {gb(m.size)}" +
                (f" ({m.kind_label})" if m.kind_hint else "") for m in self.detail.candidates[:2])
        return {
            "compositor": "desktop back buffers + composed frames",
            "wsl-vm": "all Linux GPU memory (every WSL distro + WSLg)",
            "external": "a process in another PID namespace (host or another container)",
            "kernel": "driver/kernel allocations",
            "browser": "page textures, video, canvas/WebGL",
            "video": "video frames (decode/encode)",
            "3D/game": "textures, meshes, render targets",
        }.get(self.kind, "")


@dataclass
class Snapshot:
    taken: float
    adapter: Adapter | None
    adapters: list[Adapter]
    nv: NvDevice | None
    driver: str
    cuda: str
    vram_total: int
    vram_used: int  # adapter-wide dedicated usage (WDDM counters, or NVML on Linux): the ground truth
    shared_used: int
    procs: list[ProcRow] = field(default_factory=list)
    backend: str = "wddm"  # wddm (Windows) | nvml (native Linux) | wsl (device-wide only)
    host_status: str = ""  # WSL only: Windows host counters via interop -- ok | starting | off (...) | ...
    wsl_split: str = ""  # WSL only: why Linux rows have no (inferred) VRAM, when they don't

    @property
    def host_data(self) -> bool:
        return self.host_status.startswith("ok")

    @property
    def attributed(self) -> int:
        return int(sum(p.gpu.dedicated for p in self.procs))


def _is_live(g: ProcGpu) -> bool:
    return (not g.attributed or g.dedicated >= MiB or g.gpu_pct >= 0.5 or g.shared >= 16 * MiB
            or g.paged_out >= 16 * MiB)


def select_adapter(adapters: list[Adapter], gpu_index: int) -> Adapter | None:
    hardware = [a for a in adapters if not a.software]
    choices = [a for a in hardware if a.vendor_id == VENDOR_NVIDIA] or hardware
    choices.sort(key=lambda a: -a.dedicated_vram)
    if gpu_index < 0:
        raise ValueError("--gpu must be a non-negative index")
    if gpu_index >= len(choices):
        if not choices and gpu_index == 0:
            return None
        raise ValueError(f"--gpu {gpu_index} is out of range; {len(choices)} GPU adapter(s) available")
    return choices[gpu_index]


def select_nv_device(adapter: Adapter | None, devices: list[NvDevice], gpu_index: int) -> NvDevice | None:
    if adapter:
        if adapter.vendor_id != VENDOR_NVIDIA:
            return None
        matches = [d for d in devices if d.name.casefold() == adapter.name.casefold()]
        if len(matches) == 1:
            return matches[0]
    fallback = devices[gpu_index] if gpu_index < len(devices) else None
    return None if adapter and fallback and fallback.driver_model == "MCDM" else fallback


class Collector:
    DEEP_TOP_N = 25  # memory-map candidates: the top N VRAM users plus anything busy
    SCAN_BUDGET = 0.15  # seconds of memory-map scanning per sample; leftovers wait for the next sample

    def __init__(self, gpu_index: int = 0, scan_budget: float | None = SCAN_BUDGET, host: bool = True) -> None:
        self.adapters = [a for a in enumerate_adapters() if not a.software]
        self.adapter = select_adapter(self.adapters, gpu_index)
        self.counters = GpuCounters()
        self.nvml = Nvml()
        self.inspector = ProcInspector()
        self.prober = ServerProber()
        self.scan_budget = scan_budget  # None = scan every candidate now (one-shot output)
        self.gpu_index = gpu_index
        # WSL: Linux can't see WSL's own VRAM, but the Windows host can. Ask it when interop is enabled.
        self.host: HostStream | None = None
        self.host_status = ""
        if BACKEND == "wsl":
            if not host:
                self.host_status = "off (--no-host)"
            elif not interop_enabled():
                self.host_status = "off (WSL interop is disabled)"
            else:
                self.host = HostStream()
        self._host_luid: int | None = None  # host adapter for this GPU, matched once and kept
        self.tracker = ContextTracker(self_pid=os.getpid()) if self.host is not None else None
        self._own_start = time.time()  # our NVML context is created now, not when the process started
        self._host_restarted = time.time()

    def _parent_ports(self, pid: int, depth: int = 2) -> list[int]:
        d = self.inspector.get(pid)
        if d.parents is None:
            d.parents = []
            try:
                p = psutil.Process(pid)
                for _ in range(depth):
                    p = p.parent()
                    if p is None:
                        break
                    d.parents.append(p.pid)
            except (psutil.Error, OSError):
                pass
        return [port for ppid in d.parents for port in self.inspector.ports(ppid)]

    def sample(self) -> Snapshot:
        per_luid, adapters = self.counters.sample()
        luid = self.adapter.luid if self.adapter else (next(iter(per_luid), 0))
        procs = per_luid.get(luid, {})
        a = adapters.get(luid)
        nvs = self.nvml.sample() if self.nvml.ok else []
        nv = select_nv_device(self.adapter, nvs, self.gpu_index)
        total = (nv.mem_total if nv else 0) or (self.adapter.dedicated_vram if self.adapter else 0)

        snap = Snapshot(
            taken=time.time(), adapter=self.adapter, adapters=self.adapters, nv=nv,
            driver=self.nvml.driver, cuda=self.nvml.cuda_version, vram_total=total,
            vram_used=int(a.dedicated) if a else (nv.mem_used if nv else 0),
            shared_used=int(a.shared) if a else 0, backend=BACKEND,
        )

        snap.host_status = self.host_status
        live = [(pid, g) for pid, g in procs.items() if _is_live(g)]
        live.sort(key=lambda x: -x[1].dedicated)
        deep_pids = {pid for pid, _ in live[: self.DEEP_TOP_N]} | {pid for pid, g in live if g.gpu_pct >= 2}
        deadline = None if self.scan_budget is None else time.perf_counter() + self.scan_budget
        ollama_runners = []
        for pid, g in live:  # largest VRAM first, so the budget goes to what matters
            d = self.inspector.get(pid)
            mem = g.dedicated + g.shared
            if (pid in deep_pids and (deadline is None or time.perf_counter() < deadline)
                    and self.inspector.needs_scan(d, mem)):
                self.inspector.scan(d, mem)
            row = ProcRow(pid=pid, name=d.name, kind=classify(d, g.engines), gpu=g, detail=d)
            snap.procs.append(row)
            # Old Ollama: "ollama.exe runner ..."; newer Ollama: its own lib\ollama\llama-server.exe
            exe = d.exe.lower()
            if row.kind in AI_KINDS and ("\\ollama\\" in exe or "/ollama/" in exe or
                                         (d.name.lower().startswith("ollama") and "runner" in d.cmdline)):
                ollama_runners.append(row)

        for pid, g in getattr(self.counters, "outside", {}).get(luid, {}).items():
            # A container sees the host's PIDs from NVML; unmatched ones are other containers or the host.
            snap.procs.append(ProcRow(pid, "(outside this container)", "external", g,
                                      ProcDetail(pid, "(outside this container)"), host=True, origin="outside"))
        if self.host is not None:  # before LLM views, so an inferred VRAM total feeds the breakdown
            self._add_host_rows(snap)
        for row in snap.procs:
            if row.host:
                continue
            if os.path.basename(row.detail.exe).lower() in ("handy.exe", "handy"):
                row.handy = inspect_handy(row.detail.create_time)
            if row.detail.is_diffusion or row.kind not in AI_KINDS or not (row.detail.models or row.kind == "LLM"):
                continue
            own = self.prober.get(row.detail.ports)
            host = self.prober.get(self._parent_ports(row.pid))  # Ollama / LM Studio front-end process
            log = ollama_log_path() if row in ollama_runners and len(ollama_runners) == 1 else None
            inferred = row.gpu.inferred
            row.llm = build_view(row.detail, row.gpu.dedicated if inferred is None else inferred, row.gpu.shared,
                                 own or host, log, host=host if own else None,
                                 attributed=row.gpu.attributed or inferred is not None)

        self.inspector.prune(set(procs))
        return snap

    def _add_host_rows(self, snap: Snapshot) -> None:
        """Windows processes on this GPU, plus the WSL VM itself (vmwp.exe: the exact total of Linux VRAM)."""
        if self.host.ended and time.time() - self._host_restarted > 30:
            self.host.close()  # PowerShell exited (e.g. killed, or the host was busy): try again
            self.host = HostStream()
            self._host_restarted = time.time()
        hs = self.host.latest()
        if hs is None:
            snap.host_status = "starting" if self.host.status in ("starting", "ok") else self.host.status
            return
        # Match the host adapter once: NVML is read now while the host sample can be a second old, so
        # re-matching every refresh fails whenever memory moves fast (a model loading) and the rows flicker.
        luid = self._host_luid if self._host_luid in hs.adapters else match_adapter(hs, snap.vram_used)
        if luid is None:
            # Memory can swing by GBs between the two readings while a workload loads: retry for a while.
            early = time.time() - self._host_restarted < 20
            snap.host_status = "starting" if early else "unmatched (no Windows adapter matches this GPU's usage)"
            return
        self._host_luid = luid
        age = time.time() - hs.taken
        snap.host_status = "ok" if age < 3 else f"ok (last Windows sample {age:.0f}s old)"
        a = hs.adapters.get(luid)
        if a is not None:
            snap.shared_used = int(a.shared)
        procs = hs.procs.get(luid, {})
        vm_workers = [pid for pid in procs if hs.names.get(pid, "").lower() == "vmwp"]
        host_rows: list[ProcRow] = []
        for pid, g in procs.items():
            if not _is_live(g):
                continue
            raw = hs.names.get(pid, "?")
            if raw.lower() == "vmwp":
                # One VM worker is WSL's; several could include Hyper-V VMs or Windows Sandbox.
                name = "WSL VM (vmwp.exe)" if len(vm_workers) == 1 else "VM worker (vmwp.exe)"
                kind = "wsl-vm"
            else:
                name = raw if raw in ("System", "Idle", "?") or raw.lower().endswith(".exe") else f"{raw}.exe"
                kind = "kernel" if pid == 4 else classify(ProcDetail(pid, name), g.engines)
            host_rows.append(ProcRow(pid, name, kind, g, ProcDetail(pid, name), host=True, origin="windows"))
        # Largest first; Linux rows sit right under the WSL VM row, whose total they share.
        host_rows.sort(key=lambda r: -r.gpu.dedicated)
        at = next((i + 1 for i, r in enumerate(host_rows) if r.kind == "wsl-vm"), 0)
        if len(vm_workers) == 1 and self.tracker is not None:
            self._attribute(snap, luid, hs.contexts.get(luid, {}).get(vm_workers[0], []))
        snap.procs = host_rows[:at] + snap.procs + host_rows[at:]

    def _attribute(self, snap: Snapshot, luid: int, contexts: list[float]) -> None:
        linux = [r for r in snap.procs if not r.host and not r.gpu.attributed]
        # This monitor holds a (0-byte) context too, through NVML.
        holders = [(r.pid, r.detail.create_time) for r in linux] + [(os.getpid(), self._own_start)]
        guesses, reason = self.tracker.update(luid, contexts, holders)
        snap.wsl_split = reason
        for r in linux:
            g = guesses.get(r.pid)
            if g is not None:
                r.gpu.inferred, r.gpu.inferred_by = g.bytes, g.confidence

    def close(self) -> None:
        self.counters.close()
        self.nvml.close()
        if self.host is not None:
            self.host.close()

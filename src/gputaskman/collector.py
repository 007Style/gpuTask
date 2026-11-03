"""Assembles one Snapshot from Windows counters, NVML, process inspection and LLM introspection."""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field

import psutil

from .amdsmi import AmdSmi, _bdf
from .adl import Adl
from .dxgi import VENDOR_NVIDIA, Adapter, enumerate_adapters
from .handy import HandyEvidence, inspect_handy
from .llm import LlmView, ServerProber, build_view, gb, ollama_log_path
from .nvml import Nvml, NvDevice
from .pdh import GpuCounters, ProcGpu
from .procinfo import AI_KINDS, ProcDetail, ProcInspector, classify
from .telemetry import DeviceTelemetry

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
    nv: DeviceTelemetry | None  # legacy name; may contain telemetry from either provider
    driver: str
    cuda: str
    vram_total: int
    vram_used: int | None  # WDDM adapter-wide dedicated usage
    shared_used: int | None
    procs: list[ProcRow] = field(default_factory=list)

    @property
    def attributed(self) -> int | None:
        if any("dedicated" not in p.gpu.available for p in self.procs):
            return None
        return int(sum(p.gpu.dedicated for p in self.procs))


def select_adapter(adapters: list[Adapter], gpu_index: int) -> Adapter | None:
    choices = ordered_adapters(adapters)
    if gpu_index < 0:
        raise ValueError("--gpu must be a non-negative index")
    if gpu_index >= len(choices):
        if not choices and gpu_index == 0:
            return None
        raise ValueError(f"--gpu {gpu_index} is out of range; {len(choices)} GPU adapter(s) available")
    return choices[gpu_index]


def ordered_adapters(adapters: list[Adapter]) -> list[Adapter]:
    """List all hardware adapters; retain NVIDIA-first ordering for existing users."""
    return sorted((a for a in adapters if not a.software),
                  key=lambda a: (a.vendor_id != VENDOR_NVIDIA, -a.dedicated_vram, a.luid))


def select_nv_device(adapter: Adapter | None, devices: list[NvDevice], gpu_index: int,
                     adapters: list[Adapter] | None = None) -> NvDevice | None:
    if adapter:
        if adapter.vendor_id != VENDOR_NVIDIA:
            return None
        if adapter.pci_address is None:
            return None
        if adapters is not None and sum(a.pci_address == adapter.pci_address for a in adapters) != 1:
            return None
        matches = [d for d in devices if _bdf(d.pci_bus) == adapter.pci_address and d.driver_model != "MCDM"]
        if len(matches) == 1:
            return matches[0]
        return None
    return devices[gpu_index] if gpu_index < len(devices) else None


class Collector:
    DEEP_TOP_N = 25  # memory-map candidates: the top N VRAM users plus anything busy
    SCAN_BUDGET = 0.15  # seconds of memory-map scanning per sample; leftovers wait for the next sample

    def __init__(self, gpu_index: int = 0, scan_budget: float | None = SCAN_BUDGET) -> None:
        self.adapters = [a for a in enumerate_adapters() if not a.software]
        self.adapter = select_adapter(self.adapters, gpu_index)
        self.counters = GpuCounters()
        self.nvml = Nvml()
        self.amdsmi = AmdSmi(self.adapter, self.adapters)
        self.adl = Adl(self.adapter, self.adapters)
        self.inspector = ProcInspector()
        self.prober = ServerProber()
        self.scan_budget = scan_budget  # None = scan every candidate now (one-shot output)
        self.gpu_index = gpu_index

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
        return [port for ppid in d.parents for port in self.inspector.exclusive_ports(ppid)]

    def sample(self) -> Snapshot:
        per_luid, adapters = self.counters.sample()
        luid = self.adapter.luid if self.adapter else (next(iter(per_luid), 0))
        procs = per_luid.get(luid, {})
        a = adapters.get(luid)
        nvs = self.nvml.sample() if self.nvml.ok else []
        nv = select_nv_device(self.adapter, nvs, self.gpu_index, self.adapters)
        if self.adapter and self.adapter.vendor_id == 0x1002:
            nv = self.adl.sample() or self.amdsmi.sample()
        total = (nv.mem_total if nv else 0) or (self.adapter.dedicated_vram if self.adapter else 0)

        snap = Snapshot(
            taken=time.time(), adapter=self.adapter, adapters=self.adapters, nv=nv,
            driver=(self.amdsmi.driver if nv and nv.source == "AMD SMI" else self.nvml.driver if nv and nv.source == "NVML" else ""),
            cuda=self.nvml.cuda_version if nv and nv.source == "NVML" else "", vram_total=total,
            vram_used=int(a.dedicated) if a and a.dedicated is not None else None,
            shared_used=int(a.shared) if a and a.shared is not None else None,
        )

        live = [(pid, g) for pid, g in procs.items()
                if g.dedicated >= MiB or g.gpu_pct >= 0.5 or g.shared >= 16 * MiB or g.paged_out >= 16 * MiB]
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
            if row.kind in AI_KINDS and ("\\ollama\\" in d.exe.lower() or
                                         (d.name.lower().startswith("ollama") and "runner" in d.cmdline)):
                ollama_runners.append(row)

        for row in snap.procs:
            if os.path.basename(row.detail.exe).lower() == "handy.exe":
                row.handy = inspect_handy(row.detail.create_time)
            if (row.detail.is_diffusion or row.kind not in AI_KINDS or not (row.detail.models or row.kind == "LLM")
                    or not {"dedicated", "shared"} <= row.gpu.available):
                continue
            own = self.prober.get(self.inspector.exclusive_ports(row.pid))
            host = self.prober.get(self._parent_ports(row.pid))  # Ollama / LM Studio front-end process
            log = ollama_log_path() if row in ollama_runners and len(ollama_runners) == 1 else None
            row.llm = build_view(row.detail, row.gpu.dedicated, row.gpu.shared, own or host, log,
                                 host=host if own else None)

        self.inspector.prune(set(procs))
        return snap

    def close(self) -> None:
        self.counters.close()
        self.nvml.close()
        self.amdsmi.close()
        self.adl.close()

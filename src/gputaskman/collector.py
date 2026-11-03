"""Assembles one Snapshot from Windows counters, NVML, process inspection and LLM introspection."""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field

import psutil

from .dxgi import VENDOR_NVIDIA, Adapter, enumerate_adapters
from .llm import LlmView, ServerProber, build_view, gb, ollama_log_path
from .nvml import Nvml, NvDevice
from .pdh import GpuCounters, ProcGpu
from .procinfo import AI_KINDS, ProcDetail, ProcInspector, classify

MiB = 1 << 20


@dataclass
class ProcRow:
    pid: int
    name: str
    kind: str
    gpu: ProcGpu
    detail: ProcDetail
    llm: LlmView | None = None

    @property
    def runtime(self) -> str:
        return "+".join(self.detail.runtimes[:3])

    @property
    def inside(self) -> str:
        """One-line 'what's in there' summary for the process table."""
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
    nv: NvDevice | None
    driver: str
    cuda: str
    vram_total: int
    vram_used: int  # WDDM adapter-wide dedicated usage (ground truth)
    shared_used: int
    procs: list[ProcRow] = field(default_factory=list)

    @property
    def attributed(self) -> int:
        return int(sum(p.gpu.dedicated for p in self.procs))


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

    def __init__(self, gpu_index: int = 0, scan_budget: float | None = SCAN_BUDGET) -> None:
        self.adapters = [a for a in enumerate_adapters() if not a.software]
        self.adapter = select_adapter(self.adapters, gpu_index)
        self.counters = GpuCounters()
        self.nvml = Nvml()
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
            shared_used=int(a.shared) if a else 0,
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
            if row.detail.is_diffusion or row.kind not in AI_KINDS or not (row.detail.models or row.kind == "LLM"):
                continue
            own = self.prober.get(row.detail.ports)
            host = self.prober.get(self._parent_ports(row.pid))  # Ollama / LM Studio front-end process
            log = ollama_log_path() if row in ollama_runners and len(ollama_runners) == 1 else None
            row.llm = build_view(row.detail, row.gpu.dedicated, row.gpu.shared, own or host, log,
                                 host=host if own else None)

        self.inspector.prune(set(procs))
        return snap

    def close(self) -> None:
        self.counters.close()
        self.nvml.close()

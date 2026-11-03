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
            return ", ".join(m.store_name or os.path.basename(m.path) for m in self.detail.models[:2])
        if self.detail.candidates:
            return "installed (unproven): " + ", ".join(
                f"{m.store_name or os.path.basename(m.path)} {gb(m.size)}" for m in self.detail.candidates[:2])
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


class Collector:
    DEEP_TOP_N = 25  # memory-map candidates: the top N VRAM users plus anything busy
    SCAN_BUDGET = 0.15  # seconds of memory-map scanning per sample; leftovers wait for the next sample

    def __init__(self, gpu_index: int = 0, scan_budget: float | None = SCAN_BUDGET) -> None:
        self.counters = GpuCounters()
        self.nvml = Nvml()
        self.inspector = ProcInspector()
        self.prober = ServerProber()
        self.scan_budget = scan_budget  # None = scan every candidate now (one-shot output)
        self.adapters = [a for a in enumerate_adapters() if not a.software]
        nv = [a for a in self.adapters if a.vendor_id == VENDOR_NVIDIA] or self.adapters
        nv.sort(key=lambda a: -a.dedicated_vram)
        self.adapter = nv[min(gpu_index, len(nv) - 1)] if nv else None
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
        nv = nvs[self.gpu_index] if self.gpu_index < len(nvs) else None
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
            if row.kind not in AI_KINDS or not (row.detail.models or row.kind == "LLM"):
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

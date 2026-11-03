"""Plain-text / JSON snapshot output (no TUI) - handy for scripts, agents and bug reports."""

from __future__ import annotations

import dataclasses
import json
import os

from .collector import Snapshot
from .llm import gb
from .nvml import format_metric


def to_dict(s: Snapshot) -> dict:
    nv = dataclasses.asdict(s.nv) if s.nv else None
    procs = []
    for p in s.procs:
        item = {
            "pid": p.pid, "name": p.name, "kind": p.kind, "runtimes": p.detail.runtimes,
            "vram_bytes": int(p.gpu.dedicated) if "dedicated" in p.gpu.available else None,
            "shared_bytes": int(p.gpu.shared) if "shared" in p.gpu.available else None,
            "committed_bytes": int(p.gpu.committed) if "committed" in p.gpu.available else None,
            "paged_out_bytes": int(p.gpu.paged_out) if p.gpu.paged_available else None,
            "gpu_pct": round(p.gpu.gpu_pct, 1) if "engine" in p.gpu.available else None,
            "engines": {k: round(v, 1) for k, v in p.gpu.engines.items() if v} if "engine" in p.gpu.available else None,
            "ports": p.detail.ports, "inside": p.inside,
            "model_files": [dataclasses.asdict(m) for m in p.detail.models],
            "installed_models": [dataclasses.asdict(m) for m in p.detail.candidates],
        }
        if p.handy:
            item["handy"] = dataclasses.asdict(p.handy)
        if p.llm:
            item["llm"] = {
                "models": [{"path": g.path, "name": g.name, "label": p.llm.label(i), "arch": g.arch, "kind": g.kind,
                            "kind_label": p.llm.kind_label(i),
                            "kind_hint": p.llm.files[i].kind_hint if i < len(p.llm.files) else "",
                            "decision_type": g.decision_type, "params": g.n_params,
                            "quant": g.quant, "layers": g.n_layers, "ctx_train": g.ctx_train,
                            "gpu_weight_bytes": g.gpu_weight_bytes(),
                            "sources": p.llm.files[i].sources if i < len(p.llm.files) else [],
                            "store_name": p.llm.files[i].store_name if i < len(p.llm.files) else None}
                           for i, g in enumerate(p.llm.models)],
                "segments": [dataclasses.asdict(x) for x in p.llm.segments],
                "kv_bytes_per_token": p.llm.kv_per_token, "n_ctx": p.llm.n_ctx, "flags": p.llm.flags,
                "server": dataclasses.asdict(p.llm.server) if p.llm.server else None,
                "log_source": p.llm.log_source, "notes": p.llm.notes,
            }
        procs.append(item)
    return {
        "taken": s.taken, "gpu": s.adapter.name if s.adapter else None, "driver": s.driver, "cuda": s.cuda,
        "adapter": dataclasses.asdict(s.adapter) if s.adapter else None,
        "vram_total": s.vram_total, "vram_used": s.vram_used, "shared_used": s.shared_used,
        "attributed": s.attributed, "device": nv, "processes": procs,
    }


def to_text(s: Snapshot, limit: int = 30) -> str:
    out = []
    nv = s.nv
    name = s.adapter.name if s.adapter else "GPU"
    info = f"  driver {s.driver}" if s.driver else ""
    if s.cuda:
        info += f"  CUDA {s.cuda}"
    out.append(f"{name}{info}  {nv.driver_model if nv else ''}")
    used = gb(s.vram_used) if s.vram_used is not None else "N/A"
    shared = gb(s.shared_used) if s.shared_used is not None else "N/A"
    out.append(f"VRAM {used} / {gb(s.vram_total)}   shared (in system RAM) {shared}")
    if nv:
        rx = nv.pcie_rx_kbs / 1024 if nv.pcie_rx_kbs is not None else None
        tx = nv.pcie_tx_kbs / 1024 if nv.pcie_tx_kbs is not None else None
        pstate = f"P{nv.pstate}" if nv.pstate is not None else "P N/A"
        label = "AMD gfx" if nv.source.startswith("AMD ") else "GPU"
        clock = (f"gfx {format_metric(nv.gfx_clock)}" if nv.source.startswith("AMD ") else
                 f"SM {format_metric(nv.sm_clock)}")
        power_kind = f" ({nv.power_kind})" if nv.power_kind else ""
        out.append(f"{label} {format_metric(nv.util_gpu, unit='%')}  mem-bandwidth {format_metric(nv.util_mem, unit='%')}  "
                   f"{format_metric(nv.temp, unit='C')}  {format_metric(nv.power_w, '.0f')}/"
                   f"{format_metric(nv.power_limit_w, '.0f')} W{power_kind}  {pstate}  "
                   f"{clock} MHz  "
                   f"video {format_metric(nv.video_clock)} MHz  "
                   f"PCIe rx {format_metric(rx, '.0f')} MB/s "
                   f"tx {format_metric(tx, '.0f')} MB/s")
    out.append("")
    out.append(f"{'PID':>7}  {'PROCESS':<24} {'KIND':<10} {'RUNTIME':<24} {'VRAM':>9} {'SPILL':>8} {'PAGED':>8} {'GPU%':>5}  INSIDE")
    for p in s.procs[:limit]:
        spill = "N/A" if "shared" not in p.gpu.available else gb(p.gpu.shared) if p.gpu.shared >= 1 << 20 else ""
        paged = "N/A" if not p.gpu.paged_available else gb(p.gpu.paged_out) if p.gpu.paged_out >= 1 << 20 else ""
        vram = gb(p.gpu.dedicated) if "dedicated" in p.gpu.available else "N/A"
        gpu_pct = f"{p.gpu.gpu_pct:.0f}" if "engine" in p.gpu.available else "N/A"
        out.append(f"{p.pid:>7}  {p.name[:24]:<24} {p.kind:<10} {p.runtime[:24]:<24} {vram:>9} {spill:>8} "
                   f"{paged:>8} {gpu_pct:>5}  {p.inside}")
    for p in s.procs:
        if p.handy:
            out.append("")
            if p.handy.selected_model:
                out.append(f"== {p.name} ({p.pid}) Handy selected (settings, not load proof): {p.handy.selected_model}")
            if p.handy.last_event:
                out.append(f"   last Handy event: {p.handy.last_event} {p.handy.last_model or ''} "
                           f"at {p.handy.last_event_local} (runtime-reported history, not current GPU residency)")
        if not p.llm:
            continue
        out.append("")
        out.append(f"== {p.name} ({p.pid})  VRAM {gb(p.gpu.dedicated)}")
        for i, g in enumerate(p.llm.models):
            kind = p.llm.kind_label(i)
            ev = p.llm.files[i].evidence if i < len(p.llm.files) else ""
            out.append(f"   {kind}: {p.llm.label(i)}  [{g.arch}, {g.params_label} params, {g.quant}, {g.n_layers} layers, "
                       f"ctx_train {g.ctx_train}]  {os.path.basename(g.path)}  evidence: {ev}")
        for f in p.llm.other_files:
            kind = f", {f.kind_label}" if f.kind_label else ""
            out.append(f"   file: {f.store_name or f.path}  ({gb(f.size)}{kind})  evidence: {f.evidence}")
        for seg in p.llm.segments:
            out.append(f"   {gb(seg.bytes):>9}  {seg.label}{'' if seg.exact else ''}")
        if p.llm.server:
            sv = p.llm.server
            out.append(f"   server: {sv.kind} {sv.url} ctx={sv.n_ctx} slots={sv.slots_busy}/{sv.slots_total}"
                       + (f" gen={sv.gen_tok_s:.1f} tok/s" if sv.gen_tok_s is not None else ""))
        if p.llm.log_source:
            out.append(f"   exact sizes from log: {p.llm.log_source}")
        for n in p.llm.notes:
            out.append(f"   - {n}")
    return "\n".join(out)


def dump(s: Snapshot, as_json: bool) -> str:
    return json.dumps(to_dict(s), indent=2, default=str) if as_json else to_text(s)

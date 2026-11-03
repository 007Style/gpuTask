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
            "vram_bytes": int(p.gpu.dedicated), "shared_bytes": int(p.gpu.shared),
            "committed_bytes": int(p.gpu.committed), "paged_out_bytes": int(p.gpu.paged_out),
            "gpu_pct": round(p.gpu.gpu_pct, 1), "engines": {k: round(v, 1) for k, v in p.gpu.engines.items() if v},
            "ports": p.detail.ports, "inside": p.inside,
            "model_files": [dataclasses.asdict(m) for m in p.detail.models],
            "installed_models": [dataclasses.asdict(m) for m in p.detail.candidates],
        }
        if p.llm:
            item["llm"] = {
                "models": [{"path": g.path, "name": g.name, "label": p.llm.label(i), "arch": g.arch, "kind": g.kind,
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
        "vram_total": s.vram_total, "vram_used": s.vram_used, "shared_used": s.shared_used,
        "attributed": s.attributed, "device": nv, "processes": procs,
    }


def to_text(s: Snapshot, limit: int = 30) -> str:
    out = []
    nv = s.nv
    name = s.adapter.name if s.adapter else "GPU"
    out.append(f"{name}  driver {s.driver}  CUDA {s.cuda}  {nv.driver_model if nv else ''}")
    out.append(f"VRAM {gb(s.vram_used)} / {gb(s.vram_total)}   shared (in system RAM) {gb(s.shared_used)}")
    if nv:
        rx = nv.pcie_rx_kbs / 1024 if nv.pcie_rx_kbs is not None else None
        tx = nv.pcie_tx_kbs / 1024 if nv.pcie_tx_kbs is not None else None
        pstate = f"P{nv.pstate}" if nv.pstate is not None else "P N/A"
        out.append(f"GPU {format_metric(nv.util_gpu, unit='%')}  mem-bandwidth {format_metric(nv.util_mem, unit='%')}  "
                   f"{format_metric(nv.temp, unit='C')}  {format_metric(nv.power_w, '.0f')}/"
                   f"{format_metric(nv.power_limit_w, '.0f')} W  {pstate}  "
                   f"SM {format_metric(nv.sm_clock)} MHz  PCIe rx {format_metric(rx, '.0f')} MB/s "
                   f"tx {format_metric(tx, '.0f')} MB/s")
    out.append("")
    out.append(f"{'PID':>7}  {'PROCESS':<24} {'KIND':<10} {'RUNTIME':<24} {'VRAM':>9} {'SPILL':>8} {'PAGED':>8} {'GPU%':>5}  INSIDE")
    for p in s.procs[:limit]:
        spill = gb(p.gpu.shared) if p.gpu.shared >= 1 << 20 else ""
        paged = gb(p.gpu.paged_out) if p.gpu.paged_out >= 1 << 20 else ""
        out.append(f"{p.pid:>7}  {p.name[:24]:<24} {p.kind:<10} {p.runtime[:24]:<24} {gb(p.gpu.dedicated):>9} {spill:>8} "
                   f"{paged:>8} {p.gpu.gpu_pct:>5.0f}  {p.inside}")
    for p in s.procs:
        if not p.llm:
            continue
        out.append("")
        out.append(f"== {p.name} ({p.pid})  VRAM {gb(p.gpu.dedicated)}")
        for i, g in enumerate(p.llm.models):
            kind = g.kind + (f" ({g.decision_type})" if g.decision_type else "")
            ev = p.llm.files[i].evidence if i < len(p.llm.files) else ""
            out.append(f"   {kind}: {p.llm.label(i)}  [{g.arch}, {g.params_label} params, {g.quant}, {g.n_layers} layers, "
                       f"ctx_train {g.ctx_train}]  {os.path.basename(g.path)}  evidence: {ev}")
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

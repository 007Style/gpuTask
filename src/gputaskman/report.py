"""Plain-text / JSON snapshot output (no TUI) - handy for scripts, agents and bug reports."""

from __future__ import annotations

import dataclasses
import json
import os

from .collector import ProcRow, Snapshot
from .llm import gb
from .nvml import format_metric
from .pdh import ProcGpu


WSL_HOST_NOTE = ("WSL + Windows host counters: 'WSL VM' is the exact total of Linux GPU memory (all distros). "
                 "Linux rows get their size by matching the VM's GPU contexts (* matched, ? guess). "
                 "W: rows are Windows processes.")


def wsl_note(s: Snapshot) -> str:
    if s.host_data:
        return WSL_HOST_NOTE
    status = f" Windows host counters: {s.host_status}." if s.host_status else ""
    return WSL_NOTE + status


WSL_EMPTY = ("No Linux process in this distro has the GPU open (other WSL distros and Docker Desktop "
             "are not visible from here).")
WSL_NOTE = ("WSL: VRAM and engine load are device-wide. The Windows host charges every Linux process's "
            "VRAM to the VM, so rows below are processes using the GPU, not their share of it.")


def has_wddm_counters(s: Snapshot, row: ProcRow) -> bool:
    return s.backend == "wddm" or row.origin == "windows"


def has_engine_counters(s: Snapshot, row: ProcRow) -> bool:
    return s.backend != "wsl" or row.origin == "windows"


def vram_label(g: ProcGpu) -> str:
    """Measured size; '*' / '?' when the size is exact but which local process owns it is inferred."""
    mark = {"order": "?", "consistent": "*"}.get(g.inferred_by, "")
    return "n/a" if g.shown_vram is None else gb(g.shown_vram) + mark


INFERRED_NOTE = ("* matched: the size is exact (Windows' counter for a WSL VM GPU context, or NVML for a host PID "
                 "seen from a container); which local process owns it is inferred, and only one explanation fits "
                 "(or every one that fits gives the same size).")
ORDER_NOTE = ("? order guess: the size is exact but several owners fit, so newer processes are paired with newer "
              "GPU contexts / PIDs and outside ones are assumed oldest. Cached foreign-size hints only rank extra "
              "candidates; they never prove ownership. Becomes * with evidence, such as a process starting while "
              "gputaskman runs.")


def inferred_note(g) -> str:
    return ORDER_NOTE if g.inferred_by == "order" else INFERRED_NOTE


METAL_NOTE = ("macOS: unified memory, so GPU memory is part of system RAM. Rows show each process's graphics "
              "footprint; other users' processes (WindowServer, daemons) show GPU% only, since reading their "
              "memory needs root.")


def to_dict(s: Snapshot) -> dict:
    nv = dataclasses.asdict(s.nv) if s.nv else None
    procs = []
    for p in s.procs:
        wddm = has_wddm_counters(s, p)
        engines = has_engine_counters(s, p)
        item = {
            "pid": p.pid, "host": p.origin if p.host else None, "name": p.name, "kind": p.kind, "runtimes": p.detail.runtimes,
            "vram_attributed": p.gpu.attributed, "vram_inferred_bytes": p.gpu.inferred,
            "vram_inferred_by": p.gpu.inferred_by or None,
            "vram_bytes": int(p.gpu.dedicated) if p.gpu.attributed else None,
            "shared_bytes": int(p.gpu.shared) if wddm else None,
            "committed_bytes": int(p.gpu.committed) if wddm else None,
            "paged_out_bytes": int(p.gpu.paged_out) if wddm else None,
            "gpu_pct": round(p.gpu.gpu_pct, 1) if engines else None,
            "engines": {k: round(v, 1) for k, v in p.gpu.engines.items() if v} if engines else None,
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
        "vram_total": s.vram_total, "vram_used": s.vram_used, "shared_used": s.shared_used,
        "attributed": s.attributed, "backend": s.backend, "host_status": s.host_status or None,
        "wsl_split": s.wsl_split or None, "device": nv, "processes": procs,
    }


def to_text(s: Snapshot, limit: int = 30) -> str:
    out = []
    nv = s.nv
    name = s.adapter.name if s.adapter else "GPU"
    if s.backend == "metal":
        out.append(f"{name}  {s.driver}  Metal")
        out.append(f"GPU memory {gb(s.vram_used)} allocated / {gb(s.vram_total)} unified")
        out.append(METAL_NOTE)
        if nv:
            out.append(f"GPU {format_metric(nv.util_gpu, unit='%')}")
    else:
        out.append(f"{name}  driver {s.driver}  CUDA {s.cuda}  {nv.driver_model if nv else ''}")
        shared = f"   shared (in system RAM) {gb(s.shared_used)}" if s.backend == "wddm" or s.host_data else ""
        out.append(f"VRAM {gb(s.vram_used)} / {gb(s.vram_total)}{shared}")
    if s.backend == "wsl":
        out.append(wsl_note(s))
    if nv and s.backend != "metal":
        rx = nv.pcie_rx_kbs / 1024 if nv.pcie_rx_kbs is not None else None
        tx = nv.pcie_tx_kbs / 1024 if nv.pcie_tx_kbs is not None else None
        pstate = f"P{nv.pstate}" if nv.pstate is not None else "P N/A"
        out.append(f"GPU {format_metric(nv.util_gpu, unit='%')}  mem-bandwidth {format_metric(nv.util_mem, unit='%')}  "
                   f"{format_metric(nv.temp, unit='C')}  {format_metric(nv.power_w, '.0f')}/"
                   f"{format_metric(nv.power_limit_w, '.0f')} W  {pstate}  "
                   f"SM {format_metric(nv.sm_clock)} MHz  "
                   f"video {format_metric(nv.video_clock)} MHz  "
                   f"PCIe rx {format_metric(rx, '.0f')} MB/s "
                   f"tx {format_metric(tx, '.0f')} MB/s")
    out.append("")
    shown = s.procs[:limit]
    w = max([7] + [len(p.pid_label) for p in shown])  # host PIDs like W:120880 are wider
    out.append(f"{'PID':>{w}}  {'PROCESS':<24} {'KIND':<10} {'RUNTIME':<24} {'VRAM':>9} {'SPILL':>8} {'PAGED':>8} {'GPU%':>5}  INSIDE")
    for p in shown:
        wddm = has_wddm_counters(s, p)
        spill = gb(p.gpu.shared) if wddm and p.gpu.shared >= 1 << 20 else ""
        paged = gb(p.gpu.paged_out) if wddm and p.gpu.paged_out >= 1 << 20 else ""
        pid = p.pid_label
        pct = f"{p.gpu.gpu_pct:.0f}" if has_engine_counters(s, p) else "n/a"
        out.append(f"{pid:>{w}}  {p.name[:24]:<24} {p.kind:<10} {p.runtime[:24]:<24} {vram_label(p.gpu):>9} {spill:>8} "
                   f"{paged:>8} {pct:>5}  {p.inside}")
    if any(p.gpu.inferred_by == "consistent" for p in shown):
        out.append(INFERRED_NOTE)
    if any(p.gpu.inferred_by == "order" for p in shown):
        out.append(ORDER_NOTE)
    if s.backend == "wsl" and s.wsl_split and any(not p.host and not p.gpu.attributed and p.gpu.inferred is None
                                                  for p in s.procs):
        out.append(f"Linux VRAM split: {s.wsl_split}")
    if s.backend == "wsl" and not any(not p.host for p in s.procs):
        out.append(f"{'':>{w}}  {WSL_EMPTY}")
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
        out.append(f"== {p.name} ({p.pid_label})  VRAM {vram_label(p.gpu)}")
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

"""What's inside an LLM process's VRAM: weights vs KV cache vs compute/runtime.

Evidence, best first:
  1. llama.cpp / Ollama load logs  -> exact per-device buffer sizes
  2. Local server endpoints        -> llama-server /props,/v1/models,/slots,/metrics; Ollama /api/ps;
                                      LM Studio /api/v1/models (falls back to /api/v0/models)
  3. GGUF header + command line    -> estimates (weights per layer, KV bytes per token x context)
The process's real VRAM total always comes from Windows, so the remainder is never invented.
"""

from __future__ import annotations

import dataclasses
import json
import os
import re
import time
import urllib.request
from dataclasses import dataclass, field

from .gguf import GgufInfo, is_gguf, read_gguf
from .procinfo import MappedModel, ProcDetail, make_model

MiB = 1 << 20
KV_TYPE_BYTES = {"f32": 4, "f16": 2, "bf16": 2, "q8_0": 34 / 32, "q4_0": 18 / 32, "q4_1": 20 / 32,
                 "q5_0": 22 / 32, "q5_1": 24 / 32, "iq4_nl": 18 / 32}

_FLAGS = {
    "-c": "ctx", "--ctx-size": "ctx", "-ngl": "ngl", "--n-gpu-layers": "ngl", "--gpu-layers": "ngl",
    "-np": "parallel", "--parallel": "parallel", "-ctk": "ctk", "--cache-type-k": "ctk",
    "-ctv": "ctv", "--cache-type-v": "ctv", "--port": "port", "--log-file": "log_file",
}
_BOOL_FLAGS = {"--no-mmap": "no_mmap"}


def parse_flags(cmd: list[str]) -> dict:
    out: dict = {}
    for i, arg in enumerate(cmd):
        key, val = arg, None
        if arg.startswith("--") and "=" in arg:
            key, val = arg.split("=", 1)
        if key in _BOOL_FLAGS:
            out[_BOOL_FLAGS[key]] = True
        elif key in _FLAGS:
            if val is None and i + 1 < len(cmd):
                val = cmd[i + 1]
            if val is not None:
                name = _FLAGS[key]
                if name in ("ctx", "ngl", "parallel", "port"):
                    try:
                        out[name] = int(val)
                    except ValueError:
                        if name == "ngl" and val in ("all", "auto"):
                            out[name] = -1
                else:
                    out[name] = val
    return out


# ---------------------------------------------------------------- logs (exact numbers)

_LLAMA_BUF = re.compile(r"(\S+)\s+(model|KV|compute|output|RS)\s+buffer size\s*=\s*([\d.]+)\s*MiB")
_OLLAMA_BUF = re.compile(r'msg="(model weights|kv cache|compute graph)"\s+device=(\S+)\s+size="([\d.]+)\s*(GiB|MiB|KiB|B)"')
_UNIT = {"GiB": 1 << 30, "MiB": 1 << 20, "KiB": 1 << 10, "B": 1}
_KIND = {"model": "weights", "model weights": "weights", "KV": "kv", "kv cache": "kv", "RS": "kv",
         "compute": "compute", "compute graph": "compute", "output": "compute"}


def is_gpu_device(dev: str) -> bool:
    d = dev.upper()
    return not (d.startswith("CPU") or "HOST" in d or d.startswith("MMAP"))


def parse_load_log(path: str, not_before: float = 0.0) -> dict[str, dict[str, int]] | None:
    """Most recent model-load block in a llama.cpp/Ollama log -> {device: {kind: bytes}}."""
    try:
        if os.path.getmtime(path) < not_before:
            return None
        with open(path, "rb") as f:
            f.seek(0, 2)
            f.seek(max(f.tell() - (512 << 10), 0))
            lines = f.read().decode("utf-8", "replace").splitlines()
    except OSError:
        return None
    start = None
    for i in range(len(lines) - 1, -1, -1):  # last load starts at its first weights line
        if "model buffer size" in lines[i] or 'msg="model weights"' in lines[i]:
            start = i
            while start > 0 and ("model buffer size" in lines[start - 1] or 'msg="model weights"' in lines[start - 1]):
                start -= 1
            break
    if start is None:
        return None
    out: dict[str, dict[str, int]] = {}
    for line in lines[start:]:
        m = _LLAMA_BUF.search(line)
        if m:
            dev, kind, n = m[1].rstrip(":"), _KIND[m[2]], int(float(m[3]) * MiB)
        else:
            m = _OLLAMA_BUF.search(line)
            if not m:
                continue
            dev, kind, n = m[2], _KIND[m[1]], int(float(m[3]) * _UNIT[m[4]])
        out.setdefault(dev, {}).setdefault(kind, 0)
        out[dev][kind] += n
    return out or None


def ollama_log_path() -> str:
    return os.path.join(os.environ.get("LOCALAPPDATA", ""), "Ollama", "server.log")


# ---------------------------------------------------------------- local servers

_opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def _get(url: str, timeout: float = 0.5):
    try:
        with _opener.open(urllib.request.Request(url, headers={"Accept": "application/json"}), timeout=timeout) as r:
            if r.status != 200:
                return None
            body = r.read(4 << 20)
    except Exception:
        return None
    try:
        return json.loads(body)
    except ValueError:
        return body.decode("utf-8", "replace")


@dataclass
class ServerInfo:
    kind: str  # llama-server | ollama | lmstudio
    url: str
    n_ctx: int | None = None
    slots_total: int | None = None
    slots_busy: int | None = None
    gen_tok_s: float | None = None
    prompt_tok_s: float | None = None
    model_path: str | None = None  # llama-server: the file it says it loaded
    models: list[dict] = field(default_factory=list)  # normalized: name, type, quant, ctx, size, size_vram, path


def _metric(text: str, name: str) -> float | None:
    m = re.search(rf"^llamacpp:{name}\s+([\d.eE+-]+)", text, re.M)
    return float(m[1]) if m else None


def _model_entry(name, type_=None, quant=None, ctx=None, size=None, size_vram=None, path=None) -> dict:
    return {"name": name, "type": type_, "quant": quant, "ctx": ctx, "size": size, "size_vram": size_vram, "path": path}


_MODALITY = {"text": "llm", "decisions": "decision", "embeddings": "embedding", "embedding": "embedding"}


def llama_server_models(v1: dict, model_path: str | None) -> list[dict]:
    out = []
    for m in v1.get("data") or []:
        outs = (m.get("architecture") or {}).get("output_modalities") or []
        meta = m.get("meta") or {}
        out.append(_model_entry(os.path.basename(m.get("id") or model_path or "?"),
                                next((_MODALITY.get(o, o) for o in outs), None),
                                meta.get("ftype"), meta.get("n_ctx"), meta.get("size"), None, model_path))
    return out


def ollama_models(ps: dict) -> list[dict]:
    return [_model_entry(m.get("name"), "llm", (m.get("details") or {}).get("quantization_level"),
                         m.get("context_length"), m.get("size"), m.get("size_vram"))
            for m in ps.get("models") or []]


def lmstudio_models(v1: dict | None, v0: dict | None) -> list[dict] | None:
    """Loaded models from LM Studio's /api/v1/models (preferred) or /api/v0/models."""
    if isinstance(v1, dict) and isinstance(v1.get("models"), list):
        out = []
        for m in v1["models"]:
            for inst in m.get("loaded_instances") or []:
                q = m.get("quantization") or {}
                t = m.get("type")
                if t == "decision" and m.get("decision_type"):
                    t = f"decision ({m['decision_type']})"
                out.append(_model_entry(inst.get("id") or m.get("key"), t, q.get("name"),
                                        (inst.get("config") or {}).get("context_length"), m.get("size_bytes")))
        return out
    if isinstance(v0, dict) and isinstance(v0.get("data"), list):
        return [_model_entry(m.get("id"), m.get("type"), m.get("quantization"),
                             m.get("loaded_context_length") or m.get("max_context_length"))
                for m in v0["data"] if m.get("state") == "loaded"]
    return None


def probe_server(port: int, prev: ServerInfo | None = None) -> ServerInfo | None:
    """Read-only GETs. Never calls inference, load or download endpoints."""
    base = f"http://127.0.0.1:{port}"
    props = _get(base + "/props")
    if isinstance(props, dict) and ("default_generation_settings" in props or "model_path" in props):
        s = ServerInfo("llama-server", base, model_path=props.get("model_path"))
        dgs = props.get("default_generation_settings") or {}
        s.n_ctx = dgs.get("n_ctx") or props.get("n_ctx")
        s.slots_total = props.get("total_slots")
        if prev and prev.kind == "llama-server" and prev.model_path == s.model_path:
            s.models = prev.models  # identity only changes when the model does
        else:
            v1 = _get(base + "/v1/models")
            s.models = llama_server_models(v1, s.model_path) if isinstance(v1, dict) else []
        slots = _get(base + "/slots")
        if isinstance(slots, list):
            s.slots_total = len(slots)
            s.slots_busy = sum(1 for x in slots if x.get("is_processing"))
        metrics = _get(base + "/metrics")
        if isinstance(metrics, str):
            s.gen_tok_s = _metric(metrics, "predicted_tokens_seconds")
            s.prompt_tok_s = _metric(metrics, "prompt_tokens_seconds")
        return s
    ps = _get(base + "/api/ps")
    if isinstance(ps, dict) and isinstance(ps.get("models"), list):
        return ServerInfo("ollama", base, models=ollama_models(ps))
    # Older LM Studio answers unknown endpoints with HTTP 200 + {"error": ...}, so fall back on shape, not status
    loaded = lmstudio_models(_get(base + "/api/v1/models"), None)
    if loaded is None:
        loaded = lmstudio_models(None, _get(base + "/api/v0/models"))
    if loaded is not None:
        return ServerInfo("lmstudio", base, models=loaded)
    return None


class ServerProber:
    """Throttled, cached probes. Only localhost, only read-only GET endpoints."""

    TTL_HIT, TTL_MISS = 3.0, 30.0

    def __init__(self) -> None:
        self._cache: dict[int, tuple[float, ServerInfo | None]] = {}
        self._last: dict[int, ServerInfo] = {}

    def get(self, ports: list[int]) -> ServerInfo | None:
        now = time.time()
        for port in ports:
            hit = self._cache.get(port)
            if hit and now - hit[0] < (self.TTL_HIT if hit[1] else self.TTL_MISS):
                info = hit[1]
            else:
                info = probe_server(port, self._last.get(port))
                self._cache[port] = (now, info)
                if info:
                    self._last[port] = info
            if info:
                return info
        return None


# ---------------------------------------------------------------- the breakdown


@dataclass
class Segment:
    label: str
    bytes: int
    kind: str  # weights | kv | compute | runtime | unknown
    exact: bool = False


@dataclass
class LlmView:
    models: list[GgufInfo] = field(default_factory=list)
    files: list[MappedModel] = field(default_factory=list)  # evidence for models[i] (same order)
    other_files: list[MappedModel] = field(default_factory=list)
    flags: dict = field(default_factory=dict)
    server: ServerInfo | None = None
    host: ServerInfo | None = None  # managing front-end (Ollama / LM Studio) when the runner has its own server
    log_source: str | None = None
    log: dict | None = None
    segments: list[Segment] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    kv_per_token: int | None = None
    kv_type: str = "f16"
    n_ctx: int | None = None

    def label(self, i: int) -> str:
        """Best human name for models[i]: model-store name, else GGUF basename/name."""
        f = self.files[i] if i < len(self.files) else None
        g = self.models[i]
        return (f.store_name if f and f.store_name else None) or g.metadata.get("general.basename") or g.name


def gb(n: float) -> str:
    if abs(n) >= 1 << 30:
        return f"{n / (1 << 30):.2f} GB"
    return f"{n / MiB:.0f} MB"


_SIDE_KINDS = ("vision projector", "LoRA adapter")


def _merge_api(files: list[MappedModel], server: ServerInfo | None, host: ServerInfo | None) -> list[dict]:
    """Mark files a runtime API also reports; return the API entries that matched this process."""
    matched: list[dict] = []
    if server and server.kind == "llama-server" and server.model_path:
        key = os.path.normcase(os.path.abspath(server.model_path))
        hit = next((f for f in files if os.path.normcase(os.path.abspath(f.path)) == key), None)
        if hit:
            hit.sources.append("api")
        elif os.path.isfile(server.model_path):  # loaded without mmap or via the router: API-only evidence
            files.insert(0, make_model(server.model_path, os.path.getsize(server.model_path), 0,
                                       is_gguf(server.model_path), "api"))
    for front in (s for s in (server, host) if s and s.kind == "ollama"):
        for f in files:
            names = set((f.store_name or "").split(", "))
            for om in front.models:
                if om["name"] in names:
                    f.sources.append("api")
                    matched.append(om)
    lms = host if host and host.kind == "lmstudio" else (server if server and server.kind == "lmstudio" else None)
    primary = [f for f in files if f.kind not in _SIDE_KINDS]
    if lms and len(lms.models) == 1 and len(primary) == 1:  # LM Studio keys don't map to paths; only pair 1:1
        primary[0].sources.append("api")
        matched.append(lms.models[0])
    return matched


def build_view(d: ProcDetail, vram: float, shared: float, server: ServerInfo | None,
               log_path: str | None = None, host: ServerInfo | None = None) -> LlmView:
    v = LlmView(flags=parse_flags(d.cmdline), server=server, host=host)
    files = [dataclasses.replace(m, sources=list(m.sources)) for m in d.models]  # cached details stay untouched
    matched = _merge_api(files, server, host)
    for m in files:
        g = read_gguf(m.path) if m.gguf else None
        if g is not None and not g.error:
            v.models.append(g)
            v.files.append(m)
        else:
            v.other_files.append(m)

    log_path = log_path or v.flags.get("log_file")
    if log_path:
        v.log = parse_load_log(log_path, not_before=d.create_time - 5)
        v.log_source = log_path if v.log else None

    main = next((g for g in v.models if g.kind == "LLM" and g.n_layers), None)  # only causal LLMs grow a KV cache
    ctk = str(v.flags.get("ctk", "f16")).lower()
    ctv = str(v.flags.get("ctv", "f16")).lower()
    v.kv_type = ctk if ctk == ctv else f"K {ctk}/V {ctv}"
    if main:
        elems = main.kv_bytes_per_token(1.0)  # K+V elements per token (K and V are usually equal width)
        if elems:
            v.kv_per_token = int(elems * (KV_TYPE_BYTES.get(ctk, 2) + KV_TYPE_BYTES.get(ctv, 2)) / 2)
    v.n_ctx = v.flags.get("ctx") or (server.n_ctx if server and server.kind == "llama-server" else None)
    if main:
        for om in matched or (server.models if server and server.kind == "ollama" else []):
            if om.get("ctx"):
                v.n_ctx = v.n_ctx or om["ctx"]
    primaries = [g for g in v.models if g.kind not in _SIDE_KINDS]
    no_kv = bool(primaries) and all(not g.causal for g in primaries)

    total = int(vram)
    if v.log:
        gpu = {dev: kinds for dev, kinds in v.log.items() if is_gpu_device(dev)}
        acc = 0
        for kind, label in (("weights", "weights"), ("kv", "KV cache"), ("compute", "compute buffers")):
            n = sum(k.get(kind, 0) for k in gpu.values())
            if n:
                devs = ",".join(sorted(dev for dev, k in gpu.items() if k.get(kind)))
                v.segments.append(Segment(f"{label} ({devs})", n, kind, exact=True))
                acc += n
        cpu_w = sum(k.get("weights", 0) for dev, k in v.log.items() if not is_gpu_device(dev))
        if cpu_w and main and cpu_w > main.token_embd_bytes * 1.5:
            v.notes.append(f"{gb(cpu_w)} of weights are on the CPU side (partial offload or CPU-mapped).")
        if total > acc:
            v.segments.append(Segment("CUDA context, cuBLAS & other", total - acc, "runtime"))
    else:
        ngl = v.flags.get("ngl") if len(v.models) == 1 else None
        weights = sum(g.gpu_weight_bytes(ngl) for g in v.models if g.kind != "LoRA adapter")
        weights += sum(g.tensor_bytes for g in v.models if g.kind == "LoRA adapter")
        if weights > total * 1.03 and total > 0:
            v.notes.append(f"Weights need ~{gb(weights)} but the process holds {gb(total)} of VRAM: "
                           "partial offload, paging, or placement on another GPU may explain the gap.")
            weights = total
        if weights:
            label = "weights" if len(v.models) == 1 else f"weights ({len(v.models)} files)"
            v.segments.append(Segment(label, weights, "weights"))
        rest = total - weights
        kv = v.kv_per_token * v.n_ctx if (v.kv_per_token and v.n_ctx) else 0
        if kv and kv <= rest:
            swa = main.sliding_window if main else None
            v.segments.append(Segment(f"KV cache ({v.n_ctx:,} ctx, {v.kv_type}) est.{' max' if swa else ''}", kv, "kv"))
            if swa:
                v.notes.append(f"Sliding-window attention ({swa} tokens): most layers keep a short KV window, "
                               "so the real KV cache is smaller than this estimate.")
            rest -= kv
            if rest > 0:
                v.segments.append(Segment("compute buffers + CUDA context", rest, "runtime"))
        elif rest > 0 and no_kv:
            v.segments.append(Segment("compute buffers + CUDA context", rest, "runtime"))
        elif rest > 0:
            v.segments.append(Segment("KV cache + compute + CUDA context", rest, "unknown"))

    if not v.models and not v.log and total > 0:
        v.notes.append("No GGUF mapped. Model may be loaded with --no-mmap or via another format.")
    if no_kv:
        kinds = ", ".join(sorted({f"{g.decision_type} {g.kind}" if g.decision_type else g.kind for g in primaries}))
        v.notes.append(f"Non-causal encoder ({kinds}): no KV cache grows with use; VRAM is weights + compute.")
    if main and v.kv_per_token:
        v.notes.append(f"KV cost {v.kv_per_token / 1024:.0f} KB/token ({v.kv_type}): "
                       f"4k ctx = {gb(v.kv_per_token * 4096)}, 32k ctx = {gb(v.kv_per_token * 32768)}.")
    if shared > 256 * MiB:
        v.notes.append(f"! {gb(shared)} is in SHARED GPU memory (system RAM over PCIe). "
                       "May include pinned host buffers; under VRAM pressure, fallback can slow inference.")
    if total < 64 * MiB and v.models:
        v.notes.append("Model is mapped but almost nothing is in VRAM: CPU inference, or another GPU.")
    for front in (s for s in (server, host) if s and s.kind == "ollama"):
        mine = [om for om in front.models if om in matched]
        for om in mine or front.models:  # unmatched: show everything Ollama has loaded
            if om.get("size"):
                pct = 100 * (om.get("size_vram") or 0) / om["size"]
                v.notes.append(f"Ollama{'' if mine else ' (not matched to this runner)'}: {om['name']} "
                               f"{gb(om['size'])} total, {pct:.0f}% on GPU, ctx {om.get('ctx') or '?'}.")
    if host and host.kind == "lmstudio":
        for m in host.models[:4]:
            v.notes.append(f"LM Studio: {m['name']} {m.get('type') or ''} {m.get('quant') or ''} "
                           f"ctx {m.get('ctx') or '?'}.")
    return v

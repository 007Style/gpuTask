"""Per-process introspection: which GPU runtimes are loaded, which model files are mapped."""

from __future__ import annotations

import fnmatch
import os
import re
import time
from dataclasses import dataclass, field

import psutil

from .gguf import is_gguf, read_gguf
from .stores import store_name

# (tag, dll patterns). Order = display priority (most specific first).
RUNTIMES: list[tuple[str, tuple[str, ...]]] = [
    ("llama.cpp", ("llama.dll", "libllama*.dll", "llama-*.dll", "llm_engine*")),
    ("whisper.cpp", ("whisper*.dll",)),
    ("ggml", ("ggml*.dll",)),
    ("PyTorch", ("torch_cuda*.dll", "torch_cpu.dll")),
    ("TensorRT", ("nvinfer*.dll",)),
    ("ONNX RT", ("onnxruntime*.dll",)),
    ("DirectML", ("directml.dll",)),
    ("cuDNN", ("cudnn*.dll",)),
    ("CUDA", ("nvcuda.dll", "nvcuda64.dll", "cudart64_*.dll", "cublas64_*.dll")),
    ("Vulkan", ("vulkan-1.dll",)),
    ("D3D12", ("d3d12core.dll",)),
    ("OpenGL", ("opengl32.dll",)),
    ("D3D11", ("d3d11.dll",)),
    ("NVENC", ("nvencodeapi64.dll",)),
    ("NVDEC", ("nvcuvid.dll", "nvcuvid64.dll")),
]
AI_RUNTIMES = {"llama.cpp", "whisper.cpp", "ggml", "PyTorch", "TensorRT", "ONNX RT", "DirectML", "cuDNN"}
_RUNTIME_RX = [(tag, re.compile("|".join(fnmatch.translate(p) for p in pats))) for tag, pats in RUNTIMES]
MODEL_EXTS = {".gguf", ".ggml", ".safetensors", ".onnx", ".pt", ".pth", ".bin", ".engine", ".plan", ".tflite", ".ckpt"}
AMBIGUOUS_EXTS = {".bin", ".pt", ".pth", ".ckpt"}  # also used for caches/tiles, so require size or a model store
_NOT_MODELS = {".dll", ".exe", ".sys", ".mui", ".ttf", ".ttc", ".otf", ".dat", ".db", ".nls", ".pak", ".mun"}
BROWSERS = {"msedge.exe", "chrome.exe", "firefox.exe", "msedgewebview2.exe", "brave.exe", "opera.exe", "arc.exe", "vivaldi.exe"}
LLM_NAMES = ("llama-", "llama.", "ollama", "lm studio", "lmstudio", "llmster", "koboldcpp", "jan", "gpt4all", "llamafile")
# GGUF file kind -> process kind, in display priority order
MODEL_KINDS = {"LLM": "LLM", "decision model": "decision", "embedding model": "embedding",
               "reranker": "embedding", "speech model": "speech"}
AI_KINDS = {"LLM", "decision", "decision?", "embedding", "speech", "diffusion", "AI"}
_DECISION_NAME = re.compile(r"(?<![a-z0-9])(?:openjev|laya|julia-1|lev|kev)(?![a-z0-9])", re.I)
MiB = 1 << 20
INSTALLED_TTL = 300.0
_SKIP_DIRS = {"ebwebview", "cache", "code cache", "gpucache", "node_modules", "logs", "crashpad", "locales", "__pycache__"}
_installed: dict[str, tuple[float, list[MappedModel]]] = {}


@dataclass
class MappedModel:
    path: str
    size: int
    resident: int  # bytes of the file currently in this process's RAM working set
    gguf: bool
    sources: list[str] = field(default_factory=lambda: ["mapped"])  # mapped | cmdline | api | installed
    store_name: str | None = None  # e.g. "qwen3:14b" from an Ollama manifest
    kind: str = ""  # GGUF kind: LLM, decision model, embedding model, ...
    kind_hint: str = ""  # name-based suggestion only; never changes loading evidence or GGUF math

    @property
    def kind_label(self) -> str:
        return f"likely {self.kind_hint} (name)" if self.kind_hint else self.kind

    def add_source(self, source: str) -> None:
        if source not in self.sources:
            self.sources.append(source)

    @property
    def evidence(self) -> str:
        s = set(self.sources)
        how = "+".join(x for x in ("api", "mapped", "cmdline") if x in s)
        if len(s) >= 2 and s & {"api", "mapped"}:
            return f"confirmed ({how})"
        if not how and "installed" in s:
            return "installed, not proven loaded"
        return {"mapped": "in use (mapped)", "cmdline": "launch arg", "api": "reported (api)"}.get(how, how)


def _is_diffusion_cmdline(cmd: list[str]) -> bool:
    return any(arg == "--diffusion-model" or arg.startswith("--diffusion-model=") for arg in cmd)


@dataclass
class ProcDetail:
    pid: int
    name: str = "?"
    exe: str = ""
    cmdline: list[str] = field(default_factory=list)
    create_time: float = 0.0
    runtimes: list[str] = field(default_factory=list)
    models: list[MappedModel] = field(default_factory=list)
    candidates: list[MappedModel] = field(default_factory=list)  # installed with the app, not proven loaded
    ports: list[int] = field(default_factory=list)
    parents: list[int] | None = None  # parent PIDs, looked up once (each lookup enumerates all processes)
    access_denied: bool = False
    maps_time: float = 0.0  # when memory maps were last scanned
    scan_mem: float = 0.0  # process GPU memory (dedicated + shared) at that scan

    @property
    def is_ai(self) -> bool:
        return bool(self.models) or any(r in AI_RUNTIMES for r in self.runtimes)

    @property
    def is_diffusion(self) -> bool:
        return _is_diffusion_cmdline(self.cmdline)


def _match_runtimes(dll_names: set[str]) -> list[str]:
    return [tag for tag, rx in _RUNTIME_RX if any(rx.match(d) for d in dll_names)]


def _models_from_cmdline(cmd: list[str]) -> list[str]:
    paths = []
    flags = {"-m", "--model", "--mmproj", "-md", "--model-draft", "--lora"}
    if _is_diffusion_cmdline(cmd):
        flags.update(("--diffusion-model", "--vae", "--llm"))
    for i, arg in enumerate(cmd):
        val = None
        key, sep, value = arg.partition("=")
        if key in flags:
            val = value if sep else (cmd[i + 1] if i + 1 < len(cmd) else None)
        if val and os.path.isfile(val):
            paths.append(os.path.abspath(val))
    return paths


def _is_model_file(ext: str, size: int, gg: bool, store: str | None) -> bool:
    if gg:
        return True
    if ext not in MODEL_EXTS:
        return False
    if ext in AMBIGUOUS_EXTS:
        return size >= 64 * MiB or store is not None
    return size >= 4 * MiB


def make_model(path: str, size: int, resident: int, gg: bool, source: str, store: str | None = None) -> MappedModel:
    g = read_gguf(path) if gg else None
    kind = g.kind if g is not None and not g.error else ""
    store = store if store is not None else store_name(path)
    hint = _name_kind_hint(path, store) if kind in ("", "model", "LLM") else ""
    return MappedModel(path, size, resident, gg, [source], store, kind, hint)


def _name_kind_hint(path: str, store: str | None) -> str:
    names = [os.path.basename(path), os.path.basename(os.path.dirname(path))]
    # Match the model name, not a publisher, user directory, or Ollama tag.
    names += [name.rsplit("/", 1)[-1].split(":", 1)[0] for name in (store or "").split(", ")]
    return "decision model" if any(_DECISION_NAME.search(name) for name in names) else ""


def _walk(root: str, depth: int, out: list[tuple[str, int]], budget: list[int]) -> None:
    try:
        it = os.scandir(root)
    except OSError:
        return
    with it:
        for e in it:
            budget[0] -= 1
            if budget[0] < 0:
                return
            try:
                if e.is_dir(follow_symlinks=False):
                    if depth > 0 and e.name.lower() not in _SKIP_DIRS:
                        _walk(e.path, depth - 1, out, budget)
                    continue
                ext = os.path.splitext(e.name)[1].lower()
                if ext in MODEL_EXTS:
                    size = e.stat().st_size
                    if _is_model_file(ext, size, False, None):
                        out.append((e.path, size))
            except OSError:
                continue


def installed_models(exe: str, now: float | None = None) -> list[MappedModel]:
    """Model files in an app's install and data folders. Fallback for runtimes that read a model into RAM and
    close the file (ONNX Runtime, whisper.cpp), which leaves no OS evidence of which model is loaded.
    Bounded walk (depth 4, 3000 entries) cached per exe for 5 minutes."""
    if not exe:
        return []
    now = now or time.time()
    key = os.path.normcase(exe)
    hit = _installed.get(key)
    if hit and now - hit[0] < INSTALLED_TTL:
        return hit[1]
    stem = os.path.splitext(os.path.basename(key))[0]
    roots = []
    if not key.startswith(os.path.normcase(os.environ.get("SystemRoot", r"C:\Windows"))):
        roots.append(os.path.dirname(exe))
    if len(stem) >= 4:
        for base in filter(None, (os.environ.get("APPDATA"), os.environ.get("LOCALAPPDATA"))):
            try:
                roots += [e.path for e in os.scandir(base) if stem in e.name.lower() and e.is_dir()]
            except OSError:
                pass
    files: list[tuple[str, int]] = []
    budget = [3000]
    for root in {os.path.normcase(r): r for r in roots}.values():
        _walk(root, 4, files, budget)
    by_dir: dict[str, list[tuple[str, int]]] = {}
    for path, size in files:
        by_dir.setdefault(os.path.dirname(path), []).append((path, size))
    found = []
    for folder, group in by_dir.items():
        if len(group) == 1:  # one file: name it by the file
            path, size = group[0]
            found.append(make_model(path, size, 0, path.lower().endswith(".gguf") and is_gguf(path), "installed"))
        else:  # several files (encoder + decoder, shards): name it by the folder
            found.append(make_model(folder, sum(s for _, s in group), 0, False, "installed", store_name(group[0][0])))
    found = sorted(found, key=lambda m: -m.size)[:6]
    _installed[key] = (now, found)
    return found


def _create_time(pid: int) -> float:
    try:
        return psutil.Process(pid).create_time()
    except (psutil.Error, OSError):
        return 0.0


class ProcInspector:
    """Caches per-process details. Memory-map scans are the only costly step (1-50 ms each), so they run
    for new processes, when a process's GPU memory changes noticeably, and on a slow backstop."""

    SCAN_MIN_GAP = 5.0  # never rescan one process more often than this
    SCAN_BACKSTOP = 120.0  # rescan even without a memory change
    SCAN_DELTA = 64 * MiB  # or 10% of the process's GPU memory, whichever is larger
    PORTS_INTERVAL = 5.0

    def __init__(self) -> None:
        self._cache: dict[int, ProcDetail] = {}
        self._ports: dict[int, list[int]] = {}
        self._ports_time = 0.0

    def _refresh_ports(self) -> None:
        if time.time() - self._ports_time < self.PORTS_INTERVAL:
            return
        ports: dict[int, set[int]] = {}
        try:
            for c in psutil.net_connections(kind="tcp"):
                if c.status == psutil.CONN_LISTEN and c.pid and c.laddr:
                    ports.setdefault(c.pid, set()).add(c.laddr.port)
        except (psutil.AccessDenied, OSError):
            pass
        self._ports = {pid: sorted(p) for pid, p in ports.items()}
        self._ports_time = time.time()

    def get(self, pid: int) -> ProcDetail:
        """Cheap details: name, exe, command line, listening ports. Built once per process: psutil.Process()
        enumerates every process for protected PIDs (System, csrss, dwm), ~40 ms each. Accessible processes
        are re-checked for PID reuse each call (~0.1 ms); protected ones leave through prune()."""
        self._refresh_ports()
        d = self._cache.get(pid)
        if d is not None and d.create_time and _create_time(pid) != d.create_time:
            d = None  # PID reused (e.g. while the TUI was paused)
        if d is None:
            d = self._cache[pid] = self._new(pid)
        d.ports = self._ports.get(pid, [])
        return d

    def ports(self, pid: int) -> list[int]:
        self._refresh_ports()
        return self._ports.get(pid, [])

    @staticmethod
    def _new(pid: int) -> ProcDetail:
        d = ProcDetail(pid=pid)
        if pid == 0:
            d.name = "System Idle"
            return d
        try:
            p = psutil.Process(pid)
        except (psutil.Error, OSError):
            d.access_denied = True
            return d
        try:
            d.name = p.name()
        except (psutil.Error, OSError):
            pass
        try:
            d.exe = p.exe()
            d.cmdline = p.cmdline()
            d.create_time = p.create_time()
        except (psutil.Error, OSError):
            d.access_denied = True
        return d

    def needs_scan(self, d: ProcDetail, mem: float, now: float | None = None) -> bool:
        if d.access_denied or d.pid == 0:
            return False
        if not d.maps_time:
            return True
        age = (now or time.time()) - d.maps_time
        if age < self.SCAN_MIN_GAP:
            return False
        return age > self.SCAN_BACKSTOP or abs(mem - d.scan_mem) > max(self.SCAN_DELTA, 0.1 * d.scan_mem)

    def scan(self, d: ProcDetail, mem: float = 0.0) -> None:
        """Memory maps -> loaded DLLs (runtimes) and mapped model files. Never uses open_files(): ~1-2 s/process."""
        d.maps_time, d.scan_mem = time.time(), mem
        try:
            maps = psutil.Process(d.pid).memory_maps(grouped=True)
        except (psutil.Error, OSError):
            d.access_denied = True
            return
        dlls: set[str] = set()
        models: dict[str, MappedModel] = {}
        for m in maps:
            path = m.path
            base = os.path.basename(path).lower()
            ext = os.path.splitext(base)[1]
            if ext == ".dll" or ext == ".node":
                dlls.add(base)
                continue
            if ext in _NOT_MODELS:
                continue
            try:
                size = os.path.getsize(path)
            except OSError:
                continue
            gg = is_gguf(path) if (ext in (".gguf", "") or size > 64 * MiB) else False
            if not (gg or ext in MODEL_EXTS):
                continue
            store = store_name(path)
            if _is_model_file(ext, size, gg, store):
                models[path.lower()] = make_model(path, size, m.rss, gg, "mapped", store)
        for path in _models_from_cmdline(d.cmdline):
            if path.lower() in models:
                models[path.lower()].add_source("cmdline")
            else:
                models[path.lower()] = make_model(path, os.path.getsize(path), 0, is_gguf(path), "cmdline")
        d.runtimes = _match_runtimes(dlls)
        d.models = sorted(models.values(), key=lambda m: -m.size)
        ai = any(r in AI_RUNTIMES for r in d.runtimes)
        d.candidates = installed_models(d.exe) if ai and not d.models else []

    def prune(self, live_pids: set[int]) -> None:
        for pid in list(self._cache):
            if pid not in live_pids:
                del self._cache[pid]


def classify(d: ProcDetail, engines: dict[str, float]) -> str:
    n = d.name.lower()
    if d.pid == 4:
        return "kernel"
    if n == "dwm.exe":
        return "compositor"
    if n in ("csrss.exe", "winlogon.exe", "lsass.exe", "svchost.exe"):
        return "system"
    if d.is_diffusion:
        return "diffusion"
    kinds = {MODEL_KINDS[m.kind] for m in d.models if m.kind in MODEL_KINDS and not m.kind_hint}
    for k in ("LLM", "decision", "embedding", "speech"):
        if k in kinds:
            return k
    if any(m.kind_hint == "decision model" for m in d.models):
        return "decision?"
    if "llama.cpp" in d.runtimes or n.startswith(LLM_NAMES):
        return "LLM"
    video = engines.get("decode", 0) + engines.get("encode", 0) > 1
    if d.is_ai and not video:
        return "AI"
    if n in BROWSERS:
        return "browser"
    if video or "NVENC" in d.runtimes:
        return "video"
    if "CUDA" in d.runtimes:
        return "compute"
    if ("D3D12" in d.runtimes or "Vulkan" in d.runtimes) and engines.get("3d", 0) > 10:
        return "3D/game"
    return "app"

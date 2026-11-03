"""Per-process introspection: which GPU runtimes are loaded, which model files are mapped."""

from __future__ import annotations

import fnmatch
import os
import time
from dataclasses import dataclass, field

import psutil

from .gguf import is_gguf

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
MODEL_EXTS = {".gguf", ".ggml", ".safetensors", ".onnx", ".pt", ".pth", ".bin", ".engine", ".plan", ".tflite", ".ckpt"}
_NOT_MODELS = {".dll", ".exe", ".sys", ".mui", ".ttf", ".ttc", ".otf", ".dat", ".db", ".nls", ".pak", ".mun"}
BROWSERS = {"msedge.exe", "chrome.exe", "firefox.exe", "msedgewebview2.exe", "brave.exe", "opera.exe", "arc.exe", "vivaldi.exe"}
LLM_NAMES = ("llama-", "ollama", "lm studio", "lmstudio", "llmster", "koboldcpp", "jan", "gpt4all", "llamafile")


@dataclass
class MappedModel:
    path: str
    size: int
    resident: int  # bytes of the file currently in this process's RAM working set
    gguf: bool
    source: str = "mapped"  # "mapped" (memory-mapped) or "cmdline" (passed as an argument)


@dataclass
class ProcDetail:
    pid: int
    name: str = "?"
    exe: str = ""
    cmdline: list[str] = field(default_factory=list)
    create_time: float = 0.0
    runtimes: list[str] = field(default_factory=list)
    models: list[MappedModel] = field(default_factory=list)
    ports: list[int] = field(default_factory=list)
    access_denied: bool = False
    maps_time: float = 0.0  # when memory maps were last scanned

    @property
    def is_ai(self) -> bool:
        return bool(self.models) or any(r in AI_RUNTIMES for r in self.runtimes)


def _match_runtimes(dll_names: set[str]) -> list[str]:
    found = []
    for tag, patterns in RUNTIMES:
        if any(fnmatch.fnmatch(d, p) for d in dll_names for p in patterns):
            found.append(tag)
    return found


def _models_from_cmdline(cmd: list[str]) -> list[str]:
    paths = []
    for i, arg in enumerate(cmd):
        val = None
        if arg in ("-m", "--model", "--mmproj", "-md", "--model-draft", "--lora") and i + 1 < len(cmd):
            val = cmd[i + 1]
        elif arg.startswith("--model="):
            val = arg.split("=", 1)[1]
        if val and os.path.isfile(val):
            paths.append(os.path.abspath(val))
    return paths


class ProcInspector:
    """Caches per-process details; memory-map scans are throttled because they can be slow."""

    MAPS_INTERVAL = 15.0
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

    def get(self, pid: int, deep: bool) -> ProcDetail:
        """deep=True scans memory maps (DLLs + mapped model files)."""
        self._refresh_ports()
        d = self._cache.get(pid)
        try:
            p = psutil.Process(pid)
            ct = p.create_time()
        except (psutil.NoSuchProcess, psutil.AccessDenied, OSError):
            p, ct = None, 0.0
        if d is None or d.create_time != ct:
            d = ProcDetail(pid=pid, create_time=ct)
            if pid == 0:
                d.name = "System Idle"
            elif p is not None:
                try:
                    d.name = p.name()
                except (psutil.Error, OSError):
                    pass
                try:
                    d.exe = p.exe()
                    d.cmdline = p.cmdline()
                except (psutil.Error, OSError):
                    d.access_denied = True
            self._cache[pid] = d
        d.ports = self._ports.get(pid, [])
        if deep and p is not None and not d.access_denied and time.time() - d.maps_time > self.MAPS_INTERVAL:
            self._scan_maps(p, d)
        return d

    def _scan_maps(self, p: psutil.Process, d: ProcDetail) -> None:
        d.maps_time = time.time()
        try:
            maps = p.memory_maps(grouped=True)
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
            gg = is_gguf(path) if (ext in (".gguf", "") or size > 64 << 20) else False
            if ext in MODEL_EXTS or gg:
                if size < (4 << 20) and not gg:
                    continue
                models[path.lower()] = MappedModel(path=path, size=size, resident=m.rss, gguf=gg)
        for path in _models_from_cmdline(d.cmdline):
            if path.lower() not in models:
                models[path.lower()] = MappedModel(path, os.path.getsize(path), 0, is_gguf(path), source="cmdline")
        d.runtimes = _match_runtimes(dlls)
        d.models = sorted(models.values(), key=lambda m: -m.size)

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
    if any(m.gguf for m in d.models) or "llama.cpp" in d.runtimes or n.startswith(LLM_NAMES):
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

"""Friendly model names from where a file lives on disk (Ollama, Hugging Face cache, LM Studio).

Pure path work plus small cached manifest reads: no runtime is contacted and no model is loaded.
"""

from __future__ import annotations

import json
import os
import re
import time

_OLLAMA_BLOB = re.compile(r"^(.*)[\\/]blobs[\\/]sha256-([0-9a-f]{64})$", re.I)
_HF_REPO = re.compile(r"[\\/]models--([^\\/]+?)--([^\\/]+)[\\/]snapshots[\\/]", re.I)
_TTL = 60.0
_ollama_cache: dict[str, tuple[float, dict[str, list[str]]]] = {}
_lms_cache: tuple[float, list[str]] | None = None


def ollama_names(models_root: str) -> dict[str, list[str]]:
    """{sha256 hex: ["qwen3:14b", ...]} from <root>/manifests/<registry>/<namespace>/<model>/<tag>."""
    now = time.time()
    hit = _ollama_cache.get(models_root)
    if hit and now - hit[0] < _TTL:
        return hit[1]
    names: dict[str, list[str]] = {}
    base = os.path.join(models_root, "manifests")
    for dirpath, _, files in os.walk(base):
        parts = os.path.relpath(dirpath, base).split(os.sep)
        if len(parts) != 3:
            continue
        registry, namespace, model = parts
        prefix = "" if registry == "registry.ollama.ai" else f"{registry}/"
        prefix += "" if namespace == "library" else f"{namespace}/"
        for tag in files:
            try:
                with open(os.path.join(dirpath, tag), encoding="utf-8") as f:
                    layers = json.load(f).get("layers", [])
            except (OSError, ValueError, AttributeError):
                continue
            for layer in layers:
                digest = str(layer.get("digest", "")).removeprefix("sha256:").lower()
                if digest:
                    names.setdefault(digest, []).append(f"{prefix}{model}:{tag}")
    _ollama_cache[models_root] = (now, names)
    return names


def lmstudio_dirs() -> list[str]:
    global _lms_cache
    now = time.time()
    if _lms_cache and now - _lms_cache[0] < _TTL:
        return _lms_cache[1]
    home = os.path.expanduser("~")
    dirs = [os.path.join(home, ".lmstudio", "models"), os.path.join(home, ".cache", "lm-studio", "models")]
    for settings in (os.path.join(home, ".lmstudio", "settings.json"),
                     os.path.join(home, ".cache", "lm-studio", "settings.json")):
        try:
            with open(settings, encoding="utf-8") as f:
                folder = json.load(f).get("downloadsFolder")
            if folder:
                dirs.insert(0, folder)
        except (OSError, ValueError, AttributeError):
            pass
    dirs = list(dict.fromkeys(os.path.normcase(os.path.abspath(d)) for d in dirs))
    _lms_cache = (now, dirs)
    return dirs


def store_name(path: str, lmstudio: list[str] | None = None) -> str | None:
    """'qwen3:14b' (Ollama), 'org/repo' (Hugging Face cache) or 'publisher/repo' (LM Studio)."""
    m = _OLLAMA_BLOB.match(path)
    if m:
        names = ollama_names(m[1]).get(m[2].lower())
        return ", ".join(sorted(set(names))) if names else None
    m = _HF_REPO.search(path)
    if m:
        return f"{m[1]}/{m[2]}"
    full = os.path.normcase(os.path.abspath(path))
    for root in lmstudio if lmstudio is not None else lmstudio_dirs():
        if full.startswith(root + os.sep):
            parts = os.path.relpath(full, root).split(os.sep)
            if len(parts) >= 3:
                # normcase lowercases on Windows; recover the original casing from the input path.
                orig = os.path.abspath(path).split(os.sep)[-len(parts):]
                return f"{orig[0]}/{orig[1]}"
    return None

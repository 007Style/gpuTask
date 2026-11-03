"""GGUF header reader: what model is this file, and what will it cost in VRAM?

Only the header and tensor table are read (a few MB at most), never the weights.
Tensor byte sizes come from the tensor offset table, so new quant types don't need a lookup table.
"""

from __future__ import annotations

import os
import re
import struct
from dataclasses import dataclass, field
from functools import lru_cache

_SCALARS = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i", 6: "<f", 7: "<?", 10: "<Q", 11: "<q", 12: "<d"}
_STRING, _ARRAY = 8, 9

FILE_TYPES = {
    0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1", 7: "Q8_0", 8: "Q5_0", 9: "Q5_1", 10: "Q2_K", 11: "Q3_K_S",
    12: "Q3_K_M", 13: "Q3_K_L", 14: "Q4_K_S", 15: "Q4_K_M", 16: "Q5_K_S", 17: "Q5_K_M", 18: "Q6_K",
    19: "IQ2_XXS", 20: "IQ2_XS", 21: "Q2_K_S", 22: "IQ3_XS", 23: "IQ3_XXS", 24: "IQ1_S", 25: "IQ4_NL",
    26: "IQ3_S", 27: "IQ3_M", 28: "IQ2_S", 29: "IQ2_M", 30: "IQ4_XS", 31: "IQ1_M", 32: "BF16",
    36: "TQ1_0", 37: "TQ2_0", 38: "MXFP4_MOE",
}

_BLK_RE = re.compile(r"^blk\.(\d+)\.")


@dataclass
class GgufInfo:
    path: str
    file_size: int
    version: int
    metadata: dict = field(default_factory=dict)
    n_params: int = 0
    tensor_bytes: int = 0
    layer_bytes: dict[int, int] = field(default_factory=dict)  # blk index -> bytes
    token_embd_bytes: int = 0
    output_bytes: int = 0  # output.weight (or tied token_embd copy) + output_norm
    other_bytes: int = 0
    error: str | None = None

    def meta(self, key: str, default=None):
        arch = self.metadata.get("general.architecture", "")
        return self.metadata.get(f"{arch}.{key}", default)

    @property
    def arch(self) -> str:
        return self.metadata.get("general.architecture", "?")

    @property
    def kind(self) -> str:
        t = self.metadata.get("general.type", "model")
        if self.arch == "clip" or t == "mmproj":
            return "vision projector"
        if t == "adapter":
            return "LoRA adapter"
        if self.metadata.get("bert.pooling_type") is not None or self.meta("pooling_type") is not None:
            return "embedding model"
        return "model"

    @property
    def name(self) -> str:
        return self.metadata.get("general.name") or os.path.basename(self.path)

    @property
    def quant(self) -> str:
        ft = self.metadata.get("general.file_type")
        return FILE_TYPES.get(ft, f"type{ft}") if ft is not None else "?"

    @property
    def n_layers(self) -> int:
        return int(self.meta("block_count", 0) or 0)

    @property
    def ctx_train(self) -> int | None:
        return self.meta("context_length")

    @property
    def sliding_window(self) -> int | None:
        return self.meta("attention.sliding_window")

    @property
    def params_label(self) -> str:
        n = self.n_params
        if n >= 1e9:
            return f"{n / 1e9:.1f}B"
        if n >= 1e6:
            return f"{n / 1e6:.0f}M"
        return str(n)

    def gpu_weight_bytes(self, n_gpu_layers: int | None = None) -> int:
        """Bytes llama.cpp would place on the GPU. Input embeddings always stay on the CPU."""
        L = self.n_layers or len(self.layer_bytes)
        ngl = L + 1 if n_gpu_layers is None or n_gpu_layers < 0 else n_gpu_layers
        first_gpu = max(L - ngl, 0)
        total = sum(b for i, b in self.layer_bytes.items() if i >= first_gpu)
        if ngl > L:
            total += self.output_bytes
        if not self.layer_bytes:  # non-transformer (e.g. CLIP/whisper): assume everything goes up
            total = self.tensor_bytes
        return total

    def kv_bytes_per_token(self, type_bytes: float = 2.0) -> int | None:
        """Per-token KV cache cost (K+V, all layers) at the given element size (f16 = 2)."""
        L = self.n_layers
        n_head = self.meta("attention.head_count")
        n_kv = self.meta("attention.head_count_kv", n_head)
        n_embd = self.meta("embedding_length")
        if not L or n_kv is None:
            return None
        if isinstance(n_head, list):
            n_head = max(n_head) if n_head else None
        head_dim = (n_embd // n_head) if (n_embd and n_head) else None
        k_len = self.meta("attention.key_length", head_dim)
        v_len = self.meta("attention.value_length", head_dim)
        if not k_len or not v_len:
            return None
        per_layer = n_kv if isinstance(n_kv, list) else [n_kv] * L
        elems = sum(int(h) * (k_len + v_len) for h in per_layer[:L])
        return int(elems * type_bytes)


class _Reader:
    def __init__(self, f, version: int):
        self.f = f
        self.v = version

    def unpack(self, fmt: str):
        size = struct.calcsize(fmt)
        data = self.f.read(size)
        if len(data) != size:
            raise EOFError("truncated GGUF header")
        return struct.unpack(fmt, data)[0]

    def length(self) -> int:
        return self.unpack("<Q" if self.v >= 2 else "<I")

    def string(self) -> str:
        return self.f.read(self.length()).decode("utf-8", "replace")

    def skip_string(self) -> None:
        self.f.seek(self.length(), 1)

    def value(self, vtype: int):
        if vtype in _SCALARS:
            return self.unpack(_SCALARS[vtype])
        if vtype == _STRING:
            return self.string()
        if vtype == _ARRAY:
            etype = self.unpack("<I")
            n = self.length()
            if etype in _SCALARS:
                size = struct.calcsize(_SCALARS[etype])
                if n <= 4096:
                    raw = self.f.read(size * n)
                    return list(struct.unpack("<" + _SCALARS[etype][1] * n, raw))
                self.f.seek(size * n, 1)
                return f"<array[{n}]>"
            if etype == _STRING:
                for _ in range(n):
                    self.skip_string()
                return f"<string[{n}]>"
            for _ in range(n):
                self.value(etype)
            return f"<array[{n}]>"
        raise ValueError(f"unknown GGUF value type {vtype}")


def is_gguf(path: str) -> bool:
    try:
        with open(path, "rb") as f:
            return f.read(4) == b"GGUF"
    except OSError:
        return False


@lru_cache(maxsize=64)
def _read_cached(path: str, size: int, mtime: float) -> GgufInfo:
    info = GgufInfo(path=path, file_size=size, version=0)
    try:
        with open(path, "rb", buffering=1 << 20) as f:
            if f.read(4) != b"GGUF":
                info.error = "not a GGUF file"
                return info
            info.version = struct.unpack("<I", f.read(4))[0]
            r = _Reader(f, info.version)
            n_tensors = r.length()
            n_kv = r.length()
            for _ in range(n_kv):
                key = r.string()
                info.metadata[key] = r.value(r.unpack("<I"))
            tensors = []
            for _ in range(n_tensors):
                name = r.string()
                n_dims = r.unpack("<I")
                dims = [r.length() for _ in range(n_dims)]
                r.unpack("<I")  # ggml type
                offset = r.unpack("<Q")
                n = 1
                for d in dims:
                    n *= d
                tensors.append((offset, name, n))
            align = int(info.metadata.get("general.alignment", 32))
            data_start = (f.tell() + align - 1) // align * align
    except Exception as e:  # noqa: BLE001 - show the problem in the UI rather than crash
        info.error = f"{type(e).__name__}: {e}"
        return info

    tensors.sort()
    names = set()
    for i, (offset, name, n) in enumerate(tensors):
        end = tensors[i + 1][0] if i + 1 < len(tensors) else size - data_start
        nbytes = max(end - offset, 0)
        info.n_params += n
        info.tensor_bytes += nbytes
        names.add(name)
        m = _BLK_RE.match(name)
        if m:
            idx = int(m[1])
            info.layer_bytes[idx] = info.layer_bytes.get(idx, 0) + nbytes
        elif name.startswith("token_embd."):
            info.token_embd_bytes += nbytes
        elif name.startswith("output"):
            info.output_bytes += nbytes
        else:
            info.other_bytes += nbytes
    if "output.weight" not in names and info.layer_bytes:
        info.output_bytes += info.token_embd_bytes  # tied embeddings: llama.cpp puts a copy on the GPU
    return info


def read_gguf(path: str) -> GgufInfo | None:
    try:
        st = os.stat(path)
    except OSError:
        return None
    return _read_cached(path, st.st_size, st.st_mtime)

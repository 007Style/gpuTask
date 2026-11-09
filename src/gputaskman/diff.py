"""Compare two `--json` snapshots: which measured memory buckets changed, next to what runtimes reported.

Measured counters are compared on their own. Runtime-reported and estimated breakdowns are listed
beside them, never added to measured totals. A model file appearing in a process is evidence that
the process has the file, not that it is resident in GPU memory.
"""

from __future__ import annotations

import json
import math

from .llm import gb

MiB = 1 << 20
NOISE = 16 * MiB  # smaller per-process moves are allocator and compositor churn, not a model loading
SAME_PROCESS_SECS = 2.0  # one process's start time can move this much between runs (Linux boot-time rounding, clock steps)
MEASURED = ("vram_bytes", "shared_bytes", "committed_bytes", "paged_out_bytes",
            "vram_resident_bytes", "gtt_resident_bytes")  # resident: amdgpu, shared buffers included (never summed)
DEVICE = (("vram_used", "dedicated used"), ("shared_used", "shared used"),
          ("attributed", "attributed to processes"), ("unattributed", "unattributed"))
SEGMENT_ORDER = ("weights", "kv", "compute", "runtime", "unknown")


def load(path: str) -> dict:
    with open(path, "rb") as f:
        raw = f.read()
    # Windows PowerShell 5.1 redirects (`>`) as UTF-16; PowerShell 7 and other shells write UTF-8.
    text = raw.decode("utf-16") if raw[:2] in (b"\xff\xfe", b"\xfe\xff") else raw.decode("utf-8-sig")
    data = json.loads(text)
    if not isinstance(data, dict) or not isinstance(data.get("processes"), list):
        raise ValueError(f"{path} is not a gputaskman --json snapshot")
    return data


def change(before, after) -> dict:
    return {"before": before, "after": after,
            "delta": after - before if before is not None and after is not None else None}


def _get(d: dict | None, key: str):
    return d.get(key) if d else None


def _unattributed(s: dict):
    used, attributed = s.get("vram_used"), s.get("attributed")
    return used - attributed if used is not None and attributed is not None else None


PLACEHOLDER_NAMES = ("?", "(outside this container)")  # no identity: never pair these by name


def _named(p: dict) -> bool:
    return p.get("host") != "outside" and p.get("name") not in PLACEHOLDER_NAMES


def _started(p: dict) -> float | None:
    v = p.get("started_at")
    return v if isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) and v > 0 else None


def _recycled(b: dict, a: dict) -> bool:
    """One PID with two different start times is two processes: the PID was reused in between.

    Unknown or malformed start times (older snapshots, protected processes) never conflict. Rows are only
    ever paired within one host, so the PID needs no namespace check here."""
    sb, sa = _started(b), _started(a)
    return (b.get("pid") is not None and b.get("pid") == a.get("pid") and sb is not None and sa is not None
            and abs(sb - sa) > SAME_PROCESS_SECS)


def match(before: list[dict], after: list[dict]) -> list[tuple[dict | None, dict | None, str]]:
    """Pair rows by PID, then by name and ports, then by name, when each pairing is unambiguous.

    A model reload often restarts the server under a new PID. WSL host rows have their own PID space.
    A PID that comes back with a different start time is another process and is never paired, by any rule.
    Every row comes back exactly once: paired, new, or gone.
    """
    keys = [(p.get("host"), p.get("pid")) for p in before]
    unique = {k for k in keys if k[1] is not None and keys.count(k) == 1}
    left = dict(enumerate(before))
    by_pid = {k: i for i, k in enumerate(keys) if k in unique}
    pairs: list[tuple[dict | None, dict | None, str]] = []
    rest: list[dict] = []
    for a in after:
        i = by_pid.get((a.get("host"), a.get("pid")))
        if i is not None and i in left and left[i].get("name") == a.get("name") and not _recycled(left[i], a):
            pairs.append((left.pop(i), a, "pid"))
        else:
            rest.append(a)

    def ports(p: dict):
        return (p.get("host"), p.get("name"), tuple(sorted(p["ports"]))) if p.get("ports") and _named(p) else None

    def name(p: dict):
        return (p.get("host"), p.get("name")) if _named(p) else None

    for how, key in (("name+ports", ports), ("name", name)):
        olds: dict = {}
        news: dict = {}
        for i, b in left.items():
            olds.setdefault(key(b), []).append(i)
        for a in rest:
            news.setdefault(key(a), []).append(a)
        for k, found in news.items():
            if k is not None and len(found) == 1 and len(olds.get(k, [])) == 1:
                i = olds[k][0]
                if not _recycled(left[i], found[0]):
                    pairs.append((left.pop(i), found[0], how))
                    rest.remove(found[0])
    pairs += [(None, a, "new") for a in rest]
    pairs += [(b, None, "gone") for b in left.values()]
    return pairs


def _segments(p: dict | None) -> tuple[dict, dict]:
    sums: dict[str, int] = {}
    exact: dict[str, bool] = {}
    for s in (_get(p, "llm") or {}).get("segments") or []:
        k = s.get("kind") or "unknown"
        sums[k] = sums.get(k, 0) + int(s.get("bytes") or 0)
        exact[k] = exact.get(k, True) and bool(s.get("exact"))
    return sums, exact


def _runtime(b: dict | None, a: dict | None) -> dict | None:
    lb, la = _get(b, "llm"), _get(a, "llm")
    if not lb and not la:
        return None
    sb, eb = _segments(b)
    sa, ea = _segments(a)
    kinds = sorted(set(sb) | set(sa), key=lambda k: SEGMENT_ORDER.index(k) if k in SEGMENT_ORDER else 99)
    return {
        "n_ctx": change(_get(lb, "n_ctx"), _get(la, "n_ctx")),
        "segments": {k: {**change(sb.get(k), sa.get(k)),
                         "source": "runtime-reported" if eb.get(k, True) and ea.get(k, True) else "estimated"}
                     for k in kinds},
    }


def _files(b: dict | None, a: dict | None) -> dict:
    fb = {m.get("path"): m for m in _get(b, "model_files") or []}
    fa = {m.get("path"): m for m in _get(a, "model_files") or []}
    return {"added": [{"path": p, "sources": m.get("sources")} for p, m in fa.items() if p not in fb],
            "removed": [{"path": p, "sources": m.get("sources")} for p, m in fb.items() if p not in fa]}


def _proc(b: dict | None, a: dict | None, how: str) -> dict:
    p = a or b or {}
    return {
        "name": p.get("name"), "host": p.get("host"), "match": how,
        "pid_before": _get(b, "pid"), "pid_after": _get(a, "pid"),
        "measured": {k: change(_get(b, k), _get(a, k)) for k in MEASURED},
        "inferred": {"vram_inferred_bytes": change(_get(b, "vram_inferred_bytes"), _get(a, "vram_inferred_bytes"))},
        "runtime": _runtime(b, a),
        "model_files": _files(b, a),
    }


def _size(p: dict) -> int:
    v = p["measured"]["vram_bytes"]
    if v["delta"] is not None:
        return abs(v["delta"])
    return v["after"] if v["after"] is not None else v["before"] or 0


def _num(v) -> int | None:
    return v if isinstance(v, int) and not isinstance(v, bool) else None


def _identity(s: dict) -> dict:
    """What a snapshot says about its adapter that can be compared: vendor, PCI address, adapter key.

    Missing, zero (unknown) and malformed values are left out, so they can never conflict."""
    a = s.get("adapter")
    a = a if isinstance(a, dict) else {}
    out = {name: a[field] for field, name in (("vendor_id", "vendor"), ("luid", "key")) if _num(a.get(field))}
    pci = a.get("pci_address")
    if isinstance(pci, (list, tuple)) and len(pci) == 3 and all(_num(x) is not None and x >= 0 for x in pci):
        out["pci"] = tuple(pci)
    return out


def check_adapter(before: dict, after: dict) -> tuple[str, str | None]:
    """How firmly two snapshots with the same GPU name are tied to one adapter: ("pci" | "key" | "name-only", note).

    Raises ValueError when both snapshots say which adapter they came from and the answers differ. A matching
    PCI address wins over the adapter key, which Windows reassigns at boot and when an adapter is re-enabled,
    so there a differing key is only a warning. Both identifiers are host-local: they tell two GPUs in one
    machine apart, not two computers.
    """
    b, a = _identity(before), _identity(after)

    def different(what: str) -> ValueError:
        return ValueError(f"snapshots are from different adapters, though both are named {after.get('gpu')!r}: {what}")

    if "vendor" in b and "vendor" in a and b["vendor"] != a["vendor"]:
        raise different(f"vendor {b['vendor']:#06x} vs {a['vendor']:#06x}")
    if "pci" in b and "pci" in a:
        if b["pci"] != a["pci"]:
            raise different("PCI " + " vs ".join("{:02x}:{:02x}.{}".format(*x) for x in (b["pci"], a["pci"])))
        return "pci", None
    if "key" in b and "key" in a:
        if b["key"] == a["key"]:
            return "key", None
        if after.get("backend") != "wddm":
            raise different(f"adapter key {b['key']:#x} vs {a['key']:#x}")
        return "name-only", ("Their adapter keys differ and no PCI address is available to settle it. Windows changes "
                             "those keys at boot, so this may be one GPU or two of the same model; they are matched "
                             "by GPU name and backend only.")
    return "name-only", ("These snapshots do not both say which adapter they came from (an older gputaskman, or no "
                         "adapter details), so they are matched by GPU name and backend only.")


def diff(before: dict, after: dict) -> dict:
    if before.get("gpu") != after.get("gpu") or before.get("backend") != after.get("backend"):
        raise ValueError(f"snapshots are from different GPUs or backends: "
                         f"{before.get('gpu')} ({before.get('backend')}) vs {after.get('gpu')} ({after.get('backend')})")
    adapter_check, adapter_note = check_adapter(before, after)
    device = {k: change(before.get(k), after.get(k)) for k, _ in DEVICE if k != "unattributed"}
    device["unattributed"] = change(_unattributed(before), _unattributed(after))
    procs = [_proc(b, a, how) for b, a, how in match(before["processes"], after["processes"])]
    procs.sort(key=_size, reverse=True)
    taken_b, taken_a = before.get("taken"), after.get("taken")
    seconds = round(taken_a - taken_b, 1) if isinstance(taken_b, (int, float)) and isinstance(taken_a, (int, float)) else None
    return {"gpu": after.get("gpu"), "backend": after.get("backend"),
            "adapter_check": adapter_check, "adapter_note": adapter_note, "seconds": seconds,
            "memory_note": after.get("memory_note") or before.get("memory_note"),
            "device": device, "processes": procs}


# ---------------------------------------------------------------- text


def _gb(v) -> str:
    return "N/A" if v is None else gb(v)


def _signed(v) -> str:
    if v is None:
        return "N/A"
    if abs(v) < MiB // 2:
        return "0 MB"
    return ("+" if v >= 0 else "-") + gb(abs(v))


def _moved(c: dict) -> bool:
    """A value appeared, disappeared, or moved by at least the noise threshold."""
    if c["before"] == c["after"]:
        return False
    return c["delta"] is None or abs(c["delta"]) >= NOISE


def _changed(p: dict) -> bool:
    if p["match"] in ("new", "gone"):
        return True
    if any(abs(c["delta"] or 0) >= NOISE for c in p["measured"].values()):
        return True
    if abs(p["inferred"]["vram_inferred_bytes"]["delta"] or 0) >= NOISE:
        return True
    r = p["runtime"]
    # Remainder segments follow the measured total, so they move with every sample; apply the same threshold.
    if r and (r["n_ctx"]["before"] != r["n_ctx"]["after"] or any(_moved(s) for s in r["segments"].values())):
        return True
    return bool(p["model_files"]["added"] or p["model_files"]["removed"])


def _pid(p: dict) -> str:
    b, a = p["pid_before"], p["pid_after"]
    pid = str(a if b is None else b if a is None or a == b else f"{b}->{a}")
    return f"{'W' if p['host'] == 'windows' else 'H'}:{pid}" if p["host"] else pid


def to_text(d: dict) -> str:
    when = f", {d['seconds']} s apart" if d["seconds"] is not None else ""
    out = [f"gputaskman diff: {d['gpu'] or 'unknown GPU'} ({d['backend']}){when}",
           "Measured counters are compared directly. Runtime-reported and estimated values are listed",
           "separately and are not added to measured totals."]
    if d.get("adapter_note"):
        out.append(f"Warning: {d['adapter_note']}")
    out += ["", f"{'DEVICE (measured)':<26}{'BEFORE':>11}{'AFTER':>11}{'CHANGE':>11}"]
    for key, label in DEVICE:
        c = d["device"][key]
        out.append(f"{label:<26}{_gb(c['before']):>11}{_gb(c['after']):>11}{_signed(c['delta']):>11}")
    if d.get("memory_note"):
        out.append(d["memory_note"])

    procs = [p for p in d["processes"] if _changed(p)]
    out += ["", f"{'PROCESSES (dedicated)':<22}{'PID':>14}  {'MATCH':<11}{'BEFORE':>11}{'AFTER':>11}"
                f"{'CHANGE':>11}{'SHARED +/-':>11}"]
    for p in procs:
        v, s = p["measured"]["vram_bytes"], p["measured"]["shared_bytes"]
        before = "not listed" if p["match"] == "new" else _gb(v["before"])
        after = "not listed" if p["match"] == "gone" else _gb(v["after"])
        out.append(f"{(p['name'] or '?')[:21]:<22}{_pid(p):>14}  {p['match']:<11}{before:>11}{after:>11}"
                   f"{_signed(v['delta']):>11}{_signed(s['delta']):>11}")
    if not procs:
        out.append("(no process changed by 16 MB or more)")
    if len(procs) < len(d["processes"]):
        out.append(f"{len(d['processes']) - len(procs)} process(es) changed by less than 16 MB are not shown; "
                   "--json lists all.")

    for p in procs:
        detail = []
        inf = p["inferred"]["vram_inferred_bytes"]
        if _moved(inf):
            detail.append(f"  inferred VRAM (WSL context match)  {_gb(inf['before'])} -> {_gb(inf['after'])}"
                          f"  {_signed(inf['delta'])}")
        r = p["runtime"]
        if r:
            n = r["n_ctx"]
            if n["before"] != n["after"]:
                detail.append(f"  context length  {n['before'] if n['before'] is not None else 'N/A'} -> "
                              f"{n['after'] if n['after'] is not None else 'N/A'}")
            for kind, s in r["segments"].items():
                if _moved(s):
                    detail.append(f"  {kind:<8} {_gb(s['before']):>9} -> {_gb(s['after']):>9}"
                                  f"  {_signed(s['delta']):>9}  {s['source']}")
        for verb, key in (("added", "added"), ("removed", "removed")):
            for f in p["model_files"][key]:
                detail.append(f"  model file {verb}: {f['path']} [{', '.join(f['sources'] or [])}]")
        if detail:
            out += ["", f"{p['name']} ({_pid(p)})"] + detail
    if any(p["model_files"]["added"] or p["model_files"]["removed"] for p in procs):
        out += ["", "Model files show what a process opened or named, not what is resident in GPU memory."]
    if any(p["match"] in ("new", "gone") for p in procs):
        out.append("'not listed' means the process was absent from that snapshot: not running, or using too "
                   "little GPU to be listed.")
    return "\n".join(out)

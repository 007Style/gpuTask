"""Windows host GPU counters, read from inside WSL through interop.

Linux cannot see how much VRAM WSL uses: the Windows host charges all of it to the VM worker
(vmwp.exe). When WSL interop is enabled we can still ask Windows. One long-lived powershell.exe
streams the same WDDM counters the Windows backend reads (one JSON line per second), so the
~1.5 s PowerShell startup is paid once, not per refresh.
"""

from __future__ import annotations

import base64
import glob
import itertools
import json
import math
import os
import shutil
import subprocess
import threading
import time
from collections import defaultdict
from dataclasses import dataclass, field

from .pdh import _ADAPTER_RE, _ENGINE_RE, _PROC_RE, AdapterGpu, ProcGpu, _norm_engine, luid_key

_POWERSHELL_FALLBACK = "/mnt/c/Windows/System32/WindowsPowerShell/v1.0/powershell.exe"

# PowerShell's Get-Counter resolves counters by instance *name*. vmwp.exe has one instance per guest GPU
# context, all with the same name, so Get-Counter reads the wrong one (or one twice) whenever contexts come
# and go. The PDH array API returns every instance as it exists at each collection, which is what the
# Windows backend (pdh.py) uses; this compiles the same calls once and streams one JSON line per second.
# Counter paths are English (PdhAddEnglishCounter), so this works on localized Windows too.
_SCRIPT = r"""
$ErrorActionPreference = 'Stop'
[Console]::OutputEncoding = New-Object System.Text.UTF8Encoding $false  # not the OEM code page
Add-Type -TypeDefinition @'
using System;
using System.Runtime.InteropServices;
using System.Text;
public static class GtmPdh {
    [DllImport("pdh.dll", CharSet = CharSet.Unicode)]
    static extern uint PdhOpenQueryW(string source, IntPtr user, out IntPtr query);
    [DllImport("pdh.dll", CharSet = CharSet.Unicode)]
    static extern uint PdhAddEnglishCounterW(IntPtr query, string path, IntPtr user, out IntPtr counter);
    [DllImport("pdh.dll")]
    static extern uint PdhCollectQueryData(IntPtr query);
    [DllImport("pdh.dll", CharSet = CharSet.Unicode)]
    static extern uint PdhGetFormattedCounterArrayW(IntPtr counter, uint format, ref uint size, out uint count, IntPtr items);
    static IntPtr query;
    static string[] keys;
    static IntPtr[] handles;
    public static void Open(string[] paths, string[] names) {
        PdhOpenQueryW(null, IntPtr.Zero, out query);
        keys = names;
        handles = new IntPtr[paths.Length];
        for (int i = 0; i < paths.Length; i++) PdhAddEnglishCounterW(query, paths[i], IntPtr.Zero, out handles[i]);
        PdhCollectQueryData(query);
    }
    public static string Sample() {
        PdhCollectQueryData(query);
        var sb = new StringBuilder("[");
        bool first = true;
        int stride = IntPtr.Size + 16;  // PDH_FMT_COUNTERVALUE_ITEM_W: name pointer, CStatus (+pad), double
        for (int i = 0; i < handles.Length; i++) {
            if (handles[i] == IntPtr.Zero) continue;
            uint size = 0, count;
            if (PdhGetFormattedCounterArrayW(handles[i], 0x8200, ref size, out count, IntPtr.Zero) != 0x800007D2) continue;
            IntPtr buf = Marshal.AllocHGlobal((int)size);
            try {
                if (PdhGetFormattedCounterArrayW(handles[i], 0x8200, ref size, out count, buf) != 0) continue;
                for (int k = 0; k < count; k++) {
                    IntPtr item = buf + k * stride;
                    uint status = (uint)Marshal.ReadInt32(item, IntPtr.Size);
                    double value = BitConverter.Int64BitsToDouble(Marshal.ReadInt64(item, IntPtr.Size + 8));
                    // Zero-size GPU contexts still matter: vmwp.exe's instances are matched to Linux
                    // processes by position, so "local usage" keeps them; everything else drops zeros.
                    if ((status != 0 && status != 1) || (value == 0 && i != 0)) continue;
                    string name = Marshal.PtrToStringUni(Marshal.ReadIntPtr(item)).Replace("\\", "\\\\").Replace("\"", "\\\"");
                    if (!first) sb.Append(',');
                    first = false;
                    sb.Append("[\"").Append(keys[i]).Append("\",\"").Append(name).Append("\",").Append(value.ToString("R", System.Globalization.CultureInfo.InvariantCulture)).Append(']');
                }
            } finally { Marshal.FreeHGlobal(buf); }
        }
        return sb.Append(']').ToString();
    }
}
'@
[GtmPdh]::Open(
  @('\GPU Process Memory(*)\Local Usage', '\GPU Process Memory(*)\Non Local Usage',
    '\GPU Process Memory(*)\Total Committed', '\GPU Engine(*)\Utilization Percentage',
    '\GPU Adapter Memory(*)\Dedicated Usage', '\GPU Adapter Memory(*)\Shared Usage'),
  @('local usage', 'non local usage', 'total committed', 'utilization percentage', 'dedicated usage', 'shared usage'))
while ($true) {
  Start-Sleep -Milliseconds 1000
  $s = [GtmPdh]::Sample()
  $names = @{}
  foreach ($p in [System.Diagnostics.Process]::GetProcesses()) { $names[[string]$p.Id] = $p.ProcessName }
  [Console]::Out.WriteLine('{"s":' + $s + ',"n":' + ($names | ConvertTo-Json -Compress) + '}')
  [Console]::Out.Flush()
}
"""

_PROC_FIELDS = {"local usage": "dedicated", "non local usage": "shared", "total committed": "committed"}
_ADAPTER_FIELDS = {"dedicated usage": "dedicated", "shared usage": "shared"}


def interop_enabled() -> bool:
    """True when this WSL distro can run Windows programs (interop not disabled in /etc/wsl.conf)."""
    for path in glob.glob("/proc/sys/fs/binfmt_misc/WSLInterop*"):
        try:
            with open(path, encoding="ascii") as f:
                if f.readline().strip() == "enabled":
                    return True
        except OSError:
            continue
    return False


def powershell_path() -> str | None:
    found = shutil.which("powershell.exe")
    if found:
        return found
    return _POWERSHELL_FALLBACK if os.path.exists(_POWERSHELL_FALLBACK) else None


@dataclass
class HostSample:
    taken: float
    procs: dict[int, dict[int, ProcGpu]] = field(default_factory=dict)  # {host luid: {host pid: ProcGpu}}
    adapters: dict[int, AdapterGpu] = field(default_factory=dict)  # {host luid: AdapterGpu}
    names: dict[int, str] = field(default_factory=dict)  # {host pid: process name without .exe}
    # {host luid: {host pid: [Local Usage per instance]}}. vmwp.exe has one instance per guest GPU context
    # (roughly one per Linux process using the GPU), all with the same name: sizes without owners.
    contexts: dict[int, dict[int, list[float]]] = field(default_factory=dict)


def parse_line(line: str, now: float | None = None) -> HostSample | None:
    try:
        data = json.loads(line)
    except ValueError:
        return None
    if not isinstance(data, dict):
        return None
    sample = HostSample(taken=now if now is not None else time.time())
    procs: dict[int, dict[int, ProcGpu]] = defaultdict(dict)
    adapters: dict[int, AdapterGpu] = defaultdict(AdapterGpu)
    contexts: dict[int, dict[int, list[float]]] = defaultdict(lambda: defaultdict(list))
    for row in data.get("s") or []:
        try:
            counter, instance, value = str(row[0]).lower(), str(row[1]).lower(), float(row[2])
        except (TypeError, ValueError, IndexError):
            continue
        if counter == "utilization percentage":
            m = _ENGINE_RE.match(instance)
            if m:
                g = procs[luid_key(m[2], m[3])].setdefault(int(m[1]), ProcGpu())
                g.engines[_norm_engine(m[6])] += value
        elif counter in _PROC_FIELDS:
            m = _PROC_RE.match(instance)
            if m:
                g = procs[luid_key(m[2], m[3])].setdefault(int(m[1]), ProcGpu())
                attr = _PROC_FIELDS[counter]
                setattr(g, attr, getattr(g, attr) + value)  # vmwp.exe has one instance per guest context
                if attr == "dedicated":
                    contexts[luid_key(m[2], m[3])][int(m[1])].append(value)
        elif counter in _ADAPTER_FIELDS:
            m = _ADAPTER_RE.match(instance)
            if m:
                a = adapters[luid_key(m[1], m[2])]
                attr = _ADAPTER_FIELDS[counter]
                setattr(a, attr, (getattr(a, attr) or 0.0) + value)
    for pid, name in (data.get("n") or {}).items():
        try:
            sample.names[int(pid)] = str(name)
        except ValueError:
            continue
    sample.procs, sample.adapters = dict(procs), dict(adapters)
    sample.contexts = {luid: dict(per) for luid, per in contexts.items()}
    return sample


def match_adapter(sample: HostSample, device_used: float | None) -> int | None:
    """WSL's adapter LUIDs differ from the host's, so pair the host adapter whose dedicated usage matches
    NVML's device-wide usage. Ambiguous or distant matches return None: no host data beats wrong data."""
    if not device_used or not sample.adapters:
        return None
    ranked = sorted(((luid, a) for luid, a in sample.adapters.items() if a.dedicated is not None),
                    key=lambda kv: abs(kv[1].dedicated - device_used))
    if not ranked:
        return None
    luid, best = ranked[0]
    tolerance = max(512 << 20, 0.15 * device_used)
    if abs(best.dedicated - device_used) > tolerance:
        return None
    if len(ranked) > 1 and abs(ranked[1][1].dedicated - device_used) <= tolerance:
        return None
    return luid


class HostStream:
    """Background reader of the PowerShell counter stream. status: starting | ok | off | unavailable (...)."""

    STALE_AFTER = 30.0  # keep showing the last sample through PowerShell stalls; drop it only after this

    def __init__(self, exe: str | None = None, popen=subprocess.Popen) -> None:
        self.status = "starting"
        self._latest: HostSample | None = None
        self._ready = threading.Event()
        self._proc = None
        exe = exe or powershell_path()
        if exe is None:
            self.status = "unavailable (powershell.exe not found)"
            self._ready.set()
            return
        try:
            encoded = base64.b64encode(_SCRIPT.encode("utf-16-le")).decode("ascii")
            self._proc = popen([exe, "-NoProfile", "-NonInteractive", "-EncodedCommand", encoded],
                               stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                               text=True, encoding="utf-8", errors="replace", bufsize=1)
        except (OSError, ValueError) as error:
            self.status = f"unavailable ({error})"
            self._ready.set()
            return
        threading.Thread(target=self._read, name="wsl-host-counters", daemon=True).start()

    def _read(self) -> None:
        try:
            for line in self._proc.stdout:
                try:
                    sample = parse_line(line)
                except Exception:  # one malformed line must not end the stream
                    continue
                if sample is not None and (sample.procs or sample.adapters):
                    self._latest = sample
                    self.status = "ok"
                    self._ready.set()
        except Exception:
            pass
        finally:
            # always report the end, so the collector restarts the stream instead of waiting forever
            if self.status != "off":
                self.status = "unavailable (counter stream ended)"
            self._ready.set()

    def wait(self, timeout: float) -> bool:
        return self._ready.wait(timeout)

    def latest(self) -> HostSample | None:
        s = self._latest
        return s if s is not None and time.time() - s.taken <= self.STALE_AFTER else None

    @property
    def ended(self) -> bool:
        return self.status.startswith("unavailable")

    def close(self) -> None:
        self.status = "off"
        if self._proc is not None and self._proc.poll() is None:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self._proc.kill()


# ---------------------------------------------------------------- per-process VRAM by elimination

MiB = 1 << 20


def _boot_id() -> str:
    try:
        with open("/proc/sys/kernel/random/boot_id", encoding="ascii") as f:
            return f.read().strip()
    except OSError:
        return ""


def _default_cache() -> str:
    base = os.environ.get("XDG_CACHE_HOME") or os.path.join(os.path.expanduser("~"), ".cache")
    return os.path.join(base, "gputaskman", "wsl-idle-contexts.json")


@dataclass
class ContextGuess:
    bytes: float
    confidence: str  # "consistent": every equally good explanation agrees | "order": start-order guess


FOREIGN = "foreign"  # a context from outside this distro (WSLg, other distros, Docker, hidden services)


@dataclass
class _Entry:
    value: float
    label: object = None  # holder pid | FOREIGN | ("gone", pid): its holder exited, waiting to vanish
    sure: bool = False  # established by an event or elimination, never by a guess
    same: bool = False  # this sample only: all candidates are the same size, so the bytes are right either way


class ContextTracker:
    """Matches vmwp.exe's GPU contexts to Linux processes by following them over time.

    Measured on WSL2 (driver 610.88): every Linux process that opens /dev/dxg gets one vmwp.exe context
    (an NVML-only process too, at 0 bytes), and the PDH instance array lists contexts newest first,
    keeping that order as others come and go. Contexts from outside this distro are interleaved by age.

    Contexts are aligned sample to sample (survivors keep their order). A match becomes *sure* only from
    evidence: a new Linux GPU process and a new context appearing together, elimination (as many unsure
    contexts as unsure processes, and just one process), contexts seen at startup while no other Linux process
    holds the GPU, or this monitor's own context being the only 0-byte candidate at startup.
    Cached foreign sizes help rank guesses but never establish ownership across runs.
    Everything else is an order guess (newest process <-> newest context, foreign assumed oldest),
    recomputed each sample and never used as evidence.
    """

    PENDING = 5.0  # seconds a new process may take to create its context
    GONE = 5.0  # seconds an exited process's context may linger in the (older) host sample
    TOL_MIN = 64 * MiB
    TOL_REL = 0.10

    def __init__(self, path: str | None = None, boot_id: str | None = None, self_pid: object = None) -> None:
        self.path = path or _default_cache()
        self.boot_id = _boot_id() if boot_id is None else boot_id
        self.self_pid = self_pid
        self.luid: int | None = None
        self.cached_foreign: list[float] = []
        self._reset()
        try:
            with open(self.path, encoding="utf-8") as f:
                data = json.load(f)
            if data.get("boot_id") == self.boot_id and self.boot_id:
                self.cached_foreign = [float(v) for v in data["foreign"]]
                self.luid = int(data["luid"])
        except (OSError, ValueError, KeyError, TypeError):
            pass

    def _reset(self) -> None:
        self.entries: list[_Entry] | None = None
        self.seen: dict[object, float] = {}  # holder pid -> first seen
        self.initial: set = set()  # holders already running at the first sample
        self.gone: dict[object, float] = {}

    def _tol(self, v: float) -> float:
        return max(self.TOL_MIN, self.TOL_REL * v)

    @staticmethod
    def _soft(x: float, y: float) -> float:
        return min(abs(x - y) / max(x, y, 256 * MiB), 1.0)

    EVENT_COST = 1.0  # per context appearing or vanishing beyond what process starts/exits explain

    def _align(self, prev: list[_Entry], cur: list[float], max_new: int, max_gone: int) -> list[_Entry | None]:
        """Previous context for each current one (None = new). New contexts are always a prefix (the list
        is newest first); vanished ones can be anywhere. Sizes decide among the structures that process
        starts/exits allow; other structures cost EVENT_COST per extra context, so a context that grows by
        GBs between samples (a model loading) stays the same context."""
        n, best, best_cost = len(prev), None, float("inf")
        for k in range(len(cur) + 1):
            surv = cur[k:]
            m = len(surv)
            if m > n:
                continue
            # f[i][j]: cost of matching all of surv[j:] in order to a subsequence of prev[i:]
            inf = float("inf")
            f = [[inf] * (m + 1) for _ in range(n + 1)]
            for i in range(n + 1):
                f[i][m] = 0.0
            for i in range(n - 1, -1, -1):
                for j in range(m - 1, -1, -1):
                    f[i][j] = min(f[i + 1][j], self._soft(prev[i].value, surv[j]) + f[i + 1][j + 1])
            gone = n - m
            cost = f[0][0] + self.EVENT_COST * (max(0, k - max_new) + max(0, gone - max_gone))
            if cost < best_cost - 1e-9:
                out: list[_Entry | None] = [None] * len(cur)
                i = j = 0
                while j < m:
                    if self._soft(prev[i].value, surv[j]) + f[i + 1][j + 1] <= f[i][j] + 1e-9:
                        out[k + j] = prev[i]
                        j += 1
                    i += 1
                best, best_cost = out, cost
        return best

    def _save(self) -> None:
        try:
            os.makedirs(os.path.dirname(self.path), exist_ok=True)
            tmp = f"{self.path}.{os.getpid()}.tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump({"boot_id": self.boot_id, "luid": self.luid, "foreign": self.cached_foreign}, f)
            os.replace(tmp, self.path)
        except OSError:
            pass

    def update(self, luid: int, contexts: list[float], holders: list[tuple[object, float]],
               now: float | None = None) -> tuple[dict[object, ContextGuess], str]:
        """contexts: vmwp.exe sizes, newest first. holders: (pid, start time) of every Linux process in this
        distro with /dev/dxg open, this monitor included. Returns ({pid: guess}, reason)."""
        now = time.time() if now is None else now
        if luid != self.luid:
            self.luid, self.cached_foreign = luid, []
            self._reset()
        start = dict(holders)
        first = self.entries is None
        had_foreign = any(e.label == FOREIGN and e.sure for e in self.entries or [])
        cur = [_Entry(v) for v in contexts]
        fresh: list[_Entry] = []  # contexts that appeared since the previous sample
        if not first:
            sure = {e.label for e in self.entries if e.sure}
            waiting = [p for p in start if p not in self.initial and p not in sure]  # new or lagging processes
            exited = [p for p in self.seen if p not in start]
            lingering = sum(1 for e in self.entries if isinstance(e.label, tuple))
            for e, old in zip(cur, self._align(self.entries, contexts, len(waiting), len(exited) + lingering)):
                if old is None:
                    fresh.append(e)
                else:
                    e.label, e.sure = old.label, old.sure

        for pid in start:
            if pid not in self.seen:
                self.seen[pid] = now
                self.gone.pop(pid, None)  # reused PID: not the process that exited
                if first:
                    self.initial.add(pid)
        for pid in [p for p in self.seen if p not in start]:
            del self.seen[pid]
            self.initial.discard(pid)  # a reused PID is a new process, not one that was running at startup
            self.gone[pid] = now
        for e in cur:
            if e.label in self.gone and not isinstance(e.label, tuple):
                e.label, e.sure = ("gone", e.label), False
            if isinstance(e.label, tuple) and now - self.gone.get(e.label[1], 0) > self.GONE:
                e.label = None
            if e.label is not None and not isinstance(e.label, tuple) and e.label != FOREIGN and e.label not in start:
                e.label, e.sure = None, False
        self.gone = {p: t for p, t in self.gone.items() if now - t <= self.GONE}
        for e in cur:  # guesses are recomputed below, never carried as evidence
            if not e.sure and not isinstance(e.label, tuple):
                e.label = None

        def labeled() -> set:
            return {e.label for e in cur if e.sure}

        def age(pid) -> float:  # larger = newer GPU context
            return start.get(pid, 0.0) if pid in self.initial else self.seen[pid]

        if not start:
            for e in cur:
                if e.label is None:
                    e.label, e.sure = FOREIGN, True
        elif first:
            zeros = [e for e in cur if e.value == 0]
            if self.self_pid in start and len(zeros) == 1:
                zeros[0].label, zeros[0].sure = self.self_pid, True
            if set(start) == {self.self_pid} and zeros:
                for e in cur:
                    if e.value > 0:  # NVML alone holds no VRAM; other zero-byte owners remain ambiguous
                        e.label, e.sure = FOREIGN, True
        elif fresh:
            pending = sorted((p for p in start if p not in labeled() and p not in self.initial
                              and now - self.seen[p] <= self.PENDING),
                             key=lambda p: -age(p))
            fresh = [e for e in fresh if e.label is None]
            if not pending:
                for e in fresh:
                    e.label, e.sure = FOREIGN, True
            elif len(fresh) == len(pending) == 1:
                fresh[0].label, fresh[0].sure = pending[0], True

        unsure = sorted((p for p in start if p not in labeled()), key=lambda p: -age(p))
        free = [e for e in cur if e.label is None]
        if unsure and len(free) > len(unsure):
            # Cached sizes may outlive their owners: rank guesses, never exclude candidates or pin labels.
            free.sort(key=lambda e: e.value > 0 and any(
                v > 0 and abs(e.value - v) <= self._tol(v) for v in self.cached_foreign))
        reason = ""
        if unsure:
            if len(free) < len(unsure):
                reason = (f"{len(unsure)} Linux GPU process(es) without a VM context to match "
                          "(one may not have created its GPU context yet)")
            # A process seen moments ago may not have created its context yet; the one free context could
            # then be a foreign one (WSLg, another distro), so wait for its own context instead of eliminating.
            recent = any(p not in self.initial and now - self.seen[p] <= self.PENDING for p in unsure)
            if len(free) == len(unsure) == 1 and not recent:
                free[0].label, free[0].sure = unsure[0], True
            else:
                # Exactly equal only: within tolerance, a swapped owner would show the wrong bytes.
                same = (len(free) == len(unsure) > 1 and not recent
                        and all(e.value == free[0].value for e in free))
                for pid, e in zip(unsure, free):  # newest process <-> newest context; leftovers are foreign
                    # Equal sizes make the shown bytes right, not the owner: never carried as evidence.
                    e.label, e.same = pid, same

        self.entries = cur
        foreign = [e.value for e in cur if e.label == FOREIGN and e.sure and e.value > 0]
        if not start or set(start) == {self.self_pid} or foreign or had_foreign:
            if sorted(foreign) != sorted(self.cached_foreign):
                self.cached_foreign = foreign
                self._save()
        out = {e.label: ContextGuess(e.value, "consistent" if e.sure or e.same else "order")
               for e in cur if e.label in start}
        return out, reason

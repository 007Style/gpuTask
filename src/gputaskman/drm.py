"""Linux AMD GPU backend: cards from sysfs and per-process memory/engine use from DRM fdinfo.

The amdgpu driver exposes device-wide numbers (VRAM, clocks, temperature, power) through sysfs and charges
each DRM client's resident memory and accumulated engine time to the fds that hold the render node open.
/proc/<pid>/fdinfo lists those clients, so a process can be attributed its VRAM and the engine time that
elapsed between two samples -- the Linux stand-in for WDDM's per-process counters.
"""

from __future__ import annotations

import os
import re
import time
from bisect import bisect_left
from dataclasses import dataclass
from functools import lru_cache

from .dxgi import VENDORS, Adapter
from .linux import pci_key
from .telemetry import DeviceTelemetry
from .pdh import AdapterGpu, ProcGpu

DRIVERS = ("amdgpu",)
SCAN_BUDGET = 0.05  # seconds per refresh for listing process fds (one-shot output: no limit)

_SIZE_UNITS = {
    "": 1,
    "b": 1,
    "kb": 1000,
    "kib": 1024,
    "mb": 1000**2,
    "mib": 1024**2,
    "gb": 1000**3,
    "gib": 1024**3,
    "tb": 1000**4,
    "tib": 1024**4,
}


@dataclass
class DrmCard:
    key: int
    pdev: str
    sysfs: str
    driver: str
    vendor_id: int
    device_id: int
    name: str
    vram_total: int
    integrated: bool
    # VCN 4+ (RDNA3 and later) runs decode and encode on one unified ring that fdinfo reports as "enc",
    # so its busy time is video work of either kind, not encode.
    unified_video: bool = False


def _read(path: str) -> str | None:
    """Contents of a sysfs/procfs file, stripped; None when it is missing or unreadable."""
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            return f.read().strip()
    except OSError:
        return None


def _int(path: str, base: int = 10) -> int | None:
    value = _read(path)
    if not value:
        return None
    try:
        return int(value, base)
    except ValueError:
        return None


def parse_size(value: str) -> int:
    """'2060 KiB' -> bytes; a bare number is bytes; anything unparseable is 0."""
    parts = (value or "").strip().split()
    if not parts:
        return 0
    try:
        number = int(parts[0])
    except ValueError:
        return 0
    unit = parts[1].lower() if len(parts) > 1 else ""
    multiplier = _SIZE_UNITS.get(unit)
    return number * multiplier if multiplier is not None else 0


def parse_fdinfo(text: str) -> dict | None:
    """One /proc/<pid>/fdinfo/<fd> file; None when it isn't a DRM fd."""
    fields: dict[str, str] = {}
    for line in text.splitlines():
        key, sep, value = line.partition(":")
        if sep:
            fields[key.strip()] = value.strip()
    if "drm-driver" not in fields:
        return None
    try:
        client_id = int(fields["drm-client-id"]) if "drm-client-id" in fields else None
    except ValueError:
        client_id = None

    def memory(resident: str, legacy: str) -> int:
        return parse_size(fields.get(resident, fields.get(legacy, "0")))

    def shared(name: str) -> int | None:
        return parse_size(fields[name]) if name in fields else None

    engines: dict[str, int] = {}
    for key, value in fields.items():
        if not key.startswith("drm-engine-") or key.startswith("drm-engine-capacity-"):
            continue
        token = value.split()[0] if value.split() else "0"
        try:
            engines[key[len("drm-engine-"):]] = int(token)
        except ValueError:
            engines[key[len("drm-engine-"):]] = 0
    return {
        "driver": fields["drm-driver"],
        "client_id": client_id,
        "pdev": fields.get("drm-pdev", ""),
        "vram": memory("drm-resident-vram", "drm-memory-vram"),
        "gtt": memory("drm-resident-gtt", "drm-memory-gtt"),
        # Buffers shared with other clients, also counted by every importer. None on older amdgpu
        # (before the generic DRM memory stats), whose fdinfo can't split private from shared.
        "shared_vram": shared("drm-shared-vram"),
        "shared_gtt": shared("drm-shared-gtt"),
        "cpu": memory("drm-resident-cpu", "drm-memory-cpu"),
        "engines": engines,
    }


def _uevent(device: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for line in (_read(os.path.join(device, "uevent")) or "").splitlines():
        key, sep, value = line.partition("=")
        if sep:
            out[key.strip()] = value.strip()
    return out


def _driver(device: str) -> str:
    try:
        return os.path.basename(os.readlink(os.path.join(device, "driver")).rstrip("/"))
    except OSError:
        return _uevent(device).get("DRIVER", "")


def _hex_id(path: str) -> int:
    value = _read(path)
    if not value:
        return 0
    try:
        return int(value, 16)
    except ValueError:
        return 0


@lru_cache(maxsize=4)
def _pci_ids(path: str) -> dict[int, dict[int, str]]:
    """{vendor id: {device id: name}} from pci.ids; subsystem lines (two tabs) are skipped."""
    out: dict[int, dict[int, str]] = {}
    vendor: int | None = None
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            for line in f:
                if not line.strip() or line.startswith("#"):
                    continue
                if line.startswith("\t\t"):
                    continue
                if line.startswith("\t"):
                    if vendor is None:
                        continue
                    parts = line.strip().split(None, 1)
                    if len(parts) != 2:
                        continue
                    try:
                        device = int(parts[0], 16)
                    except ValueError:
                        continue
                    out.setdefault(vendor, {})[device] = parts[1].strip()
                    continue
                parts = line.split(None, 1)
                vendor = None
                if len(parts) == 2:
                    try:
                        vendor = int(parts[0], 16)
                    except ValueError:
                        pass
    except OSError:
        return out
    return out


def _vendor_prefix(vendor_id: int) -> str:
    return VENDORS.get(vendor_id, hex(vendor_id))


PCI_IDS = ("/usr/share/hwdata/pci.ids", "/usr/share/misc/pci.ids")  # hwdata (Arch, Fedora); Debian/Ubuntu


def _card_name(device: str, vendor_id: int, device_id: int, pci_ids: str | None) -> str:
    product = _read(os.path.join(device, "product_name"))
    if product:
        return product
    prefix = _vendor_prefix(vendor_id)
    paths = (pci_ids,) if pci_ids else PCI_IDS
    entry = next((e for e in (_pci_ids(p).get(vendor_id, {}).get(device_id) for p in paths) if e), None)
    if not entry:
        return f"{prefix} GPU {device_id:04x}"
    if "[" in entry and "]" in entry:
        inner = entry[entry.index("[") + 1:entry.rindex("]")]
        if inner:
            return f"{prefix} {inner}"
    return f"{prefix} {entry}"


def cards(root: str = "/sys/class/drm", pci_ids: str | None = None) -> list[DrmCard]:
    """DRM cards backed by an amdgpu PCI device, discrete first then by VRAM desc."""
    try:
        entries = sorted(os.scandir(root), key=lambda e: e.name)
    except OSError:
        return []
    found: list[DrmCard] = []
    seen: set[str] = set()
    for entry in entries:
        if not re.fullmatch(r"card\d+", entry.name):
            continue
        device = os.path.join(entry.path, "device")
        driver = _driver(device)
        if driver not in DRIVERS:
            continue
        uevent = _uevent(device)
        pdev = uevent.get("PCI_SLOT_NAME") or os.path.basename(os.path.realpath(device))
        if pdev in seen:
            continue
        seen.add(pdev)
        vendor_id = _hex_id(os.path.join(device, "vendor"))
        device_id = _hex_id(os.path.join(device, "device"))
        vram_total = _int(os.path.join(device, "mem_info_vram_total")) or 0
        found.append(DrmCard(
            key=pci_key(pdev),
            pdev=pdev,
            sysfs=device,
            driver=driver,
            vendor_id=vendor_id,
            device_id=device_id,
            name=_card_name(device, vendor_id, device_id, pci_ids),
            vram_total=vram_total,
            # amdgpu reports the VRAM chips' vendor only for a discrete card; an APU's "VRAM" is a RAM carve-out.
            integrated=not os.path.exists(os.path.join(device, "mem_info_vram_vendor")),
            unified_video=_vcn_major(device) >= 4,
        ))
    found.sort(key=lambda c: (c.integrated, -c.vram_total))
    return found


def adapters(found: list[DrmCard]) -> list[Adapter]:
    """AMD cards as bare adapters (per-process data comes from DrmCounters)."""
    return [
        Adapter(
            luid=c.key,
            name=c.name,
            vendor_id=c.vendor_id,
            dedicated_vram=c.vram_total,
            shared_system=_int(os.path.join(c.sysfs, "mem_info_gtt_total")) or 0,
            software=False,
        )
        for c in found
    ]


def _hwmon_dir(device: str) -> str:
    hwmon = os.path.join(device, "hwmon")
    try:
        names = sorted(os.listdir(hwmon))
    except OSError:
        return ""
    return os.path.join(hwmon, names[0]) if names else ""


def _milli(path: str) -> int | None:
    value = _int(path)
    return None if value is None else value // 1000


def _microwatts(path: str) -> float | None:
    value = _int(path)
    return None if value is None else value / 1_000_000


def _hwmon_temp(hwmon: str) -> int | None:
    try:
        names = sorted(os.listdir(hwmon))
    except OSError:
        names = []
    for name in names:
        if name.startswith("temp") and name.endswith("_label") and _read(os.path.join(hwmon, name)) == "edge":
            return _milli(os.path.join(hwmon, f"{name[:-len('_label')]}_input"))
    return _milli(os.path.join(hwmon, "temp1_input"))


def _hwmon_clock(hwmon: str, label: str) -> int | None:
    try:
        names = sorted(os.listdir(hwmon))
    except OSError:
        names = []
    for name in names:
        if name.startswith("freq") and name.endswith("_label") and _read(os.path.join(hwmon, name)) == label:
            value = _int(os.path.join(hwmon, f"{name[:-len('_label')]}_input"))
            return None if value is None else value // 1_000_000
    return None


_PCIE_GEN = {2.5: 1, 5.0: 2, 8.0: 3, 16.0: 4, 32.0: 5, 64.0: 6}


def read_device(card: DrmCard, index: int) -> DeviceTelemetry:
    """NVIDIA-shaped device metrics for one AMD card, read from sysfs."""
    d = card.sysfs
    hwmon = _hwmon_dir(d)
    dev = DeviceTelemetry(
        index=index,
        name=card.name,
        pci_bus=card.pdev,
        driver_model=card.driver,
        source="AMD sysfs",
        mem_total=_int(os.path.join(d, "mem_info_vram_total")),
        mem_used=_int(os.path.join(d, "mem_info_vram_used")),
        util_gpu=_int(os.path.join(d, "gpu_busy_percent")),
        util_mem=_int(os.path.join(d, "mem_busy_percent")),
    )
    if hwmon:
        dev.temp = _hwmon_temp(hwmon)
        power = _microwatts(os.path.join(hwmon, "power1_average"))
        if power is None:
            power = _microwatts(os.path.join(hwmon, "power1_input"))
        dev.power_w = power
        if power is not None:
            dev.power_kind = _read(os.path.join(hwmon, "power1_label")) or ""  # amdgpu: "PPT" (package power)
        dev.power_limit_w = _microwatts(os.path.join(hwmon, "power1_cap"))
        fan_input = _int(os.path.join(hwmon, "fan1_input"))
        fan_max = _int(os.path.join(hwmon, "fan1_max"))
        if fan_input is not None and fan_max is not None and fan_max > 0:
            dev.fan = int(round(100 * fan_input / fan_max))
        dev.gfx_clock = _hwmon_clock(hwmon, "sclk")
        dev.mem_clock = _hwmon_clock(hwmon, "mclk")
    text = _read(os.path.join(d, "pp_dpm_sclk")) or ""
    clocks = [int(v) for v in re.findall(r"(\d+)\s*Mhz", text, re.IGNORECASE)]
    dev.sm_clock_max = max(clocks) if clocks else None
    speed = _read(os.path.join(d, "current_link_speed"))
    if speed:
        try:
            dev.pcie_gen = _PCIE_GEN.get(round(float(speed.split()[0]), 1))
        except (ValueError, IndexError):
            pass
    dev.pcie_width = _int(os.path.join(d, "current_link_width"))
    return dev


def _vcn_major(device: str) -> int:
    """Major version of the card's video block (VCN, listed as UVD in amdgpu's IP discovery); 0 if unknown."""
    return _int(os.path.join(device, "ip_discovery", "die", "0", "UVD", "0", "major")) or 0


def _engine_key(name: str, unified_video: bool = False) -> str:
    lowered = name.lower()
    if unified_video and ("enc" in lowered or "dec" in lowered):
        return "video"  # one ring for both: decode and encode can't be told apart
    if lowered == "gfx":
        return "3d"
    if lowered == "compute":
        return "compute"
    if lowered == "enc":
        return "encode"
    if lowered == "dec":
        return "decode"
    if "dec" in lowered or "jpeg" in lowered:
        return "decode"
    if "enc" in lowered:
        return "encode"
    if lowered == "dma":  # SDMA: buffer copies and uploads
        return "copy"
    return "other"  # vpe, ...: busy time still counts toward GPU%, like WDDM's unnamed engines


class DrmCounters:
    """Same interface as pdh.GpuCounters: sample() -> ({adapter key: {pid: ProcGpu}}, {adapter key: AdapterGpu})."""

    def __init__(self, found: list[DrmCard] | None = None, proc: str = "/proc",
                 clock=time.monotonic_ns, timer=time.perf_counter) -> None:
        self.cards = list(found) if found is not None else cards()
        self.ok = bool(self.cards)
        self._proc = proc
        self._clock = clock
        self._prev: dict[tuple, dict[str, int]] = {}
        self._prev_ts: int | None = None
        self._timer = timer
        self._known: dict[int, tuple[list[str], bool]] = {}
        self._cursor = 0
        self.status = ""  # what the last scan couldn't cover, for the UI and JSON; empty when it saw everything
        self.scan_budget: float | None = None  # the baseline below covers every process
        if self.ok:
            self.sample()  # engine load is a rate: take the baseline now, like the WDDM counters do
        self.scan_budget = SCAN_BUDGET

    def _adapters(self) -> dict[int, AdapterGpu]:
        out: dict[int, AdapterGpu] = {}
        for card in self.cards:
            vram = _int(os.path.join(card.sysfs, "mem_info_vram_used"))
            gtt = _int(os.path.join(card.sysfs, "mem_info_gtt_used"))
            out[card.key] = AdapterGpu(  # an unreadable counter stays None (unknown), not 0
                dedicated=float(vram) if vram is not None else None,
                shared=float(gtt) if gtt is not None else None,
            )
        return out

    def _scan_pid(self, pid: int) -> tuple[list[str], bool] | None:
        """(this process's /dev/dri fd names, whether it holds /dev/kfd); None when its fds can't be inspected."""
        dri: list[str] = []
        holds_kfd = False
        try:
            fds = os.scandir(os.path.join(self._proc, str(pid), "fd"))
        except PermissionError:
            return None
        except OSError:
            return dri, holds_kfd  # gone since /proc was listed
        with fds:
            for fd in fds:
                try:
                    target = os.readlink(fd.path)
                except OSError:
                    continue
                if target == "/dev/kfd":
                    holds_kfd = True
                elif target.startswith("/dev/dri/"):
                    dri.append(fd.name)
        return dri, holds_kfd

    def _clients(self) -> tuple[dict[tuple, tuple[set[int], dict]], set[int]]:
        """One entry per DRM client: (the pids holding it, fdinfo); plus the pids holding /dev/kfd.

        Listing a process's fds is the costly part, so it is bounded by scan_budget, resuming where the last refresh
        stopped. Past the budget a process keeps its last known fds, whose fdinfo is still read: rows and counters
        stay current, and only a GPU client opened since the last pass can show up a refresh or two late.
        """
        by_pdev = {c.pdev: c.key for c in self.cards}
        clients: dict[tuple, tuple[set[int], dict]] = {}
        kfd: set[int] = set()
        self.status = ""
        try:
            with os.scandir(self._proc) as entries:
                pids = sorted(int(e.name) for e in entries if e.name.isdigit())
        except OSError:
            self.status = f"{self._proc} is not readable"
            return clients, kfd
        start = bisect_left(pids, self._cursor)
        deadline = None if self.scan_budget is None else self._timer() + self.scan_budget
        known: dict[int, tuple[list[str], bool]] = {}
        denied = skipped = 0
        self._cursor = 0
        for pid in pids[start:] + pids[:start]:
            if deadline is not None and (skipped or self._timer() >= deadline):
                if not skipped:
                    self._cursor = pid
                skipped += 1
                entry = self._known.get(pid)
            else:
                entry = self._scan_pid(pid)
                denied += entry is None
            if entry is None:
                continue
            known[pid] = entry
            if entry[1]:
                kfd.add(pid)
            for name in entry[0]:
                try:
                    with open(os.path.join(self._proc, str(pid), "fdinfo", name),
                              encoding="utf-8", errors="replace") as f:
                        info = parse_fdinfo(f.read())
                except OSError:
                    continue
                if not info or info["driver"] not in DRIVERS or info["pdev"] not in by_pdev:
                    continue
                client_id = info["client_id"]
                key = (info["pdev"], client_id) if client_id is not None else (info["pdev"], pid, name)
                clients.setdefault(key, (set(), info))[0].add(pid)
        self._known = known
        notes = []
        if skipped:
            notes.append(f"scan budget reached, {skipped} processes keep their last known GPU fds this refresh")
        if denied:
            notes.append(f"{denied} processes not inspectable (other users' or kernel threads; sudo includes them)")
        self.status = "; ".join(notes)
        return clients, kfd

    def sample(self) -> tuple[dict[int, dict[int, ProcGpu]], dict[int, AdapterGpu]]:
        now = self._clock()
        by_pdev = {c.pdev: c.key for c in self.cards}
        unified = {c.pdev for c in self.cards if c.unified_video}
        procs: dict[int, dict[int, ProcGpu]] = {c.key: {} for c in self.cards}
        adapters = self._adapters()
        clients, kfd = self._clients()
        delta = now - self._prev_ts if self._prev_ts is not None else 0
        previous: dict[tuple, dict[str, int]] = {}
        for client_key, (owners, info) in clients.items():
            key = by_pdev[info["pdev"]]
            previous[client_key] = dict(info["engines"])
            rows = []
            for pid in owners:
                # No commit counter in fdinfo, so Paged stays unavailable rather than 0. ROCm/HIP (a /dev/kfd
                # holder) submits compute through user-mode KFD queues the kernel doesn't time per process, so
                # its engine load is unknown, not 0 (AGENTS.md rule 2); its VRAM is still in fdinfo.
                available = {"dedicated", "shared"} if pid in kfd else {"dedicated", "shared", "engine"}
                proc = procs[key].setdefault(pid, ProcGpu(attributed=True, available=available))
                proc.kfd = pid in kfd
                rows.append(proc)
            if len(owners) > 1:
                # One client held by several processes (fd passing, fork): nothing says which of them uses it,
                # so each is listed without counters instead of charging one by guess (rules 1 and 5). Its bytes
                # stay in the adapter's unattributed part, like a shared buffer's, so the other rows still add up.
                for proc in rows:
                    proc.attributed = False
                    proc.shared_client = True
                    proc.available -= {"shared", "engine"}
                continue
            proc = rows[0]
            # A buffer shared between clients is resident in each of them and fdinfo doesn't say which owns it.
            # Only private bytes are charged (AGENTS.md rule 1); shared ones are reported apart, owner unknown.
            if info["shared_vram"] is None:
                # Older amdgpu: resident bytes include every imported buffer, so summing them would count
                # shared buffers once per importer. Unknown rather than double counted (rules 1 and 9).
                proc.attributed = False
                proc.available -= {"dedicated", "shared"}
            else:
                shared = float(info["shared_vram"])
                proc.dedicated += max(float(info["vram"]) - shared, 0.0)
                proc.shared_vram += shared
                proc.shared += max(float(info["gtt"]) - float(info["shared_gtt"] or 0), 0.0)
            old = self._prev.get(client_key)
            if old is None or delta <= 0:
                proc.available.discard("engine")  # a rate needs two samples: unknown, not 0 (rule 2)
                continue
            engines: dict[str, float] = {}
            for name, ns in info["engines"].items():
                # amdgpu omits an engine until it has run, so a missing earlier value really is 0.
                pct = min(100.0, max(0.0, 100.0 * (ns - old.get(name, 0)) / delta))
                engine = _engine_key(name, info["pdev"] in unified)
                engines[engine] = engines.get(engine, 0.0) + pct
            for engine, pct in engines.items():
                proc.engines[engine] = min(100.0, proc.engines[engine] + pct)
        self._prev = previous
        self._prev_ts = now
        return procs, adapters

    def close(self) -> None:
        """No handles are held."""
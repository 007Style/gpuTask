"""Linux AMD and Intel GPU backend: cards from sysfs and per-process memory/engine use from DRM fdinfo.

The amdgpu and i915/xe drivers charge each DRM client's resident memory and accumulated engine time to
the fds that hold a render node open. /proc/<pid>/fdinfo lists those clients, so a process can be
attributed its VRAM and the engine time that elapsed between two samples -- the Linux stand-in for
WDDM's per-process counters. Device-wide numbers come from sysfs: amdgpu reports VRAM, clocks,
temperature and power directly; i915/xe expose temperature (hwmon), frequency (gt nodes) and an energy
counter, and have no per-process or device-wide memory sysfs beyond a discrete card's resizable BAR.
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

# Intel cards use resizable BARs for local memory (no mem_info_vram_total); a system-memory BAR of 256MB
# is the UHD 730's window, a multi-GB BAR is a discrete card's local VRAM. Check before trusting.
_LOCAL_BAR_MIN = 1 << 30  # 1 GiB: smallest real local-memory BAR we accept (Arc A310 is 4 GiB)

DRIVERS = ("amdgpu", "i915", "xe")
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
    # Intel (i915/xe): no device-wide memory sysfs exists. vram_total holds the local-memory (resizable
    # BAR) window for discrete cards, 0 for integrated ones; per-process bytes still come from fdinfo.
    intel: bool = False
    # Intel only: used local memory, stamped in each DrmCounters.sample() as the sum of the card's
    # clients' local bytes (shared buffers can double-count); None = no readable client, unknown.
    vram_used: int | None = None


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


def parse_size(value: str | None) -> int | None:
    """'2060 KiB' -> bytes; a bare number is bytes; anything unparseable is None (unknown, not 0)."""
    parts = (value or "").strip().split()
    if not parts:
        return None
    try:
        number = int(parts[0])
    except ValueError:
        return None
    unit = parts[1].lower() if len(parts) > 1 else ""
    multiplier = _SIZE_UNITS.get(unit)
    return number * multiplier if multiplier is not None else None


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

    def memory(resident: str, legacy: str) -> int | None:
        return parse_size(fields.get(resident, fields.get(legacy)))

    def intel_resident(*keys: str) -> int | None:
        """Resident bytes of an Intel logical region: the sum of its memory placements. A placement key
        that is absent counts as 0 (older kernels don't report stolen); all keys absent means the card
        has no such region (an iGPU has no local0) and returns None -- unknown, not 0."""
        values = [parse_size(fields.get(k)) for k in keys]
        if all(v is None for v in values):
            return None
        return sum(v for v in values if v is not None)

    # drm-shared-<region> counts buffers with more than one handle by their *preferred* region, resident or not,
    # while drm-resident-<region> counts by where buffers are now, so the two can't be subtracted. A shared
    # system-memory buffer can therefore be resident in GTT. Count every reported shared region, including CPU;
    # missing or unparseable standard regions mean unknown, not 0.
    required_shared_keys = ("drm-shared-vram", "drm-shared-gtt", "drm-shared-cpu")
    shared_keys = [k for k in fields if k.startswith("drm-shared-")]
    shared_values = [parse_size(fields[k]) for k in shared_keys]
    shared = (sum(shared_values) if all(k in fields for k in required_shared_keys)
              and shared_keys and None not in shared_values else None)

    engines: dict[str, int] | None = {}
    for key, value in fields.items():
        if not key.startswith("drm-engine-") or key.startswith("drm-engine-capacity-"):
            continue
        try:
            engines[key[len("drm-engine-"):]] = int(value.split()[0])
        except (ValueError, IndexError):
            engines = None  # a torn or malformed counter: this client's load is unknown, not 0
            break
    if fields.get("drm-driver") in ("i915", "xe"):
        if not engines:
            # i915 (gen8+) prints a drm-engine-<class> line for each engine class the device has, even at 0 ns;
            # xe reports drm-cycles-* instead and gen7 i915 nothing. No such line means the load is unknown, not 0
            # (AGENTS.md rule 2).
            engines = None
        # Intel's logical regions are local0 (device memory; the local/stolen pair covers memory that
        # shrunk out of local into system) and system0 (GPU-placed system memory, the GTT equivalent).
        # There is no separate CPU-only region: system0 feeds the shared/spill slot instead. These are
        # i915's names; xe calls them vram0/gtt/system, so its memory stays unknown until validated.
        return {
            "driver": fields["drm-driver"],
            "client_id": client_id,
            "pdev": fields.get("drm-pdev", ""),
            # The shared-buffer gate needs every shared region present and parseable: any ambiguity keeps
            # the gate closed (None) instead of risking shared buffers being charged as private.
            "shared": (sum(shared_values) if shared_keys and None not in shared_values
                       and all(k in fields for k in ("drm-shared-local0", "drm-shared-system0")) else None),
            "vram": intel_resident("drm-resident-local0", "drm-resident-stolen-local0"),
            "gtt": intel_resident("drm-resident-system0", "drm-resident-stolen-system0"),
            "cpu": None,
            "engines": engines,
        }
    return {
        "driver": fields["drm-driver"],
        "client_id": client_id,
        "pdev": fields.get("drm-pdev", ""),
        "vram": memory("drm-resident-vram", "drm-memory-vram"),
        "gtt": memory("drm-resident-gtt", "drm-memory-gtt"),
        "shared": shared,
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
        intel = vendor_id == 0x8086 and driver in ("i915", "xe")
        vram_total = _int(os.path.join(device, "mem_info_vram_total")) or 0
        if intel and not vram_total:
            # i915/xe expose no VRAM sysfs. The resource2 file IS the local-memory BAR, so its size is
            # the window (the file is binary -- its contents are not a number): >= 1 GiB marks a
            # discrete card's local VRAM, ~256 MB is an iGPU's system-memory window.
            try:
                bar = os.path.getsize(os.path.join(device, "resource2_wc"))
                if bar <= 0:
                    bar = os.path.getsize(os.path.join(device, "resource2"))
            except OSError:
                bar = 0
            vram_total = bar if bar >= _LOCAL_BAR_MIN else 0
        found.append(DrmCard(
            key=pci_key(pdev),
            pdev=pdev,
            sysfs=device,
            driver=driver,
            vendor_id=vendor_id,
            device_id=device_id,
            name=_card_name(device, vendor_id, device_id, pci_ids),
            vram_total=vram_total,
            # amdgpu reports the VRAM chips' vendor only for a discrete card; an APU's "VRAM" is a RAM
            # carve-out. Intel reports no vendor node at all: a card with a local-memory BAR is discrete.
            integrated=(vram_total == 0) if intel else not os.path.exists(os.path.join(device, "mem_info_vram_vendor")),
            unified_video=_vcn_major(device) >= 4,
            intel=intel,
        ))
    found.sort(key=lambda c: (c.integrated, -c.vram_total))
    return found


def adapters(found: list[DrmCard]) -> list[Adapter]:
    """AMD and Intel cards as bare adapters (per-process data comes from DrmCounters)."""
    return [
        Adapter(
            luid=c.key,
            name=c.name,
            vendor_id=c.vendor_id,
            # Intel: the local-memory BAR for discrete cards, 0 for an integrated GPU (its carve-out size
            # is not exposed; totals stay unknown rather than invented).
            dedicated_vram=c.vram_total,
            shared_system=_int(os.path.join(c.sysfs, "mem_info_gtt_total")) or 0 if not c.intel else 0,
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
    """NVIDIA-shaped device metrics for one AMD or Intel card, read from sysfs."""
    d = card.sysfs
    hwmon = _hwmon_dir(d)
    dev = DeviceTelemetry(
        index=index,
        name=card.name,
        pci_bus=card.pdev,
        driver_model=card.driver,
        source="AMD sysfs" if not card.intel else "Intel sysfs",
        mem_total=_int(os.path.join(d, "mem_info_vram_total")) if not card.intel else (card.vram_total or None),
        mem_used=_int(os.path.join(d, "mem_info_vram_used")) if not card.intel else card.vram_used,
        util_gpu=_int(os.path.join(d, "gpu_busy_percent")) if not card.intel else None,
        # i915/xe have no gpu_busy_percent, engine busy nodes, or memory sysfs: card-level GPU% stays
        # unknown (per-process engine deltas are the live signal); used memory is stamped in by
        # DrmCounters (sum of the card's clients' local bytes).
        util_mem=None if card.intel else _int(os.path.join(d, "mem_busy_percent")),
    )
    if hwmon:
        dev.temp = _hwmon_temp(hwmon)
        if card.intel:
            # Intel hwmon has no instantaneous power node, but energy1_input is a cumulative uJ counter;
            # its rate over the interval between samples is the card's average power draw.
            dev.power_w = _intel_power_w(hwmon)
            dev.power_kind = "Board" if dev.power_w is not None else ""
        else:
            power = _microwatts(os.path.join(hwmon, "power1_average"))
            if not power:  # some parts expose power1_average but leave it at 0
                input_power = _microwatts(os.path.join(hwmon, "power1_input"))
                if input_power is not None:
                    power = input_power
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
    if card.intel:
        # The gt nodes hang off the DRM card itself (/sys/class/drm/cardN/gt/gt0), not the PCI device.
        gt = os.path.join(os.path.dirname(d), "gt", "gt0")
        cur = _int(os.path.join(gt, "rps_cur_freq_mhz"))
        if cur is not None:
            dev.gfx_clock = cur  # engine frequency from the gt node (hwmon has no sclk on Intel)
        dev.sm_clock_max = _int(os.path.join(gt, "rps_RP0_freq_mhz"))
        # sysfs current_link_* on i915/xe can report the power-on default instead of the live link
        # (2.5 GT/s x1 shown while a measured DMA transfer ran at ~19 GB/s), so both stay unset:
        # unknown beats a confidently wrong number.
    else:
        text = _read(os.path.join(d, "pp_dpm_sclk")) or ""
        clocks = [int(v) for v in re.findall(r"(\d+)\s*Mhz", text, re.IGNORECASE)]
        dev.sm_clock_max = dev.sm_clock_max or (max(clocks) if clocks else None)
        speed = _read(os.path.join(d, "current_link_speed"))
        if speed:
            try:
                dev.pcie_gen = _PCIE_GEN.get(round(float(speed.split()[0]), 1))
            except (ValueError, IndexError):
                pass
        dev.pcie_width = _int(os.path.join(d, "current_link_width"))
    return dev


def _intel_power_w(hwmon: str) -> float | None:
    """Average watts since the previous sample, from Intel's cumulative hwmon energy counter (uJ)."""
    energy = _int(os.path.join(hwmon, "energy1_input"))
    if energy is None:
        return None
    now = time.perf_counter()
    prev = _INTEL_ENERGY_STATE.get(hwmon)
    _INTEL_ENERGY_STATE[hwmon] = (energy, now)
    if prev is None:
        return None  # first sample: the rate needs an interval
    before, at = prev
    seconds = now - at
    if seconds <= 0:
        return None
    watts = (energy - before) / 1e6 / seconds
    return watts if watts >= 0 else None  # counter wrapped or card reset: unknown, not negative


_INTEL_ENERGY_STATE: dict[str, tuple[int, float]] = {}  # hwmon path -> (last uJ, last read time); per card


def _vcn_major(device: str) -> int:
    """Major version of the card's video block (VCN, listed as UVD in amdgpu's IP discovery); 0 if unknown."""
    return _int(os.path.join(device, "ip_discovery", "die", "0", "UVD", "0", "major")) or 0


def _engine_key(name: str, unified_video: bool = False, intel: bool = False) -> str:
    if intel:
        lowered = name.lower()
        # i915's fdinfo names whole engine classes: render, copy, video, video-enhance, compute. The command-streamer
        # names (rcs, ccs, bcs, vcs, vecs, optionally with an instance number) are how xe and the engine tables spell
        # the same classes. Video-enhance (VECS) is post-processing, not encode, so it counts as video.
        if lowered == "render" or lowered.startswith("rcs"):
            return "3d"
        if lowered == "compute" or lowered.startswith("ccs"):
            return "compute"
        if lowered == "copy" or lowered.startswith("bcs"):
            return "copy"
        if lowered.startswith(("video", "vcs", "vecs")):
            return "video"  # decode and encode can't be told apart
        return "other"  # any other class: busy time still counts toward GPU%
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


_DEFERRED = object()  # _scan_pid: the scan budget ran out partway through this process's fds
_PRUNE_NS = 60 * 1_000_000_000  # forget engine baselines for clients not seen for a minute

# Why a row's DRM memory is n/a instead of measured (ProcGpu.drm_reason)
REASON_LEGACY = "legacy"  # missing or unparseable shared stats: private and shared bytes can't be told apart
REASON_SHARED_CLIENT = "shared_client"  # one DRM client held by several processes (fd passing, fork)
REASON_SHARED_BUFFERS = "shared_buffers"  # the client holds buffers shared with other clients (dma-buf)
REASON_PRIMARY = "primary"  # opened via /dev/dri/card*, whose root-held copies (logind) a normal user can't see
REASON_INCOMPLETE = "incomplete"  # some of its fds or fdinfo couldn't be read
REASON_PARTIAL_SCAN = "partial_scan"  # the budget ran out before every process was listed: a holder may be missed


class DrmCounters:
    """Same interface as pdh.GpuCounters: sample() -> ({adapter key: {pid: ProcGpu}}, {adapter key: AdapterGpu}).

    Call start() once the card is selected: engine load is a rate, so it takes the first (unbudgeted) baseline.
    """

    def __init__(self, found: list[DrmCard] | None = None, proc: str = "/proc",
                 clock=time.monotonic_ns, timer=time.perf_counter) -> None:
        self.cards = list(found) if found is not None else cards()
        self.ok = bool(self.cards)
        self._proc = proc
        self._clock = clock
        self._timer = timer
        self._prev: dict[tuple, tuple[dict[str, int], int]] = {}  # client -> (high-water engine ns, read time)
        # pid -> (start time, [(fd name, link target)], holds /dev/kfd, every fd and link was readable)
        self._known: dict[int, tuple[str, list[tuple[str, str]], bool, bool]] = {}
        self._cursor = 0
        self.status = ""  # what the last scan couldn't cover, for the UI and JSON; empty when it saw everything
        self.partial = False  # the last scan ran out of budget, so a DRM client's holders may not all be known
        self.scan_budget: float | None = SCAN_BUDGET

    def start(self) -> None:
        """Take the engine baseline over every process, like the WDDM counters do at startup."""
        if not self.ok:
            return
        budget, self.scan_budget = self.scan_budget, None
        self.sample()
        self.scan_budget = budget

    def _adapters(self) -> dict[int, AdapterGpu]:
        out: dict[int, AdapterGpu] = {}
        for card in self.cards:
            if card.intel:
                # No device-wide sysfs: used memory is the client sum stamped in _stamp_intel_memory
                # (None stays None); no device-wide system-memory counter exists at all.
                out[card.key] = AdapterGpu(
                    dedicated=float(card.vram_used) if card.vram_used is not None else None,
                    shared=None,
                )
                continue
            vram = _int(os.path.join(card.sysfs, "mem_info_vram_used"))
            gtt = _int(os.path.join(card.sysfs, "mem_info_gtt_used"))
            out[card.key] = AdapterGpu(  # an unreadable counter stays None (unknown), not 0
                dedicated=float(vram) if vram is not None else None,
                shared=float(gtt) if gtt is not None else None,
            )
        return out

    def _stamp_intel_memory(self, clients: dict[tuple, list]) -> None:
        """Intel cards expose no device-wide memory sysfs: used local memory is the sum of the card's
        clients' resident local bytes, counted once per client even when several processes share it
        (buffers shared between clients can double-count, and clients that can't be inspected are missing;
        report.INTEL_MEMORY_NOTE says so). With no readable client the sum is None -- unknown, not 0."""
        for card in self.cards:
            if not card.intel:
                continue
            used = [c[1]["vram"] for _key, c in clients.items() if c[1]["pdev"] == card.pdev]
            card.vram_used = sum(used) if used and None not in used else None

    def _start_time(self, pid: int) -> str | None:
        """Field 22 of /proc/<pid>/stat: tells a reused pid from the process cached under it."""
        try:
            with open(os.path.join(self._proc, str(pid), "stat"), encoding="utf-8", errors="replace") as f:
                fields = f.read().rsplit(")", 1)[1].split()
            return fields[19]
        except (OSError, IndexError):
            return None

    def _scan_pid(self, pid: int, deadline: float | None):
        """(start time, [(fd name, target)] for /dev/dri, holds /dev/kfd, complete); None when its fds can't be
        listed; _DEFERRED when the budget ran out partway through."""
        start = self._start_time(pid)
        dri: list[tuple[str, str]] = []
        holds_kfd = False
        complete = True
        try:
            fds = os.scandir(os.path.join(self._proc, str(pid), "fd"))
        except PermissionError:
            return None
        except OSError:
            return start, dri, holds_kfd, complete  # gone since /proc was listed
        with fds:
            for n, fd in enumerate(fds, 1):
                if deadline is not None and n % 64 == 0 and self._timer() >= deadline:
                    return _DEFERRED
                try:
                    target = os.readlink(fd.path)
                except PermissionError:
                    complete = False
                    continue
                except OSError:
                    continue  # closed since the listing
                if target == "/dev/kfd":
                    holds_kfd = True
                elif target.startswith("/dev/dri/"):
                    dri.append((fd.name, target))
        return start, dri, holds_kfd, complete

    def _clients(self):
        """One entry per DRM client: [pids holding it, fdinfo, via a render node, read time]; plus the pids holding
        /dev/kfd and the pids whose fds or fdinfo couldn't all be read.

        Listing a process's fds is the costly part, so it is bounded by scan_budget: processes not seen before go
        first, then the rest resume where the last refresh stopped. Past the budget a process keeps its last known
        fds (dropped if its pid was reused), whose fdinfo is still read: rows and counters stay current, and only an
        fd handed to an already-running process shows up a refresh or two late.
        """
        by_pdev = {c.pdev: c.key for c in self.cards}
        clients: dict[tuple, list] = {}
        kfd: set[int] = set()
        incomplete: set[int] = set()
        self.status = ""
        self.partial = False
        try:
            with os.scandir(self._proc) as entries:
                pids = sorted(int(e.name) for e in entries if e.name.isdigit())
        except OSError:
            self.status = f"{self._proc} is not readable"
            return clients, kfd, incomplete
        start = bisect_left(pids, self._cursor)
        ordered = pids[start:] + pids[:start]
        ordered = [p for p in ordered if p not in self._known] + [p for p in ordered if p in self._known]
        deadline = None if self.scan_budget is None else self._timer() + self.scan_budget
        known: dict[int, tuple] = {}
        denied = deferred = 0
        out_of_time = False
        cursor = None  # first known process not rescanned: where the rotation resumes (new ones go first anyway)
        scanned = 0
        for pid in ordered:
            entry = None
            if not out_of_time and (deadline is None or self._timer() < deadline):
                # The first process of a refresh is always listed in full, so one huge fd table can't stall the rotation.
                entry = self._scan_pid(pid, deadline if scanned else None)
                scanned += 1
                if entry is None:
                    denied += 1
                    continue
                if entry is _DEFERRED:
                    entry = None
            if entry is None:  # not scanned this refresh: reuse the cached fds if this is still the same process
                out_of_time = True
                if cursor is None and pid in self._known:
                    cursor = pid
                deferred += 1
                cached = self._known.get(pid)
                if cached is None or cached[0] != self._start_time(pid):
                    continue
                entry = cached
            known[pid] = entry
            _start, dri, holds_kfd, complete = entry
            if holds_kfd:
                kfd.add(pid)
            if not complete:
                incomplete.add(pid)
            for name, target in dri:
                try:
                    with open(os.path.join(self._proc, str(pid), "fdinfo", name),
                              encoding="utf-8", errors="replace") as f:
                        text = f.read()
                except PermissionError:
                    incomplete.add(pid)
                    continue
                except OSError:
                    continue  # closed since the listing
                read_at = self._clock()
                info = parse_fdinfo(text)
                if not info or info["driver"] not in DRIVERS or info["pdev"] not in by_pdev:
                    continue
                client_id = info["client_id"]
                key = (info["pdev"], client_id) if client_id is not None else (info["pdev"], pid, name)
                client = clients.setdefault(key, [set(), info, True, read_at])
                client[0].add(pid)
                client[2] = client[2] and target.startswith("/dev/dri/renderD")
        self._known = known
        self.partial = out_of_time
        if not out_of_time:
            self._cursor = 0
        elif cursor is not None:
            self._cursor = cursor  # else only new processes were deferred: the known ones keep their place
        notes = []
        if out_of_time:
            notes.append(f"scan budget reached, {deferred} processes not rescanned this refresh (last known GPU fds kept)")
        if denied:
            notes.append(f"{denied} processes not inspectable (other users' or kernel threads; sudo includes them)")
        if incomplete:
            notes.append(f"{len(incomplete)} processes whose fds or fdinfo couldn't all be read")
        self.status = "; ".join(notes)
        return clients, kfd, incomplete

    def sample(self) -> tuple[dict[int, dict[int, ProcGpu]], dict[int, AdapterGpu]]:
        by_pdev = {c.pdev: c.key for c in self.cards}
        unified = {c.pdev for c in self.cards if c.unified_video}
        intel_pdevs = {c.pdev for c in self.cards if c.intel}
        procs: dict[int, dict[int, ProcGpu]] = {c.key: {} for c in self.cards}
        clients, kfd, incomplete = self._clients()
        self._stamp_intel_memory(clients)
        adapters = self._adapters()

        by_pid: dict[tuple[int, int], list[tuple[tuple, list]]] = {}
        for client_key, client in clients.items():
            for pid in client[0]:
                by_pid.setdefault((by_pdev[client[1]["pdev"]], pid), []).append((client_key, client))

        now = self._clock()
        current: dict[tuple, tuple[dict[str, int], int]] = {}
        for client_key, (owners, info, _render, read_at) in clients.items():
            old = self._prev.get(client_key)
            if info["engines"] is None:
                if old is not None:
                    current[client_key] = old
                continue
            high = dict(old[0]) if old else {}
            for name, ns in info["engines"].items():
                high[name] = max(ns, high.get(name, 0))  # the DRM spec allows a counter to step back briefly
            current[client_key] = (high, read_at)

        for (key, pid), held in by_pid.items():
            # No commit counter in fdinfo, so Paged stays unavailable rather than 0. ROCm/HIP (a /dev/kfd holder)
            # submits compute through user-mode KFD queues the kernel doesn't time per process, so its engine load is
            # unknown, not 0 (AGENTS.md rule 2); its VRAM is still in fdinfo.
            proc = ProcGpu(attributed=True, available={"dedicated", "shared", "engine"})
            proc.kfd = pid in kfd
            infos = [c[1] for _k, c in held]
            vram = [i["vram"] for i in infos]
            gtt = [i["gtt"] for i in infos]
            # Every buffer this process can reach, shared ones included: a reference, never summed (rule 1).
            complete = pid not in incomplete  # otherwise any sum would be a partial total that looks whole
            proc.resident_vram = float(sum(vram)) if complete and None not in vram else None
            proc.resident_gtt = float(sum(gtt)) if complete and None not in gtt else None

            # Bytes are charged to this process only when nothing else can hold them: one visible holder, reached
            # through a render node, no shared buffers at all, and every value known. Anything else leaves the whole
            # process's bytes in the adapter's unattributed part instead of charging one by guess (rules 1 and 5).
            reason = ""
            if pid in incomplete:
                reason = REASON_INCOMPLETE
            elif any(len(c[0]) > 1 for _k, c in held):
                reason = REASON_SHARED_CLIENT
            elif any(i["shared"] is None for i in infos):
                reason = REASON_LEGACY
            elif any(i["shared"] for i in infos):
                reason = REASON_SHARED_BUFFERS
            elif not all(c[2] for _k, c in held):
                reason = REASON_PRIMARY
            elif proc.resident_vram is None or proc.resident_gtt is None:
                reason = REASON_LEGACY
            elif self.partial:
                reason = REASON_PARTIAL_SCAN
            proc.drm_reason = reason
            if reason:
                proc.attributed = False
                proc.available -= {"dedicated", "shared"}
            else:
                proc.dedicated = proc.resident_vram
                proc.shared = proc.resident_gtt

            engines: dict[str, float] = {}
            for client_key, (owners, info, _render, read_at) in held:
                old = self._prev.get(client_key)
                if (proc.kfd or not complete or self.partial or len(owners) > 1 or info["engines"] is None
                        or old is None or read_at - old[1] <= 0):
                    engines = {}
                    proc.available.discard("engine")  # unknown, not 0 (rule 2)
                    break
                delta = read_at - old[1]
                for name, ns in current[client_key][0].items():
                    # amdgpu omits an engine until it has run, so a missing earlier value really is 0.
                    pct = min(100.0, max(0.0, 100.0 * (ns - old[0].get(name, 0)) / delta))
                    engine = _engine_key(name, info["pdev"] in unified, info["pdev"] in intel_pdevs)
                    engines[engine] = min(100.0, engines.get(engine, 0.0) + pct)
            for engine, pct in engines.items():
                proc.engines[engine] = pct
            procs[key][pid] = proc

        for client_key, value in self._prev.items():  # keep baselines of clients a deferred scan missed, briefly
            if client_key not in current and now - value[1] < _PRUNE_NS:
                current[client_key] = value
        self._prev = current
        return procs, adapters

    def close(self) -> None:
        """No handles are held."""
"""Device-level NVIDIA metrics via NVML (nvml.dll ships with the driver)."""

from __future__ import annotations

from .telemetry import DeviceTelemetry as NvDevice

try:
    import pynvml as N
except ImportError:  # pragma: no cover
    N = None


def format_metric(value: float | int | None, spec: str = "", unit: str = "") -> str:
    return f"{format(value, spec)}{unit}" if value is not None else "N/A"


_THROTTLE = {
    0x2: "app-clocks",
    0x4: "sw-power-cap",
    0x8: "hw-slowdown",
    0x20: "sw-thermal",
    0x40: "hw-thermal",
    0x80: "power-brake",
}
_RESET_ERRORS = (1, 9, 15)  # uninitialized, driver not loaded, GPU lost

# NVML is one session per process. After Windows disables and re-enables a GPU, reads for that GPU
# can fault inside nvml.dll and nvmlShutdown() can hang (reported on an RTX 3060, driver 610.88).
LOST: set[str] = set()  # PCI bus IDs whose reads faulted; never read again in this process
_unsafe = False


def mark_unsafe() -> None:
    """An NVIDIA adapter went away: skip nvmlShutdown() from now on and let process exit clean up."""
    global _unsafe
    _unsafe = True


def _try(fn, *a, default=None):
    try:
        return fn(*a)
    except Exception:
        return default


class Nvml:
    def __init__(self) -> None:
        self.ok = False
        self.driver = ""
        self.cuda_version = ""
        self.handles = []
        self.needs_reset = False
        self._faulted = False
        self.refresh()

    def refresh(self) -> None:
        """Reacquire handles without sampling telemetry or restarting a healthy NVML session."""
        if N is None:
            return
        if self.needs_reset:
            self.close()
            self.needs_reset = False
        if not self.ok:
            try:
                N.nvmlInit()
            except Exception:
                return
            self.ok = True
        try:
            handles = []
            reset_needed = False
            for i in range(N.nvmlDeviceGetCount()):
                try:
                    handles.append((i, N.nvmlDeviceGetHandleByIndex(i)))
                except Exception as error:
                    reset_needed |= getattr(error, "value", None) in _RESET_ERRORS
                    continue  # one unavailable device must not hide its surviving peers
            self.handles = handles
            if not handles and reset_needed:
                self.needs_reset = True
        except Exception:
            self.handles = []
            self.close()  # next low-frequency refresh retries driver initialization
            return
        self.driver = _try(N.nvmlSystemGetDriverVersion, default="") or ""
        cv = _try(N.nvmlSystemGetCudaDriverVersion_v2) or _try(N.nvmlSystemGetCudaDriverVersion)
        if cv:
            self.cuda_version = f"{cv // 1000}.{(cv % 1000) // 10}"

    def _read(self, fn, *args, default=None):
        try:
            return fn(*args)
        except OSError:  # access violation inside nvml.dll, not an NVML error code
            self._faulted = True
            return default
        except Exception as error:
            if getattr(error, "value", None) in _RESET_ERRORS:
                self.needs_reset = True
            return default

    def sample(self) -> list[NvDevice]:
        out = []
        for i, h in self.handles:
            self._faulted = False
            d = NvDevice(index=i)
            d.name = self._read(N.nvmlDeviceGetName, h, default="?")
            pci = self._read(N.nvmlDeviceGetPciInfo, h)
            if pci is not None:
                d.pci_bus = pci.busId.decode() if isinstance(pci.busId, bytes) else str(pci.busId)
            if d.pci_bus in LOST or self._faulted:
                continue
            dm = self._read(N.nvmlDeviceGetDriverModel, h)
            if dm:
                d.driver_model = {0: "WDDM", 1: "TCC", 2: "MCDM"}.get(dm[0], str(dm[0]))
            mem = self._read(N.nvmlDeviceGetMemoryInfo, h, N.nvmlMemory_v2) or self._read(N.nvmlDeviceGetMemoryInfo, h)
            if mem is not None:
                d.mem_total, d.mem_used = mem.total, mem.used
                d.mem_reserved = getattr(mem, "reserved", 0)
            u = self._read(N.nvmlDeviceGetUtilizationRates, h)
            if u is not None:
                d.util_gpu, d.util_mem = u.gpu, u.memory
            enc = self._read(N.nvmlDeviceGetEncoderUtilization, h)
            dec = self._read(N.nvmlDeviceGetDecoderUtilization, h)
            d.enc = enc[0] if enc else None
            d.dec = dec[0] if dec else None
            d.temp = self._read(N.nvmlDeviceGetTemperature, h, N.NVML_TEMPERATURE_GPU)
            p = self._read(N.nvmlDeviceGetPowerUsage, h)
            d.power_w = p / 1000 if p is not None else None
            pl = self._read(N.nvmlDeviceGetEnforcedPowerLimit, h)
            d.power_limit_w = pl / 1000 if pl is not None else None
            d.fan = self._read(N.nvmlDeviceGetFanSpeed, h)
            d.pstate = self._read(N.nvmlDeviceGetPerformanceState, h)
            d.sm_clock = self._read(N.nvmlDeviceGetClockInfo, h, N.NVML_CLOCK_SM)
            d.sm_clock_max = self._read(N.nvmlDeviceGetMaxClockInfo, h, N.NVML_CLOCK_SM)
            d.mem_clock = self._read(N.nvmlDeviceGetClockInfo, h, N.NVML_CLOCK_MEM)
            d.video_clock = self._read(N.nvmlDeviceGetClockInfo, h, N.NVML_CLOCK_VIDEO)
            d.pcie_rx_kbs = self._read(N.nvmlDeviceGetPcieThroughput, h, N.NVML_PCIE_UTIL_RX_BYTES)
            d.pcie_tx_kbs = self._read(N.nvmlDeviceGetPcieThroughput, h, N.NVML_PCIE_UTIL_TX_BYTES)
            d.pcie_gen = self._read(N.nvmlDeviceGetCurrPcieLinkGeneration, h)
            d.pcie_width = self._read(N.nvmlDeviceGetCurrPcieLinkWidth, h)
            reasons = self._read(N.nvmlDeviceGetCurrentClocksEventReasons, h)
            if reasons is None:
                reasons = self._read(N.nvmlDeviceGetCurrentClocksThrottleReasons, h)
            if reasons:
                d.throttle = [n for bit, n in _THROTTLE.items() if reasons & bit]
            cc = self._read(N.nvmlDeviceGetCudaComputeCapability, h)
            if cc:
                d.compute_cap = f"{cc[0]}.{cc[1]}"
            if self._faulted:
                if d.pci_bus:
                    LOST.add(d.pci_bus)
                mark_unsafe()
                continue
            out.append(d)
        return out

    def close(self) -> None:
        if self.ok:
            if not _unsafe:
                _try(N.nvmlShutdown)
            self.ok = False

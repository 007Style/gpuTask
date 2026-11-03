"""Device-level NVIDIA metrics via NVML (nvml.dll ships with the driver)."""

from __future__ import annotations

from dataclasses import dataclass, field

try:
    import pynvml as N
except ImportError:  # pragma: no cover
    N = None


@dataclass
class NvDevice:
    index: int
    name: str = "?"
    pci_bus: str = ""
    driver_model: str = ""  # WDDM / TCC / MCDM
    mem_total: int = 0
    mem_used: int = 0
    mem_reserved: int = 0
    util_gpu: float | None = None  # % of time a kernel was running
    util_mem: float | None = None  # % of time device memory was being read/written (bandwidth pressure)
    enc: float | None = None
    dec: float | None = None
    temp: int | None = None
    power_w: float | None = None
    power_limit_w: float | None = None
    fan: int | None = None
    pstate: int | None = None
    sm_clock: int | None = None
    sm_clock_max: int | None = None
    mem_clock: int | None = None
    video_clock: int | None = None
    pcie_rx_kbs: int | None = None  # host -> GPU
    pcie_tx_kbs: int | None = None  # GPU -> host
    pcie_gen: int | None = None
    pcie_width: int | None = None
    throttle: list[str] = field(default_factory=list)
    compute_cap: str = ""


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
        if N is None:
            return
        try:
            N.nvmlInit()
        except Exception:
            return
        self.ok = True
        self.driver = _try(N.nvmlSystemGetDriverVersion, default="") or ""
        cv = _try(N.nvmlSystemGetCudaDriverVersion_v2) or _try(N.nvmlSystemGetCudaDriverVersion)
        if cv:
            self.cuda_version = f"{cv // 1000}.{(cv % 1000) // 10}"
        for i in range(_try(N.nvmlDeviceGetCount, default=0)):
            self.handles.append(N.nvmlDeviceGetHandleByIndex(i))

    def sample(self) -> list[NvDevice]:
        out = []
        for i, h in enumerate(self.handles):
            d = NvDevice(index=i)
            d.name = _try(N.nvmlDeviceGetName, h, default="?")
            pci = _try(N.nvmlDeviceGetPciInfo, h)
            if pci is not None:
                d.pci_bus = pci.busId.decode() if isinstance(pci.busId, bytes) else str(pci.busId)
            dm = _try(N.nvmlDeviceGetDriverModel, h)
            if dm:
                d.driver_model = {0: "WDDM", 1: "TCC", 2: "MCDM"}.get(dm[0], str(dm[0]))
            mem = _try(N.nvmlDeviceGetMemoryInfo, h, N.nvmlMemory_v2) or _try(N.nvmlDeviceGetMemoryInfo, h)
            if mem is not None:
                d.mem_total, d.mem_used = mem.total, mem.used
                d.mem_reserved = getattr(mem, "reserved", 0)
            u = _try(N.nvmlDeviceGetUtilizationRates, h)
            if u is not None:
                d.util_gpu, d.util_mem = u.gpu, u.memory
            enc = _try(N.nvmlDeviceGetEncoderUtilization, h)
            dec = _try(N.nvmlDeviceGetDecoderUtilization, h)
            d.enc = enc[0] if enc else None
            d.dec = dec[0] if dec else None
            d.temp = _try(N.nvmlDeviceGetTemperature, h, N.NVML_TEMPERATURE_GPU)
            p = _try(N.nvmlDeviceGetPowerUsage, h)
            d.power_w = p / 1000 if p is not None else None
            pl = _try(N.nvmlDeviceGetEnforcedPowerLimit, h)
            d.power_limit_w = pl / 1000 if pl is not None else None
            d.fan = _try(N.nvmlDeviceGetFanSpeed, h)
            d.pstate = _try(N.nvmlDeviceGetPerformanceState, h)
            d.sm_clock = _try(N.nvmlDeviceGetClockInfo, h, N.NVML_CLOCK_SM)
            d.sm_clock_max = _try(N.nvmlDeviceGetMaxClockInfo, h, N.NVML_CLOCK_SM)
            d.mem_clock = _try(N.nvmlDeviceGetClockInfo, h, N.NVML_CLOCK_MEM)
            d.video_clock = _try(N.nvmlDeviceGetClockInfo, h, N.NVML_CLOCK_VIDEO)
            d.pcie_rx_kbs = _try(N.nvmlDeviceGetPcieThroughput, h, N.NVML_PCIE_UTIL_RX_BYTES)
            d.pcie_tx_kbs = _try(N.nvmlDeviceGetPcieThroughput, h, N.NVML_PCIE_UTIL_TX_BYTES)
            d.pcie_gen = _try(N.nvmlDeviceGetCurrPcieLinkGeneration, h)
            d.pcie_width = _try(N.nvmlDeviceGetCurrPcieLinkWidth, h)
            reasons = _try(N.nvmlDeviceGetCurrentClocksEventReasons, h)
            if reasons is None:
                reasons = _try(N.nvmlDeviceGetCurrentClocksThrottleReasons, h)
            if reasons:
                d.throttle = [n for bit, n in _THROTTLE.items() if reasons & bit]
            cc = _try(N.nvmlDeviceGetCudaComputeCapability, h)
            if cc:
                d.compute_cap = f"{cc[0]}.{cc[1]}"
            out.append(d)
        return out

    def close(self) -> None:
        if self.ok:
            _try(N.nvmlShutdown)
            self.ok = False

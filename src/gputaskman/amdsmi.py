"""Optional read-only AMD SMI telemetry. WDDM collection works without this module."""
from __future__ import annotations

import importlib
import math
import re

from .dxgi import Adapter, VENDOR_AMD
from .telemetry import DeviceTelemetry


def _number(value):
    return value if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) else None


def _bdf(value):
    match = re.fullmatch(r"([0-9a-fA-F]+):([0-9a-fA-F]+):([0-9a-fA-F]+)\.([0-9a-fA-F]+)", str(value))
    if not match:
        return None
    domain, bus, device, function = (int(part, 16) for part in match.groups())
    # WDDM's ADAPTERADDRESS has no domain. Refuse nonzero domains rather than guess.
    return (bus, device, function) if domain == 0 else None


class AmdSmi:
    def __init__(self, adapter: Adapter | None, adapters: list[Adapter], api=None):
        self.api = None
        self.handle = None
        self.driver = ""
        self.initialized = False
        if not adapter or adapter.vendor_id != VENDOR_AMD or adapter.pci_address is None:
            return
        if sum(a.pci_address == adapter.pci_address for a in adapters if not a.software) != 1:
            return
        try:
            self.api = api or importlib.import_module("amdsmi")
            self.api.amdsmi_init()
            self.initialized = True
            matches = [h for h in self.api.amdsmi_get_processor_handles()
                       if _bdf(self._call("amdsmi_get_gpu_device_bdf", h)) == adapter.pci_address]
            if len(matches) == 1:
                self.handle = matches[0]
                info = self._call("amdsmi_get_gpu_driver_info", self.handle) or {}
                if isinstance(info, dict):
                    self.driver = str(info.get("driver_version") or "")
        except Exception:
            self.close()

    def _call(self, name, *args):
        try:
            return getattr(self.api, name)(*args)
        except Exception:
            return None

    def sample(self) -> DeviceTelemetry | None:
        if self.handle is None:
            return None
        d = DeviceTelemetry(index=0, source="AMD SMI")
        activity = self._call("amdsmi_get_gpu_activity", self.handle)
        if isinstance(activity, dict):
            d.util_gpu = _number(activity.get("gfx_activity"))
        power = self._call("amdsmi_get_power_info", self.handle)
        if isinstance(power, dict):
            d.power_w = _number(power.get("socket_power"))
            d.power_kind = "socket" if d.power_w is not None else ""
            d.power_limit_w = _number(power.get("power_limit"))
        try:
            temp = self._call("amdsmi_get_temp_metric", self.handle,
                              self.api.AmdSmiTemperatureType.EDGE, self.api.AmdSmiTemperatureMetric.CURRENT)
            value = _number(temp)
            d.temp = value  # AMD SMI reports degrees Celsius
        except AttributeError:
            pass
        return d

    def close(self):
        self.handle = None
        if self.initialized:
            self._call("amdsmi_shut_down")
            self.initialized = False

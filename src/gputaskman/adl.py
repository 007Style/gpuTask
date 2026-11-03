"""Read-only AMD Display Library telemetry supplied by the Windows driver.

ADL2_New_QueryPMLogData_Get is deprecated by AMD; failures leave data unavailable.
Definitions: https://github.com/GPUOpen-LibrariesAndSDKs/display-library/tree/master/include
"""
from __future__ import annotations

import ctypes
import os

from .dxgi import Adapter, VENDOR_AMD
from .telemetry import DeviceTelemetry


class _AdapterInfo(ctypes.Structure):
    _fields_ = [("size", ctypes.c_int), ("index", ctypes.c_int), ("udid", ctypes.c_char * 256),
                ("bus", ctypes.c_int), ("device", ctypes.c_int), ("function", ctypes.c_int),
                ("vendor", ctypes.c_int), ("name", ctypes.c_char * 256),
                ("display", ctypes.c_char * 256), ("present", ctypes.c_int), ("exists", ctypes.c_int),
                ("driver", ctypes.c_char * 256), ("driver_ext", ctypes.c_char * 256),
                ("pnp", ctypes.c_char * 256), ("os_index", ctypes.c_int)]


class _Sensor(ctypes.Structure):
    _fields_ = [("supported", ctypes.c_int), ("value", ctypes.c_int)]


class _Metrics(ctypes.Structure):
    _fields_ = [("size", ctypes.c_int), ("sensors", _Sensor * 256)]


class Adl:
    def __init__(self, adapter: Adapter | None, adapters: list[Adapter]):
        self.dll = None
        self.context = ctypes.c_void_p()
        self.index = None
        self.driver = ""
        if not adapter or adapter.vendor_id != VENDOR_AMD or adapter.pci_address is None:
            return
        if sum(a.pci_address == adapter.pci_address for a in adapters if not a.software) != 1:
            return
        try:
            # System32 is the AMD driver's installation location, not the working directory.
            self.dll = ctypes.CDLL(os.path.join(os.environ.get("SystemRoot", r"C:\Windows"), "System32", "atiadlxx.dll"))
            crt = ctypes.CDLL("msvcrt")
            crt.malloc.argtypes = [ctypes.c_size_t]
            crt.malloc.restype = ctypes.c_void_p
            self._malloc = ctypes.CFUNCTYPE(ctypes.c_void_p, ctypes.c_int)(crt.malloc)
            self.dll.ADL2_Main_Control_Create.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.POINTER(ctypes.c_void_p)]
            self.dll.ADL2_Main_Control_Destroy.argtypes = [ctypes.c_void_p]
            self.dll.ADL2_Adapter_NumberOfAdapters_Get.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_int)]
            self.dll.ADL2_Adapter_AdapterInfo_Get.argtypes = [ctypes.c_void_p, ctypes.POINTER(_AdapterInfo), ctypes.c_int]
            self.dll.ADL2_New_QueryPMLogData_Get.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.POINTER(_Metrics)]
            if self.dll.ADL2_Main_Control_Create(self._malloc, 1, ctypes.byref(self.context)) != 0:
                return
            count = ctypes.c_int()
            if self.dll.ADL2_Adapter_NumberOfAdapters_Get(self.context, ctypes.byref(count)) != 0 or not 0 < count.value <= 256:
                return
            entries = (_AdapterInfo * count.value)()
            for entry in entries:
                entry.size = ctypes.sizeof(_AdapterInfo)
            if self.dll.ADL2_Adapter_AdapterInfo_Get(self.context, entries, ctypes.sizeof(entries)) != 0:
                return
            matches = {entry.index for entry in entries if (entry.bus, entry.device, entry.function) == adapter.pci_address}
            # ADL may return several logical display handles for the same physical GPU.
            if matches:
                self.index = min(matches)
        except (OSError, AttributeError, ValueError):
            self.close()
        finally:
            if self.index is None:
                self.close()

    def sample(self) -> DeviceTelemetry | None:
        if self.index is None:
            return None
        metrics = _Metrics()
        metrics.size = ctypes.sizeof(metrics)
        try:
            status = self.dll.ADL2_New_QueryPMLogData_Get(self.context, self.index, ctypes.byref(metrics))
        except (OSError, AttributeError):
            return None
        if status != 0 or not any(item.supported for item in metrics.sensors):
            return None
        def sensor(index):
            item = metrics.sensors[index]
            return item.value if item.supported and item.value >= 0 else None
        board_power = sensor(73)
        asic_power = sensor(23)
        return DeviceTelemetry(index=self.index, source="AMD ADL", gfx_clock=sensor(1), mem_clock=sensor(2),
                               temp=sensor(8), fan=sensor(15), util_gpu=sensor(19),
                               power_w=board_power if board_power is not None else asic_power,
                               power_kind="board" if board_power is not None else "ASIC" if asic_power is not None else "",
                               pcie_width=sensor(41))

    def close(self):
        self.index = None
        if self.dll is not None and self.context.value:
            context = self.context
            self.context = ctypes.c_void_p()
            self.dll.ADL2_Main_Control_Destroy(context)

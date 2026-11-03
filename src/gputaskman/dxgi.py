"""Enumerate GPU adapters via DXGI so performance-counter LUIDs map to real GPU names."""

from __future__ import annotations

import ctypes
from ctypes import wintypes
from dataclasses import dataclass


class _LUID(ctypes.Structure):
    _fields_ = [("LowPart", wintypes.DWORD), ("HighPart", wintypes.LONG)]


class _DESC1(ctypes.Structure):
    _fields_ = [
        ("Description", ctypes.c_wchar * 128),
        ("VendorId", wintypes.UINT),
        ("DeviceId", wintypes.UINT),
        ("SubSysId", wintypes.UINT),
        ("Revision", wintypes.UINT),
        ("DedicatedVideoMemory", ctypes.c_size_t),
        ("DedicatedSystemMemory", ctypes.c_size_t),
        ("SharedSystemMemory", ctypes.c_size_t),
        ("AdapterLuid", _LUID),
        ("Flags", wintypes.UINT),
    ]


class _GUID(ctypes.Structure):
    _fields_ = [("d1", wintypes.DWORD), ("d2", wintypes.WORD), ("d3", wintypes.WORD), ("d4", ctypes.c_ubyte * 8)]


_IID_IDXGIFactory1 = _GUID(0x770AAE78, 0xF26F, 0x4DBA, (ctypes.c_ubyte * 8)(0xA8, 0x29, 0x25, 0x3C, 0x83, 0xD1, 0xB3, 0x87))
VENDOR_NVIDIA = 0x10DE
VENDOR_AMD = 0x1002
VENDORS = {0x10DE: "NVIDIA", 0x1002: "AMD", 0x8086: "Intel", 0x1414: "Microsoft", 0x5143: "Qualcomm"}


@dataclass
class Adapter:
    luid: int
    name: str
    vendor_id: int
    dedicated_vram: int
    shared_system: int
    software: bool
    pci_address: tuple[int, int, int] | None = None

    @property
    def vendor(self) -> str:
        return VENDORS.get(self.vendor_id, hex(self.vendor_id))


def _vcall(obj: ctypes.c_void_p, index: int, restype, *argtypes):
    vtbl = ctypes.cast(obj, ctypes.POINTER(ctypes.POINTER(ctypes.c_void_p)))[0]
    return ctypes.WINFUNCTYPE(restype, ctypes.c_void_p, *argtypes)(vtbl[index])


class _OpenAdapter(ctypes.Structure):
    _fields_ = [("luid", _LUID), ("handle", wintypes.UINT)]


class _AdapterAddress(ctypes.Structure):
    _fields_ = [("bus", wintypes.UINT), ("device", wintypes.UINT), ("function", wintypes.UINT)]


class _QueryAdapter(ctypes.Structure):
    _fields_ = [("handle", wintypes.UINT), ("kind", wintypes.UINT),
                ("data", ctypes.c_void_p), ("size", wintypes.UINT)]


def adapter_pci_address(luid: int) -> tuple[int, int, int] | None:
    """Obtain the WDDM adapter's bus/device/function; the PCI domain is not exposed."""
    try:
        gdi = ctypes.WinDLL("gdi32")
        opened = _OpenAdapter(_LUID(luid & 0xffffffff, ctypes.c_int32(luid >> 32).value), 0)
        if gdi.D3DKMTOpenAdapterFromLuid(ctypes.byref(opened)) != 0:
            return None
        try:
            address = _AdapterAddress()
            query = _QueryAdapter(opened.handle, 6, ctypes.addressof(address), ctypes.sizeof(address))
            if gdi.D3DKMTQueryAdapterInfo(ctypes.byref(query)) != 0:
                return None
            return address.bus, address.device, address.function
        finally:
            gdi.D3DKMTCloseAdapter(ctypes.byref(wintypes.UINT(opened.handle)))
    except (OSError, AttributeError):
        return None


def enumerate_adapters() -> list[Adapter]:
    try:
        dxgi = ctypes.WinDLL("dxgi")
    except OSError:
        return []
    factory = ctypes.c_void_p()
    if dxgi.CreateDXGIFactory1(ctypes.byref(_IID_IDXGIFactory1), ctypes.byref(factory)) != 0:
        return []
    out: list[Adapter] = []
    try:
        enum_adapters1 = _vcall(factory, 12, ctypes.c_long, wintypes.UINT, ctypes.POINTER(ctypes.c_void_p))
        i = 0
        while True:
            adapter = ctypes.c_void_p()
            if enum_adapters1(factory, i, ctypes.byref(adapter)) != 0:
                break  # DXGI_ERROR_NOT_FOUND
            try:
                desc = _DESC1()
                if _vcall(adapter, 10, ctypes.c_long, ctypes.POINTER(_DESC1))(adapter, ctypes.byref(desc)) == 0:
                    luid = ((desc.AdapterLuid.HighPart & 0xffffffff) << 32) | desc.AdapterLuid.LowPart
                    out.append(Adapter(
                        luid=luid,
                        name=desc.Description,
                        vendor_id=desc.VendorId,
                        dedicated_vram=desc.DedicatedVideoMemory,
                        shared_system=desc.SharedSystemMemory,
                        software=bool(desc.Flags & 2),
                        pci_address=adapter_pci_address(luid),
                    ))
            finally:
                _vcall(adapter, 2, wintypes.ULONG)(adapter)  # Release
            i += 1
    finally:
        _vcall(factory, 2, wintypes.ULONG)(factory)
    return out

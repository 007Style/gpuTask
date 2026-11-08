"""Low-frequency Windows discovery in the existing sampling worker, with stable display slots."""

from __future__ import annotations

import time
from dataclasses import dataclass

from .collector import Collector, Snapshot, ordered_adapters, select_adapter
from .dxgi import Adapter, enumerate_adapters


@dataclass
class _Slot:
    adapter: Adapter | None
    collector: Collector | None = None
    retry: bool = False
    had_telemetry: bool = False
    retry_after: float = 0.0
    retry_delay: float = 0.0
    warning: str = ""


def _location(adapter: Adapter) -> tuple | None:
    # LUIDs may change after re-enabling. A unique PCI location plus device description
    # can bridge that change; never match by name or enumeration index alone.
    if adapter.pci_address is not None:
        return adapter.pci_address, adapter.vendor_id, adapter.name
    return None


class LiveCollectors:
    DISCOVERY_INTERVAL = 5.0
    MAX_RETRY_INTERVAL = 60.0

    def __init__(self, gpu_index: int | None = None, *, collector_factory=Collector,
                 discover=enumerate_adapters) -> None:
        self.factory = collector_factory
        self.discover = discover
        self.gpu_index = gpu_index
        self.discovery_error = ""
        try:
            self.adapters = ordered_adapters(discover(strict=True))
        except OSError as error:
            self.adapters = []
            self.discovery_error = f"Windows adapter discovery unavailable ({error}); retaining last known topology."
        selected = ([select_adapter(self.adapters, gpu_index)] if gpu_index is not None and not self.discovery_error
                    else self.adapters or [None])
        self.slots = [_Slot(a) for a in selected]
        now = time.monotonic()
        self.next_discovery = now + self.DISCOVERY_INTERVAL
        self._reconcile(self.adapters, now)

    def _close_slot(self, slot: _Slot) -> None:
        if slot.collector is not None:
            collector = slot.collector
            slot.collector = None
            try:
                collector.close()
            except (OSError, ValueError) as error:
                slot.warning = f"GPU provider cleanup failed: {error}."

    def _schedule_retry(self, slot: _Slot, now: float) -> None:
        slot.retry_delay = min(slot.retry_delay * 2 or self.DISCOVERY_INTERVAL, self.MAX_RETRY_INTERVAL)
        slot.retry_after = now + slot.retry_delay

    def _start_slot(self, slot: _Slot, adapters: list[Adapter], now: float, *, retrying: bool = False) -> None:
        if retrying:
            self._schedule_retry(slot, now)
        try:
            slot.collector = self.factory(adapters.index(slot.adapter), adapters=adapters,
                                          scan_budget=Collector.SCAN_BUDGET / max(len(adapters), 1))
            slot.retry = False
        except (OSError, ValueError) as error:
            slot.warning = " ".join(filter(None, (slot.warning, f"GPU provider initialization failed: {error}.")))
            slot.retry = True
            if not retrying:
                self._schedule_retry(slot, now)

    def _reconcile(self, adapters: list[Adapter], now: float) -> None:
        remaining = list(adapters)
        for slot in self.slots:
            old = slot.adapter
            match = next((a for a in remaining if old and a.luid == old.luid), None)
            if match is None and old is not None:
                location = _location(old)
                candidates = [a for a in remaining if location is not None and _location(a) == location]
                if (len(candidates) == 1
                        and sum(a.pci_address == old.pci_address for a in adapters) == 1
                        and sum(s.adapter.pci_address == old.pci_address
                                for s in self.slots if s.adapter) == 1):
                    match = candidates[0]
            if old is None and remaining:
                index = self.gpu_index or 0
                if index < len(adapters):
                    match = adapters[index]
            if match is None:
                self._close_slot(slot)
                slot.retry = False
                slot.retry_after = slot.retry_delay = 0.0
                continue
            remaining.remove(match)
            changed = old != match
            slot.adapter = match
            if changed:
                self._close_slot(slot)
                slot.retry = False
                slot.retry_after = slot.retry_delay = 0.0
            retrying = slot.retry and now >= slot.retry_after
            if retrying:
                self._close_slot(slot)
            if slot.collector is None and (not slot.retry or retrying):
                self._start_slot(slot, adapters, now, retrying=retrying)
            elif slot.collector is not None:
                slot.collector.adapters = adapters
                slot.collector.nvml.refresh()
        if self.gpu_index is None:
            for adapter in remaining:
                self.slots.append(_Slot(adapter))
            if remaining:
                # Existing slots have already been reconciled; initialize only the new ones.
                for slot in self.slots[-len(remaining):]:
                    self._start_slot(slot, adapters, now)
        active = sum(s.collector is not None for s in self.slots)
        for slot in self.slots:
            if slot.collector:
                slot.collector.scan_budget = Collector.SCAN_BUDGET / max(active, 1)
        self.adapters = adapters

    def sample(self, now: float | None = None) -> list[Snapshot]:
        now = time.monotonic() if now is None else now
        if now >= self.next_discovery:
            self.next_discovery = now + self.DISCOVERY_INTERVAL
            try:
                adapters = ordered_adapters(self.discover(strict=True))
            except OSError as error:
                self.discovery_error = f"Windows adapter discovery unavailable ({error}); retaining last known topology."
            else:
                self.discovery_error = ""
                self._reconcile(adapters, now)
        snapshots = []
        for slot in self.slots:
            snap = None
            if slot.collector is not None:
                try:
                    snap = slot.collector.sample()
                    slot.retry = (snap.counter_adapter_matched is False
                                  or (slot.had_telemetry and snap.nv is None))
                    slot.had_telemetry |= snap.nv is not None
                    if not slot.retry:
                        slot.retry_after = slot.retry_delay = 0.0
                except (OSError, ValueError) as error:
                    slot.retry = True
                    slot.warning = f"GPU sampling failed: {error}."
            if snap is None:
                snap = Snapshot(time.time(), slot.adapter, self.adapters, None, "", "",
                                slot.adapter.dedicated_vram if slot.adapter else 0, None, None,
                                counter_status="Adapter unavailable; waiting for Windows rediscovery.",
                                counter_adapter_matched=False)
            snap.counter_status = " ".join(filter(None, (snap.counter_status, slot.warning, self.discovery_error)))
            slot.warning = ""
            snapshots.append(snap)
        return snapshots

    def close(self) -> None:
        for slot in self.slots:
            self._close_slot(slot)

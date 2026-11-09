"""Hardware-independent Windows disable/re-enable and provider recovery sequences."""

from types import SimpleNamespace
from unittest import IsolatedAsyncioTestCase, TestCase
from unittest.mock import Mock, patch

from gputaskman import nvml
from gputaskman.app import GpuTaskManager
from gputaskman.collector import Collector, ProcRow, Snapshot, select_nv_device
from gputaskman.dxgi import Adapter, enumerate_adapters
from gputaskman.nvml import Nvml
from gputaskman.pdh import ProcGpu
from gputaskman.procinfo import ProcDetail
from gputaskman.recovery import LiveCollectors
from gputaskman.report import to_dict
from gputaskman.telemetry import DeviceTelemetry


def adapter(luid, vendor=0x10DE, bus=1):
    return Adapter(luid, f"device at {bus}", vendor, 4 << 30, 0, False, (bus, 0, 0))


def snapshot(a, value=10):
    return Snapshot(1, a, [a], DeviceTelemetry(index=0, util_gpu=value), "", "",
                    a.dedicated_vram, value << 20, 0,
                    [ProcRow(10, "app", "app", ProcGpu(dedicated=value << 20), ProcDetail(10, "app"))],
                    counter_adapter_matched=True)


class RecoveryTests(TestCase):
    def setUp(self):
        for name, value in (("_unsafe", False), ("LOST", set())):
            patcher = patch(f"gputaskman.nvml.{name}", value)
            patcher.start()
            self.addCleanup(patcher.stop)
        self.first, self.second = adapter(1), adapter(2, 0x1002, 2)
        self.topology = [self.first, self.second]
        self.created = []
        self.discover = Mock(side_effect=lambda **_kwargs: list(self.topology))

        def factory(index, *, adapters, scan_budget):
            a = adapters[index]
            c = Mock(adapter=a, scan_budget=scan_budget)
            c.sample.side_effect = lambda: snapshot(a, a.luid)
            self.created.append(c)
            return c

        self.factory = Mock(side_effect=factory)

    def live(self, index=None):
        live = LiveCollectors(index, collector_factory=self.factory, discover=self.discover)
        self.addCleanup(live.close)
        return live

    def rediscover(self, live):
        return live.sample(now=live.next_discovery)

    def test_removed_readded_and_new_adapter_keep_slots_and_surviving_collector(self):
        live = self.live()
        survivor = self.created[1]
        self.topology = [self.second]
        snaps = self.rediscover(live)
        self.assertIsNone(snaps[0].vram_used)
        self.assertIsNone(snaps[0].attributed)
        self.assertEqual(snaps[0].procs, [])
        self.assertFalse(snaps[0].counter_adapter_matched)
        self.assertEqual(snaps[1].adapter, self.second)
        self.created[0].close.assert_called_once()
        self.assertIs(live.slots[1].collector, survivor)
        # Re-enabling can produce a new LUID; bridge only a unique PCI+description match.
        readded, new = adapter(11), adapter(3, bus=3)
        self.topology = [new, self.second, readded]
        snaps = self.rediscover(live)
        self.assertEqual([s.adapter.luid for s in snaps], [11, 2, 3])
        self.assertIs(live.slots[1].collector, survivor)
        self.assertEqual(len(self.created), 4)
        self.assertAlmostEqual(survivor.scan_budget, Collector.SCAN_BUDGET / 3)

    def test_nvidia_adapter_loss_marks_nvml_unsafe_and_lost_telemetry_does_not_rebuild(self):
        live = self.live()
        self.topology = [self.second]
        self.rediscover(live)
        self.assertTrue(nvml._unsafe)
        readded = adapter(11)
        self.topology = [readded, self.second]
        self.rediscover(live)
        lost = snapshot(readded)
        lost.nv, lost.nv_lost = None, True
        live.slots[0].had_telemetry = True
        self.created[-1].sample.side_effect = lambda: lost
        live.sample(now=live.next_discovery - 1)
        self.assertFalse(live.slots[0].retry)

    def test_amd_adapter_loss_keeps_nvml_shutdown(self):
        live = self.live()
        self.topology = [self.first]
        self.rediscover(live)
        self.assertFalse(nvml._unsafe)

    def test_explicit_gpu_pins_original_identity_not_current_index(self):
        live = self.live(1)
        self.topology = [self.first]
        snaps = self.rediscover(live)
        self.assertEqual(len(snaps), 1)
        self.assertEqual(snaps[0].adapter, self.second)
        self.assertIsNone(snaps[0].vram_used)
        self.topology = [adapter(12, 0x1002, 2), self.first, adapter(3, bus=3)]
        self.assertEqual(self.rediscover(live)[0].adapter.luid, 12)
        self.assertEqual(len(live.slots), 1)

    def test_reordering_does_not_recreate_collectors_or_swap_ownership(self):
        live = self.live()
        original = list(self.created)
        self.topology.reverse()
        snaps = self.rediscover(live)
        self.assertEqual([s.adapter.luid for s in snaps], [1, 2])
        self.assertEqual([s.collector for s in live.slots], original)
        self.assertEqual(len(self.created), 2)

    def test_discovery_failure_keeps_surviving_samples_and_retries_later(self):
        live = self.live()
        self.discover.side_effect = OSError("temporary DXGI failure")
        snaps = self.rediscover(live)
        self.assertEqual([s.vram_used for s in snaps], [1 << 20, 2 << 20])
        self.assertIn("discovery unavailable", snaps[0].counter_status)
        self.discover.side_effect = lambda **_kwargs: list(self.topology)
        self.assertEqual(self.rediscover(live)[0].counter_status, "")
        self.assertEqual(len(self.created), 2)

    def test_sample_failure_is_unknown_and_rebuilds_only_failed_provider(self):
        live = self.live()
        first, second = self.created
        first.sample.side_effect = OSError("lost provider")
        snaps = live.sample(now=live.next_discovery - 1)
        self.assertIsNone(snaps[0].attributed)
        self.assertEqual(snaps[1].vram_used, 2 << 20)
        first.close.side_effect = OSError("cleanup failed")
        snaps = self.rediscover(live)
        self.assertEqual(snaps[0].vram_used, 1 << 20)
        self.assertIn("cleanup failed", snaps[0].counter_status)
        self.assertIs(live.slots[1].collector, second)
        first.close.assert_called_once()

    def test_unmatched_counters_trigger_bounded_rebuild(self):
        live = self.live()
        failed = snapshot(self.first)
        failed.counter_adapter_matched = False
        self.created[0].sample.side_effect = lambda: failed
        live.sample(now=live.next_discovery - 1)
        self.assertEqual(len(self.created), 2)
        self.rediscover(live)
        self.assertEqual(len(self.created), 3)

    def test_persistent_missing_counters_back_off_and_recover(self):
        live = self.live()
        failed = snapshot(self.first)
        failed.counter_adapter_matched = False
        self.created[0].sample.side_effect = lambda: failed
        original_factory = self.factory.side_effect

        def failing_factory(*args, **kwargs):
            collector = original_factory(*args, **kwargs)
            if collector.adapter.luid == self.first.luid:
                collector.sample.side_effect = lambda: failed
            return collector

        self.factory.side_effect = failing_factory
        live.sample(now=live.next_discovery - 1)
        for _ in range(12):
            self.rediscover(live)
        self.assertEqual(sum(c.adapter.luid == self.first.luid for c in self.created), 5)
        self.assertEqual(sum(c.adapter.luid == self.second.luid for c in self.created), 1)

        recovered = self.created[-1]
        recovered.sample.side_effect = lambda: snapshot(self.first)
        self.assertTrue(live.sample(now=live.next_discovery - 1)[0].counter_adapter_matched)
        recovered.sample.side_effect = lambda: failed
        live.sample(now=live.next_discovery - 1)
        self.rediscover(live)
        self.assertEqual(sum(c.adapter.luid == self.first.luid for c in self.created), 6)

    def test_returning_adapter_retries_immediately_during_backoff(self):
        live = self.live()
        failed = snapshot(self.first)
        failed.counter_adapter_matched = False
        self.created[0].sample.side_effect = lambda: failed
        live.sample(now=live.next_discovery - 1)
        self.rediscover(live)
        self.created[-1].sample.side_effect = lambda: failed
        live.sample(now=live.next_discovery - 1)
        self.rediscover(live)
        self.created[-1].sample.side_effect = lambda: failed
        live.sample(now=live.next_discovery - 1)

        self.topology = [self.second]
        self.assertIsNone(self.rediscover(live)[0].vram_used)
        self.topology = [self.first, self.second]
        self.assertEqual(self.rediscover(live)[0].vram_used, 1 << 20)

    def test_discovery_runs_only_every_five_seconds_without_catchup_burst(self):
        live = self.live()
        for _ in range(20):
            live.sample(now=live.next_discovery - 1)
        self.discover.assert_called_once_with(strict=True)
        self.rediscover(live)
        self.assertEqual(self.discover.call_count, 2)
        live.sample(now=live.next_discovery + 100)
        self.assertEqual(self.discover.call_count, 3)

    def test_no_hardware_at_startup_can_discover_first_device(self):
        self.topology = []
        live = self.live(0)
        self.assertIsNone(live.sample()[0].vram_used)
        self.topology = [self.first]
        self.assertEqual(self.rediscover(live)[0].adapter, self.first)

    def test_initial_discovery_error_recovers_without_binding_wrong_explicit_index(self):
        self.discover.side_effect = OSError("temporary DXGI failure")
        live = self.live(1)
        self.assertIsNone(live.sample()[0].adapter)
        self.discover.side_effect = lambda **_kwargs: list(self.topology)
        self.assertEqual(self.rediscover(live)[0].adapter, self.second)

    def test_out_of_range_explicit_gpu_still_rejected(self):
        with self.assertRaisesRegex(ValueError, "out of range"):
            self.live(2)
        self.factory.assert_not_called()

    def test_ambiguous_pci_match_never_reassigns_removed_device(self):
        live = self.live(0)
        self.topology = [adapter(11), adapter(12)]
        snap = self.rediscover(live)[0]
        self.assertEqual(snap.adapter, self.first)
        self.assertIsNone(snap.vram_used)

    def test_duplicate_pci_address_is_ambiguous_even_with_different_vendors(self):
        live = self.live(0)
        self.topology = [adapter(11), adapter(12, 0x1002)]
        snap = self.rediscover(live)[0]
        self.assertEqual(snap.adapter, self.first)
        self.assertIsNone(snap.vram_used)

    def test_missing_pci_does_not_match_by_name(self):
        self.first.pci_address = None
        live = self.live(0)
        replacement = adapter(11)
        replacement.pci_address = None
        self.topology = [replacement]
        self.assertIsNone(self.rediscover(live)[0].vram_used)

    def test_lost_telemetry_rebuilds_provider_without_recreating_survivor(self):
        live = self.live()
        live.sample()
        failed = snapshot(self.first)
        failed.nv = None
        self.created[0].sample.side_effect = lambda: failed
        live.sample()
        self.rediscover(live)
        self.assertEqual(len(self.created), 3)
        self.assertIs(live.slots[1].collector, self.created[1])

    def test_collector_creation_error_retries_and_keeps_survivor(self):
        survivor = Mock()
        survivor.sample.return_value = snapshot(self.second)
        self.factory.side_effect = [OSError("provider unavailable"), survivor]
        live = self.live()
        self.assertIs(live.slots[1].collector, survivor)
        replacement = Mock()
        replacement.sample.return_value = snapshot(self.first)
        self.factory.side_effect = [replacement]
        self.assertEqual(self.rediscover(live)[0].vram_used, 10 << 20)
        self.assertIs(live.slots[1].collector, survivor)

    def test_close_failure_does_not_skip_other_collectors(self):
        live = self.live()
        self.created[0].close.side_effect = OSError("device gone")
        live.close()
        for collector in self.created:
            collector.close.assert_called_once()
        live.close()
        for collector in self.created:
            collector.close.assert_called_once()

    def test_software_adapter_is_not_added_as_a_gpu(self):
        live = self.live()
        software = adapter(30)
        software.software = True
        self.topology.append(software)
        self.assertEqual(len(self.rediscover(live)), 2)


class NvmlRecoveryTests(TestCase):
    def setUp(self):
        for name, value in (("_unsafe", False), ("LOST", set())):
            patcher = patch(f"gputaskman.nvml.{name}", value)
            patcher.start()
            self.addCleanup(patcher.stop)

    def api(self):
        return SimpleNamespace(
            nvmlInit=Mock(), nvmlShutdown=Mock(), nvmlDeviceGetCount=Mock(return_value=2),
            nvmlDeviceGetHandleByIndex=Mock(side_effect=["first", "second"]),
            nvmlSystemGetDriverVersion=Mock(return_value="test"),
            nvmlSystemGetCudaDriverVersion_v2=Mock(return_value=12000),
            nvmlSystemGetCudaDriverVersion=Mock(return_value=12000),
        )

    def test_rebind_handles_without_reinitializing_healthy_session(self):
        api = self.api()
        with patch("gputaskman.nvml.N", api):
            nv = Nvml()
            api.nvmlDeviceGetHandleByIndex.side_effect = ["new-first", "new-second"]
            nv.refresh()
            self.assertEqual(nv.handles, [(0, "new-first"), (1, "new-second")])
            api.nvmlInit.assert_called_once()
            api.nvmlShutdown.assert_not_called()
            nv.close()

    def test_one_failed_handle_does_not_hide_other_devices(self):
        api = self.api()
        api.nvmlDeviceGetHandleByIndex.side_effect = [OSError("GPU lost"), "second"]
        with patch("gputaskman.nvml.N", api):
            nv = Nvml()
            self.assertEqual(nv.handles, [(1, "second")])
            nv.close()

    def test_all_lost_handles_request_reset_without_resetting_surviving_peers(self):
        class LostGpu(Exception):
            value = 15

        api = self.api()
        api.nvmlDeviceGetHandleByIndex.side_effect = [LostGpu(), "second"]
        with patch("gputaskman.nvml.N", api):
            nv = Nvml()
            self.assertFalse(nv.needs_reset)
            api.nvmlDeviceGetHandleByIndex.side_effect = [LostGpu(), LostGpu()]
            nv.refresh()
            self.assertTrue(nv.needs_reset)
            api.nvmlDeviceGetHandleByIndex.side_effect = ["first", "second"]
            nv.refresh()
            self.assertEqual(nv.handles, [(0, "first"), (1, "second")])
            self.assertEqual(api.nvmlInit.call_count, 2)
            api.nvmlShutdown.assert_called_once()
            nv.close()

    def test_count_error_clears_stale_handles_and_next_refresh_recovers(self):
        api = self.api()
        with patch("gputaskman.nvml.N", api):
            nv = Nvml()
            api.nvmlDeviceGetCount.side_effect = OSError("driver lost")
            nv.refresh()
            self.assertFalse(nv.ok)
            self.assertEqual(nv.handles, [])
            api.nvmlDeviceGetCount.side_effect = None
            api.nvmlDeviceGetHandleByIndex.side_effect = ["new-first", "new-second"]
            nv.refresh()
            self.assertTrue(nv.ok)
            self.assertEqual(nv.handles, [(0, "new-first"), (1, "new-second")])
            nv.close()

    def test_failed_initialization_can_retry(self):
        api = self.api()
        api.nvmlInit.side_effect = OSError("driver unavailable")
        with patch("gputaskman.nvml.N", api):
            nv = Nvml()
            self.assertFalse(nv.ok)
            api.nvmlInit.side_effect = None
            nv.refresh()
            self.assertTrue(nv.ok)
            nv.close()

    def test_failed_first_handle_preserves_survivor_index_json_and_fallback_selection(self):
        class Api(SimpleNamespace):
            def __getattr__(self, name):
                return 0 if name.startswith("NVML_") else lambda *args: None

        api = Api(**vars(self.api()))
        api.nvmlDeviceGetHandleByIndex.side_effect = [OSError("GPU lost"), "second"]
        api.nvmlDeviceGetName = Mock(return_value="surviving GPU")
        api.nvmlDeviceGetPciInfo = Mock(return_value=SimpleNamespace(busId="00000000:02:00.0"))
        with patch("gputaskman.nvml.N", api):
            nv = Nvml()
            try:
                devices = nv.sample()
                self.assertEqual(len(devices), 1)
                survivor = devices[0]
                self.assertEqual(survivor.index, 1)
                snap = snapshot(adapter(2))
                snap.adapter = None
                snap.nv = survivor
                self.assertEqual(to_dict(snap)["device"]["index"], 1)
                self.assertIsNone(select_nv_device(None, devices, 0))
                self.assertIs(select_nv_device(None, devices, 1), survivor)
                self.assertIsNone(select_nv_device(None, devices, 2))
                api.nvmlDeviceGetHandleByIndex.side_effect = ["new-first", "new-second"]
                nv.refresh()
                recovered = nv.sample()
                self.assertEqual([d.index for d in recovered], [0, 1])
                self.assertIs(select_nv_device(None, recovered, 0), recovered[0])
                self.assertIs(select_nv_device(None, recovered, 1), recovered[1])
            finally:
                nv.close()

    def test_adapter_none_fallback_uses_index_even_when_devices_are_reordered(self):
        first = DeviceTelemetry(index=0)
        second = DeviceTelemetry(index=1)
        self.assertIs(select_nv_device(None, [second, first], 0), first)
        self.assertIs(select_nv_device(None, [second, first], 1), second)
        self.assertIsNone(select_nv_device(None, [first, second], -1))
        self.assertIsNone(select_nv_device(None, [], 0))

    def test_lost_pci_identity_requests_session_reset_on_refresh(self):
        api = self.api()
        with patch("gputaskman.nvml.N", api):
            nv = Nvml()
            nv.needs_reset = True
            api.nvmlDeviceGetHandleByIndex.side_effect = ["new-first", "new-second"]
            nv.refresh()
            self.assertEqual(api.nvmlInit.call_count, 2)
            api.nvmlShutdown.assert_called_once()
            nv.close()

    def test_unsupported_pci_query_does_not_restart_healthy_nvml(self):
        class Api(SimpleNamespace):
            def __getattr__(self, name):
                return 0 if name.startswith("NVML_") else lambda *args: None

        class Unsupported(Exception):
            value = 3

        api = Api(**vars(self.api()))
        api.nvmlDeviceGetPciInfo = Mock(side_effect=Unsupported())
        with patch("gputaskman.nvml.N", api):
            nv = Nvml()
            nv.sample()
            self.assertFalse(nv.needs_reset)
            api.nvmlDeviceGetHandleByIndex.side_effect = ["first", "second"]
            nv.refresh()
            api.nvmlInit.assert_called_once()
            api.nvmlShutdown.assert_not_called()
            nv.close()

    def test_gpu_lost_metric_error_resets_even_when_pci_identity_still_responds(self):
        class Api(SimpleNamespace):
            def __getattr__(self, name):
                return 0 if name.startswith("NVML_") else lambda *args: None

        class Error(Exception):
            value = 15

        api = Api(**vars(self.api()))
        api.nvmlDeviceGetPciInfo = Mock(return_value=SimpleNamespace(busId="00000000:01:00.0"))
        api.nvmlDeviceGetUtilizationRates = Mock(side_effect=Error("GPU lost"))
        with patch("gputaskman.nvml.N", api):
            nv = Nvml()
            self.assertIsNone(nv.sample()[0].util_gpu)
            self.assertTrue(nv.needs_reset)
            api.nvmlDeviceGetHandleByIndex.side_effect = ["new-first", "new-second"]
            nv.refresh()
            api.nvmlShutdown.assert_called_once()
            nv.close()

    def test_unsupported_optional_metric_does_not_request_reset(self):
        class Error(Exception):
            value = 3

        api = self.api()
        with patch("gputaskman.nvml.N", api):
            nv = Nvml()
            self.assertIsNone(nv._read(Mock(side_effect=Error("not supported"))))
            self.assertFalse(nv.needs_reset)
            nv.close()

    def test_access_violation_stops_reading_that_gpu_and_skips_hanging_shutdown(self):
        class Api(SimpleNamespace):
            def __getattr__(self, name):
                return 0 if name.startswith("NVML_") else lambda *args: None

        def temperature(handle, _kind):
            if handle == "second":
                raise OSError("exception: access violation reading 0x20C25D65908")
            return 40

        api = Api(**vars(self.api()))
        api.nvmlDeviceGetPciInfo = Mock(side_effect=lambda h: SimpleNamespace(
            busId=b"00000000:01:00.0" if h == "first" else b"00000000:05:00.0"))
        api.nvmlDeviceGetTemperature = Mock(side_effect=temperature)
        with patch("gputaskman.nvml.N", api):
            nv = Nvml()
            self.assertEqual([(d.pci_bus, d.temp) for d in nv.sample()], [("00000000:01:00.0", 40)])
            self.assertEqual(nvml.LOST, {"00000000:05:00.0"})
            calls = api.nvmlDeviceGetTemperature.call_count
            self.assertEqual(len(nv.sample()), 1)
            self.assertEqual(api.nvmlDeviceGetTemperature.call_count, calls + 1)  # the lost GPU is skipped
            nv.close()
            api.nvmlShutdown.assert_not_called()

    def test_healthy_session_still_shuts_down(self):
        api = self.api()
        with patch("gputaskman.nvml.N", api):
            Nvml().close()
            api.nvmlShutdown.assert_called_once()


class DiscoveryTests(TestCase):
    def test_discovery_error_is_not_confused_with_empty_topology(self):
        with patch("gputaskman.dxgi.ctypes.WinDLL", side_effect=OSError("DXGI unavailable"), create=True):
            self.assertEqual(enumerate_adapters(), [])
            with self.assertRaises(OSError):
                enumerate_adapters(strict=True)

    def test_factory_error_is_reported_to_live_discovery(self):
        dll = SimpleNamespace(CreateDXGIFactory1=Mock(return_value=-1))
        with patch("gputaskman.dxgi.ctypes.WinDLL", return_value=dll, create=True):
            with self.assertRaisesRegex(OSError, "discovery failed"):
                enumerate_adapters(strict=True)

    def test_enumeration_error_is_not_reported_as_successful_empty_list(self):
        dll = SimpleNamespace(CreateDXGIFactory1=Mock(return_value=0))
        with patch("gputaskman.dxgi.ctypes.WinDLL", return_value=dll, create=True), \
                patch("gputaskman.dxgi._vcall", side_effect=[Mock(return_value=-1), Mock()]):
            with self.assertRaisesRegex(OSError, "enumeration failed"):
                enumerate_adapters(strict=True)


class CollectorResourceTests(TestCase):
    def test_partial_provider_initialization_closes_previously_created_resources(self):
        with patch("gputaskman.collector.GpuCounters") as counters, \
                patch("gputaskman.collector.Nvml") as nvml, \
                patch("gputaskman.collector.AmdSmi") as amd, \
                patch("gputaskman.collector.Adl", side_effect=OSError("device unavailable")):
            with self.assertRaises(OSError):
                Collector(adapters=[adapter(1)])
            counters.return_value.close.assert_called_once()
            nvml.return_value.close.assert_called_once()
            amd.return_value.close.assert_called_once()

    def test_close_failure_still_releases_all_providers(self):
        c = Collector.__new__(Collector)
        c.host = None
        c.counters, c.nvml, c.amdsmi, c.adl = Mock(), Mock(), Mock(), Mock()
        c.counters.close.side_effect = OSError("query lost")
        with self.assertRaises(OSError):
            c.close()
        for provider in (c.counters, c.nvml, c.amdsmi, c.adl):
            provider.close.assert_called_once()


class RecoveryUiTests(IsolatedAsyncioTestCase):
    def test_queued_row_highlight_is_ignored_after_shutdown(self):
        app = GpuTaskManager()
        event = SimpleNamespace(row_key=SimpleNamespace(value="10"))
        app.on_data_table_row_highlighted(event)
        self.assertIsNone(app.selected_key)

    async def test_histories_and_row_ownership_stay_in_stable_slots_across_recovery(self):
        first, second = adapter(1), adapter(2, 0x1002, 2)
        topology = [first, second]

        def factory(index, *, adapters, **_kwargs):
            a = adapters[index]
            c = Mock()
            c.sample.side_effect = lambda: snapshot(a, a.luid)
            return c

        live = LiveCollectors(collector_factory=factory, discover=lambda **_kwargs: list(topology))
        try:
            with patch.object(GpuTaskManager, "poll"):
                app = GpuTaskManager()
                app.show_all = True
                async with app.run_test() as pilot:
                    app.apply_all(live.sample())
                    await pilot.pause()
                    app.selected_key = "1:10"
                    topology[:] = [second]
                    app.apply_all(live.sample(now=live.next_discovery))
                    await pilot.pause()
                    topology[:] = [second, adapter(11)]
                    app.apply_all(live.sample(now=live.next_discovery))
                    await pilot.pause()
                    self.assertEqual(list(app._histories[0]["util"]), [1, None, 11])
                    self.assertEqual(list(app._histories[1]["util"]), [2, 2, 2])
                    self.assertEqual(app.selected_key, "1:10")
                    self.assertEqual(app._row_snapshots["1:10"].adapter, second)
                    self.assertEqual(app._row_snapshots["0:10"].adapter.luid, 11)
        finally:
            live.close()

    async def test_new_second_gpu_preserves_selected_process_on_original_gpu(self):
        first, second = adapter(1), adapter(2, 0x1002, 2)
        with patch.object(GpuTaskManager, "poll"):
            app = GpuTaskManager()
            app.show_all = True
            async with app.run_test() as pilot:
                app.apply(snapshot(first))
                await pilot.pause()
                app.selected_key = "10"
                app.apply_all([snapshot(first), snapshot(second, 80)])
                await pilot.pause()
                self.assertEqual(app.selected_key, "0:10")
                self.assertEqual(app._row_snapshots[app.selected_key].adapter, first)

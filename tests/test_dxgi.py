import contextlib
import ctypes
import io
import sys
import unittest
from unittest.mock import Mock, patch

from rich.console import Console

from gputaskman import dxgi
from gputaskman.app import GpuTaskManager
from gputaskman.cli import main
from gputaskman.collector import Collector, ordered_adapters, select_adapter
from gputaskman.pdh import AdapterGpu, ProcGpu
from gputaskman.report import to_dict, to_text


ROOT = "ROOT\\DISPLAY\\0001"
PATH = r"\\?\ROOT#DISPLAY#0001#{5b45201d-f2f2-4f3b-85bb-30ff1f953599}"
STARDESK = "StarDesk Virtual Display Adapter"


class IdentityApiTests(unittest.TestCase):
    def setUp(self):
        self.user, self.setup = Mock(), Mock()
        self.user.DisplayConfigGetDeviceInfo.side_effect = self.get_name
        self.setup.SetupDiCreateDeviceInfoList.return_value = 123
        self.setup.SetupDiOpenDeviceInterfaceW.return_value = True
        self.setup.SetupDiGetDeviceInterfaceDetailW.return_value = True
        self.setup.SetupDiGetDeviceInstanceIdW.side_effect = self.get_instance
        self.setup.SetupDiGetDeviceRegistryPropertyW.side_effect = self.get_property
        self.dll = patch.object(dxgi.ctypes, "WinDLL", side_effect=self.load, create=True)
        self.dll.start()
        self.addCleanup(self.dll.stop)
        self.last_error = patch.object(dxgi.ctypes, "get_last_error", return_value=5, create=True)
        self.last_error.start()
        self.addCleanup(self.last_error.stop)

    def load(self, name, **kwargs):
        return {"user32": self.user, "setupapi": self.setup}[name]

    def get_name(self, header):
        request = ctypes.cast(header, ctypes.POINTER(dxgi._AdapterName)).contents
        self.assertEqual(request.header.kind, 4)
        self.assertEqual(request.header.size, ctypes.sizeof(dxgi._AdapterName))
        self.assertEqual(request.header.id, 0)
        self.requested_luid = ((request.header.luid.HighPart & 0xffffffff) << 32
                               | request.header.luid.LowPart)
        request.path = PATH
        return 0

    def get_instance(self, handle, info, buffer, length, required):
        buffer.value = ROOT
        return True

    def get_property(self, handle, info, prop, kind, buffer, length, required):
        self.assertEqual(prop, 12)
        buffer.value = STARDESK
        kind._obj.value = 1
        required._obj.value = (len(STARDESK) + 1) * ctypes.sizeof(ctypes.c_wchar)
        return True

    def test_exact_luid_lookup_needs_no_active_display_path(self):
        self.assertEqual(dxgi.adapter_identity(0x20452), (ROOT, STARDESK))
        self.assertEqual(self.requested_luid, 0x20452)
        self.setup.SetupDiOpenDeviceInterfaceW.assert_called_once()
        self.assertEqual(self.setup.SetupDiOpenDeviceInterfaceW.call_args.args[1], PATH)
        self.setup.SetupDiDestroyDeviceInfoList.assert_called_once_with(123)
        self.assertEqual(self.user.method_calls[0][0], "DisplayConfigGetDeviceInfo")

    def test_signed_high_luid_preserves_all_bits(self):
        dxgi.adapter_identity(0xfedcba9800020452)
        self.assertEqual(self.requested_luid, 0xfedcba9800020452)

    def test_friendly_name_falls_back_to_device_description(self):
        original = self.get_property

        def property_value(*args):
            if args[2] == 12:
                return False
            return original(*args[:2], 12, *args[3:])

        self.setup.SetupDiGetDeviceRegistryPropertyW.side_effect = property_value
        self.assertEqual(dxgi.adapter_identity(0x20452), (ROOT, STARDESK))
        self.assertEqual([c.args[2] for c in self.setup.SetupDiGetDeviceRegistryPropertyW.call_args_list], [12, 0])

    def test_missing_or_invalid_names_keep_verified_instance(self):
        for kind, size, text in ((7, 100, STARDESK), (1, 100000, STARDESK), (1, 0, STARDESK), (1, 4, " ")):
            with self.subTest(kind=kind, size=size, text=text):
                def invalid(*args):
                    args[3]._obj.value = kind
                    args[4].value = text
                    args[6]._obj.value = size
                    return True

                self.setup.SetupDiGetDeviceRegistryPropertyW.side_effect = invalid
                self.assertEqual(dxgi.adapter_identity(0x20452), (ROOT, ""))

    def test_api_failures_are_bounded_and_close_device_list(self):
        for function in ("SetupDiOpenDeviceInterfaceW", "SetupDiGetDeviceInterfaceDetailW",
                         "SetupDiGetDeviceInstanceIdW"):
            with self.subTest(function=function):
                api = getattr(self.setup, function)
                previous_effect, previous_value = api.side_effect, api.return_value
                api.side_effect, api.return_value = None, False
                self.setup.SetupDiDestroyDeviceInfoList.reset_mock()
                with self.assertRaisesRegex(OSError, "error 5"):
                    dxgi.adapter_identity(0x20452)
                self.setup.SetupDiDestroyDeviceInfoList.assert_called_once_with(123)
                api.side_effect, api.return_value = previous_effect, previous_value

    def test_displayconfig_failure_does_not_open_pnp_list(self):
        self.user.DisplayConfigGetDeviceInfo.side_effect = None
        self.user.DisplayConfigGetDeviceInfo.return_value = 87
        with self.assertRaisesRegex(OSError, "error 87"):
            dxgi.adapter_identity(0x20452)
        self.setup.SetupDiCreateDeviceInfoList.assert_not_called()

    def test_invalid_device_list_is_not_used_or_closed(self):
        self.setup.SetupDiCreateDeviceInfoList.return_value = ctypes.c_void_p(-1).value
        with self.assertRaisesRegex(OSError, "device list unavailable"):
            dxgi.adapter_identity(0x20452)
        self.setup.SetupDiOpenDeviceInterfaceW.assert_not_called()
        self.setup.SetupDiDestroyDeviceInfoList.assert_not_called()

    def test_empty_instance_is_not_accepted(self):
        self.setup.SetupDiGetDeviceInstanceIdW.side_effect = None
        with self.assertRaisesRegex(OSError, "instance is empty"):
            dxgi.adapter_identity(0x20452)

    @unittest.skipUnless(sys.platform == "win32", "Windows ABI")
    def test_windows_structure_sizes(self):
        self.assertEqual(ctypes.sizeof(dxgi._DeviceInfoHeader), 20)
        self.assertEqual(ctypes.sizeof(dxgi._AdapterName), 276)
        size = 32 if ctypes.sizeof(ctypes.c_void_p) == 8 else 28
        self.assertEqual(ctypes.sizeof(dxgi._DeviceInterface), size)
        self.assertEqual(ctypes.sizeof(dxgi._DeviceInfo), size)


class AdapterIdentityTests(unittest.TestCase):
    def adapter(self, luid=0x20452, vendor=0x8086):
        return dxgi.Adapter(luid, "Intel(R) UHD Graphics 630", vendor, 128 << 20, 8 << 30, False)

    def identify(self, adapter, instance=ROOT, name=STARDESK):
        with patch.object(dxgi, "adapter_identity", return_value=(instance, name)):
            dxgi._identify_adapter(adapter)
        return adapter

    def test_root_name_comes_from_exact_pnp_identity_not_dxgi_vendor(self):
        for vendor in (0x8086, 0x10de, 0x1002, 0):
            with self.subTest(vendor=vendor):
                a = self.identify(self.adapter(vendor=vendor))
                self.assertEqual(a.name, STARDESK)
                self.assertEqual(a.vendor_id, vendor)
                self.assertEqual(a.vendor, "ROOT display adapter")
                self.assertEqual(a.device_instance_id, ROOT)

    def test_physical_name_and_vendor_are_unchanged(self):
        a = self.identify(self.adapter(0x1295f), "PCI\\VEN_8086&DEV_3E9B\\123", "Generic PnP description")
        self.assertEqual(a.name, "Intel(R) UHD Graphics 630")
        self.assertEqual(a.vendor, "Intel")
        self.assertFalse(a.root_display)

    def test_root_instance_not_duplicate_name_proves_root_display(self):
        for instance, root in (("root\\display\\0001", True), ("ROOT\\DISPLAYOTHER\\0001", False),
                               ("PCI\\VEN_8086\\0001", False), ("", False)):
            with self.subTest(instance=instance):
                a = self.identify(self.adapter(), instance)
                self.assertEqual(a.root_display, root)
                self.assertEqual(a.name, STARDESK if root else "Intel(R) UHD Graphics 630")

    def test_missing_pnp_name_has_explicit_root_label_and_reason(self):
        a = self.identify(self.adapter(), name="")
        self.assertEqual(a.name, f"ROOT display adapter ({ROOT})")
        self.assertIn("name unavailable", a.identity_status)
        self.assertEqual(a.vendor, "ROOT display adapter")

    def test_probe_failure_preserves_dxgi_and_explains_uncertainty(self):
        for error in (OSError("access denied"), AttributeError("API missing")):
            with self.subTest(error=error), patch.object(dxgi, "adapter_identity", side_effect=error):
                a = self.adapter()
                dxgi._identify_adapter(a)
                self.assertEqual(a.name, "Intel(R) UHD Graphics 630")
                self.assertEqual(a.device_instance_id, "")
                self.assertIn(str(error), a.identity_status)
                self.assertIn("DXGI-reported", a.identity_status)

    def test_list_explains_failed_identity_lookup(self):
        a = self.adapter()
        a.identity_status = "Windows PnP identity unavailable; name and vendor are DXGI-reported."
        output = io.StringIO()
        with patch("sys.argv", ["gputaskman", "--list-gpus"]), patch("gputaskman.cli._check_platform"), \
                patch("gputaskman.collector.enumerate_adapters", return_value=[a]), \
                contextlib.redirect_stdout(output):
            main()
        self.assertIn(a.name, output.getvalue())
        self.assertIn(a.identity_status, output.getvalue())

    def test_enumeration_keeps_slots_and_does_not_query_root_pci_location(self):
        luids = [0x142f1, 0x1295f, 0x20452]

        def enum(factory, index, out):
            if index == len(luids):
                return 0x887a0002
            out._obj.value = index + 10
            return 0

        def description(adapter, desc):
            index = adapter.value - 10
            desc._obj.Description = "NVIDIA Quadro P620" if index == 0 else "Intel(R) UHD Graphics 630"
            desc._obj.AdapterLuid.LowPart = luids[index]
            desc._obj.VendorId = 0x10de if index == 0 else 0x8086
            desc._obj.DedicatedVideoMemory = 128 << 20
            return 0

        def vcall(obj, index, *types):
            return {12: enum, 10: description, 2: lambda obj: None}[index]

        def identity(luid):
            return (ROOT, STARDESK) if luid == 0x20452 else ("PCI\\physical", "PnP name")

        api = Mock()
        api.CreateDXGIFactory1.return_value = 0
        with patch.object(dxgi.ctypes, "WinDLL", return_value=api, create=True), \
                patch.object(dxgi, "_vcall", side_effect=vcall), \
                patch.object(dxgi, "adapter_identity", side_effect=identity), \
                patch.object(dxgi, "adapter_pci_address", return_value=(1, 0, 0)) as pci:
            adapters = ordered_adapters(dxgi.enumerate_adapters(strict=True))
        self.assertEqual([a.luid for a in adapters], luids)
        self.assertEqual([a.name for a in adapters], ["NVIDIA Quadro P620", "Intel(R) UHD Graphics 630", STARDESK])
        self.assertIsNone(select_adapter(adapters, 2).pci_address)
        self.assertEqual([call.args[0] for call in pci.call_args_list], luids[:2])
        output = io.StringIO()
        with patch("sys.argv", ["gputaskman", "--list-gpus"]), patch("gputaskman.cli._check_platform"), \
                patch("gputaskman.collector.enumerate_adapters", return_value=adapters), \
                contextlib.redirect_stdout(output):
            main()
        self.assertIn(f"2: {STARDESK} (ROOT display adapter, LUID 0x0000000000020452)", output.getvalue())

    def test_unmatched_report_stays_unknown_in_json_text_and_tui(self):
        c = Collector.__new__(Collector)
        c.adapter = self.identify(self.adapter())
        c.adapters = [c.adapter]
        c.gpu_index, c.scan_budget, c.host, c.host_status = 0, None, None, ""
        c.nvml = Mock(ok=False)
        c.counters = Mock(outside={})
        c.counters.sample.return_value = (
            {0x1295f: {4: ProcGpu(dedicated=1024, engines={"compute": 10})}},
            {0x1295f: AdapterGpu(dedicated=1024, shared=0)})
        c.inspector, c.prober = Mock(), Mock()
        for reason in ("", "Windows PnP name unavailable"):
            with self.subTest(reason=reason), patch("gputaskman.collector.BACKEND", "wddm"):
                c.adapter.identity_status = reason
                s = c.sample()
                data = to_dict(s)
                self.assertEqual(data["gpu"], STARDESK)
                self.assertEqual(data["adapter"]["device_instance_id"], ROOT)
                for key in ("vram_used", "shared_used", "attributed"):
                    self.assertIsNone(data[key])
                self.assertFalse(data["counter_adapter_matched"])
                self.assertEqual(data["processes"], [])
                self.assertIn("could not be matched", data["counter_status"])
                self.assertIn(reason, data["counter_status"])
                self.assertIn(STARDESK, to_text(s))
                output = io.StringIO()
                Console(file=output, width=200).print(GpuTaskManager().render_device(s))
                self.assertIn(STARDESK, output.getvalue())
                self.assertIn(reason, output.getvalue())

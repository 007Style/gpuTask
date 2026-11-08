from __future__ import annotations

import contextlib
import asyncio
import ctypes
import io
import json
import os
from types import SimpleNamespace
import tempfile
import threading
import unittest
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import Mock, patch

from rich.console import Console
from rich.table import Table
from textual.widgets import Footer, Select, Static

from gputaskman.amdsmi import AmdSmi
from gputaskman.adl import Adl, _Metrics
from gputaskman.app import GpuTaskManager, spark, stacked_bar
from gputaskman.cli import main
from gputaskman.collector import Collector, ProcRow, Snapshot, select_adapter, select_nv_device
from gputaskman.dxgi import Adapter
from gputaskman.gguf import GgufInfo, read_gguf
from gputaskman.handy import HandyEvidence, inspect_handy, _log_cache, _settings_cache
from gputaskman.llm import (LlmView, ServerInfo, build_view, is_gpu_device, llama_server_models, lmstudio_models,
                            parse_flags, parse_load_log, probe_server)
from gputaskman.nvml import NvDevice, Nvml
from gputaskman.pdh import AdapterGpu, GpuCounters, ProcGpu
from gputaskman.procinfo import (AI_KINDS, MappedModel, ProcDetail, ProcInspector, _is_model_file, _match_runtimes,
                                _models_from_cmdline, classify, installed_models, make_model)
from gputaskman.report import to_dict, to_text
from gputaskman.stores import store_name
from tools.capture_screenshots import for_publication


def native(path: str) -> str:
    """Windows-style test paths as native paths, so path logic is exercised the same way on Linux."""
    return path if os.sep == "\\" else "/" + path.split(":", 1)[-1].lstrip("\\").replace("\\", "/")

GiB = 1 << 30
MiB = 1 << 20


def snapshot() -> Snapshot:
    return Snapshot(
        taken=1, adapter=None, adapters=[], nv=None, driver="", cuda="",
        vram_total=16 * GiB, vram_used=10 * GiB, shared_used=GiB,
        procs=[
            ProcRow(10, "llama-server.exe", "LLM",
                    ProcGpu(dedicated=8 * GiB, shared=GiB, committed=9 * GiB),
                    ProcDetail(10, "llama-server.exe", cmdline=["server.exe", "--api-key", "private-test-value"]),
                    LlmView(server=ServerInfo("llama-server", "http://127.0.0.1:8080"))),
            ProcRow(20, "dwm.exe", "compositor",
                    ProcGpu(dedicated=2 * GiB, committed=3 * GiB),
                    ProcDetail(20, "dwm.exe")),
        ],
    )


class CliTests(unittest.TestCase):
    def test_invalid_options_fail_before_collecting(self):
        for args in (["--gpu", "-1"], ["-i", "0"], ["-i", "-1"],
                     ["-i", "nan"], ["-i", "inf"]):
            with self.subTest(args=args), patch("sys.argv", ["gputaskman", *args]):
                with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
                    main()
                self.assertEqual(error.exception.code, 2)

    def test_help_needs_no_gpu(self):
        output = io.StringIO()
        with patch("sys.argv", ["gputaskman", "--help"]), contextlib.redirect_stdout(output):
            with self.assertRaises(SystemExit) as error:
                main()
        self.assertEqual(error.exception.code, 0)
        self.assertIn("implies --once", output.getvalue())

    def test_json_implies_once(self):
        output = io.StringIO()
        with patch("sys.argv", ["gputaskman", "--json"]), patch("gputaskman.cli.time.sleep"):
            with patch("gputaskman.collector.Collector") as collector, contextlib.redirect_stdout(output):
                collector.return_value.sample.return_value = snapshot()
                main()
                collector.return_value.close.assert_called_once()
        self.assertEqual(len(json.loads(output.getvalue())["processes"]), 2)

    def test_snapshot_failure_closes_collector(self):
        with patch("sys.argv", ["gputaskman", "--once"]), patch("gputaskman.cli.time.sleep"):
            with patch("gputaskman.collector.Collector") as collector:
                collector.return_value.sample.side_effect = OSError("counter unavailable")
                with self.assertRaisesRegex(OSError, "counter unavailable"):
                    main()
                collector.return_value.close.assert_called_once()

    def test_out_of_range_gpu_is_rejected(self):
        adapter = Adapter(1, "NVIDIA GPU", 0x10DE, 16 * GiB, 0, False)
        output = io.StringIO()
        with patch("sys.argv", ["gputaskman", "--gpu", "1", "--json"]), \
                patch("gputaskman.collector.enumerate_adapters", return_value=[adapter]), \
                patch("gputaskman.collector.Collector") as collector, contextlib.redirect_stderr(output):
            with self.assertRaises(SystemExit) as error:
                main()
        self.assertEqual(error.exception.code, 2)
        self.assertIn("out of range", output.getvalue())
        collector.assert_not_called()


class DeviceSelectionTests(unittest.TestCase):
    def test_adl_empty_or_failed_sample_allows_fallback(self):
        provider = Adl.__new__(Adl)
        provider.context, provider.index = ctypes.c_void_p(1), 4
        provider.dll = SimpleNamespace(ADL2_New_QueryPMLogData_Get=lambda *args: 0,
                                       ADL2_Main_Control_Destroy=lambda context: None)
        self.assertIsNone(provider.sample())
        provider.dll.ADL2_New_QueryPMLogData_Get = lambda *args: -1
        self.assertIsNone(provider.sample())
        provider.close()
        provider.close()
        self.assertIsNone(provider.sample())

    def test_shared_listen_port_is_not_used_for_server_attribution(self):
        inspector = ProcInspector()
        inspector._ports = {1: [9090, 9091], 2: [9090]}
        with patch.object(inspector, "_refresh_ports"):
            self.assertEqual(inspector.exclusive_ports(1), [9091])
            self.assertEqual(inspector.exclusive_ports(2), [])

    def test_adl_uses_only_supported_sensors_and_reports_board_power(self):
        def read(_context, _index, pointer):
            sensors = ctypes.cast(pointer, ctypes.POINTER(_Metrics)).contents.sensors
            for index, value in ((1, 2500), (8, 55), (19, 0), (73, 120)):
                sensors[index].supported, sensors[index].value = 1, value
            return 0

        provider = Adl.__new__(Adl)
        provider.context, provider.index = ctypes.c_void_p(1), 4
        provider.dll = SimpleNamespace(ADL2_New_QueryPMLogData_Get=read)
        sample = provider.sample()
        self.assertEqual((sample.gfx_clock, sample.temp, sample.util_gpu), (2500, 55, 0))
        self.assertEqual((sample.power_w, sample.power_kind), (120, "board"))
        self.assertIsNone(sample.mem_clock)
        self.assertIsNone(sample.util_mem)

    def test_amdsmi_matches_pci_and_handles_partial_metrics(self):
        amd = Adapter(1, "AMD", 0x1002, 16 * GiB, 0, False, (10, 0, 0))
        closed = []
        api = SimpleNamespace(
            amdsmi_init=lambda: None, amdsmi_shut_down=lambda: closed.append(True),
            amdsmi_get_processor_handles=lambda: ["other", "target"],
            amdsmi_get_gpu_device_bdf=lambda h: "0000:0a:00.0" if h == "target" else "0000:0b:00.0",
            amdsmi_get_gpu_driver_info=lambda h: {"driver_version": "test"},
            amdsmi_get_gpu_activity=lambda h: {"gfx_activity": 0},
            amdsmi_get_power_info=lambda h: {"socket_power": 42, "power_limit": "N/A"},
            AmdSmiTemperatureType=SimpleNamespace(EDGE=1),
            AmdSmiTemperatureMetric=SimpleNamespace(CURRENT=2),
            amdsmi_get_temp_metric=lambda *args: 47,  # AMD SMI reports degrees Celsius
        )
        provider = AmdSmi(amd, [amd], api)
        sample = provider.sample()
        self.assertEqual((sample.source, sample.util_gpu, sample.temp, sample.power_w), ("AMD SMI", 0, 47, 42))
        self.assertIsNone(sample.power_limit_w)
        self.assertIsNone(sample.util_mem)
        provider.close()
        provider.close()
        self.assertEqual(closed, [True])
        self.assertIsNone(provider.sample())

    def test_amdsmi_rejects_ambiguous_and_nonzero_domain(self):
        amd = Adapter(1, "AMD", 0x1002, 16 * GiB, 0, False, (10, 0, 0))
        other = Adapter(2, "AMD", 0x1002, 16 * GiB, 0, False, (10, 0, 0))
        api = SimpleNamespace(amdsmi_init=lambda: None, amdsmi_shut_down=lambda: None,
                              amdsmi_get_processor_handles=lambda: [1],
                              amdsmi_get_gpu_device_bdf=lambda h: "0001:0a:00.0")
        self.assertIsNone(AmdSmi(amd, [amd, other], api).sample())
        provider = AmdSmi(amd, [amd], api)
        self.assertIsNone(provider.sample())
        provider.close()

    def test_mixed_vendor_adapters_are_selectable(self):
        amd = Adapter(2, "AMD GPU", 0x1002, 24 * GiB, 0, False)
        nvidia = Adapter(1, "NVIDIA GPU", 0x10DE, 8 * GiB, 0, False)
        self.assertIs(select_adapter([amd, nvidia], 0), nvidia)
        self.assertIs(select_adapter([amd, nvidia], 1), amd)

    def test_amd_snapshot_does_not_show_cuda_header(self):
        s = snapshot()
        s.adapter = Adapter(2, "AMD GPU", 0x1002, 24 * GiB, 0, False)
        self.assertNotIn("CUDA", to_text(s).splitlines()[0])
        self.assertEqual(to_dict(s)["adapter"]["vendor_id"], 0x1002)

    def test_hip_runtime_is_compute_evidence(self):
        self.assertEqual(classify(ProcDetail(1, "worker.exe", runtimes=["HIP"]), {}), "compute")
        for dll in ("amdhip64.dll", "amdhip64_6.dll"):
            self.assertIn("HIP", _match_runtimes({dll}))
        self.assertNotIn("HIP", _match_runtimes({"amd_comgr_2.dll"}))

    def test_adapter_indices_do_not_clamp_to_another_gpu(self):
        gpu = Adapter(1, "NVIDIA GPU", 0x10DE, 16 * GiB, 0, False)
        software = Adapter(2, "Software", 0x1414, 0, 0, True)
        self.assertIs(select_adapter([software, gpu], 0), gpu)
        with self.assertRaisesRegex(ValueError, "out of range"):
            select_adapter([software, gpu], 1)
        self.assertIsNone(select_adapter([], 0))
        with self.assertRaisesRegex(ValueError, "out of range"):
            select_adapter([], 1)

    @patch("gputaskman.collector.sys.platform", "win32")
    def test_nvml_matches_pci_address_not_name_or_index(self):
        gpu = Adapter(1, "NVIDIA RTX Spark", 0x10DE, 64 * GiB, 0, False, (10, 0, 0))
        npu = NvDevice(index=0, name="NVIDIA NPU", driver_model="MCDM")
        nv_gpu = NvDevice(index=1, name="NVIDIA RTX SPARK", pci_bus="00000000:0A:00.0", driver_model="WDDM")
        self.assertIs(select_nv_device(gpu, [npu, nv_gpu], 0), nv_gpu)
        self.assertIs(select_nv_device(None, [npu, nv_gpu], 0), npu)
        self.assertIsNone(select_nv_device(gpu, [npu], 0))
        self.assertIsNone(select_nv_device(gpu, [nv_gpu, nv_gpu], 0))
        intel = Adapter(2, "Intel GPU", 0x8086, 8 * GiB, 0, False)
        self.assertIsNone(select_nv_device(intel, [nv_gpu], 0))


class MemoryTests(unittest.TestCase):
    def test_windows_counter_luid_mismatch_is_not_attributed_to_selected_adapter(self):
        c = Collector.__new__(Collector)
        c.adapter = Adapter(0x4bb7, "GPU", 0x10DE, 16 * GiB, 0, False)
        c.adapters = [c.adapter]
        c.gpu_index, c.scan_budget, c.host, c.host_status = 0, None, None, ""
        c.nvml = Mock(ok=False)
        c.counters = Mock(outside={})
        c.counters.sample.return_value = (
            {0x4bf1: {4: ProcGpu(dedicated=GiB, engines={"compute": 10})}},
            {0x4bf1: AdapterGpu(dedicated=GiB, shared=0)})
        c.inspector, c.prober = Mock(), Mock()
        with patch("gputaskman.collector.BACKEND", "wddm"):
            s = c.sample()
        self.assertIsNone(s.vram_used)
        self.assertIsNone(s.attributed)
        self.assertEqual(s.procs, [])
        self.assertFalse(s.counter_adapter_matched)
        self.assertIn("could not be matched", s.counter_status)
        self.assertIn("GPU-PV", to_text(s))
        data = to_dict(s)
        self.assertIsNone(data["attributed"])
        self.assertFalse(data["counter_adapter_matched"])
        self.assertEqual(data["counter_status"], s.counter_status)
        output = io.StringIO()
        Console(file=output, width=160).print(GpuTaskManager().render_device(s))
        self.assertIn("could not be matched", output.getvalue())

    def test_nvml_is_swept_only_for_an_nvidia_adapter(self):
        amd_telemetry = NvDevice(index=0, source="AMD ADL", util_gpu=5)
        for vendor, swept, expected in ((0x1002, False, amd_telemetry), (0x8086, False, None),
                                        (0x10DE, True, None)):
            with self.subTest(vendor=hex(vendor)):
                c = Collector.__new__(Collector)
                c.adapter = Adapter(1, "GPU", vendor, 16 * GiB, 0, False, (1, 0, 0))
                c.adapters = [c.adapter]
                c.gpu_index, c.scan_budget, c.host, c.host_status = 0, None, None, ""
                c.nvml = Mock(ok=True)
                c.nvml.sample.return_value = []
                c.adl, c.amdsmi = Mock(), Mock()
                c.adl.sample.return_value = amd_telemetry
                c.counters = Mock(outside={})
                c.counters.sample.return_value = ({1: {}}, {1: AdapterGpu(dedicated=0, shared=0)})
                c.inspector, c.prober = Mock(), Mock()
                with patch("gputaskman.collector.BACKEND", "wddm"):
                    s = c.sample()
                self.assertEqual(c.nvml.sample.called, swept)
                self.assertIs(s.nv, expected)

    def test_windows_matching_counter_zero_is_not_unavailable(self):
        c = Collector.__new__(Collector)
        c.adapter = Adapter(1, "GPU", 0x10DE, 16 * GiB, 0, False)
        c.adapters = [c.adapter]
        c.gpu_index, c.scan_budget, c.host, c.host_status = 0, None, None, ""
        c.nvml = Mock(ok=False)
        c.counters = Mock(outside={})
        c.counters.sample.return_value = (
            {2: {4: ProcGpu(dedicated=GiB)}},
            {2: AdapterGpu(dedicated=GiB, shared=GiB), 1: AdapterGpu(dedicated=0, shared=0)})
        c.inspector, c.prober = Mock(), Mock()
        with patch("gputaskman.collector.BACKEND", "wddm"):
            s = c.sample()
        self.assertEqual((s.vram_used, s.shared_used), (0, 0))
        self.assertEqual(s.procs, [])
        self.assertTrue(s.counter_adapter_matched)
        self.assertEqual(s.counter_status, "")

    def test_windows_missing_adapter_counter_explains_unavailable_memory(self):
        c = Collector.__new__(Collector)
        c.adapter = Adapter(1, "GPU", 0x10DE, 16 * GiB, 0, False)
        c.adapters = [c.adapter]
        c.gpu_index, c.scan_budget, c.host, c.host_status = 0, None, None, ""
        c.nvml = Mock(ok=False)
        c.counters = Mock(outside={})
        c.counters.sample.return_value = ({1: {}}, {})
        c.inspector, c.prober = Mock(), Mock()
        with patch("gputaskman.collector.BACKEND", "wddm"):
            s = c.sample()
        self.assertTrue(s.counter_adapter_matched)
        self.assertIsNone(s.vram_used)
        self.assertIn("adapter memory counters are unavailable", s.counter_status)
        self.assertIn("adapter memory counters are unavailable", to_text(s))

    def test_failed_collection_does_not_reuse_previous_counters(self):
        counters = GpuCounters.__new__(GpuCounters)
        counters._query = None
        with patch("gputaskman.pdh._pdh", SimpleNamespace(PdhCollectQueryData=lambda query: 1)), \
                patch.object(counters, "_read") as read:
            self.assertEqual(counters.sample(), ({}, {}))
            read.assert_not_called()

    def test_counter_samples_track_each_process_field(self):
        counters = GpuCounters.__new__(GpuCounters)
        counters._query = None
        data = {"dedicated": [("pid_42_luid_0x0_0x1_phys_0", 0.0)],
                "engine": [("pid_42_luid_0x0_0x1_phys_0_eng_0_engtype_Compute", 10.0)]}
        with patch.object(counters, "_read", side_effect=lambda key: data.get(key, [])), \
                patch("gputaskman.pdh._pdh", SimpleNamespace(PdhCollectQueryData=lambda query: 0)):
            processes, _ = counters.sample()
        gpu = processes[1][42]
        self.assertEqual(gpu.available, {"dedicated", "engine"})
        self.assertEqual(gpu.dedicated, 0)
        self.assertEqual(gpu.gpu_pct, 10)
        self.assertFalse(gpu.paged_available)

    def test_missing_process_counters_are_not_reported_as_zero(self):
        s = snapshot()
        s.procs[0].gpu.available = {"dedicated"}
        row = to_dict(s)["processes"][0]
        self.assertEqual(row["vram_bytes"], 8 * GiB)
        for name in ("shared_bytes", "committed_bytes", "paged_out_bytes", "gpu_pct", "engines"):
            self.assertIsNone(row[name])
        self.assertIn("N/A", to_text(s))
        s.procs[0].gpu.available.clear()
        self.assertIsNone(to_dict(s)["attributed"])

    def test_missing_adapter_memory_is_not_reported_as_zero(self):
        s = snapshot()
        s.vram_used = None
        s.shared_used = None
        self.assertIsNone(to_dict(s)["vram_used"])
        self.assertIsNone(to_dict(s)["shared_used"])
        self.assertIn("VRAM N/A", to_text(s))
        self.assertIn("shared (in system RAM) N/A", to_text(s))
        output = io.StringIO()
        Console(file=output, width=120).print(GpuTaskManager().render_device(s))
        self.assertIn("N/A", output.getvalue())
    def test_best_vram_keeps_measured_and_inferred_zero_distinct_from_unknown(self):
        gpu = ProcGpu(dedicated=0, inferred=8 * GiB)
        self.assertEqual(gpu.shown_vram, 0)
        gpu.attributed = False
        self.assertEqual(gpu.shown_vram, 8 * GiB)
        gpu.inferred = 0
        self.assertEqual(gpu.shown_vram, 0)
        gpu.inferred = None
        self.assertIsNone(gpu.shown_vram)

    def test_local_counters_are_preferred(self):
        self.assertTrue(GpuCounters._COUNTERS["dedicated"][0].endswith(r"\Local Usage"))
        self.assertTrue(GpuCounters._COUNTERS["shared"][0].endswith(r"\Non Local Usage"))

    def test_paged_is_clamped_commit_difference(self):
        gpu = ProcGpu(dedicated=100, shared=20, committed=150)
        self.assertEqual(gpu.paged_out, 30)
        gpu.committed = 10
        self.assertEqual(gpu.paged_out, 0)

    def test_compositor_is_in_attribution(self):
        self.assertEqual(snapshot().attributed, 10 * GiB)

    def test_json_and_text_match_the_snapshot(self):
        data = to_dict(snapshot())
        self.assertEqual(data["processes"][1]["paged_out_bytes"], GiB)
        self.assertEqual(data["processes"][0]["shared_bytes"], GiB)
        self.assertEqual(data["attributed"], 10 * GiB)
        text = to_text(snapshot())
        self.assertIn("PAGED", text)
        self.assertNotIn("gen=None", text)

    def test_text_shows_unavailable_device_metrics_without_fabricating_zeroes(self):
        s = snapshot()
        s.nv = NvDevice(index=0, util_gpu=0, power_w=2.4, pcie_tx_kbs=0, video_clock=656)
        text = to_text(s)
        self.assertIn("GPU 0%", text)
        self.assertIn("mem-bandwidth N/A", text)
        self.assertIn("2/N/A W", text)
        self.assertIn("P N/A", text)
        self.assertIn("video 656 MHz", text)
        self.assertIn("PCIe rx N/A MB/s tx 0 MB/s", text)
        self.assertNotIn("None", text.splitlines()[2])
        self.assertIsNone(to_dict(s)["device"]["power_limit_w"])
        self.assertEqual(to_dict(s)["device"]["video_clock"], 656)

    def test_video_clock_is_optional_nvml_metric(self):
        memory = SimpleNamespace(total=16 * GiB, used=GiB, reserved=0)
        clock_calls = []

        def clock_info(_handle, clock):
            clock_calls.append(clock)
            if clock == 8:
                return 656
            raise RuntimeError("unsupported")

        api = SimpleNamespace(
            NVML_CLOCK_SM=1, NVML_CLOCK_MEM=2, NVML_CLOCK_VIDEO=8,
            NVML_TEMPERATURE_GPU=0, nvmlMemory_v2=object(),
            NVML_PCIE_UTIL_RX_BYTES=0, NVML_PCIE_UTIL_TX_BYTES=1,
            nvmlInit=lambda: None, nvmlShutdown=lambda: None,
            nvmlSystemGetDriverVersion=lambda: "621.01",
            nvmlSystemGetCudaDriverVersion_v2=lambda: None,
            nvmlSystemGetCudaDriverVersion=lambda: None,
            nvmlDeviceGetCount=lambda: 1, nvmlDeviceGetHandleByIndex=lambda _i: "gpu",
            nvmlDeviceGetName=lambda _h: "NVIDIA test GPU",
            nvmlDeviceGetPciInfo=lambda _h: SimpleNamespace(busId=b"0000:01:00.0"),
            nvmlDeviceGetDriverModel=lambda _h: (0, 0),
            nvmlDeviceGetMemoryInfo=lambda *_args: memory,
            nvmlDeviceGetUtilizationRates=lambda _h: None,
            nvmlDeviceGetEncoderUtilization=lambda _h: None,
            nvmlDeviceGetDecoderUtilization=lambda _h: None,
            nvmlDeviceGetTemperature=lambda *_args: None,
            nvmlDeviceGetPowerUsage=lambda _h: None,
            nvmlDeviceGetEnforcedPowerLimit=lambda _h: None,
            nvmlDeviceGetFanSpeed=lambda _h: None,
            nvmlDeviceGetPerformanceState=lambda _h: None,
            nvmlDeviceGetClockInfo=clock_info,
            nvmlDeviceGetMaxClockInfo=lambda *_args: None,
            nvmlDeviceGetPcieThroughput=lambda *_args: None,
            nvmlDeviceGetCurrPcieLinkGeneration=lambda _h: None,
            nvmlDeviceGetCurrPcieLinkWidth=lambda _h: None,
            nvmlDeviceGetCurrentClocksEventReasons=lambda _h: None,
            nvmlDeviceGetCurrentClocksThrottleReasons=lambda _h: None,
            nvmlDeviceGetCudaComputeCapability=lambda _h: None,
        )
        with patch("gputaskman.nvml.N", api):
            nvml = Nvml()
            device = nvml.sample()[0]
            nvml.close()
        self.assertEqual(device.video_clock, 656)
        self.assertIn(api.NVML_CLOCK_VIDEO, clock_calls)
        self.assertIsNone(device.mem_clock)

    def test_capture_does_not_change_original_or_publish_command_line(self):
        original = snapshot()
        original.procs[0].llm.models = [
            GgufInfo(path=native(r"C:\Users\example\models\model.gguf"), file_size=1, version=3)
        ]
        public = for_publication(original)
        self.assertEqual(public.procs[0].detail.cmdline, [])
        self.assertEqual(public.procs[0].llm.models[0].path, "model.gguf")
        self.assertIn("private-test-value", original.procs[0].detail.cmdline)
        self.assertIn("Users", original.procs[0].llm.models[0].path)


class HandyTests(unittest.TestCase):
    def setUp(self):
        _log_cache.clear()
        _settings_cache.clear()

    def test_selected_model_is_distinct_from_runtime_load_and_unload_events(self):
        with tempfile.TemporaryDirectory() as directory:
            appdata, localappdata = Path(directory) / "Roaming", Path(directory) / "Local"
            settings_dir = appdata / "com.pais.handy"
            log_dir = localappdata / "com.pais.handy" / "logs"
            settings_dir.mkdir(parents=True)
            log_dir.mkdir(parents=True)
            selected = "handy-computer/nemotron/model-Q8_0.gguf"
            (settings_dir / "settings_store.json").write_text(
                json.dumps({"settings": {"selected_model": selected}}), encoding="utf-8"
            )
            (log_dir / "handy.log").write_text(
                "[2026-11-02][22:59:00][transcription][INFO] Loaded whisper model 'old.gguf'\n"
                "[2026-11-02][23:01:00][transcription][INFO] Loaded whisper model 'handy-computer/nemotron/model-Q8_0.gguf' (backend 'Vulkan0')\n"
                "[2026-11-02][23:01:01][transcription][INFO] Live streaming transcription started (model 'handy-computer/nemotron/model-Q8_0.gguf', backend 'Vulkan0')\n"
                "[2026-11-02][23:01:02][transcription][INFO] Model idle for 3606s (limit: 3600s), unloading\n"
                "[2026-11-02][23:01:03][transcription][INFO] Model unloaded due to inactivity (took 20ms)\n"
                "[2026-11-02][23:01:04][transcription][INFO] Partial transcription: private dictation text\n",
                encoding="utf-8",
            )
            started = datetime(2026, 11, 2, 23, 0, tzinfo=timezone.utc).timestamp()
            evidence = inspect_handy(started, str(appdata), str(localappdata))
        self.assertEqual(evidence.selected_model, selected)
        self.assertEqual(evidence.last_model, "handy-computer/nemotron/model-Q8_0.gguf")
        self.assertEqual(evidence.last_event, "unloaded")
        self.assertTrue(evidence.last_event_at.endswith("+00:00"))
        local = datetime(2026, 11, 2, 23, 1, 3, tzinfo=timezone.utc).astimezone()
        self.assertEqual(evidence.last_event_local, local.strftime("%Y-%m-%d %H:%M:%S"))
        self.assertNotIn("private dictation text", repr(evidence))

    def test_selected_model_without_fresh_log_is_not_reported_as_loaded(self):
        with tempfile.TemporaryDirectory() as directory:
            appdata, localappdata = Path(directory) / "Roaming", Path(directory) / "Local"
            settings_dir = appdata / "com.pais.handy"
            log_dir = localappdata / "com.pais.handy" / "logs"
            settings_dir.mkdir(parents=True)
            log_dir.mkdir(parents=True)
            selected = "handy-computer/nemotron/model-Q8_0.gguf"
            (settings_dir / "settings_store.json").write_text(
                json.dumps({"settings": {"selected_model": selected}}), encoding="utf-8"
            )
            (log_dir / "handy.log").write_text(
                "[2026-11-02][22:59:00][transcription][INFO] Loaded whisper model 'old.gguf'\n",
                encoding="utf-8",
            )
            started = datetime(2026, 11, 2, 23, 0, tzinfo=timezone.utc).timestamp()
            evidence = inspect_handy(started, str(appdata), str(localappdata))
        self.assertEqual(evidence.selected_model, selected)
        self.assertIsNone(evidence.last_model)
        self.assertIsNone(evidence.last_event)

    def test_handy_evidence_is_visible_in_text_and_json(self):
        s = snapshot()
        s.procs[0] = ProcRow(
            10, "handy.exe", "AI", ProcGpu(dedicated=GiB), ProcDetail(10, "handy.exe"),
            handy=HandyEvidence("handy-computer/nemotron/model.gguf", "model.gguf", "loaded",
                                "2026-11-02T23:01:00+00:00"),
        )
        self.assertIn("Handy last log (loaded): model.gguf", s.procs[0].inside)
        self.assertIn("not current GPU residency", to_text(s))
        self.assertEqual(to_dict(s)["processes"][0]["handy"]["last_event"], "loaded")


class ModelTests(unittest.TestCase):
    def test_diffusion_arguments_do_not_classify_text_encoder_as_llm(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = [Path(directory) / name for name in ("diffusion.gguf", "encoder.gguf", "vae.safetensors")]
            for path in paths:
                path.touch()
            cmd = ["sd-cli.exe", "--diffusion-model", str(paths[0]),
                   f"--llm={paths[1]}", "--vae", str(paths[2])]
            self.assertEqual(_models_from_cmdline(cmd), [str(path) for path in paths])
            self.assertEqual(_models_from_cmdline(["sd-cli.exe", f"--diffusion-model={paths[0]}"]), [str(paths[0])])
            d = ProcDetail(1, "sd-cli.exe", cmdline=cmd, runtimes=["ggml"],
                           models=[MappedModel(str(paths[1]), 0, 0, True, kind="LLM")])
            self.assertEqual(classify(d, {}), "diffusion")
            d.models[0].kind_hint = "decision model"
            self.assertEqual(classify(d, {}), "diffusion")
            self.assertIn("diffusion", AI_KINDS)

    def test_layer_and_kv_estimates(self):
        model = GgufInfo(
            path="model.gguf", file_size=1000, version=3,
            metadata={"general.architecture": "test", "test.block_count": 2,
                      "test.attention.head_count": 4, "test.attention.head_count_kv": 2,
                      "test.embedding_length": 16},
            layer_bytes={0: 100, 1: 100}, output_bytes=50, token_embd_bytes=50,
        )
        self.assertEqual(model.gpu_weight_bytes(), 250)
        self.assertEqual(model.gpu_weight_bytes(1), 100)
        self.assertEqual(model.gpu_weight_bytes(0), 0)
        self.assertEqual(model.kv_bytes_per_token(), 64)

    def test_truncated_model_reports_error(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "truncated.gguf"
            path.write_bytes(b"GGUF")
            self.assertIsNotNone(read_gguf(str(path)).error)

    def test_flags_include_equals_and_gpu_layers(self):
        self.assertEqual(parse_flags(["server", "--ctx-size=8192", "-ngl", "all", "--cache-type-k=q8_0"]),
                         {"ctx": 8192, "ngl": -1, "ctk": "q8_0"})

    def test_gpu_buffers_exclude_host_and_cpu(self):
        for device in ("CPU", "CPU_Mapped", "CUDA_Host", "Vulkan_Host"):
            self.assertFalse(is_gpu_device(device))
        for device in ("CUDA0", "Vulkan0"):
            self.assertTrue(is_gpu_device(device))

    def test_latest_load_and_multiple_kv_buffers(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "server.log"
            path.write_text(
                "load_tensors: CUDA0 model buffer size = 900 MiB\n"
                "llama_context: CUDA0 KV buffer size = 800 MiB\n"
                "new load\n"
                "load_tensors: CPU_Mapped model buffer size = 30 MiB\n"
                "load_tensors: CUDA0 model buffer size = 100 MiB\n"
                "llama_context: CUDA0 KV buffer size = 20 MiB\n"
                "llama_context: CUDA0 KV buffer size = 10 MiB\n"
                "llama_context: CUDA0 compute buffer size = 5 MiB\n",
                encoding="utf-8",
            )
            self.assertEqual(parse_load_log(str(path)), {
                "CPU_Mapped": {"weights": 30 * MiB},
                "CUDA0": {"weights": 100 * MiB, "kv": 30 * MiB, "compute": 5 * MiB},
            })
            self.assertIsNone(parse_load_log(str(path), not_before=path.stat().st_mtime + 10))


def gguf(arch: str, **meta) -> GgufInfo:
    md = {"general.architecture": arch}
    md.update({k if k.startswith(("general.", "stt.")) else f"{arch}.{k}": v for k, v in meta.items()})
    return GgufInfo(path=f"{arch}.gguf", file_size=1000, version=3, metadata=md, layer_bytes={0: 400},
                    tensor_bytes=400)


LAYA = dict(block_count=30, context_length=8192, **{"decision.type": "laya", "attention.causal": False})


class IdentityTests(unittest.TestCase):
    def test_decision_name_hints_use_model_names_not_substrings(self):
        matches = [
            (r"C:\models\OpenJev-Q4_K_M.gguf", None),
            (r"C:\models\Laya-Q8_0.gguf", None),
            (r"C:\models\julia-1.gguf", None),
            (r"C:\models\LEV-Q4_K_M.gguf", None),
            (r"C:\models\kev.gguf", None),
            (r"C:\models\OpenJev\model.safetensors", None),
            (r"C:\models\blob", "ggml-org/OpenJev-GGUF"),
            (r"C:\models\blob", "openjev:latest"),
            (r"C:\models\blob", "qwen3:14b, openjev:latest"),
        ]
        misses = [
            (r"C:\models\clever.gguf", None),
            (r"C:\models\kevin.gguf", None),
            (r"C:\models\openjevish.gguf", None),
            (r"C:\Users\laya\models\qwen.gguf", None),
            (r"C:\models\model.gguf", "openjev/qwen3"),
            (r"C:\models\model.gguf", "qwen3:openjev"),
            (r"C:\models\model.gguf", "unsloth/Qwen3-GGUF"),
            (r"C:\models\jev.gguf", None),
        ]
        for cases, expected in ((matches, "decision model"), (misses, "")):
            for path, store in cases:
                with self.subTest(path=path, store=store), \
                        patch("gputaskman.procinfo.store_name", return_value=store):
                    m = make_model(native(path), 1000, 0, False, "mapped", store)
                    self.assertEqual(m.kind_hint, expected)
                    self.assertEqual(m.sources, ["mapped"])
                    self.assertEqual(m.evidence, "in use (mapped)")

    def test_specific_header_kinds_override_name_hints(self):
        kinds = [
            gguf("modern-bert", **LAYA), gguf("clip"), gguf("bert", pooling_type=1),
            gguf("bert", pooling_type=4), gguf("parakeet", **{"stt.variant": "tdt"}),
            gguf("qwen3", **{"general.type": "adapter"}),
        ]
        for g in kinds:
            with self.subTest(kind=g.kind), patch("gputaskman.procinfo.read_gguf", return_value=g):
                m = make_model("OpenJev.gguf", 1000, 0, True, "mapped", "openjev/openjev-GGUF")
            self.assertEqual(m.kind, g.kind)
            self.assertEqual(m.kind_hint, "")
            self.assertEqual(m.kind_label, g.kind)

    def test_openjev_hint_does_not_change_header_kv_or_loading_evidence(self):
        g = gguf("qwen35", block_count=2, context_length=4096, **{
            "general.name": "Snapshot_Cfg", "attention.head_count": 4,
            "attention.head_count_kv": 2, "embedding_length": 16})
        with patch("gputaskman.procinfo.read_gguf", return_value=g):
            m = make_model("OpenJev-Q4_K_M.gguf", 1000, 0, True, "mapped", "")
        d = ProcDetail(10, "llama-server.exe", models=[m])
        self.assertEqual(classify(d, {}), "decision?")
        self.assertIn("decision?", AI_KINDS)
        self.assertEqual(m.kind, "LLM")
        self.assertEqual(m.kind_label, "likely decision model (name)")
        server = ServerInfo("llama-server", "http://127.0.0.1:1", n_ctx=4, model_path=m.path)
        with patch("gputaskman.llm.read_gguf", return_value=g):
            v = build_view(d, 1000, 0, server)
        self.assertEqual(v.label(0), "OpenJev-Q4_K_M.gguf")
        self.assertEqual(v.kind_label(0), "likely decision model (name)")
        self.assertEqual(v.files[0].evidence, "confirmed (api+mapped)")
        self.assertEqual(m.sources, ["mapped"])
        self.assertEqual(g.kind, "LLM")
        self.assertIsNone(g.decision_type)
        self.assertEqual(v.kv_per_token, 64)
        self.assertTrue(any(s.kind == "kv" for s in v.segments))
        self.assertFalse(any("Non-causal" in n for n in v.notes))
        s = snapshot()
        s.procs[0] = ProcRow(10, d.name, classify(d, {}), ProcGpu(dedicated=1000), d, v)
        data = to_dict(s)["processes"][0]
        self.assertEqual(data["kind"], "decision?")
        self.assertEqual(data["model_files"][0]["kind_hint"], "decision model")
        self.assertEqual(data["llm"]["models"][0]["kind"], "LLM")
        self.assertEqual(data["llm"]["models"][0]["kind_hint"], "decision model")
        self.assertEqual(data["llm"]["models"][0]["kind_label"], "likely decision model (name)")
        self.assertIn("likely decision model (name)", to_text(s))
        self.assertIn("confirmed (api+mapped)", to_text(s))

    def test_installed_kind_hints_do_not_classify_the_process(self):
        m = make_model("OpenJev.safetensors", 1000, 0, False, "installed", "")
        d = ProcDetail(10, "python.exe", runtimes=["PyTorch"], candidates=[m])
        self.assertEqual(m.kind_label, "likely decision model (name)")
        self.assertEqual(m.evidence, "installed, not proven loaded")
        self.assertEqual(classify(d, {}), "AI")
        row = ProcRow(10, d.name, classify(d, {}), ProcGpu(), d)
        self.assertIn("installed (unproven)", row.inside)
        self.assertIn("likely decision model (name)", row.inside)

    def test_non_gguf_hints_are_visible_in_reports(self):
        m = make_model("OpenJev.safetensors", 1000, 0, False, "mapped", "")
        d = ProcDetail(10, "python.exe", runtimes=["PyTorch"], models=[m])
        v = build_view(d, 1000, 0, None)
        s = snapshot()
        s.procs[0] = ProcRow(10, d.name, classify(d, {}), ProcGpu(dedicated=1000), d, v)
        self.assertEqual(v.models, [])
        self.assertEqual(v.other_files, [m])
        self.assertIn("likely decision model (name)", s.procs[0].inside)
        self.assertIn("likely decision model (name)", to_text(s))
        self.assertEqual(to_dict(s)["processes"][0]["model_files"][0]["kind_hint"], "decision model")

    def test_gguf_kinds_come_from_header_keys(self):
        cases = {
            "LLM": gguf("qwen3", block_count=28, context_length=40960),
            "decision model": gguf("modern-bert", **LAYA),
            "embedding model": gguf("bert", pooling_type=1),
            "reranker": gguf("bert", pooling_type=4),
            "speech model": gguf("parakeet", **{"stt.variant": "tdt"}),
            "vision projector": gguf("clip"),
            "model": gguf("modern-bert", block_count=30, context_length=8192, **{"attention.causal": False}),
        }
        for kind, info in cases.items():
            with self.subTest(kind=kind):
                self.assertEqual(info.kind, kind)
        self.assertEqual(cases["decision model"].decision_type, "laya")

    def test_store_names(self):
        digest = "ab" * 32
        with tempfile.TemporaryDirectory() as root:
            for tag_dir, tag in (("registry.ollama.ai/library/qwen3", "14b"), ("hf.co/org/thing", "latest")):
                folder = Path(root, "manifests", *tag_dir.split("/"))
                folder.mkdir(parents=True)
                (folder / tag).write_text(json.dumps({"layers": [{"digest": f"sha256:{digest}"}]}), encoding="utf-8")
            self.assertEqual(store_name(str(Path(root, "blobs", f"sha256-{digest}"))), "hf.co/org/thing:latest, qwen3:14b")
            self.assertIsNone(store_name(str(Path(root, "blobs", "sha256-" + "cd" * 32))))
            lms = Path(root, "lms")
            path = lms / "lmstudio-community" / "Qwen3-GGUF" / "Qwen3-Q4_K_M.gguf"
            self.assertEqual(store_name(str(path), [os.path.normcase(str(lms))]), "lmstudio-community/Qwen3-GGUF")
            self.assertIsNone(store_name(str(Path(root, "elsewhere", "a", "b.gguf")), [os.path.normcase(str(lms))]))
        hf = r"C:\x\hub\models--ggml-org--Laya-GGUF\snapshots\0123\Laya-Q8_0.gguf"
        self.assertEqual(store_name(hf, []), "ggml-org/Laya-GGUF")

    def test_runtime_api_normalization(self):
        v1 = {"models": [
            {"type": "decision", "key": "laya", "decision_type": "laya", "quantization": {"name": "Q8_0"},
             "size_bytes": 449, "loaded_instances": [{"id": "laya", "config": {"context_length": 8192}}]},
            {"type": "llm", "key": "idle", "loaded_instances": []},
        ]}
        self.assertEqual(lmstudio_models(v1, None), [{"name": "laya", "type": "decision (laya)", "quant": "Q8_0",
                                                      "ctx": 8192, "size": 449, "size_vram": None, "path": None}])
        v0 = {"data": [{"id": "a", "type": "llm", "state": "loaded", "quantization": "Q4_K_M", "loaded_context_length": 4096},
                       {"id": "b", "state": "not-loaded"}]}
        self.assertEqual([(m["name"], m["ctx"]) for m in lmstudio_models(None, v0)], [("a", 4096)])
        self.assertIsNone(lmstudio_models(None, None))
        llama = {"data": [{"id": r"C:\m\Laya-Q8_0.gguf", "architecture": {"output_modalities": ["decisions"]},
                           "meta": {"ftype": "Q8_0", "n_ctx": 8192, "size": 447}}]}
        self.assertEqual(llama_server_models(llama, r"C:\m\Laya-Q8_0.gguf")[0]["type"], "decision")

    def test_lmstudio_falls_back_to_v0_when_v1_is_an_error(self):
        replies = {"/api/v1/models": {"error": "Unexpected endpoint or method. (GET /api/v1/models)"},
                   "/api/v0/models": {"data": [{"id": "qwen", "state": "loaded", "max_context_length": 8192}]}}
        with patch("gputaskman.llm._get", side_effect=lambda url: next(
                (v for k, v in replies.items() if url.endswith(k)), None)):
            s = probe_server(1234)
        self.assertEqual((s.kind, s.models[0]["name"]), ("lmstudio", "qwen"))

    def test_reused_pid_gets_fresh_details(self):
        inspector = ProcInspector()
        inspector._ports_time = float("inf")  # no port scan in this test
        inspector._cache[42] = ProcDetail(42, "llama-server.exe", cmdline=["-m", "old.gguf"], create_time=100.0)
        with patch("gputaskman.procinfo._create_time", return_value=100.0):
            self.assertEqual(inspector.get(42).name, "llama-server.exe")
        fresh = ProcDetail(42, "msedge.exe", create_time=200.0)
        with patch("gputaskman.procinfo._create_time", return_value=200.0), \
                patch.object(ProcInspector, "_new", return_value=fresh):
            self.assertIs(inspector.get(42), fresh)

    def test_repeated_model_arguments_do_not_duplicate_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            path = str(Path(directory) / "model.gguf")
            Path(path).write_bytes(b"not a GGUF")
            detail = ProcDetail(6, "llama-server.exe", cmdline=["server", "--model", path, "--mmproj", path])
            with patch("gputaskman.procinfo._memory_maps", return_value=[]):
                ProcInspector().scan(detail)
        self.assertEqual(len(detail.models), 1)
        self.assertEqual(detail.models[0].sources, ["cmdline"])

    def test_ambiguous_extensions_need_size_or_store(self):
        self.assertFalse(_is_model_file(".bin", 10 * MiB, False, None))  # e.g. a Start menu tile cache
        self.assertTrue(_is_model_file(".bin", 10 * MiB, False, "org/repo"))
        self.assertTrue(_is_model_file(".bin", 100 * MiB, False, None))
        self.assertTrue(_is_model_file(".safetensors", 10 * MiB, False, None))
        self.assertTrue(_is_model_file("", 1, True, None))
        self.assertFalse(_is_model_file(".txt", GiB, False, None))

    def test_scans_follow_memory_changes(self):
        inspector, now = ProcInspector(), 1000.0
        self.assertTrue(inspector.needs_scan(ProcDetail(1), GiB, now))
        self.assertFalse(inspector.needs_scan(ProcDetail(1, maps_time=now - 1, scan_mem=GiB), 4 * GiB, now))
        self.assertFalse(inspector.needs_scan(ProcDetail(1, maps_time=now - 10, scan_mem=GiB), GiB + 50 * MiB, now))
        self.assertTrue(inspector.needs_scan(ProcDetail(1, maps_time=now - 10, scan_mem=GiB), 2 * GiB, now))
        self.assertTrue(inspector.needs_scan(ProcDetail(1, maps_time=now - 200, scan_mem=GiB), GiB, now))
        self.assertFalse(inspector.needs_scan(ProcDetail(1, access_denied=True), GiB, now))

    def test_decision_model_has_no_kv_cache_and_api_evidence(self):
        laya = gguf("modern-bert", **LAYA)
        d = ProcDetail(5, "llama-server.exe", models=[MappedModel("laya.gguf", 1000, 0, True, kind="decision model")])
        self.assertEqual(classify(d, {}), "decision")
        server = ServerInfo("llama-server", "http://127.0.0.1:1", n_ctx=8192, model_path="laya.gguf")
        with patch("gputaskman.llm.read_gguf", return_value=laya):
            v = build_view(d, 1000, 0, server)
        self.assertEqual(v.files[0].evidence, "confirmed (api+mapped)")
        self.assertEqual(d.models[0].sources, ["mapped"])  # cached detail is not mutated
        self.assertIsNone(v.kv_per_token)
        self.assertEqual([s.label for s in v.segments], ["weights", "compute buffers + runtime"])
        self.assertTrue(any("laya decision model" in n for n in v.notes))

    def test_vulkan_log_does_not_label_unknown_overhead_as_cuda(self):
        laya = gguf("modern-bert", **LAYA)
        detail = ProcDetail(5, "llama-server.exe", models=[
            MappedModel("laya.gguf", 1000, 0, True, kind="decision model")
        ])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "server.log"
            path.write_text(
                "load_tensors: Vulkan0 model buffer size = 10 MiB\n"
                "load_tensors: Vulkan_Host model buffer size = 3 MiB\n"
                "llama_context: Vulkan0 compute buffer size = 2 MiB\n",
                encoding="utf-8",
            )
            with patch("gputaskman.llm.read_gguf", return_value=laya):
                view = build_view(detail, 20 * MiB, 0, None, str(path))
        self.assertEqual([(s.label, s.bytes) for s in view.segments], [
            ("weights (Vulkan0)", 10 * MiB),
            ("compute buffers (Vulkan0)", 2 * MiB),
            ("runtime and other overhead", 8 * MiB),
        ])

    def test_installed_models_are_candidates_only(self):
        with tempfile.TemporaryDirectory() as root:
            app = Path(root, "Speechy")
            (app / "models" / "parakeet").mkdir(parents=True)
            (app / "cache").mkdir()
            exe = app / "speechy.exe"
            exe.write_bytes(b"")
            for path, size in (("models/whisper-large.bin", 70 * MiB), ("models/parakeet/encoder.onnx", 8 * MiB),
                               ("models/parakeet/decoder.onnx", 5 * MiB), ("models/tiles.bin", 5 * MiB),
                               ("cache/big.onnx", 9 * MiB)):
                with open(app / path, "wb") as f:
                    f.truncate(size)
            with patch.dict(os.environ, {"APPDATA": root, "LOCALAPPDATA": root, "SystemRoot": r"C:\Windows"}):
                found = installed_models(str(exe), now=1.0)
            self.assertEqual([(os.path.basename(m.path), m.size) for m in found],
                             [("whisper-large.bin", 70 * MiB), ("parakeet", 13 * MiB)])
            self.assertEqual(found[0].evidence, "installed, not proven loaded")
            self.assertIs(installed_models(str(exe), now=2.0), found)  # cached
            d = ProcDetail(7, "speechy.exe", runtimes=["ONNX RT"], candidates=found)
            self.assertEqual(classify(d, {}), "AI")  # candidates never decide the kind
            self.assertTrue(ProcRow(7, "speechy.exe", "AI", ProcGpu(), d).inside.startswith("installed (unproven): "))

    def test_ollama_runner_shows_only_its_model(self):
        qwen = gguf("qwen3", block_count=1, context_length=4096)
        d = ProcDetail(6, "llama-server.exe", models=[MappedModel("blob", 1000, 0, True, store_name="qwen3:14b", kind="LLM")])
        host = ServerInfo("ollama", "http://127.0.0.1:11434", models=[
            {"name": "qwen3:14b", "size": GiB, "size_vram": GiB, "ctx": 4096},
            {"name": "other:1b", "size": GiB, "size_vram": 0, "ctx": 2048}])
        with patch("gputaskman.llm.read_gguf", return_value=qwen):
            v = build_view(d, GiB, 0, host)
        self.assertEqual(v.label(0), "qwen3:14b")
        self.assertIn("api", v.files[0].sources)
        self.assertEqual(v.n_ctx, 4096)
        ollama = [n for n in v.notes if n.startswith("Ollama")]
        self.assertEqual(len(ollama), 1)
        self.assertIn("qwen3:14b", ollama[0])

    def test_runner_and_host_apis_do_not_duplicate_evidence(self):
        qwen = gguf("qwen3", block_count=1, context_length=4096)
        model = MappedModel("blob", 1000, 0, True, ["cmdline"], "qwen3:4b", "LLM")
        detail = ProcDetail(6, "llama-server.exe", models=[model])
        runner = ServerInfo("llama-server", "http://127.0.0.1:1", model_path="blob")
        host = ServerInfo("ollama", "http://127.0.0.1:2", models=[{"name": "qwen3:4b", "ctx": 4096}])
        with patch("gputaskman.llm.read_gguf", return_value=qwen):
            view = build_view(detail, GiB, 0, runner, host=host)
        self.assertEqual(view.files[0].sources, ["cmdline", "api"])
        self.assertEqual(model.sources, ["cmdline"])


class ChartTests(unittest.TestCase):
    def test_spark_distinguishes_zero_from_missing_and_clamps_values(self):
        self.assertEqual(spark([None, 0, 50, 100], 4), " ▁▅█")
        self.assertEqual(spark([-10, 200], 2), "▁█")
        self.assertEqual(spark([100, None, 0], 2), " ▁")

    def test_chart_palette_uses_distinct_explicit_colors_consistently(self):
        app = GpuTaskManager()
        colors = app.CHART_COLORS
        self.assertEqual(len(set(colors.values())), len(app.CHARTS))
        for index, name in enumerate(app.CHARTS):
            self.assertRegex(colors[name], r"^#[0-9a-f]{6}$")
            self.assertEqual(app.chart_series(index, None)[3], colors[name])
        s = snapshot()
        s.nv = NvDevice(index=0, util_gpu=0, util_mem=0)
        metrics = next(line for line in app.render_device(s).renderables if "Mem activity" in line.plain)
        self.assertTrue(any(span.style == colors["Mem activity"] for span in metrics.spans))


class UiTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        # These snapshots are NVIDIA/WDDM ones: render them with that backend's charts on every OS (macOS too).
        backend = patch("gputaskman.app.BACKEND", "wddm")
        backend.start()
        self.addCleanup(backend.stop)

    def chart_text(self, app):
        widget = app.query_one("#charts", Static)
        output = io.StringIO()
        Console(file=output, width=widget.content_size.width, color_system=None).print(widget.content)
        return output.getvalue()

    async def test_monitoring_toggle_hides_diagnostics_without_changing_live_state(self):
        s = snapshot()
        s.driver, s.cuda = "621.01", "13.5"
        s.nv = NvDevice(index=0, name="Test GPU", driver_model="WDDM", pci_bus="00000000:01:00.0",
                        util_gpu=47, util_mem=44, temp=59, power_w=126, power_limit_w=300,
                        sm_clock=2595, sm_clock_max=3090, mem_clock=15801, video_clock=2242,
                        pstate=1, pcie_gen=5, pcie_width=8, compute_cap="12.0",
                        pcie_rx_kbs=35 * 1024, pcie_tx_kbs=31 * 1024, throttle=["thermal"])
        s.counter_status = "Counters unavailable for one protected process"
        with patch.object(GpuTaskManager, "poll"):
            app = GpuTaskManager()
            async with app.run_test(size=(180, 50)) as pilot:
                app.apply(s)
                await pilot.pause()
                device = app.query_one("#device", Static)
                full = "\n".join(line.plain for line in device.content.renderables)
                self.assertFalse(app.monitoring)
                self.assertEqual(str(device.border_title), "Full diagnostics")
                for diagnostic in ("driver 621.01", "CUDA 13.5", "WDDM", "00000000:01:00.0",
                                   "PCIe gen5 x8", "CC 12.0", "126/300 W", "P1",
                                   "2595/3090 MHz", "mem 15801 MHz", "video 2242 MHz"):
                    self.assertIn(diagnostic, full)
                await pilot.press("down", "p")
                selected = app.selected_key
                histories = list(app.hist_util)
                table = app.query_one("#procs")
                row_count = table.row_count
                charts = self.chart_text(app)
                inspector = app.query_one("#inspector", Static).content
                await pilot.press("m")
                await pilot.pause()
                slim = "\n".join(line.plain for line in device.content.renderables)
                self.assertTrue(app.monitoring)
                self.assertEqual(str(device.border_title), "Slim monitoring")
                for hidden in ("621.01", "13.5", "WDDM", "00000000:01:00.0", "PCIe gen", "CC ",
                               "/300 W", "P1", "/3090", "mem 15801", "video 2242"):
                    self.assertNotIn(hidden, slim)
                for live in ("Test GPU", "VRAM", "10.00 GB / 16.00 GB", "shared(RAM) 1.00 GB",
                             "GPU 47%", "Mem activity 44%", "PCIe↓", "59°C", "126 W", "SM 2595 MHz",
                             "throttle: thermal", s.counter_status, "[PAUSED]"):
                    self.assertIn(live, slim)
                self.assertEqual(app.selected_key, selected)
                self.assertEqual(list(app.hist_util), histories)
                self.assertEqual(table.row_count, row_count)
                self.assertEqual(self.chart_text(app), charts)
                self.assertIs(app.query_one("#inspector", Static).content, inspector)
                await pilot.press("m")
                await pilot.pause()
                restored = "\n".join(line.plain for line in device.content.renderables)
                self.assertEqual(restored.replace("   [PAUSED]", ""), full)

    async def test_monitoring_applies_to_every_gpu_and_survives_new_samples(self):
        with patch.object(GpuTaskManager, "poll"):
            app = GpuTaskManager()
            async with app.run_test(size=(140, 42)) as pilot:
                await pilot.press("m")
                first, second = snapshot(), snapshot()
                first.driver = second.driver = "test-driver"
                first.nv = NvDevice(index=0, name="First GPU", util_gpu=10)
                second.nv = NvDevice(index=1, name="Second GPU", util_gpu=80)
                app.apply_all([first, second])
                await pilot.pause()
                device = app.query_one("#device", Static)
                rendered = io.StringIO()
                Console(file=rendered, width=140).print(device.content)
                for name in ("GPU 0", "GPU 1", "First GPU", "Second GPU"):
                    self.assertIn(name, rendered.getvalue())
                self.assertNotIn("test-driver", rendered.getvalue())
                self.assertNotIn("PCIe gen", rendered.getvalue())
                self.assertEqual(app.query_one("#procs").row_count, 4)
                self.assertEqual(list(app._histories[0]["util"]), [10])
                self.assertEqual(list(app._histories[1]["util"]), [80])
                first.nv.util_gpu = 20
                app.apply_all([first, second])
                await pilot.pause()
                self.assertTrue(app.monitoring)
                self.assertEqual(list(app._histories[0]["util"]), [10, 20])
                self.assertEqual(list(app._histories[1]["util"]), [80, 80])
                await pilot.resize_terminal(80, 24)
                await pilot.pause()
                self.assertGreaterEqual(app.query_one("#procs").region.height, 4)
                self.assertLessEqual(app.query_one("#procs").region.bottom, app.query_one(Footer).region.y)
                screenshot = "".join(ET.fromstring(app.export_screenshot()).itertext())
                self.assertIn("Slim/Full", screenshot)
                await pilot.press("m")
                await pilot.pause()
                rendered = io.StringIO()
                Console(file=rendered, width=140).print(device.content)
                self.assertIn("test-driver", rendered.getvalue())

    async def test_monitoring_keeps_missing_values_and_backend_provenance(self):
        with patch.object(GpuTaskManager, "poll"):
            app = GpuTaskManager()
            async with app.run_test(size=(140, 42)) as pilot:
                await pilot.press("m")
                for backend in ("wddm", "wsl", "nvml", "metal"):
                    with self.subTest(backend=backend):
                        s = snapshot()
                        s.backend, s.driver = backend, "hidden-driver"
                        s.vram_used = None
                        s.nv = NvDevice(index=0, util_gpu=0)
                        s.counter_status = "Counter pairing unavailable"
                        app.apply(s)
                        await pilot.pause()
                        lines = "\n".join(line.plain for line in app.render_device(s).renderables)
                        self.assertNotIn("hidden-driver", lines)
                        self.assertIn("N/A / 16.00 GB", lines)
                        self.assertIn("GPU 0%", lines)
                        self.assertIn(s.counter_status, lines)
                        self.assertNotIn("None", lines)
                        if backend != "metal":
                            self.assertIn("N/A  N/A W  SM N/A MHz", lines)
                s = snapshot()
                s.nv = NvDevice(index=0, source="AMD ADL", util_gpu=25, gfx_clock=2200,
                                power_w=115, power_kind="ASIC")
                app.apply(s)
                await pilot.pause()
                lines = "\n".join(line.plain for line in app.render_device(s).renderables)
                self.assertIn("AMD gfx 25%", lines)
                self.assertIn("gfx 2200 MHz", lines)
                self.assertIn("115 W ASIC", lines)
                s.nv = None
                app.apply(s)
                await pilot.pause()
                self.assertIn("VRAM", "\n".join(line.plain for line in app.render_device(s).renderables))

    async def test_wsl_sort_selection_and_bar_use_inferred_vram_without_double_counting(self):
        s = snapshot()
        s.backend = "wsl"
        small = ProcRow(10, "small", "LLM", ProcGpu(attributed=False, inferred=2 * GiB, inferred_by="order"),
                        ProcDetail(10, "small"), LlmView())
        large = ProcRow(20, "large", "LLM", ProcGpu(attributed=False, inferred=8 * GiB, inferred_by="order"),
                        ProcDetail(20, "large"), LlmView())
        vm = ProcRow(30, "WSL VM", "wsl-vm", ProcGpu(dedicated=10 * GiB), ProcDetail(30, "WSL VM"),
                     host=True, origin="windows")
        host = ProcRow(40, "browser.exe", "browser", ProcGpu(dedicated=3 * GiB), ProcDetail(40, "browser.exe"),
                       host=True, origin="windows")
        s.procs = [small, large, host, vm]
        with patch.object(GpuTaskManager, "poll"), patch("gputaskman.app.stacked_bar", wraps=stacked_bar) as bar:
            app = GpuTaskManager()
            async with app.run_test(size=(180, 42)) as pilot:
                app.apply(s)
                await pilot.pause()
                self.assertEqual(app.selected_key, large.key)
                self.assertEqual([r.key for r in app.visible_rows(s)], [vm.key, large.key, small.key, host.key])
                self.assertEqual(bar.call_args.args[1], 8 * GiB)
                table = app.query_one("#procs")
                self.assertEqual(table.get_cell(large.key, "VRAM").plain, "8.00 GB?")
                for column in ("GPU%", "3D", "Copy", "Video"):
                    self.assertEqual(table.get_cell(large.key, column).plain, "n/a")
                self.assertEqual(s.attributed, 13 * GiB)

    async def test_native_hidden_vram_keeps_engine_reading_in_ui(self):
        s = snapshot()
        s.backend = "nvml"
        s.procs = [ProcRow(10, "python3", "AI", ProcGpu(attributed=False, engines={"compute": 65}),
                           ProcDetail(10, "python3"))]
        with patch.object(GpuTaskManager, "poll"):
            app = GpuTaskManager()
            app.selected_key = "10"
            async with app.run_test(size=(180, 42)) as pilot:
                app.apply(s)
                await pilot.pause()
                self.assertEqual(app.query_one("#procs").get_cell("10", "GPU%").plain, "65")
                hdr = app.query_one("#inspector", Static).content.renderables[0].plain
                self.assertIn("compute 65%", hdr)
                self.assertIn("spill(RAM) n/a", hdr)

    async def test_unknown_process_memory_preserves_unattributed_adapter_usage(self):
        s = snapshot()
        s.procs[0].gpu.dedicated = 2 * GiB
        unknown = ProcGpu(attributed=False, engines={"3d": 20})
        unknown.available.clear()
        s.procs.append(ProcRow(30, "unknown", "app", unknown, ProcDetail(30, "unknown")))
        with patch.object(GpuTaskManager, "poll"), patch("gputaskman.app.stacked_bar", wraps=stacked_bar) as bar:
            app = GpuTaskManager()
            async with app.run_test(size=(180, 42)) as pilot:
                app.apply(s)
                await pilot.pause()
                self.assertIsNone(s.attributed)
                self.assertTrue(any((6 * GiB, "grey35") in call.args[0] for call in bar.call_args_list))

    async def test_eighth_chart_records_sm_clock_with_scale_and_missing_readings(self):
        with patch.object(GpuTaskManager, "poll"):
            app = GpuTaskManager()
            async with app.run_test(size=(117, 59)) as pilot:
                self.assertEqual(len(app.CHARTS), 8)
                index = app.CHARTS.index("SM clock")
                s = snapshot()
                s.nv = NvDevice(index=0, sm_clock=1500, sm_clock_max=3003)
                app.apply(s)
                await pilot.pause()
                self.assertIn("1500 MHz", self.chart_text(app))
                self.assertIn("SM clock 0–3003 MHz", str(app.query_one("#chart-note", Static).content))
                self.assertEqual(app.chart_series(index, s.nv)[1:3], (3003, " MHz"))
                self.assertEqual(len(app.query_one("#charts", Static).content.rows), 4)

                s.nv.sm_clock = 0
                app.apply(s)
                s.nv.sm_clock = None
                s.nv.sm_clock_max = None
                app.apply(s)
                self.assertEqual(list(app.hist_clock), [1500, 0, None])
                self.assertEqual(app.chart_series(index, s.nv)[1], 1500)
                app.query_one("#chart-mode", Select).value = index
                await pilot.pause()
                self.assertIn("N/A", self.chart_text(app))
                self.assertIn("█▁ ", self.chart_text(app))

                s.nv = None
                app.apply(s)
                self.assertEqual(list(app.hist_clock), [1500, 0, None, None])
                app.hist_clock.clear()
                self.assertEqual(app.chart_series(index, None)[1], 1)

    async def test_unavailable_device_metrics_render_as_na(self):
        with patch.object(GpuTaskManager, "poll"):
            app = GpuTaskManager()
            async with app.run_test(size=(140, 42)):
                s = snapshot()
                s.nv = NvDevice(index=0, util_mem=0, power_w=2.4, pcie_tx_kbs=0)
                app.apply(s)
                lines = "\n".join(line.plain for line in app.render_device(s).renderables)
                self.assertIn("2/N/A W", lines)
                self.assertIn("P N/A", lines)
                self.assertIn("PCIe↓  N/A", lines)
                self.assertNotIn("None", lines)
                self.assertIsNone(app.hist_util[-1])
                self.assertEqual(app.hist_bw[-1], 0)

    async def test_chart_cycle_shows_selected_metric_and_preserves_gaps(self):
        with patch.object(GpuTaskManager, "poll"):
            app = GpuTaskManager()
            async with app.run_test(size=(180, 42)) as pilot:
                s = snapshot()
                s.nv = NvDevice(index=0, util_gpu=50, util_mem=0, temp=49, power_w=72,
                                power_limit_w=150, enc=None, dec=0, video_clock=656)
                app.apply(s)
                await pilot.pause()
                self.assertEqual(app.chart_idx, -1)
                self.assertEqual(app.query_one("#chart-mode", Select).value, -1)
                for name in app.CHARTS:
                    self.assertIn(name, self.chart_text(app))
                for expected in app.CHARTS:
                    await pilot.press("c")
                    await pilot.pause()
                    self.assertEqual(app.CHARTS[app.chart_idx], expected)
                    rendered = self.chart_text(app)
                    self.assertIn(expected, rendered)
                    self.assertEqual(len(app.query_one("#charts", Static).content.columns), 1)
                    if expected == "Encoder":
                        self.assertIn("N/A", rendered)
                        screenshot = ET.fromstring(app.export_screenshot())
                        self.assertIn("Encoder", "".join(screenshot.itertext()))
                await pilot.press("c")
                self.assertEqual(app.chart_idx, -1)
                device = "\n".join(line.plain for line in app.render_device(s).renderables)
                self.assertIn("video 656 MHz", device)
                self.assertIsNone(app.hist_enc[-1])
                self.assertEqual(app.hist_dec[-1], 0)
                self.assertIn("0%", rendered)
                self.assertIn("▁", rendered)

    async def test_chart_dropdown_is_clickable_and_keyboard_accessible(self):
        with patch.object(GpuTaskManager, "poll"):
            app = GpuTaskManager()
            async with app.run_test(size=(117, 59)) as pilot:
                app.apply(snapshot())
                await pilot.pause()
                self.assertIs(app.focused, app.query_one("#procs"))
                await pilot.click("#chart-mode", offset=(2, 0))
                selector = app.query_one("#chart-mode", Select)
                self.assertTrue(selector.expanded)
                overlay = selector.query_one("SelectOverlay")
                self.assertEqual(overlay.styles.border_top[0], "solid")
                self.assertEqual(overlay.styles.padding.left, 1)
                await pilot.press("down", "down", "enter")
                await pilot.pause()
                self.assertEqual(selector.value, 1)
                self.assertEqual(app.chart_idx, 1)
                self.assertIn(app.CHARTS[1], self.chart_text(app))
                await pilot.press("enter", "home", "enter")
                await pilot.pause()
                self.assertEqual(selector.value, -1)
                for name in app.CHARTS:
                    self.assertIn(name, self.chart_text(app))

    async def test_multi_gpu_rows_histories_and_selection_stay_separate(self):
        with patch.object(GpuTaskManager, "poll"):
            app = GpuTaskManager()
            async with app.run_test(size=(140, 42)) as pilot:
                first, second = snapshot(), snapshot()
                first.nv = NvDevice(index=0, util_gpu=10)
                second.nv = NvDevice(index=1, util_gpu=80)
                second.procs[0].gpu.dedicated = 3 * GiB
                second.procs[0].gpu.available = {"dedicated"}
                app.apply_all([first, second])
                await pilot.pause()
                table = app.query_one("#procs")
                self.assertEqual(table.row_count, 4)
                self.assertEqual(table.get_cell("0:10", "GPU").plain, "0")
                self.assertEqual(table.get_cell("1:10", "GPU").plain, "1")
                self.assertEqual(table.get_cell("1:10", "VRAM").plain, "3.00 GB")
                self.assertEqual(table.get_cell("1:10", "GPU%").plain, "n/a")
                self.assertEqual(list(app._histories[0]["util"]), [10])
                self.assertEqual(list(app._histories[1]["util"]), [80])
                self.assertIn("GPU 0", self.chart_text(app))
                self.assertIn("GPU 1", self.chart_text(app))
                app.selected_key = "1:10"
                app.render_inspector()
                self.assertEqual(app.selected_pid, 10)
                self.assertIn("GPU 1", str(app.query_one("#inspector").border_title))
                app.apply_all([first, second])
                await pilot.pause()
                self.assertEqual(app.selected_key, "1:10")
                await pilot.press("c", "p")
                await pilot.pause()
                await pilot.resize_terminal(80, 24)
                await pilot.pause()
                self.assertIn("GPU 1", self.chart_text(app))
                self.assertEqual(list(app._histories[1]["util"]), [80, 80])
                self.assertGreaterEqual(table.region.height, 4)
                app.apply(first)
                await pilot.pause()
                self.assertNotIn("GPU", table.columns)
                self.assertEqual(table.row_count, 2)

    async def test_poll_collects_all_adapters_and_closes_each_collector(self):
        adapters = [Adapter(1, "first", 0x10DE, 16 * GiB, 0, False),
                    Adapter(2, "second", 0x10DE, 8 * GiB, 0, False)]
        with patch("gputaskman.app.enumerate_adapters", return_value=adapters), \
                patch("gputaskman.app.Collector") as factory:
            factory.SCAN_BUDGET = 0.15
            collectors = [Mock(), Mock()]
            closed = [threading.Event(), threading.Event()]
            for collector, event in zip(collectors, closed):
                collector.sample.side_effect = snapshot
                collector.close.side_effect = event.set
            factory.side_effect = collectors
            app = GpuTaskManager(interval=0.05)
            async with app.run_test(size=(140, 42)) as pilot:
                await pilot.pause(0.8)
                self.assertEqual(len(app.snapshots), 2)
                self.assertEqual(factory.call_args_list[0].args, (0,))
                self.assertEqual(factory.call_args_list[1].args, (1,))
                self.assertEqual(factory.call_args_list[0].kwargs["scan_budget"], 0.075)
            for collector, event in zip(collectors, closed):
                self.assertTrue(await asyncio.to_thread(event.wait, 2))
                collector.close.assert_called_once()

    async def test_explicit_gpu_keeps_single_device_layout(self):
        with patch("gputaskman.app.Collector") as factory:
            factory.SCAN_BUDGET = 0.15
            factory.return_value.sample.side_effect = snapshot
            closed = threading.Event()
            factory.return_value.close.side_effect = closed.set
            app = GpuTaskManager(gpu_index=1, interval=0.05)
            async with app.run_test(size=(140, 42)) as pilot:
                await pilot.pause(0.8)
                self.assertEqual(len(app.snapshots), 1)
                self.assertNotIn("GPU", app.query_one("#procs").columns)
                self.assertEqual(factory.call_args.args, (1,))
            self.assertTrue(await asyncio.to_thread(closed.wait, 2))
            factory.return_value.close.assert_called_once()

    async def test_multi_gpu_wsl_does_not_claim_process_device_ownership(self):
        with patch.object(GpuTaskManager, "poll"):
            app = GpuTaskManager()
            async with app.run_test(size=(140, 42)) as pilot:
                first, second = snapshot(), snapshot()
                for s in (first, second):
                    s.backend = "wsl"
                    for row in s.procs:
                        row.gpu = ProcGpu(attributed=False)
                app.apply_all([first, second])
                await pilot.pause()
                table = app.query_one("#procs")
                self.assertEqual(table.row_count, 2)
                self.assertEqual(table.get_cell("0:10", "GPU").plain, "?")
                self.assertEqual(table.get_cell("0:10", "VRAM").plain, "n/a")
                app.selected_key = "0:10"
                app.render_inspector()
                self.assertIn("GPU unknown", str(app.query_one("#inspector").border_title))

    async def test_chart_layout_reflows_while_paused_and_keeps_processes_visible(self):
        with patch.object(GpuTaskManager, "poll"):
            app = GpuTaskManager()
            async with app.run_test(size=(180, 50)) as pilot:
                s = snapshot()
                s.nv = NvDevice(index=0, util_gpu=50, temp=49, power_w=200, power_limit_w=150)
                app.apply(s)
                await pilot.pause()
                await pilot.press("down", "p")
                history = list(app.hist_util)
                for width, height, columns in ((180, 50, 3), (117, 59, 2), (80, 40, 1),
                                               (60, 24, 1), (80, 24, 1), (180, 50, 3)):
                    with self.subTest(width=width, height=height):
                        await pilot.resize_terminal(width, height)
                        await pilot.pause()
                        grid = app.query_one("#charts", Static).content
                        self.assertIsInstance(grid, Table)
                        self.assertEqual(len(grid.columns), columns)
                        self.assertEqual(len(grid.rows), (len(app.CHARTS) + columns - 1) // columns)
                        self.assertEqual(list(app.hist_util), history)
                        self.assertEqual(app.selected_pid, 20)
                        self.assertTrue(app.paused)
                        processes = app.query_one("#procs")
                        footer = app.query_one(Footer)
                        self.assertGreaterEqual(processes.region.height, 4)
                        self.assertLessEqual(processes.region.bottom, footer.region.y)
                        self.assertLessEqual(app.query_one("#inspector").region.bottom, footer.region.y)
                        self.assertLessEqual(footer.region.bottom, height)
                        for name in app.CHARTS:
                            self.assertIn(name, self.chart_text(app))
                        if height == 24:
                            dashboard = app.query_one("#dashboard")
                            self.assertGreater(dashboard.max_scroll_y, 0)
                            dashboard.scroll_end(animate=False)
                            await pilot.pause()
                            screenshot = "".join(ET.fromstring(app.export_screenshot()).itertext())
                            self.assertIn("Decoder", screenshot)
                            if width == 60:
                                note = app.query_one("#chart-note", Static)
                                self.assertGreaterEqual(note.region.height, 2)
                self.assertIn("0–200 W", str(app.query_one("#chart-note", Static).content))

    async def test_all_charts_work_without_nvml_or_vram_capacity(self):
        with patch.object(GpuTaskManager, "poll"):
            app = GpuTaskManager()
            async with app.run_test(size=(117, 59)) as pilot:
                s = snapshot()
                s.vram_total = 0
                app.apply(s)
                await pilot.pause()
                self.assertIsNone(app.hist_vram[-1])
                self.assertIsNone(app.hist_util[-1])
                rendered = self.chart_text(app)
                self.assertEqual(rendered.count("N/A"), len(app.CHARTS))
                self.assertNotIn("▁", rendered)

    async def test_handy_evidence_is_explicit_in_inspector(self):
        with patch.object(GpuTaskManager, "poll"):
            app = GpuTaskManager()
            async with app.run_test(size=(180, 42)) as pilot:
                s = snapshot()
                s.procs[0] = ProcRow(
                    10, "handy.exe", "AI", ProcGpu(dedicated=GiB), ProcDetail(10, "handy.exe"),
                    handy=HandyEvidence("handy-computer/nemotron/model.gguf", "model.gguf", "loaded",
                                        "2026-11-02T23:01:00+00:00"),
                )
                app.apply(s)
                await pilot.pause()
                await pilot.press("down")
                await pilot.pause()
                text = " ".join(ET.fromstring(app.export_screenshot()).itertext()).replace("\u00a0", " ")
                self.assertIn("Handy selected (settings; not load proof)", text)
                self.assertIn("runtime-reported history, not current GPU residency", text)

    async def test_likely_kind_is_explicit_in_process_and_inspector(self):
        g = gguf("qwen35", block_count=2, context_length=4096, **{"general.name": "Snapshot_Cfg"})
        with patch.object(GpuTaskManager, "poll"):
            app = GpuTaskManager()
            async with app.run_test(size=(240, 50)) as pilot:
                for ext in ("gguf", "safetensors"):
                    with self.subTest(format=ext):
                        with patch("gputaskman.procinfo.read_gguf", return_value=g):
                            m = make_model(f"OpenJev.{ext}", 1000, 0, ext == "gguf", "mapped", "")
                        d = ProcDetail(10, "llama-server.exe", models=[m])
                        with patch("gputaskman.llm.read_gguf", return_value=g):
                            v = build_view(d, GiB, 0, None)
                        s = snapshot()
                        s.procs[0] = ProcRow(10, d.name, classify(d, {}), ProcGpu(dedicated=GiB), d, v)
                        app.apply(s)
                        await pilot.pause()
                        self.assertEqual(app.query_one("#procs").get_row("10")[2].plain, "decision?")
                        text = " ".join(ET.fromstring(app.export_screenshot()).itertext()).replace("\u00a0", " ")
                        self.assertIn("likely decision model (name)", text)
                        self.assertIn("in use (mapped)", text)

    async def test_navigation_sort_pause_and_help(self):
        with patch.object(GpuTaskManager, "poll"):
            app = GpuTaskManager()
            async with app.run_test(size=(140, 42)) as pilot:
                app.apply(snapshot())
                await pilot.pause()
                self.assertEqual(app.query_one("#procs").row_count, 2)
                await pilot.press("down")
                await pilot.pause()
                self.assertEqual(app.selected_pid, 20)
                app.apply(snapshot())
                await pilot.pause()
                self.assertEqual(app.selected_pid, 20)
                for _ in app.SORTS:
                    await pilot.press("s")
                self.assertEqual(app.sort_idx, 0)
                await pilot.press("a", "p")
                self.assertTrue(app.show_all)
                self.assertTrue(app.paused)
                await pilot.press("question_mark")
                self.assertEqual(len(app.screen_stack), 2)
                await pilot.press("escape", "p")
                self.assertEqual(len(app.screen_stack), 1)
                self.assertFalse(app.paused)


if __name__ == "__main__":
    unittest.main()

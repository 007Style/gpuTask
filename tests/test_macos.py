"""macOS backend. Fixtures follow ioreg/footprint output measured on an M4 Mac mini (macOS 26.3.1)."""

from __future__ import annotations

import json
import plistlib
import unittest
from unittest.mock import patch

from gputaskman import macos
from gputaskman.collector import ProcRow, Snapshot
from gputaskman.pdh import ProcGpu
from gputaskman.procinfo import ProcDetail, _lib_name, _match_runtimes
from gputaskman.report import to_dict, to_text

GiB = 1 << 30
ME, ROOT = 501, 0


def accel(util=28, alloc=2202157056):
    return [{"IOClass": "AGXAcceleratorG16G", "model": "Apple M4", "gpu-core-count": 10,
             "PerformanceStatistics": {"Device Utilization %": util, "Renderer Utilization %": util,
                                       "Tiler Utilization %": util, "Alloc system memory": alloc,
                                       "In use system memory": 13598720}}]


def client(pid, name, *times):
    return {"IOUserClientCreator": f"pid {pid}, {name}",
            "AppUsage": [{"API": "Metal", "accumulatedGPUTime": t} for t in times]}


class FakePoller:
    def wait_pass(self, timeout):
        return True

    def __init__(self, sizes=None):
        self.sizes = sizes or {}
        self.targets = []

    def start(self):
        pass

    def set_targets(self, pids):
        self.targets = pids

    def get(self, pid):
        return self.sizes.get(pid)

    def close(self):
        pass


class ParseTests(unittest.TestCase):
    def test_ioreg_plist_round_trip(self):
        entries = plistlib.loads(plistlib.dumps(accel(99, 5494423552)))
        dev = macos.parse_accelerator(entries)
        self.assertEqual((dev.model, dev.cores, dev.utilization, dev.allocated), ("Apple M4", 10, 99, 5494423552))
        self.assertIsNone(macos.parse_accelerator([{"model": "x"}]))
        self.assertIsNone(macos.parse_accelerator(None))

    def test_gpu_time_is_summed_per_process(self):
        # WindowServer has several Metal clients; runningboardd has one with no usage yet
        times = macos.parse_clients([client(167, "WindowServer", 53012153708, 17704333),
                                     client(167, "WindowServer", 262833), client(176, "runningboardd"),
                                     {"IOUserClientCreator": "garbage"}])
        self.assertEqual(times, {167: 53012153708 + 17704333 + 262833, 176: 0})

    def test_graphics_footprint_counts_only_graphics_categories(self):
        process = {"pid": 79379, "categories": {
            "Owned physical footprint (unmapped) (graphics)": {"dirty": 2162163712, "swapped": 0},
            "IOAccelerator (graphics)": {"dirty": 8880128, "swapped": 0},
            "IOAccelerator": {"dirty": 65536}, "MALLOC_SMALL": {"dirty": 58 << 20}}}
        self.assertEqual(macos.graphics_bytes(process), 2162163712 + 8880128)


class CountersTests(unittest.TestCase):
    def counters(self, sizes, owners):
        poller = FakePoller(sizes)
        with patch.object(macos, "_owner", side_effect=lambda pid: owners.get(pid)), \
                patch.object(macos, "_ioreg", return_value=[]):  # the constructor's baseline sees nothing
            c = macos.MacCounters(poller=poller, uid=ME)
        c._owners = dict(owners)
        c._names = {167: "WindowServer", 176: "runningboardd", 900: "backupd"}
        return c, poller

    def test_gpu_percent_memory_and_honest_unknowns(self):
        owners = {79379: ME, 486: ME, 167: ROOT, 176: ROOT, 900: ROOT}
        c, poller = self.counters({79379: 2162163712 + 8880128}, owners)
        first = [client(79379, "python3.13", 10 * 10**9), client(486, "wezterm-gui", 4 * 10**9),
                 client(167, "WindowServer", 50 * 10**9), client(176, "runningboardd", 0),
                 client(900, "backupd", 7 * 10**9)]
        c.sample(accel(), first, now=100.0)
        second = [client(79379, "python3.13", 10 * 10**9 + 1_968_000_000), client(486, "wezterm-gui", 4 * 10**9),
                  client(167, "WindowServer", 50 * 10**9 + 40_000_000), client(176, "runningboardd", 0),
                  client(900, "backupd", 7 * 10**9)]
        procs, adapters = c.sample(accel(99, 5494423552), second, now=102.0)
        per = procs[macos.ADAPTER_KEY]
        self.assertAlmostEqual(per[79379].gpu_pct, 98.4)  # measured: 98.4% vs Device Utilization 99%
        self.assertEqual(per[79379].dedicated, 2162163712 + 8880128)
        self.assertTrue(per[79379].attributed)
        self.assertFalse(per[486].attributed)  # own process, not measured yet: unknown, not 0
        self.assertFalse(per[167].attributed)  # root's WindowServer: GPU% only
        self.assertAlmostEqual(per[167].gpu_pct, 2.0)
        self.assertNotIn(176, per)  # idle daemons of other users stay out of the table
        self.assertNotIn(900, per)
        self.assertEqual(poller.targets, [79379, 486])  # busiest own process measured first
        self.assertEqual(adapters[macos.ADAPTER_KEY].dedicated, 5494423552)
        self.assertEqual((c.device.name, c.device.util_gpu, c.device.driver_model), ("Apple M4", 99, "Metal"))

    def test_counter_reset_is_not_negative(self):
        c, _ = self.counters({}, {1: ME})
        c.sample(accel(), [client(1, "a", 5 * 10**9)], now=1.0)
        procs, _ = c.sample(accel(), [client(1, "a", 1 * 10**9)], now=2.0)  # PID reused
        self.assertEqual(procs[macos.ADAPTER_KEY][1].gpu_pct, 0.0)


class PollerTests(unittest.TestCase):
    def test_footprint_json_is_read_per_pid(self):
        def run(args, **kw):
            path = args[args.index("-j") + 1]
            pid = int(args[-1])
            with open(path, "w", encoding="utf-8") as f:
                json.dump({"processes": [{"pid": pid, "categories": {
                    "IOAccelerator (graphics)": {"dirty": 3 * GiB}}}]}, f)

        p = macos.FootprintPoller(run=run)
        p.set_targets([42])
        p.poll_once()
        self.assertEqual(p.get(42), 3 * GiB)

    def test_denied_processes_are_retried_later_not_every_pass(self):
        calls = []

        def run(args, **kw):
            calls.append(args)  # writes nothing, like footprint on a root process

        p = macos.FootprintPoller(run=run)
        p.set_targets([170])
        p.poll_once()
        p.poll_once()
        self.assertEqual(len(calls), 1)
        self.assertIsNone(p.get(170))


class ProcInfoTests(unittest.TestCase):
    def test_macos_libraries_and_metal_driver(self):
        names = {_lib_name(b) for b in ("libggml-metal.dylib", "libllama.dylib", "libmlx.dylib",
                                        "agxmetalg16g_b0", "libtorch_cpu.dylib", "assets.car")}
        self.assertNotIn("assets.car", names)
        self.assertEqual(_match_runtimes(names - {None}), ["llama.cpp", "ggml", "PyTorch", "MLX", "Metal"])


class SkipPathTests(unittest.TestCase):
    def test_protected_containers_and_system_files_are_never_opened(self):
        from gputaskman import procinfo
        with patch.object(procinfo.sys, "platform", "darwin"):
            self.assertTrue(procinfo._skip_path(
                "/Users/me/Library/Containers/com.apple.mediaanalysisd/Data/Library/Caches/PFSceneGeographyData.index"))
            self.assertTrue(procinfo._skip_path("/Users/me/Library/Group Containers/group.x/db"))
            self.assertTrue(procinfo._skip_path("/System/Library/Fonts/SFNS.ttf"))
            self.assertFalse(procinfo._skip_path("/Users/me/.ollama/models/blobs/sha256-abc"))
        with patch.object(procinfo.sys, "platform", "linux"):
            self.assertFalse(procinfo._skip_path("/usr/lib/x86_64-linux-gnu/libcuda.so.1"))


class ReportTests(unittest.TestCase):
    def test_metal_header_and_json(self):
        rows = [ProcRow(79379, "python3.13", "AI", ProcGpu(dedicated=2 * GiB, engines={"gpu": 98.4}),
                        ProcDetail(79379, "python3.13")),
                ProcRow(167, "WindowServer", "compositor", ProcGpu(attributed=False, engines={"gpu": 2.0}),
                        ProcDetail(167, "WindowServer"))]
        snap = Snapshot(taken=1, adapter=None, adapters=[], nv=None, driver="macOS 26.3.1", cuda="",
                        vram_total=16 * GiB, vram_used=5 * GiB, shared_used=0, backend="metal", procs=rows)
        text = to_text(snap)
        self.assertIn("unified", text)
        self.assertNotIn("CUDA", text)
        ws = next(line for line in text.splitlines() if line.lstrip().startswith("167"))
        self.assertIn("n/a", ws)
        data = to_dict(snap)["processes"]
        self.assertEqual((data[0]["vram_bytes"], data[1]["vram_bytes"], data[1]["gpu_pct"]), (2 * GiB, None, 2.0))
        self.assertIsNone(data[0]["shared_bytes"])


if __name__ == "__main__":
    unittest.main()

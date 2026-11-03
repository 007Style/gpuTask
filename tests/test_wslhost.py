from __future__ import annotations

import io
import json
import os
import tempfile
import time
import unittest
from unittest.mock import Mock, mock_open, patch

from gputaskman import collector as collector_mod
from gputaskman import wslhost
from gputaskman.collector import Collector, ProcRow, Snapshot
from gputaskman.llm import LlmView
from gputaskman.pdh import AdapterGpu, ProcGpu
from gputaskman.procinfo import MappedModel, ProcDetail
from gputaskman.report import to_dict, to_text
from gputaskman.wslhost import ContextTracker, HostSample, HostStream, match_adapter, parse_line

GiB = 1 << 30
MiB = 1 << 20
LUID = "luid_0x00000000_0x0000ff9b_phys_0"
RTX = 0xFF9B


def line(rows, names):
    return json.dumps({"s": rows, "n": names})


SAMPLE = line(
    [
        ["local usage", f"pid_24820_{LUID}", 2.0 * GiB],  # vmwp: one instance per guest context, summed
        ["local usage", f"pid_24820_{LUID}", 0.4 * GiB],
        ["local usage", f"pid_2744_{LUID}", 300 * MiB],
        ["non local usage", f"pid_2744_{LUID}", 20 * MiB],
        ["utilization percentage", f"pid_24820_{LUID}_eng_0_engtype_3D", 40.0],
        ["utilization percentage", f"pid_2744_{LUID}_eng_0_engtype_3D", 3.0],
        ["dedicated usage", LUID, 3.0 * GiB],
        ["dedicated usage", "luid_0x00000000_0x00011431_phys_0", 2 * MiB],
        ["local usage", "garbage", 1.0],
    ],
    {"24820": "vmwp", "2744": "dwm", "4": "System"},
)


class ParseTests(unittest.TestCase):
    def test_parses_counters_like_the_windows_backend(self):
        s = parse_line(SAMPLE, now=1.0)
        self.assertEqual(s.procs[RTX][24820].dedicated, 2.4 * GiB)
        self.assertEqual(s.procs[RTX][24820].engines["3d"], 40.0)
        self.assertEqual(s.procs[RTX][2744].shared, 20 * MiB)
        self.assertEqual(s.adapters[RTX].dedicated, 3.0 * GiB)
        self.assertEqual(s.names[24820], "vmwp")

    def test_bad_lines_are_ignored(self):
        self.assertIsNone(parse_line("WARNING: something"))
        self.assertIsNone(parse_line("[1, 2]"))
        self.assertEqual(parse_line(line([["local usage"]], {})).procs, {})


class MatchTests(unittest.TestCase):
    def sample(self, *usages):
        return HostSample(taken=0, adapters={i + 1: AdapterGpu(dedicated=u) for i, u in enumerate(usages)})

    def test_matches_adapter_by_device_usage(self):
        self.assertEqual(match_adapter(self.sample(5.1 * GiB, 0), 5.3 * GiB), 1)

    def test_refuses_distant_or_ambiguous_matches(self):
        self.assertIsNone(match_adapter(self.sample(1 * GiB), 5 * GiB))
        self.assertIsNone(match_adapter(self.sample(5.0 * GiB, 5.1 * GiB), 5.05 * GiB))
        self.assertIsNone(match_adapter(self.sample(5 * GiB), 0))


class InteropTests(unittest.TestCase):
    def test_interop_flag_reads_binfmt_entry(self):
        with patch("gputaskman.wslhost.glob.glob", return_value=["/proc/sys/fs/binfmt_misc/WSLInterop"]):
            with patch("builtins.open", mock_open(read_data="enabled\ninterpreter /init\n")):
                self.assertTrue(wslhost.interop_enabled())
            with patch("builtins.open", mock_open(read_data="disabled\n")):
                self.assertFalse(wslhost.interop_enabled())
        with patch("gputaskman.wslhost.glob.glob", return_value=[]):
            self.assertFalse(wslhost.interop_enabled())

    def test_stream_reads_lines_from_powershell(self):
        calls = []

        class FakeProc:
            def __init__(self, args, **kw):
                calls.append(args)
                self.stdout = io.StringIO(SAMPLE + "\n")

            def poll(self):
                return 0

        h = HostStream(exe="powershell.exe", popen=FakeProc)
        self.assertTrue(h.wait(2))
        deadline = time.time() + 2
        while h.status == "starting" and time.time() < deadline:
            time.sleep(0.01)
        self.assertIsNotNone(h.latest())
        self.assertEqual(calls[0][:4], ["powershell.exe", "-NoProfile", "-NonInteractive", "-EncodedCommand"])

    def test_a_bad_byte_ends_the_stream_visibly_so_it_restarts(self):
        class BadStdout:
            def __iter__(self):
                yield SAMPLE + "\n"
                raise UnicodeDecodeError("utf-8", b"\x82", 0, 1, "invalid start byte")

        class FakeProc:
            def __init__(self, args, **kw):
                self.encoding = kw.get("encoding")
                self.stdout = BadStdout()

            def poll(self):
                return 0

        h = HostStream(exe="powershell.exe", popen=FakeProc)
        deadline = time.time() + 2
        while not h.ended and time.time() < deadline:
            time.sleep(0.01)
        self.assertTrue(h.ended)
        self.assertEqual(h._proc.encoding, "utf-8")

    def test_missing_powershell_is_reported(self):
        with patch("gputaskman.wslhost.powershell_path", return_value=None):
            h = HostStream()
        self.assertTrue(h.status.startswith("unavailable"))
        self.assertIsNone(h.latest())


def at(values_gib):
    return [v * GiB for v in values_gib]


class TrackerTests(unittest.TestCase):
    """Replays context sequences measured on WSL2: contexts newest first, one per /dev/dxg holder."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = os.path.join(self.tmp.name, "ctx.json")
        self.clock = 1000.0

    def make(self, boot="boot-1"):
        return ContextTracker(path=self.path, boot_id=boot, self_pid="S")

    def step(self, t, contexts, holders):
        self.clock += 1.1
        return t.update(RTX, at(contexts), holders, now=self.clock)

    def check(self, result, expected, reason=""):
        guesses, why = result
        self.assertIn(reason, why)
        got = {pid: (round(g.bytes / GiB, 2), "*" if g.confidence == "consistent" else "?")
               for pid, g in guesses.items() if pid != "S"}
        self.assertEqual(got, expected)

    def test_processes_started_while_watching_are_pinned(self):
        t = self.make()
        h = [("S", 900)]
        self.check(self.step(t, [0.0, 0.0], h), {})  # our context (newest) and one foreign
        h.append(("A", 950))
        self.check(self.step(t, [1.19, 0.0, 0.0], h), {"A": (1.19, "*")})
        h.append(("F1", 960))
        self.check(self.step(t, [0.4, 1.19, 0.0, 0.0], h), {"A": (1.19, "*"), "F1": (0.4, "*")})
        h.append(("B", 970))
        self.check(self.step(t, [3.05, 0.4, 1.19, 0.0, 0.0], h),
                   {"A": (1.19, "*"), "F1": (0.4, "*"), "B": (3.05, "*")})
        h.remove(("A", 950))  # the oldest job exits; the others keep their contexts
        self.check(self.step(t, [3.05, 0.4, 0.0, 0.0], h), {"F1": (0.4, "*"), "B": (3.05, "*")})

    def test_a_workload_started_after_us_is_not_taken_for_our_context(self):
        t = self.make()
        h = [("S", 900), ("W", 950)]  # W created its CUDA context between our NVML init and the first sample
        self.check(self.step(t, [20 / 1024, 0.0], h), {"W": (0.02, "*")})
        self.check(self.step(t, [5.0, 0.0], h), {"W": (5.0, "*")})

    def test_cache_forgets_a_foreign_context_that_exited(self):
        t = self.make()
        self.check(self.step(t, [0.0, 2.0], [("S", 900)]), {})  # only us: 2.0 is foreign
        self.check(self.step(t, [0.0], [("S", 900)]), {})  # it exits
        with open(self.path, encoding="utf-8") as f:
            self.assertEqual(json.load(f)["foreign"], [])
        t = self.make()  # next run, same boot: a local 2 GiB job must not be taken for the old foreign one
        self.check(self.step(t, [0.0, 2.0], [("S", 900), ("A", 800)]), {"A": (2.0, "*")})

    def test_equal_sizes_are_right_now_but_never_evidence_later(self):
        t = self.make()
        h = [("S", 900), ("A", 13), ("B", 14)]
        self.check(self.step(t, [0.0, 0.2, 0.2], h), {"B": (0.2, "*"), "A": (0.2, "*")})
        # one of them loads a model: which one is unknown, so neither keeps the "*"
        self.check(self.step(t, [0.0, 4.0, 0.2], h), {"B": (4.0, "?"), "A": (0.2, "?")})

    def test_new_process_waits_for_its_own_context_instead_of_eliminating(self):
        t = self.make()
        self.check(self.step(t, [0.0, 2.0, 0.15], [("S", 900), ("B", 10)]), {"B": (2.0, "?")})
        h = [("S", 900)]  # B exits; 0.15 (WSLg) was never seen without B, so it isn't known as foreign
        self.check(self.step(t, [0.0, 0.15], h), {})
        h.append(("A", 1001))
        self.check(self.step(t, [0.0, 0.15], h), {"A": (0.15, "?")})  # A's context isn't there yet
        self.check(self.step(t, [6.0, 0.0, 0.15], h), {"A": (6.0, "*")})

    def test_cached_foreign_sizes_only_rank_startup_guesses(self):
        self.step(self.make(), [0.0, 0.15], [("S", 900)])  # only gputaskman holds the GPU: 0.15 is foreign
        t = self.make()  # a later start in the same boot, with a job already running
        self.check(self.step(t, [0.0, 0.15, 2.0], [("S", 900), ("X", 5)]), {"X": (2.0, "?")})
        self.check(self.step(t, [0.0, 0.15, 2.5], [("S", 900), ("X", 5)]), {"X": (2.5, "?")})

    def test_foreign_cache_clears_immediately_when_last_foreign_context_exits(self):
        t = self.make()
        self.step(t, [0.0, 2.0], [("S", 900)])
        with open(self.path, encoding="utf-8") as f:
            self.assertEqual(json.load(f)["foreign"], [2 * GiB])
        self.step(t, [0.0], [("S", 900)])
        self.assertEqual(t.cached_foreign, [])
        with open(self.path, encoding="utf-8") as f:
            self.assertEqual(json.load(f)["foreign"], [])
        self.check(self.step(self.make(), [0.0, 2.0], [("S", 900), ("X", 5)]), {"X": (2.0, "*")})

    def test_foreign_cache_clears_while_local_workload_remains(self):
        t = self.make()
        self.step(t, [0.0, 0.4], [("S", 900)])
        h = [("S", 900), ("X", 950)]
        self.step(t, [2.0, 0.0, 0.4], h)
        self.check(self.step(t, [2.0, 0.0], h), {"X": (2.0, "*")})
        with open(self.path, encoding="utf-8") as f:
            self.assertEqual(json.load(f)["foreign"], [])

    def test_stale_cache_never_takes_a_local_context(self):
        self.step(self.make(), [0.0, 2.0], [("S", 900)])
        # The foreign process exits while the monitor is off; a local process reuses that size.
        self.check(self.step(self.make(), [0.0, 2.0], [("S", 900), ("X", 5)]), {"X": (2.0, "*")})

    def test_first_sample_finds_monitor_zero_instead_of_newest_small_context(self):
        t = self.make()
        h = [("S", 900), ("X", 950)]
        guesses, _ = t.update(RTX, [20 * MiB, 0], h, now=self.clock)
        self.assertEqual(guesses["S"].bytes, 0)
        self.assertEqual(guesses["X"].bytes, 20 * MiB)
        self.check(self.step(t, [5.0, 0.0], h), {"X": (5.0, "*")})

    def test_equal_zero_contexts_never_pin_monitor_ownership(self):
        t = self.make()
        h = [("S", 900), ("X", 950)]
        self.check(self.step(t, [0.0, 0.0], h), {"X": (0.0, "*")})
        self.assertFalse(any(e.sure for e in t.entries))
        self.check(self.step(t, [5.0, 0.0], h), {"X": (5.0, "?")})
        self.assertFalse(any(e.label == "S" and e.value for e in t.entries))

    def test_idle_zero_ambiguity_does_not_pollute_foreign_cache(self):
        t = self.make()
        self.step(t, [0.0, 0.15, 0.0], [("S", 900)])
        self.assertEqual(t.cached_foreign, [0.15 * GiB])
        self.assertFalse(any(e.label == "S" and e.sure for e in t.entries))
        self.check(self.step(t, [2.0, 0.0, 0.15, 0.0], [("S", 900), ("X", 950)]), {"X": (2.0, "*")})

    def test_a_reused_pid_is_a_new_process(self):
        t = self.make()
        self.check(self.step(t, [0.0, 1.0], [("S", 900), ("P", 10)]), {"P": (1.0, "*")})
        self.check(self.step(t, [0.0], [("S", 900)]), {})
        self.check(self.step(t, [2.0, 0.0], [("S", 900), ("P", 2000)]), {"P": (2.0, "*")})
        self.check(self.step(t, [2.0, 0.0], [("S", 900), ("P", 2000)]), {"P": (2.0, "*")})

    def test_already_running_processes_are_guesses_until_evidence(self):
        t = self.make()
        h = [("S", 900), ("C", 14), ("B", 13)]
        self.check(self.step(t, [0.0, 2.12, 3.05, 0.0], h), {"C": (2.12, "?"), "B": (3.05, "?")})
        h.append(("D", 950))
        self.check(self.step(t, [0.5, 0.0, 2.12, 3.05, 0.0], h), {"D": (0.5, "*"), "C": (2.12, "?"), "B": (3.05, "?")})
        # the guesses never promote themselves, however long they stay unchanged
        for _ in range(5):
            self.check(self.step(t, [0.5, 0.0, 2.12, 3.05, 0.0], h), {"D": (0.5, "*"), "C": (2.12, "?"), "B": (3.05, "?")})

    def test_no_holders_teaches_foreign_without_confirming_future_owners(self):
        self.make().update(RTX, at([0.4]), [], now=self.clock)  # nothing in this distro: 0.4 is foreign
        t = self.make()  # a fresh start in the same boot, with a job already running
        self.check(self.step(t, [0.0, 0.4, 2.0], [("S", 900), ("X", 5)]), {"X": (2.0, "?")})
        t = self.make(boot="boot-2")  # a new boot knows nothing
        self.check(self.step(t, [0.0, 0.4, 2.0], [("S", 900), ("X", 5)]), {"X": (0.4, "?")})

    def test_context_that_lags_its_process_is_still_pinned(self):
        t = self.make()
        self.step(t, [0.0], [("S", 900)])
        h = [("S", 900), ("P", 950)]
        self.check(self.step(t, [0.0], h), {}, reason="without a VM context")
        self.check(self.step(t, [0.8, 0.0], h), {"P": (0.8, "*")})

    def test_exited_process_context_is_not_handed_to_a_newcomer(self):
        t = self.make()
        self.step(t, [0.0], [("S", 900)])
        self.step(t, [1.19, 0.0], [("S", 900), ("A", 950)])
        # A exits and B starts; the host sample still shows A's context, B's hasn't appeared yet
        h = [("S", 900), ("B", 960)]
        guesses, _ = self.step(t, [1.19, 0.0], h)
        self.assertNotIn("B", guesses)
        self.check(self.step(t, [2.5, 0.0], h), {"B": (2.5, "*")})

    def test_model_loading_keeps_its_context(self):
        t = self.make()
        h = [("S", 900)]
        self.step(t, [0.0], h)
        h.append(("L", 950))
        self.check(self.step(t, [0.3, 0.0], h), {"L": (0.3, "*")})
        for gib in (2.5, 6.0, 11.0):  # GBs per refresh while weights upload
            self.check(self.step(t, [gib, 0.0], h), {"L": (gib, "*")})

    def test_context_without_a_new_process_is_foreign(self):
        t = self.make()
        h = [("S", 900)]
        self.step(t, [0.0], h)
        h.append(("A", 950))
        self.step(t, [1.0, 0.0], h)
        # another distro / Docker starts something: a new context, but no new Linux process here
        self.check(self.step(t, [0.7, 1.0, 0.0], h), {"A": (1.0, "*")})
        h.append(("B", 990))
        self.check(self.step(t, [2.0, 0.7, 1.0, 0.0], h), {"A": (1.0, "*"), "B": (2.0, "*")})

    def test_equal_sizes_are_sure_even_when_order_is_not(self):
        t = self.make()
        self.check(self.step(t, [0.0, 0.4, 0.4], [("S", 900), ("F1", 5), ("F2", 6)]),
                   {"F1": (0.4, "*"), "F2": (0.4, "*")})

    def test_nearly_equal_sizes_stay_guesses(self):
        t = self.make()  # 0.40 vs 0.42 GiB: within tolerance, but swapping owners changes the shown bytes
        self.check(self.step(t, [0.0, 0.4, 0.42], [("S", 900), ("F1", 5), ("F2", 6)]),
                   {"F1": (0.42, "?"), "F2": (0.4, "?")})

    def test_different_adapter_resets_history(self):
        self.make().update(RTX, at([0.4]), [], now=self.clock)
        t = self.make()
        self.check(self.step(t, [0.0, 0.4, 2.0], [("S", 900), ("X", 5)]), {"X": (2.0, "?")})
        guesses, _ = t.update(0x1234, at([0.0, 0.4, 2.0]), [("S", 900), ("X", 5)], now=self.clock + 5)
        self.assertEqual(guesses["X"].bytes, 0.4 * GiB)
        self.assertEqual(guesses["X"].confidence, "order")


class FakeStream:
    def __init__(self, sample, status="ok"):
        self.sample, self.status = sample, status
        self.closed = False

    @property
    def ended(self):
        return self.status.startswith("unavailable")

    def latest(self):
        return self.sample

    def close(self):
        self.closed = True


class HostRowTests(unittest.TestCase):
    def collector(self, stream):
        c = Collector.__new__(Collector)  # only _add_host_rows is exercised
        c.host = stream
        c._host_luid = None
        c._host_restarted = time.time()
        c.tracker = ContextTracker(path=os.path.join(self.tmp.name, "ctx.json"), boot_id="boot", self_pid=os.getpid())
        c._own_start = time.time()
        return c

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)

    def snap(self, used=3.1 * GiB):
        linux = ProcRow(2744, "python3", "AI", ProcGpu(attributed=False), ProcDetail(2744, "python3"))
        return Snapshot(taken=1, adapter=None, adapters=[], nv=None, driver="", cuda="", vram_total=24 * GiB,
                        vram_used=int(used), shared_used=0, backend="wsl", procs=[linux], host_status="")

    def test_wsl_vm_and_windows_rows(self):
        snap = self.snap()
        self.collector(FakeStream(parse_line(SAMPLE, now=time.time())))._add_host_rows(snap)
        self.assertEqual(snap.host_status, "ok")
        rows = {r.key: r for r in snap.procs}
        # Linux PID 2744 and Windows dwm PID 2744 coexist: distinct keys
        self.assertEqual(set(rows), {"2744", "w2744", "w24820"})
        vm = rows["w24820"]
        self.assertEqual((vm.name, vm.kind, vm.gpu.dedicated), ("WSL VM (vmwp.exe)", "wsl-vm", 2.4 * GiB))
        self.assertEqual((rows["w2744"].name, rows["w2744"].kind), ("dwm.exe", "compositor"))
        self.assertEqual(snap.shared_used, 0)

        text = to_text(snap)
        self.assertIn("W:24820", text)
        self.assertIn("WSL VM (vmwp.exe)", text)
        self.assertIn("exact total", text)
        data = to_dict(snap)
        self.assertEqual({p["host"] for p in data["processes"]}, {None, "windows"})
        self.assertEqual(data["host_status"], "ok")

    def test_linux_process_gets_matched_vram(self):
        # contexts newest first: this monitor's own 0-byte NVML context, then the workload (3 GB)
        busy = line([["local usage", f"pid_24820_{LUID}", v * GiB] for v in (0.0, 3.0)]
                    + [["dedicated usage", LUID, 3.1 * GiB]], {"24820": "vmwp"})
        c = self.collector(FakeStream(parse_line(busy, now=time.time())))
        snap = self.snap()
        c._add_host_rows(snap)
        row = next(r for r in snap.procs if not r.host)
        self.assertEqual((row.gpu.inferred, row.gpu.inferred_by), (3.0 * GiB, "consistent"))
        self.assertFalse(row.gpu.attributed)  # engines are still unknown
        row.llm = LlmView()
        text = to_text(snap)
        self.assertIn("3.00 GB*", text)
        self.assertIn("== python3 (2744)  VRAM 3.00 GB*", text)
        self.assertIn("* matched", text)
        data = to_dict(snap)["processes"][1]
        self.assertEqual((data["vram_inferred_bytes"], data["vram_inferred_by"]), (3.0 * GiB, "consistent"))
        self.assertEqual(snap.attributed, 3 * GiB)  # the Linux child is not added to the VM total again

    def test_current_inferred_vram_sets_first_sample_scan_priority(self):
        big_pid = Collector.DEEP_TOP_N + 1
        total = 8 * GiB + Collector.DEEP_TOP_N * MiB
        busy = line([["local usage", f"pid_24820_{LUID}", v]
                     for v in [0, 8 * GiB] + [MiB] * Collector.DEEP_TOP_N]
                    + [["dedicated usage", LUID, total]], {"24820": "vmwp"})
        c = self.collector(FakeStream(parse_line(busy, now=time.time())))
        c.adapter, c.adapters, c.gpu_index, c.scan_budget = None, [], 0, None
        c.host_status = ""
        c.nvml = Mock(ok=False, driver="", cuda_version="")
        c.counters = Mock(outside={})
        c.counters.sample.return_value = (
            {RTX: {pid: ProcGpu(attributed=False) for pid in range(1, big_pid + 1)}},
            {RTX: AdapterGpu(dedicated=total)})
        details = {pid: ProcDetail(pid, "python3", create_time=pid, parents=[])
                   for pid in range(1, big_pid + 1)}
        c.inspector = Mock()
        c.inspector.get.side_effect = details.__getitem__
        c.inspector.needs_scan.return_value = True

        def scan(d, mem):
            if d.pid == big_pid:
                d.models = [MappedModel("large.gguf", 8 * GiB, 0, True, kind="LLM")]

        c.inspector.scan.side_effect = scan
        c.prober = Mock()
        c.prober.get.return_value = None
        with patch.object(collector_mod, "BACKEND", "wsl"), \
                patch.object(collector_mod, "build_view", return_value=LlmView()) as view:
            snap = c.sample()
            self.assertEqual(c.inspector.scan.call_count, Collector.DEEP_TOP_N)
            self.assertEqual(c.inspector.scan.call_args_list[0].args, (details[big_pid], 8 * GiB))
            self.assertNotIn(Collector.DEEP_TOP_N, [call.args[0].pid for call in c.inspector.scan.call_args_list])
            self.assertEqual([r.pid for r in snap.procs[:2]], [24820, big_pid])
            self.assertEqual(snap.procs[1].kind, "LLM")  # classification follows the scan
            self.assertEqual(view.call_args.args[1], 8 * GiB)
            self.assertTrue(view.call_args.kwargs["attributed"])
            self.assertEqual(snap.attributed, total)
            c.scan_budget = 0.0
            c.inspector.scan.reset_mock()
            c.sample()
            c.inspector.scan.assert_not_called()

    def test_order_guess_is_marked(self):
        # an extra (foreign) context and no history: placement is a guess
        busy = line([["local usage", f"pid_24820_{LUID}", v * GiB] for v in (0.0, 3.0, 0.4)]
                    + [["dedicated usage", LUID, 3.1 * GiB]], {"24820": "vmwp"})
        c = self.collector(FakeStream(parse_line(busy, now=time.time())))
        snap = self.snap()
        c._add_host_rows(snap)
        row = next(r for r in snap.procs if not r.host)
        self.assertEqual(row.gpu.inferred_by, "order")
        self.assertIn("3.00 GB?", to_text(snap))
        self.assertIn("order guess", to_text(snap))

    def test_split_reason_is_reported(self):
        snap = self.snap()  # SAMPLE has 2 vmwp contexts; add Linux rows until there are more holders
        snap.procs += [ProcRow(p, "python3", "AI", ProcGpu(attributed=False), ProcDetail(p, "python3"))
                       for p in (2745, 2746)]
        self.collector(FakeStream(parse_line(SAMPLE, now=time.time())))._add_host_rows(snap)
        self.assertIn("without a VM context", snap.wsl_split)
        self.assertIn("Linux VRAM split:", to_text(snap))

    def test_unmatched_adapter_adds_nothing(self):
        snap = self.snap(used=12 * GiB)
        c = self.collector(FakeStream(parse_line(SAMPLE, now=time.time())))
        c._add_host_rows(snap)
        self.assertEqual(snap.host_status, "starting")  # retried for a while first
        c._host_restarted -= 60
        snap = self.snap(used=12 * GiB)
        c._add_host_rows(snap)
        self.assertTrue(snap.host_status.startswith("unmatched"))
        self.assertEqual(len(snap.procs), 1)
        self.assertIn("unmatched", to_text(snap))

    def test_adapter_match_is_kept_while_memory_moves(self):
        c = self.collector(FakeStream(parse_line(SAMPLE, now=time.time())))
        c._add_host_rows(self.snap())
        moving = self.snap(used=12 * GiB)  # a model loading between the NVML and host readings
        c._add_host_rows(moving)
        self.assertEqual(moving.host_status, "ok")
        self.assertEqual(len(moving.procs), 3)

    def test_late_samples_stay_visible_and_say_so(self):
        snap = self.snap()
        self.collector(FakeStream(parse_line(SAMPLE, now=time.time() - 8)))._add_host_rows(snap)
        self.assertTrue(snap.host_data)
        self.assertIn("8s old", snap.host_status)
        self.assertEqual(len(snap.procs), 3)

    def test_ended_stream_is_restarted(self):
        dead = FakeStream(None, status="unavailable (counter stream ended)")
        c = self.collector(dead)
        c._host_restarted = time.time() - 60
        with patch.object(collector_mod, "HostStream", return_value=FakeStream(None, "starting")) as stream:
            c._add_host_rows(self.snap())
        self.assertTrue(dead.closed)
        stream.assert_called_once()

    def test_no_sample_yet_is_starting(self):
        snap = self.snap()
        self.collector(FakeStream(None, status="starting"))._add_host_rows(snap)
        self.assertEqual(snap.host_status, "starting")

    def test_interop_off_and_flag_off_never_start_powershell(self):
        with patch.object(collector_mod, "BACKEND", "wsl"), patch.object(collector_mod, "enumerate_adapters",
                                                                          return_value=[]), \
                patch.object(collector_mod, "GpuCounters"), patch.object(collector_mod, "Nvml"), \
                patch.object(collector_mod, "HostStream") as stream:
            with patch.object(collector_mod, "interop_enabled", return_value=False):
                self.assertEqual(Collector().host_status, "off (WSL interop is disabled)")
            with patch.object(collector_mod, "interop_enabled", return_value=True):
                self.assertEqual(Collector(host=False).host_status, "off (--no-host)")
                stream.assert_not_called()
                Collector()
                stream.assert_called_once()


if __name__ == "__main__":
    unittest.main()

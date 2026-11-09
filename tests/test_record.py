"""gputaskman record: one --json snapshot per interval, appended to a JSON Lines file.

Hardware-independent: samples are fakes, time is a fake clock and the collectors are mocks. A recording is the
same data as --json, so unknown stays null (never 0) and every line must be one complete JSON object.
"""

from __future__ import annotations

import contextlib
import errno
import io
import json
import os
import tempfile
import types
import unittest
from unittest.mock import MagicMock, patch

from gputaskman import collector as collector_module
from gputaskman import record
from gputaskman.cli import main
from gputaskman.collector import ProcRow, Snapshot
from gputaskman.diff import diff, load
from gputaskman.dxgi import Adapter
from gputaskman.host import HostProc, HostSample
from gputaskman.host import to_dict as host_to_dict
from gputaskman.pdh import ProcGpu
from gputaskman.procinfo import ProcDetail
from gputaskman.report import dump, to_dict, to_json_dict

GiB = 1 << 30


def row(pid: int, name: str, kind: str = "other", dedicated: int = GiB) -> ProcRow:
    return ProcRow(pid, name, kind, ProcGpu(dedicated=dedicated), ProcDetail(pid, name))


def snapshot(taken: float = 1.0, used: int = 10 * GiB) -> Snapshot:
    return Snapshot(taken=taken, adapter=None, adapters=[], nv=None, driver="", cuda="",
                    vram_total=16 * GiB, vram_used=used, shared_used=GiB,
                    procs=[row(10, "llama-server.exe", "LLM", 8 * GiB), row(20, "dwm.exe", "compositor", 2 * GiB)])


def host_sample() -> HostSample:
    return HostSample(cpu_pct=37.0, per_core=[10.0, 90.0], logical=2, physical=2, ram_total=32 * GiB,
                      ram_used=12 * GiB, ram_available=20 * GiB, swap_total=4 * GiB, swap_used=GiB,
                      procs=[HostProc(10, "llama-server.exe", 6 * GiB, 5 * GiB, 12.0)])


class Clock:
    """Time only moves when something sleeps or a sample takes time."""

    def __init__(self):
        self.now = 1000.0
        self.sleeps = []

    def __call__(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds


class Sink:
    """A file that keeps what was flushed apart from what is only buffered."""

    def __init__(self):
        self.pending = self.flushed = ""

    def write(self, text: str) -> None:
        self.pending += text

    def flush(self) -> None:
        self.flushed, self.pending = self.flushed + self.pending, ""

    def lines(self) -> list[str]:
        return self.flushed.splitlines()


class RunTests(unittest.TestCase):
    def setUp(self):
        self.clock = Clock()
        self.sink = Sink()

    def run_for(self, sample, **kwargs) -> int:
        return record.run(sample, self.sink, clock=self.clock, sleep=self.clock.sleep, **kwargs)

    def test_one_compact_json_line_per_sample_flushed_as_it_goes(self):
        seen = []

        def sample():
            seen.append(len(self.sink.lines()))  # lines already flushed when this sample is taken
            return {"taken": len(seen), "vram_used": None}

        self.assertEqual(self.run_for(sample, interval=1.0, duration=3.0), 3)
        self.assertEqual(seen, [0, 1, 2])
        self.assertEqual(self.sink.pending, "")
        self.assertEqual(self.sink.lines(), ['{"taken":1,"vram_used":null}', '{"taken":2,"vram_used":null}',
                                             '{"taken":3,"vram_used":null}'])

    def test_a_duration_sets_the_sample_count_and_nothing_waits_after_the_last_one(self):
        self.assertEqual(self.run_for(lambda: {}, interval=1.0, duration=300.0), 300)
        self.assertEqual(len(self.clock.sleeps), 299)
        self.assertEqual(self.clock.now - 1000.0, 299.0)

    def test_a_duration_shorter_than_the_interval_records_one_sample(self):
        self.assertEqual(self.run_for(lambda: {}, interval=5.0, duration=1.0), 1)
        self.assertEqual(self.clock.sleeps, [])

    def test_samples_keep_their_schedule_however_long_each_one_takes(self):
        starts = []

        def sample():
            starts.append(self.clock.now - 1000.0)
            self.clock.now += 0.25  # collecting takes time too
            return {}

        self.run_for(sample, interval=1.0, duration=4.0)
        self.assertEqual(starts, [0.0, 1.0, 2.0, 3.0])  # not 0, 1.25, 2.5 ...

    def test_a_slow_sample_delays_the_next_one_with_no_burst_to_catch_up(self):
        starts, took = [], iter([0.0, 3.5, 0.0, 0.0])

        def sample():
            starts.append(self.clock.now - 1000.0)
            self.clock.now += next(took)
            return {}

        self.assertEqual(self.run_for(sample, interval=1.0, duration=6.0), 4)
        self.assertEqual(starts, [0.0, 1.0, 4.5, 5.5])

    def test_ctrl_c_while_waiting_leaves_every_line_complete(self):
        def interrupt(seconds):
            raise KeyboardInterrupt

        self.assertEqual(record.run(lambda: {"n": 1}, self.sink, interval=1.0, clock=self.clock, sleep=interrupt), 1)
        self.assertEqual(self.sink.flushed, '{"n":1}\n')

    def test_ctrl_c_during_a_sample_writes_nothing_for_it(self):
        calls = []

        def sample():
            calls.append(1)
            if len(calls) == 3:
                raise KeyboardInterrupt
            return {"n": len(calls)}

        self.assertEqual(self.run_for(sample, interval=1.0), 2)  # with no duration only Ctrl+C ends it
        self.assertEqual(self.sink.flushed, '{"n":1}\n{"n":2}\n')
        self.assertEqual(self.sink.pending, "")

    def test_each_record_is_one_ascii_line_even_with_odd_names_and_newlines(self):
        data = {"name": "mödel-日本.exe", "cmd": "a\nb\r\nc", "path": "C:\\Users\\x", "unknown": None}
        self.run_for(lambda: data, interval=1.0, duration=0.5)
        self.assertTrue(self.sink.flushed.isascii())
        self.assertEqual(self.sink.flushed.count("\n"), 1)
        self.assertEqual(json.loads(self.sink.flushed), data)

    def test_a_value_json_cannot_encode_is_written_as_its_text_not_a_traceback(self):
        class Odd:
            def __str__(self):
                return "odd value"

        self.run_for(lambda: {"n": 1, "odd": Odd()}, interval=1.0, duration=0.5)
        self.assertEqual(json.loads(self.sink.flushed), {"n": 1, "odd": "odd value"})

    def test_a_full_disk_stops_the_recording_with_its_error_and_keeps_the_earlier_lines(self):
        class FullDisk(Sink):
            def write(self, text: str) -> None:
                if len(self.lines()) == 2:
                    raise OSError(errno.ENOSPC, "No space left on device")
                super().write(text)

        sink = FullDisk()
        with self.assertRaises(OSError) as stop:
            record.run(lambda: {"n": 1}, sink, interval=1.0, clock=self.clock, sleep=self.clock.sleep)
        self.assertEqual(stop.exception.errno, errno.ENOSPC)
        self.assertEqual(sink.lines(), ['{"n":1}', '{"n":1}'])


class MakeSampleTests(unittest.TestCase):
    def test_a_record_is_the_object_json_prints(self):
        sample = record.make_sample(snapshot, host_sample)
        self.assertEqual(json.loads(json.dumps(sample(), default=str)),
                         json.loads(dump(snapshot(), True, host_sample())))

    def test_system_figures_are_optional_and_their_failure_is_recorded_not_fatal(self):
        def broken():
            raise OSError("layout not recognised")

        failed = record.make_sample(snapshot, broken)()
        self.assertEqual(failed["system"], {"error": "OSError: layout not recognised"})
        self.assertEqual(failed["processes"], to_dict(snapshot())["processes"])  # the GPU part is intact
        self.assertNotIn("system", record.make_sample(snapshot)())

    def test_a_gpu_that_is_gone_is_recorded_as_unavailable_not_zero(self):
        from gputaskman.recovery import LiveCollectors

        live = LiveCollectors(0, collector_factory=MagicMock(), discover=lambda strict=True: [])
        data = record.make_sample(lambda: live.sample()[0])()
        self.assertIn("Adapter unavailable", data["counter_status"])
        self.assertIs(data["counter_adapter_matched"], False)
        for key in ("vram_total", "vram_used", "shared_used", "attributed", "device"):
            self.assertIsNone(data[key], key)
        self.assertEqual(data["processes"], [])


class JsonDumpTests(unittest.TestCase):
    def test_dump_json_is_still_the_indented_object_with_its_optional_system_section(self):
        s, h = snapshot(), host_sample()
        self.assertEqual(dump(s, True), json.dumps(to_dict(s), indent=2, default=str))
        self.assertEqual(dump(s, True, h), json.dumps({**to_dict(s), "system": host_to_dict(h)}, indent=2, default=str))
        self.assertEqual(dump(s, True, None, "OSError: boom"),
                         json.dumps({**to_dict(s), "system": {"error": "OSError: boom"}}, indent=2, default=str))
        self.assertEqual(to_json_dict(s, h), {**to_dict(s), "system": host_to_dict(h)})
        self.assertNotIn("system", to_json_dict(s))


class FileTestCase(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.dir.cleanup)
        self.path = os.path.join(self.dir.name, "session.jsonl")

    def content(self) -> bytes:
        with open(self.path, "rb") as f:
            return f.read()


class AppendTests(FileTestCase):
    def append(self, text: str) -> None:
        with record.append_to(self.path) as out:
            out.write(text)

    def test_creates_the_file_and_appends_without_touching_what_is_there(self):
        self.append('{"n":1}\n')
        self.append('{"n":2}\n')
        self.assertEqual(self.content(), b'{"n":1}\n{"n":2}\n')

    def test_lines_end_with_a_bare_newline_on_every_platform(self):
        self.append('{"n":1}\n')
        self.assertNotIn(b"\r", self.content())

    def test_a_last_line_cut_short_by_a_crash_is_closed_before_the_next_record(self):
        with open(self.path, "wb") as f:
            f.write(b'{"n":1}\n{"n":2')
        self.append('{"n":3}\n')
        self.assertEqual(self.content(), b'{"n":1}\n{"n":2\n{"n":3}\n')

    def test_an_empty_file_gets_no_stray_newline(self):
        open(self.path, "wb").close()
        self.append('{"n":1}\n')
        self.assertEqual(self.content(), b'{"n":1}\n')

    def test_a_path_that_cannot_be_written_raises_for_the_cli_to_report(self):
        with self.assertRaises(OSError):
            record.append_to(self.dir.name)  # a directory

    def test_a_recording_written_to_disk_is_valid_json_lines_that_diff_accepts(self):
        snapshots = iter([snapshot(taken=1.0, used=4 * GiB), snapshot(taken=2.0, used=6 * GiB),
                          snapshot(taken=3.0, used=9 * GiB)])
        clock = Clock()
        with record.append_to(self.path) as out:
            count = record.run(record.make_sample(lambda: next(snapshots), host_sample), out,
                               interval=1.0, duration=3.0, clock=clock, sleep=clock.sleep)
        self.assertEqual(count, 3)
        with open(self.path, encoding="utf-8") as f:
            records = [json.loads(line) for line in f]
        self.assertEqual([r["taken"] for r in records], [1.0, 2.0, 3.0])
        self.assertTrue(all(r["system"]["ram"]["total"] == 32 * GiB for r in records))
        first, last = (os.path.join(self.dir.name, name) for name in ("first.json", "last.json"))
        for target, line in ((first, records[0]), (last, records[-1])):
            with open(target, "w", encoding="utf-8") as f:
                f.write(json.dumps(line))  # one line saved to its own file is a normal --json snapshot
        self.assertEqual(diff(load(first), load(last))["device"]["vram_used"]["delta"], 5 * GiB)


class SourceTests(unittest.TestCase):
    def setUp(self):
        self.sleep = patch("gputaskman.record.time.sleep").start()
        self.sampler = patch("gputaskman.host.HostSampler").start()
        self.addCleanup(patch.stopall)
        self.sampler.return_value.sample.return_value = host_sample()

    def collector(self) -> MagicMock:
        c = MagicMock()
        c.host = None
        c.counters = types.SimpleNamespace()  # no warm_up, like Windows and native Linux
        c.sample.return_value = snapshot()
        return c

    def test_native_linux_samples_one_collector_with_budgeted_scanning_and_closes_it(self):
        c = self.collector()
        with patch("gputaskman.collector.BACKEND", "nvml"), \
                patch("gputaskman.collector.Collector", return_value=c) as factory:
            with record.source(2) as sample:
                data = sample()
        factory.assert_called_once_with(2, host=True)  # the TUI's budget, not scan_budget=None
        c.close.assert_called_once_with()
        self.sleep.assert_called_once_with(1.0)  # engine utilization is a rate
        self.assertEqual(data["taken"], 1.0)
        self.assertEqual(self.sampler.return_value.sample.call_count, 2)  # the first reading only primes CPU%
        self.assertEqual(data["system"]["ram"]["total"], 32 * GiB)

    def test_no_host_is_passed_on(self):
        with patch("gputaskman.collector.BACKEND", "wsl"), \
                patch("gputaskman.collector.Collector", return_value=self.collector()) as factory:
            with record.source(0, host=False):
                pass
        factory.assert_called_once_with(0, host=False)

    def test_wsl_host_counters_and_macos_footprints_get_their_warm_up(self):
        c = self.collector()
        c.host, c.counters = MagicMock(), MagicMock()
        with patch("gputaskman.collector.BACKEND", "wsl"), patch("gputaskman.collector.Collector", return_value=c):
            with record.source(0):
                pass
        c.host.wait.assert_called_once_with(15.0)
        c.counters.warm_up.assert_called_once_with(15.0)

    def test_windows_rediscovers_a_gpu_that_was_disabled_and_records_the_selected_slot(self):
        with patch("gputaskman.collector.BACKEND", "wddm"), patch("gputaskman.recovery.LiveCollectors") as live:
            live.return_value.sample.return_value = [snapshot(taken=1.0), snapshot(taken=2.0)]
            with record.source(1, system=False) as sample:
                data = sample()
        args, kwargs = live.call_args
        self.assertEqual(args, (1,))
        self.assertIs(kwargs["collector_factory"], collector_module.Collector)
        self.assertIs(kwargs["discover"], collector_module.enumerate_adapters)
        live.return_value.close.assert_called_once_with()
        self.assertEqual(data["taken"], 1.0)
        self.assertNotIn("system", data)
        self.sampler.assert_not_called()

    def test_collectors_are_closed_even_when_recording_fails(self):
        c = self.collector()
        with patch("gputaskman.collector.BACKEND", "nvml"), patch("gputaskman.collector.Collector", return_value=c):
            with self.assertRaises(RuntimeError), record.source(0):
                raise RuntimeError("disk full")
        c.close.assert_called_once_with()

    def test_a_broken_system_sampler_never_stops_the_gpu_records(self):
        self.sampler.side_effect = OSError("layout not recognised")
        with patch("gputaskman.collector.BACKEND", "nvml"), \
                patch("gputaskman.collector.Collector", return_value=self.collector()):
            with record.source(0) as sample:
                data = sample()
        self.assertEqual(data["system"], {"error": "OSError: layout not recognised"})
        self.assertEqual(len(data["processes"]), 2)

    def test_one_system_sampler_serves_every_record_because_cpu_percent_is_a_rate_between_readings(self):
        with patch("gputaskman.collector.BACKEND", "nvml"), \
                patch("gputaskman.collector.Collector", return_value=self.collector()):
            with record.source(0) as sample:
                sample()
                sample()
        self.sampler.assert_called_once_with()
        self.assertEqual(self.sampler.return_value.sample.call_count, 3)  # one priming read, then one per record

    def test_no_system_never_builds_a_sampler_and_leaves_no_section(self):
        with patch("gputaskman.collector.BACKEND", "nvml"), \
                patch("gputaskman.collector.Collector", return_value=self.collector()):
            with record.source(0, system=False) as sample:
                data = sample()
        self.sampler.assert_not_called()
        self.assertNotIn("system", data)


class CliTests(FileTestCase):
    def setUp(self):
        super().setUp()
        self.calls = []
        self.samples = 0
        self.interrupt_at = None

    def fake_source(self, fail=None):
        calls = self.calls

        @contextlib.contextmanager
        def source(gpu, *, host, system):
            calls.append((gpu, host, system))
            if fail is not None:
                raise fail

            def sample():
                self.samples += 1
                if self.samples == self.interrupt_at:
                    raise KeyboardInterrupt
                if self.samples > 5:  # a recording that ignores its --duration must fail the test, not hang it
                    raise AssertionError("the recording ran on past what the test asked for")
                return to_json_dict(snapshot(taken=float(self.samples)))

            yield sample

        return patch("gputaskman.record.source", source)

    def refused(self):
        """A source that fails the test at once: a recording must never start for input the command refuses."""
        return self.fake_source(fail=AssertionError("a recording started for input that must be refused"))

    def run_cli(self, *args, fail=None) -> str:
        err = io.StringIO()
        with self.fake_source(fail), patch("sys.argv", ["gputaskman", "record", *args]), \
                contextlib.redirect_stderr(err):
            main()
        return err.getvalue()

    def lines(self) -> list[str]:
        return self.content().decode("utf-8").splitlines()

    def test_records_a_snapshot_and_says_so_on_stderr(self):
        err = self.run_cli("-o", self.path, "--duration", "0.5")
        self.assertEqual(len(self.lines()), 1)
        self.assertEqual(json.loads(self.lines()[0]), json.loads(dump(snapshot(taken=1.0), True)))
        self.assertEqual(self.calls, [(0, True, True)])
        self.assertIn("Recording GPU 0 every 1 s for 0.5 s", err)
        self.assertIn("Recorded 1 sample(s)", err)

    def test_the_flags_reach_the_sampling_side(self):
        self.run_cli("--output", self.path, "--gpu", "0", "--interval", "2", "--duration", "1", "--no-host", "--no-system")
        self.assertEqual(self.calls, [(0, False, False)])

    def test_every_value_reaches_the_side_that_uses_it(self):
        adapters = [Adapter(1, "GPU 0", 0x10DE, GiB, 0, False), Adapter(2, "GPU 1", 0x10DE, GiB, 0, False)]
        for name, args, source_args, run_kwargs in (
                ("given", ["--gpu", "1", "--interval", "2", "--duration", "1", "--no-host"],
                 (1, False, True), {"interval": 2.0, "duration": 1.0}),
                ("defaults", [], (0, True, True), {"interval": 1.0, "duration": None})):
            with self.subTest(name):
                self.calls.clear()
                with patch("gputaskman.collector.enumerate_adapters", return_value=adapters), \
                        patch("gputaskman.record.run", return_value=1) as run:
                    self.run_cli("-o", self.path, *args)
                self.assertEqual(self.calls, [source_args])
                self.assertEqual(run.call_args.kwargs, run_kwargs)

    def test_a_second_recording_appends_and_a_cut_last_line_is_closed_first(self):
        with open(self.path, "wb") as f:
            f.write(b'{"taken":0.5}\n{"taken"')
        self.run_cli("-o", self.path, "--duration", "0.5")
        self.run_cli("-o", self.path, "--duration", "0.5")
        lines = self.lines()
        self.assertEqual(lines[:2], ['{"taken":0.5}', '{"taken"'])
        self.assertEqual([json.loads(line)["taken"] for line in lines[2:]], [1.0, 2.0])  # no blank lines in between

    def test_ctrl_c_ends_a_recording_cleanly_and_keeps_what_was_written(self):
        self.interrupt_at = 3
        err = self.run_cli("-o", self.path, "-i", "0.01")
        self.assertEqual([json.loads(line)["taken"] for line in self.lines()], [1.0, 2.0])
        self.assertIn("Recorded 2 sample(s)", err)

    def test_ctrl_c_while_starting_up_exits_quietly(self):
        err = self.run_cli("-o", self.path, fail=KeyboardInterrupt())
        self.assertIn("Recorded 0 sample(s)", err)
        self.assertEqual(self.content(), b"")

    def test_bad_arguments_are_refused_before_anything_is_opened(self):
        cases = {
            "--interval 0": ["-o", self.path, "--interval", "0"],
            "--interval -1": ["-o", self.path, "--interval", "-1"],
            "--interval nan": ["-o", self.path, "--interval", "nan"],
            "--interval inf": ["-o", self.path, "--interval", "inf"],
            "--duration 0": ["-o", self.path, "--duration", "0"],
            "--duration -1": ["-o", self.path, "--duration", "-1"],
            "--duration nan": ["-o", self.path, "--duration", "nan"],
            "--duration inf": ["-o", self.path, "--duration", "inf"],
            "--gpu -1": ["-o", self.path, "--gpu", "-1"],
            "no output": [],
        }
        for name, args in cases.items():
            with self.subTest(name):
                err = io.StringIO()
                with self.refused(), patch("sys.argv", ["gputaskman", "record", *args]), \
                        patch("gputaskman.collector.enumerate_adapters",
                              side_effect=AssertionError("no GPU may be touched while refusing the arguments")), \
                        contextlib.redirect_stderr(err), self.assertRaises(SystemExit) as stop:
                    main()
                self.assertEqual(stop.exception.code, 2)
                self.assertEqual(self.calls, [])
                self.assertFalse(os.path.exists(self.path))

    def test_a_gpu_that_does_not_exist_is_refused_up_front(self):
        err = io.StringIO()
        with self.refused(), patch("gputaskman.collector.enumerate_adapters", return_value=[]), \
                patch("sys.argv", ["gputaskman", "record", "-o", self.path, "--gpu", "2"]), \
                contextlib.redirect_stderr(err), self.assertRaises(SystemExit) as stop:
            main()
        self.assertEqual(stop.exception.code, 2)
        self.assertIn("--gpu 2 is out of range", err.getvalue())
        self.assertEqual(self.calls, [])
        self.assertFalse(os.path.exists(self.path))

    def test_an_output_that_cannot_be_written_is_refused(self):
        err = io.StringIO()
        with self.refused(), patch("sys.argv", ["gputaskman", "record", "-o", self.dir.name]), \
                contextlib.redirect_stderr(err), self.assertRaises(SystemExit) as stop:
            main()
        self.assertEqual(stop.exception.code, 2)
        self.assertIn("cannot write", err.getvalue())
        self.assertEqual(self.calls, [])

    def test_help_finds_record_from_the_main_command(self):
        for argv, expected in ((["gputaskman", "--help"], "gputaskman record -o FILE"),
                               (["gputaskman", "record", "--help"], "--duration")):
            out = io.StringIO()
            with patch("sys.argv", argv), contextlib.redirect_stdout(out), self.assertRaises(SystemExit):
                main()
            self.assertIn(expected, out.getvalue())


if __name__ == "__main__":
    unittest.main()

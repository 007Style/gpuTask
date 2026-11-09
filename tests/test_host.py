import contextlib
import os
import struct
import sys
import unittest
from unittest.mock import patch

import psutil

from gputaskman import host

SECOND = 10_000_000  # one second of CPU time, in the 100 ns units NtQuerySystemInformation reports


def sample(**override) -> host.HostSample:
    fields = dict(cpu_pct=5, per_core=[5], logical=1, physical=1, ram_total=100, ram_used=40,
                  ram_available=60, swap_total=0, swap_used=0, procs=[])
    return host.HostSample(**{**fields, **override})


class Clock:
    """Stands in for the time module inside host.py, so rates come out exact."""

    def __init__(self) -> None:
        self.now = 1000.0

    def perf_counter(self) -> float:
        return self.now


class SampleTests(unittest.TestCase):
    def test_unknown_is_not_zero(self):
        h = sample(procs=[host.HostProc(1, "a", None, None, None), host.HostProc(2, "b", 10, 5, 0.0)])
        self.assertEqual([p.pid for p in h.top("rss")], [2])  # the unreadable process is not ranked as 0
        self.assertEqual([p.pid for p in h.top("cpu_pct")], [2])  # a measured 0.0 is ranked, unknown is not
        self.assertIsNone(h.proc(1).rss)
        self.assertIsNone(h.proc(None))
        d = host.to_dict(h)
        self.assertEqual([p["pid"] for p in d["top_ram"]], [2])
        self.assertEqual(d["top_cpu"][0]["cpu_pct"], 0.0)
        self.assertNotIn("mhz", d["cpu"])  # psutil reports a nominal frequency, not a measurement

    def test_a_pid_taken_by_another_process_is_not_returned(self):
        h = sample(procs=[host.HostProc(7, "new.exe", 10, 5, 1.0, 6000.0), host.HostProc(8, "p.exe", 10, 5, 1.0, 5000.0),
                          host.HostProc(9, "q.exe", 10, 5, 1.0, None)])
        self.assertIsNone(h.proc(7, 5000.0))  # pid 7 now belongs to a process that started later
        self.assertEqual(h.proc(7, 6000.0).name, "new.exe")
        self.assertEqual(h.proc(8, 5000.000002).name, "p.exe")  # psutil and the NT list differ by microseconds
        self.assertEqual(h.proc(7).name, "new.exe")  # a caller with no creation time (pid 0 or 4) keeps the pid match
        self.assertEqual(h.proc(9, 5000.0).name, "q.exe")  # a reading with no creation time cannot disagree

    def test_json_keeps_unknown_as_null_and_rounds_rates(self):
        h = sample(procs=[host.HostProc(2, "b", 10, None, 12.3456)])
        d = host.to_dict(h)
        self.assertEqual(d["top_ram"], [{"pid": 2, "name": "b", "rss": 10, "private": None, "cpu_pct": 12.3}])

    def test_text_labels_the_source_and_unknowns(self):
        h = sample(procs=[host.HostProc(2, "b", 10, None, None)], denied=3)
        text = host.to_text(h)
        self.assertIn("OS-reported; not GPU memory", text)
        self.assertIn("n/a", text)
        self.assertIn("3 processes not readable", text)


class NtPathTests(unittest.TestCase):
    """The Windows process-list path, driven by canned rows so every OS runs it."""

    def setUp(self):
        self.rows: dict = {}
        self.clock = Clock()
        for target in (patch.object(host, "_nt_query", lambda: dict(self.rows)),
                       patch.object(host, "_check_nt_layout"),
                       patch.object(host, "time", self.clock)):
            target.start()
            self.addCleanup(target.stop)
        self.sampler = host.HostSampler()
        self.sampler._logical = 4

    def read(self, rows: dict, seconds: float = 0) -> host.HostSample:
        self.rows = rows
        self.clock.now += seconds
        return self.sampler.sample()

    def test_cpu_is_a_rate_that_needs_two_readings(self):
        first = self.read({7: ("a.exe", 100, 50, 10 * SECOND, 5000.0)})
        self.assertIsNone(first.proc(7).cpu_pct)
        self.assertEqual(first.proc(7).created, 5000.0)  # lets the caller tell this process from a reuse of its pid
        second = self.read({7: ("a.exe", 100, 50, 12 * SECOND, 5000.0)}, seconds=4)
        # 2 s of CPU time in 4 s of wall clock on 4 logical CPUs: 12.5% of the whole machine
        self.assertAlmostEqual(second.proc(7).cpu_pct, 12.5)
        self.assertEqual((second.proc(7).rss, second.proc(7).private), (100, 50))
        self.assertEqual(second.denied, 0)  # no process handle is opened, so nothing is denied

    def test_new_and_reused_pids_have_no_rate(self):
        self.read({7: ("a.exe", 100, 50, 10 * SECOND, 5000.0)})
        second = self.read({7: ("b.exe", 100, 50, 11 * SECOND, 6000.0),  # pid 7 reused: new creation time
                            8: ("c.exe", 100, 50, SECOND, 6000.0)}, seconds=1)  # pid 8 appeared
        self.assertIsNone(second.proc(7).cpu_pct)
        self.assertIsNone(second.proc(8).cpu_pct)
        third = self.read({7: ("b.exe", 100, 50, 12 * SECOND, 6000.0)}, seconds=1)
        self.assertAlmostEqual(third.proc(7).cpu_pct, 25.0)  # the new process is rated against itself

    def test_cpu_time_going_backwards_is_unknown(self):
        self.read({7: ("a.exe", 100, 50, 10 * SECOND, 5000.0)})
        second = self.read({7: ("a.exe", 100, 50, 9 * SECOND, 5000.0)}, seconds=1)
        self.assertIsNone(second.proc(7).cpu_pct)

    def test_idle_process_is_not_a_process(self):
        h = self.read({0: ("System Idle Process", 8, 0, 900 * SECOND, 0.0), 7: ("a.exe", 100, 50, SECOND, 5000.0)})
        self.assertIsNone(h.proc(0))
        self.assertEqual([p.pid for p in h.procs], [7])

    def test_machine_without_a_usable_query_says_why(self):
        with patch.object(host, "_nt_query", host._nt_unavailable):
            with self.assertRaisesRegex(OSError, "64-bit Python on Windows"):
                host.HostSampler()


class LayoutCheckTests(unittest.TestCase):
    """Rows are parsed by byte offset, so they are checked against our own process before anything is shown."""

    def row(self, **change) -> dict:
        me = psutil.Process()
        fields = {"name": me.name(), "ws": me.memory_info().rss, "created": me.create_time(), **change}
        return {me.pid: (fields["name"], fields["ws"], 0, 0, fields["created"])}

    def test_a_row_that_matches_psutil_is_accepted(self):
        host._check_nt_layout(self.row())

    def test_a_row_that_disagrees_with_psutil_is_rejected(self):
        me = psutil.Process()
        for change in ({"name": "other.exe"}, {"created": me.create_time() + 60}, {"ws": 0},
                       {"ws": me.memory_info().rss * 10}):
            with self.subTest(change=change), self.assertRaisesRegex(OSError, "layout not recognised"):
                host._check_nt_layout(self.row(**change))

    def test_missing_own_process_is_rejected(self):
        with self.assertRaisesRegex(OSError, "layout not recognised"):
            host._check_nt_layout({})


class FakeProc:
    def __init__(self, pid, name="p.exe", created=1.0, rss=10, cpu=40.0, denied=False, gone=False):
        self.pid, self._name, self._created, self.rss = pid, name, created, rss
        self.cpu, self.denied, self.gone = cpu, denied, gone

    def oneshot(self):
        return contextlib.nullcontext()

    def name(self):
        if self.gone:
            raise psutil.NoSuchProcess(self.pid)
        return self._name

    def memory_info(self):
        if self.denied:
            raise psutil.AccessDenied(self.pid)
        return type("MemoryInfo", (), {"rss": self.rss})()

    def cpu_percent(self, interval=None):
        return self.cpu

    def create_time(self):
        return self._created


class PsutilPathTests(unittest.TestCase):
    """Linux, macOS and WSL read each process through psutil."""

    def setUp(self):
        self.procs: list = []
        for target in (patch.object(host, "_nt_query", None),
                       patch.object(host.psutil, "process_iter", lambda: iter(self.procs))):
            target.start()
            self.addCleanup(target.stop)
        self.sampler = host.HostSampler()
        self.sampler._logical = 4

    def test_cpu_is_a_share_of_the_machine_after_the_second_reading(self):
        self.procs = [FakeProc(1, cpu=100.0)]
        self.assertIsNone(self.sampler.sample().proc(1).cpu_pct)
        self.assertAlmostEqual(self.sampler.sample().proc(1).cpu_pct, 25.0)  # psutil's 100% is one CPU of 4

    def test_a_reused_pid_is_not_rated_against_the_previous_process(self):
        self.procs = [FakeProc(1, created=1.0, cpu=100.0)]
        self.sampler.sample()
        self.procs = [FakeProc(1, created=2.0, cpu=100.0)]
        self.assertIsNone(self.sampler.sample().proc(1).cpu_pct)

    def test_protected_and_vanished_processes_are_unknown_or_skipped(self):
        self.procs = [FakeProc(1, denied=True), FakeProc(2, gone=True), FakeProc(3, rss=7)]
        h = self.sampler.sample()
        self.assertEqual(h.denied, 1)
        self.assertIsNone(h.proc(1).rss)  # not readable: unknown, not 0
        self.assertIsNone(h.proc(1).created)
        self.assertIsNone(h.proc(2))
        self.assertEqual(h.proc(3).rss, 7)
        self.assertEqual(h.proc(3).created, 1.0)


class LiveTests(unittest.TestCase):
    @unittest.skipUnless(sys.platform == "win32" and struct.calcsize("P") == 8,
                         "NtQuerySystemInformation is 64-bit Windows only")
    def test_nt_query_matches_psutil(self):
        me = psutil.Process(os.getpid())
        name, ws, private, cpu, created = host._nt_query()[os.getpid()]
        mi = me.memory_info()
        self.assertEqual(name.lower(), me.name().lower())
        self.assertLess(abs(private - mi.private), 8 << 20)
        self.assertLess(abs(ws - mi.rss), 16 << 20)
        self.assertGreater(cpu, 0)
        self.assertLess(abs(created - me.create_time()), 2)

    def test_sampler_rates_cpu_after_second_sample(self):
        s = host.HostSampler()
        first = s.sample()
        self.assertTrue(all(p.cpu_pct is None for p in first.procs))  # a rate needs two readings
        second = s.sample()
        self.assertTrue(any(p.cpu_pct is not None for p in second.procs))
        self.assertTrue(0 <= second.cpu_pct <= 100)
        self.assertGreater(second.ram_used, 0)
        self.assertGreaterEqual(second.ram_total, second.ram_used)


if __name__ == "__main__":
    unittest.main()

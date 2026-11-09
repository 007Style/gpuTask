import os
import sys
import unittest

import psutil

from gputaskman import host


class HostTests(unittest.TestCase):
    def test_unknown_is_not_zero(self):
        h = host.HostSample(cpu_pct=5, per_core=[5], cpu_mhz=None, logical=1, physical=1, ram_total=100,
                            ram_used=40, ram_available=60, swap_total=0, swap_used=0,
                            procs=[host.HostProc(1, "a", None, None, None), host.HostProc(2, "b", 10, 5, 0.0)])
        self.assertEqual([p.pid for p in h.top("rss")], [2])  # the unreadable process is not ranked as 0
        self.assertIsNone(h.proc(1).rss)
        self.assertIsNone(host.to_dict(h)["cpu"]["mhz"])

    @unittest.skipUnless(sys.platform == "win32", "NtQuerySystemInformation is Windows-only")
    def test_nt_query_matches_psutil(self):
        me = psutil.Process(os.getpid())
        name, ws, private, cpu = host._nt_query()[os.getpid()]
        mi = me.memory_info()
        self.assertEqual(name.lower(), me.name().lower())
        self.assertLess(abs(private - mi.private), 8 << 20)
        self.assertLess(abs(ws - mi.rss), 16 << 20)
        self.assertGreater(cpu, 0)

    def test_sampler_rates_cpu_after_second_sample(self):
        s = host.HostSampler()
        first = s.sample()
        self.assertTrue(all(p.cpu_pct is None for p in first.procs))  # a rate needs two readings
        second = s.sample()
        self.assertTrue(any(p.cpu_pct is not None for p in second.procs))
        self.assertTrue(0 <= second.cpu_pct <= 100)
        self.assertGreater(second.ram_total, second.ram_used > 0)


if __name__ == "__main__":
    unittest.main()

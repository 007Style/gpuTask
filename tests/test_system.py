"""System RAM and CPU: the panel, the table columns, the one-shot report, and the rules they must keep.

These are OS-reported system-memory figures. They are never GPU bytes and never part of Snapshot.attributed.
"""

from __future__ import annotations

import contextlib
import io
import json
import unittest
from unittest.mock import patch

from textual.widgets import Static

from gputaskman.app import GpuTaskManager
from gputaskman.cli import main
from gputaskman.collector import ProcRow, Snapshot
from gputaskman.diff import diff
from gputaskman.host import HostProc, HostSample
from gputaskman.pdh import ProcGpu
from gputaskman.procinfo import ProcDetail
from gputaskman.report import dump, to_dict

GiB = 1 << 30
MiB = 1 << 20


def row(pid: int, name: str, kind: str = "other", dedicated: int = GiB, **extra) -> ProcRow:
    return ProcRow(pid, name, kind, ProcGpu(dedicated=dedicated), ProcDetail(pid, name), **extra)


def snapshot() -> Snapshot:
    return Snapshot(taken=1, adapter=None, adapters=[], nv=None, driver="", cuda="",
                    vram_total=16 * GiB, vram_used=10 * GiB, shared_used=GiB,
                    procs=[row(10, "llama-server.exe", "LLM", 8 * GiB), row(20, "dwm.exe", "compositor", 2 * GiB)])


def host_sample(**override) -> HostSample:
    fields = dict(cpu_pct=37.0, per_core=[10.0, 90.0], logical=2, physical=2, ram_total=32 * GiB,
                  ram_used=12 * GiB, ram_available=20 * GiB, swap_total=4 * GiB, swap_used=GiB,
                  procs=[HostProc(10, "llama-server.exe", 6 * GiB, 5 * GiB, 12.0),
                         HostProc(20, "dwm.exe", 300 * MiB, 200 * MiB, None)])
    return HostSample(**{**fields, **override})


class SystemUiTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        backend = patch("gputaskman.app.BACKEND", "wddm")  # these are WDDM snapshots, rendered the same on every OS
        sampler = patch("gputaskman.app.HostSampler")
        backend.start()
        self.sampler = sampler.start()
        self.sampler.return_value.sample.return_value = host_sample()
        self.addCleanup(backend.stop)
        self.addCleanup(sampler.stop)

    def panel(self, app) -> list[str]:
        content = app.query_one("#system", Static).content
        return [text.plain for text in getattr(content, "renderables", [content])]

    async def test_panel_and_columns_show_os_reported_values(self):
        with patch.object(GpuTaskManager, "poll"):
            app = GpuTaskManager()
            async with app.run_test(size=(140, 42)) as pilot:
                app.apply_all([snapshot()], host_sample())
                await pilot.pause()
                table = app.query_one("#procs")
                self.assertEqual(table.get_cell("10", "RAM").plain, "6.00 GB")
                self.assertEqual(table.get_cell("10", "CPU%").plain, "12")
                self.assertEqual(table.get_cell("20", "RAM").plain, "300 MB")
                self.assertEqual(table.get_cell("20", "CPU%").plain, "…")  # readable, but one more sample is needed
                self.assertEqual(str(app.query_one("#system", Static).border_title), "System (OS-reported)")
                text = "\n".join(self.panel(app))
                for shown in ("12.00 GB / 32.00 GB", "free 20.00 GB", "1.00 GB/4.00 GB", "37%",
                              "2 threads/2 cores", "top RAM: llama-server.exe 6.00 GB", "top CPU: llama-server.exe 12%"):
                    self.assertIn(shown, text)

    async def test_system_memory_is_never_added_to_gpu_totals(self):
        s = snapshot()
        attributed = s.attributed
        self.assertEqual(attributed, 10 * GiB)
        with patch.object(GpuTaskManager, "poll"):
            app = GpuTaskManager()
            async with app.run_test(size=(140, 42)) as pilot:
                app.apply_all([s], host_sample())
                await pilot.pause()
        self.assertEqual(s.attributed, attributed)
        self.assertEqual(s.vram_used, 10 * GiB)

    async def test_other_pid_spaces_never_borrow_a_local_process_reading(self):
        s = snapshot()
        foreign = row(10, "foreign.exe", host=True, origin="windows")  # a Windows pid that equals a local pid
        s.procs.append(foreign)
        with patch.object(GpuTaskManager, "poll"):
            app = GpuTaskManager()
            async with app.run_test(size=(140, 42)) as pilot:
                app.apply_all([s], host_sample())
                await pilot.pause()
                table = app.query_one("#procs")
                self.assertEqual(table.get_cell(foreign.key, "RAM").plain, "n/a")
                self.assertEqual(table.get_cell(foreign.key, "CPU%").plain, "n/a")
                self.assertEqual(table.get_cell("10", "RAM").plain, "6.00 GB")

    async def test_a_pid_that_now_belongs_to_another_process_shows_n_a(self):
        s = snapshot()
        s.procs[0].detail.create_time = 1000.0  # llama-server as the GPU sample saw it
        s.procs[1].detail.create_time = 2000.0
        sample = host_sample(procs=[HostProc(10, "other.exe", 6 * GiB, 5 * GiB, 12.0, created=1500.0),  # pid 10 reused
                                    HostProc(20, "dwm.exe", 300 * MiB, 200 * MiB, 3.0, created=2000.0)])
        with patch.object(GpuTaskManager, "poll"):
            app = GpuTaskManager()
            async with app.run_test(size=(140, 42)) as pilot:
                app.apply_all([s], sample)
                await pilot.pause()
                table = app.query_one("#procs")
                self.assertEqual(table.get_cell("10", "RAM").plain, "n/a")
                self.assertEqual(table.get_cell("10", "CPU%").plain, "n/a")
                self.assertEqual(table.get_cell("20", "RAM").plain, "300 MB")  # the same process still reads
                self.assertEqual(table.get_cell("20", "CPU%").plain, "3")

    async def test_processes_the_os_would_not_describe_show_n_a(self):
        s = snapshot()
        s.procs.append(row(30, "protected.exe"))  # not in the host sample
        with patch.object(GpuTaskManager, "poll"):
            app = GpuTaskManager()
            async with app.run_test(size=(140, 42)) as pilot:
                app.apply_all([s], host_sample())
                await pilot.pause()
                table = app.query_one("#procs")
                self.assertEqual(table.get_cell("30", "RAM").plain, "n/a")
                self.assertEqual(table.get_cell("30", "CPU%").plain, "n/a")

    async def test_failed_sample_keeps_the_gpu_view_and_says_why(self):
        self.sampler.return_value.sample.side_effect = OSError("denied")
        with patch.object(GpuTaskManager, "poll"):
            app = GpuTaskManager()
            async with app.run_test(size=(140, 42)) as pilot:
                host = app._sample_host()
                self.assertIsNone(host)
                app.apply_all([snapshot()], host)
                await pilot.pause()
                self.assertIn("RAM/CPU unavailable: OSError: denied", "\n".join(self.panel(app)))
                table = app.query_one("#procs")
                self.assertEqual(table.row_count, 2)
                self.assertEqual(table.get_cell("10", "RAM").plain, "n/a")
                self.assertIn("10.00 GB / 16.00 GB",
                              "\n".join(t.plain for t in app.query_one("#device", Static).content.renderables))
                self.sampler.return_value.sample.side_effect = None  # a later sample recovers
                host = app._sample_host()
                self.assertIsNotNone(host)
                self.assertIsNone(app.host_error)
                app.apply_all([snapshot()], host)
                await pilot.pause()
                self.assertEqual(table.get_cell("10", "RAM").plain, "6.00 GB")

    async def test_a_sampler_that_cannot_start_says_why_and_is_tried_again(self):
        self.sampler.side_effect = [OSError("layout not recognised"), self.sampler.return_value]
        with patch.object(GpuTaskManager, "poll"):
            app = GpuTaskManager()
            async with app.run_test(size=(140, 42)):
                self.assertIsNone(app._sample_host())
                self.assertEqual(app.host_error, "OSError: layout not recognised")
                self.assertIsNotNone(app._sample_host())  # the failure was not cached
                self.assertIsNone(app.host_error)

    async def test_no_system_removes_the_panel_columns_and_sorts(self):
        with patch.object(GpuTaskManager, "poll"):
            app = GpuTaskManager(system=False)
            async with app.run_test(size=(140, 42)) as pilot:
                app.apply(snapshot())
                await pilot.pause()
                self.assertEqual(len(app.query("#system")), 0)
                for column in ("RAM", "CPU%"):
                    self.assertNotIn(column, app.query_one("#procs").columns)
                self.assertEqual(app.sorts, GpuTaskManager.SORTS)
                self.assertIsNone(app._sample_host())
                self.sampler.assert_not_called()

    async def test_ram_and_cpu_sorts_rank_unknown_last(self):
        s = snapshot()
        s.procs.insert(0, row(30, "protected.exe"))  # not in the host sample
        s.procs.append(row(40, "idle.exe"))  # measured: nothing resident, no CPU time. That is not unknown
        sample = host_sample(procs=[HostProc(10, "llama-server.exe", 6 * GiB, 5 * GiB, 12.0),
                                    HostProc(20, "dwm.exe", 300 * MiB, 200 * MiB, None),
                                    HostProc(40, "idle.exe", 0, 0, 0.0)])
        with patch.object(GpuTaskManager, "poll"):
            app = GpuTaskManager()
            async with app.run_test(size=(140, 42)) as pilot:
                app.apply_all([s], sample)
                await pilot.pause()
                for _ in GpuTaskManager.SORTS:
                    await pilot.press("s")
                self.assertEqual(app.query_one("#procs").border_subtitle.split("  ·")[0], "sort: RAM")
                self.assertEqual([r.pid for r in app.visible_rows(s)], [10, 20, 40, 30])
                await pilot.press("s")
                self.assertEqual(app.query_one("#procs").border_subtitle.split("  ·")[0], "sort: CPU%")
                self.assertEqual([r.pid for r in app.visible_rows(s)], [10, 40, 30, 20])

    async def test_cpu_below_half_a_percent_is_blank_and_unknown_is_not(self):
        s = snapshot()
        s.procs += [row(40, "quiet.exe"), row(50, "busy.exe")]
        sample = host_sample(procs=[HostProc(10, "llama-server.exe", 6 * GiB, 5 * GiB, 12.0),
                                    HostProc(20, "dwm.exe", 300 * MiB, 200 * MiB, None),
                                    HostProc(40, "quiet.exe", 10 * MiB, MiB, 0.49),
                                    HostProc(50, "busy.exe", 10 * MiB, MiB, 0.6)])
        with patch.object(GpuTaskManager, "poll"):
            app = GpuTaskManager()
            async with app.run_test(size=(140, 42)) as pilot:
                app.apply_all([s], sample)
                await pilot.pause()
                table = app.query_one("#procs")
                self.assertEqual(table.get_cell("40", "CPU%").plain, "")
                self.assertEqual(table.get_cell("50", "CPU%").plain, "1")
                self.assertEqual(table.get_cell("20", "CPU%").plain, "…")

    async def test_panel_title_says_when_it_describes_the_wsl_vm(self):
        with patch.object(GpuTaskManager, "poll"), patch("gputaskman.app.BACKEND", "wsl"):
            app = GpuTaskManager()
            async with app.run_test(size=(140, 42)) as pilot:
                app.apply_all([snapshot()], host_sample())
                await pilot.pause()
                self.assertEqual(str(app.query_one("#system", Static).border_title), "System (WSL VM, OS-reported)")

    async def test_gpu_panel_stays_above_the_system_panel_in_a_small_window(self):
        with patch.object(GpuTaskManager, "poll"):
            app = GpuTaskManager()
            async with app.run_test(size=(100, 24)) as pilot:
                app.apply_all([snapshot()], host_sample())
                await pilot.pause()
                device, system = app.query_one("#device"), app.query_one("#system")
                self.assertEqual(device.region.y, app.query_one("#dashboard").region.y)
                self.assertLess(device.region.y, system.region.y)

    async def test_slim_mode_keeps_two_lines_and_full_mode_keeps_the_detail(self):
        with patch.object(GpuTaskManager, "poll"):
            app = GpuTaskManager()
            async with app.run_test(size=(140, 42)) as pilot:
                app.apply_all([snapshot()], host_sample())
                await pilot.pause()
                self.assertEqual(len(self.panel(app)), 6)
                await pilot.press("m")
                await pilot.pause()
                slim = self.panel(app)
                self.assertEqual([line[:3] for line in slim], ["RAM", "CPU"])
                self.assertIn("12.00 GB / 32.00 GB", slim[0])
                self.assertNotIn("threads", slim[1])

    async def test_lines_never_wrap_at_any_width(self):
        # 128 threads: the per-thread strip alone is wider than any of these terminals
        sample = host_sample(per_core=[50.0] * 128, logical=128, physical=64)
        for width in (60, 80, 100, 140, 200):
            for slim in (False, True):
                with self.subTest(width=width, slim=slim), patch.object(GpuTaskManager, "poll"):
                    app = GpuTaskManager()
                    async with app.run_test(size=(width, 42)) as pilot:
                        if slim:
                            await pilot.press("m")
                        app.apply_all([snapshot()], sample)
                        await pilot.pause()
                        panel = app.query_one("#system", Static)
                        lines = list(panel.content.renderables)
                        self.assertEqual(panel.content_size.height, len(lines))  # one row per line: nothing wrapped
                        # the RAM and CPU bars and the sparkline row are sized to fit whole, not cut off
                        for index in (0, 1, 3)[:len(lines)]:
                            self.assertLessEqual(lines[index].cell_len, panel.content_size.width, lines[index].plain)


class SystemReportTests(unittest.TestCase):
    def run_cli(self, *args) -> str:
        output = io.StringIO()
        with patch("sys.argv", ["gputaskman", *args]), patch("gputaskman.cli.time.sleep"), \
                patch("gputaskman.collector.Collector") as collector, contextlib.redirect_stdout(output):
            collector.return_value.sample.return_value = snapshot()
            main()
        return output.getvalue()

    def test_json_carries_a_system_section_with_unknowns_as_null(self):
        h = host_sample(procs=[HostProc(1, "a.exe", 10, None, 3.14159)])
        text = dump(snapshot(), True, h)
        system = json.loads(text)["system"]
        self.assertEqual(system["ram"]["total"], 32 * GiB)
        self.assertIn("OS-reported", system["source"])
        self.assertEqual(system["top_ram"], [{"pid": 1, "name": "a.exe", "rss": 10, "private": None, "cpu_pct": 3.1}])
        self.assertEqual(json.loads(text)["processes"], to_dict(snapshot())["processes"])  # the GPU part is untouched

    def test_text_report_appends_the_system_section(self):
        text = dump(snapshot(), False, host_sample())
        self.assertIn("System RAM and CPU (OS-reported; not GPU memory)", text)
        self.assertIn("llama-server.exe", text)

    def test_unavailable_system_figures_say_why_in_json_and_text(self):
        self.assertEqual(json.loads(dump(snapshot(), True, None, "OSError: boom"))["system"], {"error": "OSError: boom"})
        self.assertIn("System RAM/CPU unavailable: OSError: boom", dump(snapshot(), False, None, "OSError: boom"))
        self.assertNotIn("system", json.loads(dump(snapshot(), True)))  # not asked for: no section at all

    def test_json_one_shot_samples_twice_so_cpu_has_a_rate(self):
        with patch("gputaskman.host.HostSampler") as factory:
            factory.return_value.sample.return_value = host_sample()
            output = self.run_cli("--json")
        self.assertEqual(factory.return_value.sample.call_count, 2)  # one primes the rates, one is reported
        self.assertEqual(json.loads(output)["system"]["cpu"]["pct"], 37.0)

    def test_no_system_flag_skips_the_sampler_and_the_section(self):
        with patch("gputaskman.host.HostSampler") as factory:
            output = self.run_cli("--json", "--no-system")
        factory.assert_not_called()
        self.assertNotIn("system", json.loads(output))

    def test_a_failing_sampler_does_not_cost_the_gpu_report(self):
        with patch("gputaskman.host.HostSampler") as factory:
            factory.side_effect = OSError("layout not recognised")
            output = self.run_cli("--json")
        data = json.loads(output)
        self.assertEqual(data["system"], {"error": "OSError: layout not recognised"})
        self.assertEqual(len(data["processes"]), 2)

    def test_a_sampler_that_fails_later_does_not_cost_the_gpu_report(self):
        with patch("gputaskman.host.HostSampler") as factory:
            factory.return_value.sample.side_effect = [host_sample(), OSError("gone")]
            output = self.run_cli("--once")
        self.assertIn("System RAM/CPU unavailable: OSError: gone", output)
        self.assertIn("llama-server.exe", output)

    def test_diff_ignores_the_system_section(self):
        before, after = to_dict(snapshot()), to_dict(snapshot())
        plain = diff(before, after)
        with_system = diff({**before, "system": {"ram": {"used": 1}}}, {**after, "system": {"ram": {"used": 2}}})
        self.assertEqual(with_system, plain)


if __name__ == "__main__":
    unittest.main()

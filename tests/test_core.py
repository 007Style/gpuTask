from __future__ import annotations

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from gputaskman.app import GpuTaskManager
from gputaskman.cli import main
from gputaskman.collector import ProcRow, Snapshot
from gputaskman.gguf import GgufInfo, read_gguf
from gputaskman.llm import LlmView, ServerInfo, is_gpu_device, parse_flags, parse_load_log
from gputaskman.pdh import GpuCounters, ProcGpu
from gputaskman.procinfo import ProcDetail
from gputaskman.report import to_dict, to_text
from tools.capture_screenshots import for_publication

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


class MemoryTests(unittest.TestCase):
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

    def test_capture_does_not_change_original_or_publish_command_line(self):
        original = snapshot()
        original.procs[0].llm.models = [
            GgufInfo(path=r"C:\Users\example\models\model.gguf", file_size=1, version=3)
        ]
        public = for_publication(original)
        self.assertEqual(public.procs[0].detail.cmdline, [])
        self.assertEqual(public.procs[0].llm.models[0].path, "model.gguf")
        self.assertIn("private-test-value", original.procs[0].detail.cmdline)
        self.assertIn("Users", original.procs[0].llm.models[0].path)


class ModelTests(unittest.TestCase):
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
        for device in ("CPU", "CPU_Mapped", "CUDA_Host"):
            self.assertFalse(is_gpu_device(device))
        self.assertTrue(is_gpu_device("CUDA0"))

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


class UiTests(unittest.IsolatedAsyncioTestCase):
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

from __future__ import annotations

import contextlib
import io
import json
import os
import tempfile
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path
from unittest.mock import patch

from gputaskman.app import GpuTaskManager
from gputaskman.cli import main
from gputaskman.collector import ProcRow, Snapshot, select_adapter, select_nv_device
from gputaskman.dxgi import Adapter
from gputaskman.gguf import GgufInfo, read_gguf
from gputaskman.llm import (LlmView, ServerInfo, build_view, is_gpu_device, llama_server_models, lmstudio_models,
                            parse_flags, parse_load_log, probe_server)
from gputaskman.nvml import NvDevice
from gputaskman.pdh import GpuCounters, ProcGpu
from gputaskman.procinfo import (AI_KINDS, MappedModel, ProcDetail, ProcInspector, _is_model_file,
                                _models_from_cmdline, classify, installed_models, make_model)
from gputaskman.report import to_dict, to_text
from gputaskman.stores import store_name
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
    def test_adapter_indices_do_not_clamp_to_another_gpu(self):
        gpu = Adapter(1, "NVIDIA GPU", 0x10DE, 16 * GiB, 0, False)
        software = Adapter(2, "Software", 0x1414, 0, 0, True)
        self.assertIs(select_adapter([software, gpu], 0), gpu)
        with self.assertRaisesRegex(ValueError, "out of range"):
            select_adapter([software, gpu], 1)
        self.assertIsNone(select_adapter([], 0))
        with self.assertRaisesRegex(ValueError, "out of range"):
            select_adapter([], 1)

    def test_nvml_matches_unique_adapter_name_before_index(self):
        gpu = Adapter(1, "NVIDIA RTX Spark", 0x10DE, 64 * GiB, 0, False)
        npu = NvDevice(index=0, name="NVIDIA NPU", driver_model="MCDM")
        nv_gpu = NvDevice(index=1, name="NVIDIA RTX SPARK", driver_model="WDDM")
        self.assertIs(select_nv_device(gpu, [npu, nv_gpu], 0), nv_gpu)
        self.assertIs(select_nv_device(None, [npu, nv_gpu], 0), npu)
        self.assertIsNone(select_nv_device(gpu, [npu], 0))
        intel = Adapter(2, "Intel GPU", 0x8086, 8 * GiB, 0, False)
        self.assertIsNone(select_nv_device(intel, [nv_gpu], 0))


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

    def test_text_shows_unavailable_device_metrics_without_fabricating_zeroes(self):
        s = snapshot()
        s.nv = NvDevice(index=0, util_gpu=0, power_w=2.4, pcie_tx_kbs=0)
        text = to_text(s)
        self.assertIn("GPU 0%", text)
        self.assertIn("mem-bandwidth N/A", text)
        self.assertIn("2/N/A W", text)
        self.assertIn("P N/A", text)
        self.assertIn("PCIe rx N/A MB/s tx 0 MB/s", text)
        self.assertNotIn("None", text.splitlines()[2])
        self.assertIsNone(to_dict(s)["device"]["power_limit_w"])

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
                    m = make_model(path, 1000, 0, False, "mapped", store)
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
            with patch("gputaskman.procinfo.psutil.Process") as process:
                process.return_value.memory_maps.return_value = []
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


class UiTests(unittest.IsolatedAsyncioTestCase):
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

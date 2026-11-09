from __future__ import annotations

import contextlib
import io
import json
import os
import tempfile
import unittest
from unittest.mock import patch

from gputaskman.cli import main
from gputaskman.diff import diff, load, match, to_text

GiB = 1 << 30
MiB = 1 << 20


def proc(pid, name, vram=None, shared=0, host=None, ports=(), llm=None, files=(), inferred=None):
    return {"pid": pid, "host": host, "name": name, "vram_bytes": vram, "shared_bytes": shared,
            "committed_bytes": None, "paged_out_bytes": None, "vram_inferred_bytes": inferred,
            "ports": list(ports), "model_files": [{"path": f, "sources": ["mapped"]} for f in files],
            **({"llm": llm} if llm else {})}


def snap(procs, used, attributed, taken=100.0, gpu="RTX", backend="wddm"):
    return {"taken": taken, "gpu": gpu, "backend": backend, "vram_used": used, "shared_used": 0,
            "attributed": attributed, "processes": procs}


def llm(n_ctx, kv, exact=False):
    return {"n_ctx": n_ctx, "segments": [{"label": "weights", "bytes": 4 * GiB, "kind": "weights", "exact": exact},
                                         {"label": "KV", "bytes": kv, "kind": "kv", "exact": exact}]}


class MatchTests(unittest.TestCase):
    def test_pid_then_name_and_ports_then_unique_name(self):
        before = [proc(1, "a.exe"), proc(2, "llama-server", ports=[8080]), proc(3, "solo.exe")]
        after = [proc(1, "a.exe"), proc(20, "llama-server", ports=[8080]), proc(30, "solo.exe")]
        self.assertEqual([(b["pid"], a["pid"], how) for b, a, how in match(before, after)],
                         [(1, 1, "pid"), (2, 20, "name+ports"), (3, 30, "name")])

    def test_reused_pid_ambiguous_names_and_host_pid_space_are_not_paired(self):
        before = [proc(5, "old.exe"), proc(7, "w.exe"), proc(8, "w.exe"), proc(9, "x", host="windows")]
        after = [proc(5, "new.exe"), proc(17, "w.exe"), proc(18, "w.exe"), proc(9, "x")]
        result = {(b and b["pid"], a and a["pid"], how) for b, a, how in match(before, after)}
        self.assertEqual(result, {(None, 5, "new"), (None, 17, "new"), (None, 18, "new"), (None, 9, "new"),
                                  (5, None, "gone"), (7, None, "gone"), (8, None, "gone"), (9, None, "gone")})

    def test_placeholder_rows_and_missing_pids_are_never_paired_or_dropped(self):
        before = [proc(7, "(outside this container)", host="outside"), proc(None, "a"), proc(None, "b"),
                  proc(40, "?", host="windows")]
        after = [proc(9, "(outside this container)", host="outside"), proc(None, "a"), proc(41, "?", host="windows")]
        result = sorted(((b and b["name"], a and a["name"], how) for b, a, how in match(before, after)), key=str)
        self.assertEqual(result, sorted([("(outside this container)", None, "gone"), (None, "(outside this container)", "new"),
                                         ("?", None, "gone"), (None, "?", "new"), ("a", "a", "name"),
                                         ("b", None, "gone")], key=str))
        d = diff(snap(before, 0, 0), snap(after, 0, 0))
        self.assertIn("H:9", to_text(d))
        self.assertIn("W:41", to_text(d))


class DiffTests(unittest.TestCase):
    def test_measured_runtime_and_mapped_evidence_stay_separate(self):
        before = snap([proc(1, "dwm.exe", vram=2 * GiB),
                       proc(2, "llama-server", vram=5 * GiB, ports=[8080], llm=llm(4096, GiB // 2))],
                      used=8 * GiB, attributed=7 * GiB)
        after = snap([proc(1, "dwm.exe", vram=2 * GiB),
                      proc(22, "llama-server", vram=9 * GiB, ports=[8080], llm=llm(32768, 4 * GiB),
                           files=[r"D:\models\q.gguf"]),
                      proc(3, "python.exe", vram=GiB)],
                     used=13 * GiB, attributed=12 * GiB, taken=142.5)
        d = diff(before, after)
        self.assertEqual(d["seconds"], 42.5)
        self.assertEqual(d["device"]["vram_used"]["delta"], 5 * GiB)
        self.assertEqual(d["device"]["unattributed"], {"before": GiB, "after": GiB, "delta": 0})
        server, new, dwm = d["processes"]
        self.assertEqual((server["match"], server["measured"]["vram_bytes"]["delta"]), ("name+ports", 4 * GiB))
        self.assertEqual(server["runtime"]["n_ctx"], {"before": 4096, "after": 32768, "delta": 28672})
        self.assertEqual(server["runtime"]["segments"]["kv"]["delta"], 3584 * MiB)
        self.assertEqual(server["runtime"]["segments"]["kv"]["source"], "estimated")
        self.assertEqual(server["model_files"]["added"], [{"path": r"D:\models\q.gguf", "sources": ["mapped"]}])
        self.assertEqual(new["match"], "new")
        self.assertEqual(new["measured"]["vram_bytes"], {"before": None, "after": GiB, "delta": None})

        text = to_text(d)
        self.assertIn("2->22", text)
        self.assertIn("context length  4096 -> 32768", text)
        self.assertIn("+3.50 GB  estimated", text)
        self.assertIn("not listed", text)
        self.assertIn("not what is resident in GPU memory", text)
        self.assertNotIn("dwm.exe", text)
        self.assertIn("1 process(es) changed by less than 16 MB", text)

    def test_runtime_reported_segments_and_unknown_measurements(self):
        before = snap([proc(1, "llama", vram=None, llm=llm(4096, GiB, exact=True))], used=None, attributed=None)
        after = snap([proc(1, "llama", vram=None, llm=llm(8192, 2 * GiB, exact=True))], used=None, attributed=None)
        d = diff(before, after)
        self.assertEqual(d["device"]["unattributed"]["delta"], None)
        self.assertEqual(d["processes"][0]["measured"]["vram_bytes"]["delta"], None)
        self.assertEqual(d["processes"][0]["runtime"]["segments"]["kv"]["source"], "runtime-reported")
        self.assertIn("N/A", to_text(d))

    def test_rejects_snapshots_from_different_gpus(self):
        with self.assertRaisesRegex(ValueError, "different GPUs"):
            diff(snap([], 0, 0, gpu="A"), snap([], 0, 0, gpu="B"))

    def test_memory_note_follows_the_snapshots_into_the_comparison(self):
        note = "Intel VRAM used is a sum over the processes gputaskman can inspect."
        noted = {**snap([], 3 * GiB, 0, taken=130.0, backend="drm", gpu="Arc"), "memory_note": note}
        older = snap([], 2 * GiB, 0, backend="drm", gpu="Arc")  # saved by a version that had no note
        for before, after in ((older, noted), (noted, older), (noted, noted)):
            d = diff(before, after)
            self.assertEqual(d["memory_note"], note)
            self.assertIn(note, to_text(d))
        plain = diff(snap([], 2 * GiB, 0), snap([], 3 * GiB, 0))  # every other GPU: nothing to add
        self.assertIsNone(plain["memory_note"])
        self.assertNotIn(note, to_text(plain))

    def test_remainder_segments_follow_the_noise_threshold(self):
        def server(runtime):
            view = {"n_ctx": 4096, "segments": [{"label": "rest", "bytes": runtime, "kind": "runtime", "exact": False}]}
            return proc(1, "llama", vram=GiB + runtime, llm=view)
        d = diff(snap([server(GiB)], 3 * GiB, 2 * GiB), snap([server(GiB + 4096)], 3 * GiB, 2 * GiB))
        self.assertIn("(no process changed by 16 MB or more)", to_text(d))
        self.assertNotIn("llama (1)", to_text(d))


class LoadAndCliTests(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.dir.cleanup)

    def write(self, name, data, encoding):
        path = os.path.join(self.dir.name, name)
        with open(path, "w", encoding=encoding) as f:
            f.write(json.dumps(data))
        return path

    def test_reads_powershell_utf16_and_utf8_bom(self):
        data = snap([proc(1, "a", vram=MiB)], used=MiB, attributed=MiB)
        self.assertEqual(load(self.write("a.json", data, "utf-16")), data)
        self.assertEqual(load(self.write("b.json", data, "utf-8-sig")), data)
        with self.assertRaisesRegex(ValueError, "not a gputaskman --json snapshot"):
            load(self.write("c.json", {"hello": 1}, "utf-8"))

    def test_cli_text_json_and_errors(self):
        a = self.write("a.json", snap([], used=GiB, attributed=GiB), "utf-8")
        b = self.write("b.json", snap([proc(1, "x", vram=GiB)], used=2 * GiB, attributed=2 * GiB), "utf-8")
        out = io.StringIO()
        with patch("sys.argv", ["gputaskman", "diff", a, b]), contextlib.redirect_stdout(out):
            main()
        self.assertIn("dedicated used", out.getvalue())
        out = io.StringIO()
        with patch("sys.argv", ["gputaskman", "diff", a, b, "--json"]), contextlib.redirect_stdout(out):
            main()
        self.assertEqual(json.loads(out.getvalue())["device"]["vram_used"]["delta"], GiB)
        err = io.StringIO()
        with patch("sys.argv", ["gputaskman", "diff", a, os.path.join(self.dir.name, "missing.json")]), \
                contextlib.redirect_stderr(err), self.assertRaises(SystemExit):
            main()
        self.assertIn("missing.json", err.getvalue())


if __name__ == "__main__":
    unittest.main()

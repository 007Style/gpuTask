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


def proc(pid, name, vram=None, shared=0, host=None, ports=(), llm=None, files=(), inferred=None, started=...):
    return {"pid": pid, "host": host, "name": name, "vram_bytes": vram, "shared_bytes": shared,
            "committed_bytes": None, "paged_out_bytes": None, "vram_inferred_bytes": inferred,
            "ports": list(ports), "model_files": [{"path": f, "sources": ["mapped"]} for f in files],
            **({} if started is ... else {"started_at": started}),  # absent: a snapshot from before started_at existed
            **({"llm": llm} if llm else {})}


def pairs(before, after):
    return {(b and b["pid"], a and a["pid"], how) for b, a, how in match(before, after)}


def snap(procs, used, attributed, taken=100.0, gpu="RTX", backend="wddm", adapter=...):
    return {"taken": taken, "gpu": gpu, "backend": backend, "vram_used": used, "shared_used": 0,
            "attributed": attributed, "processes": procs, **({} if adapter is ... else {"adapter": adapter})}


def gpu_adapter(luid=0x4BB7, pci=(1, 0, 0), vendor=0x10DE):
    """The adapter block of a --json snapshot (a dataclass dump: the PCI address is a list)."""
    return {"luid": luid, "name": "RTX", "vendor_id": vendor, "dedicated_vram": 16 * GiB, "shared_system": 0,
            "software": False, "pci_address": list(pci) if pci else None}


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


class PidReuseTests(unittest.TestCase):
    """A PID that comes back with a different start time is another process, however alike the rows look."""

    def test_a_reused_pid_is_gone_and_new_with_or_without_ports(self):
        for ports in ([8080], []):  # name+ports and unique-name fallbacks must not pair them either
            with self.subTest(ports=ports):
                before = [proc(42, "llama-server", ports=ports, started=100.0)]
                after = [proc(42, "llama-server", ports=ports, started=200.0)]
                self.assertEqual(pairs(before, after), {(None, 42, "new"), (42, None, "gone")})

    def test_one_lifetime_still_pairs_by_pid_within_the_rounding_tolerance(self):
        for later in (100.0, 101.9, 102.0, 98.0):
            with self.subTest(later=later):
                self.assertEqual(pairs([proc(42, "a.exe", started=100.0)], [proc(42, "a.exe", started=later)]),
                                 {(42, 42, "pid")})
        self.assertEqual(pairs([proc(42, "a.exe", started=100.0)], [proc(42, "a.exe", started=102.5)]),
                         {(None, 42, "new"), (42, None, "gone")})

    def test_unknown_or_malformed_start_times_keep_the_old_heuristic_pairing(self):
        # ... means the key is absent (an older snapshot); the rest are values a snapshot could carry
        for before_value, after_value in ((..., 200.0), (100.0, ...), (None, None), (None, 200.0), (0.0, 200.0),
                                          (-5.0, 200.0), (True, 200.0), ("100", 200.0), ([100.0], 200.0),
                                          (float("nan"), 200.0), (float("inf"), 200.0)):
            with self.subTest(before=before_value, after=after_value):
                before = [proc(42, "a.exe", started=before_value)]
                after = [proc(42, "a.exe", started=after_value)]
                self.assertEqual(pairs(before, after), {(42, 42, "pid")})
        # no PID means no reuse to speak of: rows without one are still paired by name alone
        self.assertEqual({how for _, _, how in match([proc(None, "a", started=100.0)], [proc(None, "a", started=200.0)])},
                         {"name"})
        before = [proc(42, "llama-server", ports=[8080], started=100.0), proc(50, "solo.exe", started=100.0)]
        after = [proc(77, "llama-server", ports=[8080], started=200.0), proc(60, "solo.exe", started=200.0)]
        self.assertEqual(pairs(before, after), {(42, 77, "name+ports"), (50, 60, "name")})

    def test_a_refused_pair_does_not_make_the_other_rows_less_ambiguous(self):
        before = [proc(42, "llama-server", started=100.0)]
        after = [proc(42, "llama-server", started=200.0), proc(77, "llama-server", started=300.0)]
        self.assertEqual(pairs(before, after), {(None, 42, "new"), (None, 77, "new"), (42, None, "gone")})

    def test_reuse_is_judged_inside_one_pid_namespace(self):
        before = [proc(42, "x", started=100.0), proc(42, "x", host="windows")]
        after = [proc(42, "x", started=200.0), proc(42, "x", host="windows")]
        self.assertEqual(pairs(before, after), {(None, 42, "new"), (42, None, "gone"), (42, 42, "pid")})


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

    def test_a_reused_pid_keeps_the_device_change_but_makes_no_process_delta(self):
        before = snap([proc(42, "llama-server", vram=4 * GiB, ports=[8080], started=100.0)], 5 * GiB, 4 * GiB)
        after = snap([proc(42, "llama-server", vram=8 * GiB, ports=[8080], started=200.0)], 9 * GiB, 8 * GiB)
        d = diff(before, after)
        self.assertEqual(d["device"]["vram_used"]["delta"], 4 * GiB)  # the device-wide change is still measured
        self.assertEqual(sorted(p["match"] for p in d["processes"]), ["gone", "new"])
        self.assertTrue(all(p["measured"]["vram_bytes"]["delta"] is None for p in d["processes"]))
        rows = [line for line in to_text(d).splitlines() if line.startswith("llama-server")]
        self.assertEqual(len(rows), 2)  # one row per lifetime, each absent from the other snapshot
        self.assertTrue(all("not listed" in row for row in rows))


class AdapterGuardTests(unittest.TestCase):
    def pair(self, before, after, backend="wddm"):
        return diff(snap([], 0, 0, backend=backend, adapter=before), snap([], 0, 0, backend=backend, adapter=after))

    def test_same_name_but_different_pci_addresses_are_not_one_gpu(self):
        with self.assertRaisesRegex(ValueError, r"different adapters.*'RTX'.*PCI 01:00\.0 vs 02:00\.0"):
            self.pair(gpu_adapter(pci=(1, 0, 0)), gpu_adapter(pci=(2, 0, 0)))

    def test_a_matching_pci_address_survives_a_changed_luid(self):
        d = self.pair(gpu_adapter(luid=0x4BB7), gpu_adapter(luid=0x5A01))  # a reboot, or the adapter was re-enabled
        self.assertEqual((d["adapter_check"], d["adapter_note"]), ("pci", None))
        self.assertNotIn("Warning", to_text(d))

    def test_conflicting_vendors_are_refused_without_pci_addresses(self):
        with self.assertRaisesRegex(ValueError, "vendor 0x10de vs 0x1002"):
            self.pair(gpu_adapter(pci=None, vendor=0x10DE), gpu_adapter(pci=None, vendor=0x1002))

    def test_stable_adapter_keys_separate_identical_gpus_off_windows(self):
        # Linux and macOS keys derive from the PCI slot, so they hold from one snapshot to the next
        with self.assertRaisesRegex(ValueError, "adapter key 0x1000 vs 0x2000"):
            self.pair(gpu_adapter(luid=0x1000, pci=None), gpu_adapter(luid=0x2000, pci=None), backend="nvml")
        d = self.pair(gpu_adapter(luid=0x1000, pci=None), gpu_adapter(luid=0x1000, pci=None), backend="nvml")
        self.assertEqual((d["adapter_check"], d["adapter_note"]), ("key", None))

    def test_a_windows_luid_alone_never_refuses_a_diff(self):
        # Windows reassigns LUIDs at boot: without a PCI address on both sides a changed one proves nothing
        d = self.pair(gpu_adapter(luid=0x4BB7, pci=None), gpu_adapter(luid=0x5A01, pci=None))
        self.assertEqual(d["adapter_check"], "name-only")
        self.assertIn("one GPU or two of the same model", d["adapter_note"])
        self.assertIn("Warning: Their adapter keys differ", to_text(d))
        mixed = self.pair(gpu_adapter(pci=(1, 0, 0)), gpu_adapter(pci=None))  # a PCI address on one side only
        self.assertEqual(mixed["adapter_check"], "key")

    def test_snapshots_with_nothing_to_compare_fall_back_to_name_and_backend_with_a_warning(self):
        legacy = snap([], 0, 0)  # saved by a version that wrote no adapter details
        for before, after in ((legacy, legacy), (legacy, {**legacy, "adapter": gpu_adapter()}),
                              ({**legacy, "adapter": None}, {**legacy, "adapter": None})):
            d = diff(before, after)
            self.assertEqual(d["adapter_check"], "name-only")
            self.assertIn("matched by GPU name and backend only", d["adapter_note"])
            self.assertIn("Warning: These snapshots do not both say which adapter", to_text(d))

    def test_zero_and_malformed_details_are_unknown_not_identities(self):
        unknown = [{"luid": 0, "vendor_id": 0, "pci_address": None},  # zero means unknown, not "adapter 0"
                   {"luid": "4BB7", "vendor_id": True, "pci_address": [1, 0]},  # wrong types, wrong length
                   {"luid": None, "vendor_id": None, "pci_address": [1, 0, -1]}, "RTX", 7, []]
        for backend in ("wddm", "nvml"):  # on nvml a real key mismatch would be refused
            for bad in unknown:
                d = self.pair(bad, gpu_adapter(luid=0x4BB7), backend=backend)
                self.assertEqual(d["adapter_check"], "name-only", (backend, bad))


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

    def test_cli_refuses_conflicting_adapters_and_reports_what_was_compared(self):
        a = self.write("a.json", snap([], 0, 0, adapter=gpu_adapter(pci=(1, 0, 0))), "utf-8")
        b = self.write("b.json", snap([], 0, 0, adapter=gpu_adapter(pci=(2, 0, 0))), "utf-8")
        err = io.StringIO()
        with patch("sys.argv", ["gputaskman", "diff", a, b]), contextlib.redirect_stderr(err), \
                self.assertRaises(SystemExit) as stop:
            main()
        self.assertEqual(stop.exception.code, 2)
        self.assertIn("different adapters", err.getvalue())
        c = self.write("c.json", snap([], 0, 0, adapter=gpu_adapter(luid=0x5A01)), "utf-16")  # same PCI slot, new boot
        out = io.StringIO()
        with patch("sys.argv", ["gputaskman", "diff", a, c, "--json"]), contextlib.redirect_stdout(out):
            main()
        self.assertEqual(json.loads(out.getvalue())["adapter_check"], "pci")


if __name__ == "__main__":
    unittest.main()

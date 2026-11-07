from __future__ import annotations

import os
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

from gputaskman import drm

GiB = 1 << 30

FDINFO = """pos:\t0
flags:\t02100002
mnt_id:\t28
ino:\t415
drm-driver:\tamdgpu
drm-client-id:\t26
drm-pdev:\t0000:03:00.0
pasid:\t25
drm-total-cpu:\t0
drm-shared-cpu:\t0
drm-resident-cpu:\t0
drm-purgeable-cpu:\t0
drm-total-gtt:\t2 MiB
drm-shared-gtt:\t0
drm-resident-gtt:\t2060 KiB
drm-purgeable-gtt:\t0
drm-total-vram:\t12 KiB
drm-shared-vram:\t0
drm-resident-vram:\t0
drm-purgeable-vram:\t0
drm-memory-vram:\t0 KiB
drm-memory-gtt: \t2060 KiB
drm-memory-cpu: \t0 KiB
amd-evicted-vram:\t12 KiB
amd-requested-vram:\t12 KiB
amd-requested-gtt:\t2048 KiB
drm-engine-gfx:\t5308400172 ns
drm-engine-compute:\t0 ns
drm-engine-enc:\t1200 ns
drm-engine-capacity-gfx:\t1
"""


def _make_card(root: Path, name, pdev, vendor, device, vram, driver="amdgpu", gtt=0, apu=False) -> Path:
    dev = root / "drm" / name / "device"
    dev.mkdir(parents=True)
    (dev / "vendor").write_text(f"0x{vendor:04x}\n")
    (dev / "device").write_text(f"0x{device:04x}\n")
    (dev / "mem_info_vram_total").write_text(str(vram))
    (dev / "mem_info_gtt_total").write_text(str(gtt))
    if not apu:
        (dev / "mem_info_vram_vendor").write_text("samsung\n")  # only discrete cards have VRAM chips
    driver_dir = root / "drivers" / driver
    driver_dir.mkdir(parents=True, exist_ok=True)
    os.symlink(driver_dir, dev / "driver")
    (dev / "uevent").write_text(f"DRIVER={driver}\nPCI_SLOT_NAME={pdev}\n")
    return dev


def _card(dev, pdev, name="AMD GPU", vram=16 * GiB, vendor=0x1002, device=0x7550) -> drm.DrmCard:
    return drm.DrmCard(
        key=drm.pci_key(pdev), pdev=pdev, sysfs=str(dev), driver="amdgpu",
        vendor_id=vendor, device_id=device, name=name, vram_total=vram,
        integrated=vram <= 4 * GiB,
    )


def _write_fd(proc: Path, pid, fd, text, target="/dev/dri/renderD128") -> None:
    """One process fd: the fdinfo contents plus the /proc/<pid>/fd symlink DrmCounters filters on."""
    fdinfo = proc / str(pid) / "fdinfo"
    fdinfo.mkdir(parents=True, exist_ok=True)
    (fdinfo / str(fd)).write_text(text)
    fds = proc / str(pid) / "fd"
    fds.mkdir(parents=True, exist_ok=True)
    link = fds / str(fd)
    if not link.is_symlink():
        os.symlink(target, link)


def _fdinfo(client, pdev, gfx=0, vram=0, gtt=0, shared_vram=0, shared_gtt=0, shared_cpu=0) -> str:
    return (
        "pos:\t0\n"
        "drm-driver:\tamdgpu\n"
        f"drm-client-id:\t{client}\n"
        f"drm-pdev:\t{pdev}\n"
        f"drm-shared-vram:\t{shared_vram} KiB\n"
        f"drm-resident-vram:\t{vram} KiB\n"
        f"drm-shared-gtt:\t{shared_gtt} KiB\n"
        f"drm-resident-gtt:\t{gtt} KiB\n"
        f"drm-shared-cpu:\t{shared_cpu} KiB\n"
        "drm-resident-cpu:\t0\n"
        f"drm-engine-gfx:\t{gfx} ns\n"
        f"drm-engine-capacity-gfx:\t1\n"
    )


class Clock:
    """A monotonic_ns stand-in the test advances by hand: every fdinfo read in one refresh sees the same time."""

    def __init__(self) -> None:
        self.t = 0

    def __call__(self) -> int:
        return self.t

    def tick(self, seconds: float = 1.0) -> None:
        self.t += int(seconds * 1_000_000_000)


class ParseTests(unittest.TestCase):
    def test_parse_size(self):
        self.assertEqual(drm.parse_size("2060 KiB"), 2060 * 1024)
        self.assertEqual(drm.parse_size("2 MiB"), 2 * 1024**2)
        self.assertEqual(drm.parse_size("1 GiB"), 1024**3)
        self.assertEqual(drm.parse_size("12 KiB"), 12 * 1024)
        self.assertEqual(drm.parse_size("0"), 0)
        self.assertEqual(drm.parse_size("512"), 512)
        self.assertEqual(drm.parse_size("  4 KiB  "), 4096)
        self.assertIsNone(drm.parse_size("bogus"))  # unknown, not 0
        self.assertIsNone(drm.parse_size("12 bananas"))
        self.assertIsNone(drm.parse_size(None))

    def test_parse_fdinfo_example(self):
        info = drm.parse_fdinfo(FDINFO)
        self.assertIsNotNone(info)
        self.assertEqual(info["driver"], "amdgpu")
        self.assertEqual(info["client_id"], 26)
        self.assertEqual(info["pdev"], "0000:03:00.0")
        self.assertEqual(info["vram"], 0)
        self.assertEqual(info["gtt"], 2060 * 1024)
        self.assertEqual(info["cpu"], 0)
        self.assertEqual(info["engines"], {"gfx": 5308400172, "compute": 0, "enc": 1200})

    def test_parse_fdinfo_non_drm(self):
        self.assertIsNone(drm.parse_fdinfo("pos:\t0\nflags:\t02\nmnt_id:\t28\n"))

    def test_parse_fdinfo_resident_fallback(self):
        info = drm.parse_fdinfo(
            "drm-driver:\tamdgpu\n"
            "drm-resident-vram:\t8 KiB\n"
            "drm-resident-gtt:\t3 MiB\n"
            "drm-resident-cpu:\t0\n"
        )
        self.assertIsNotNone(info)
        self.assertEqual(info["vram"], 8 * 1024)
        self.assertEqual(info["gtt"], 3 * 1024**2)
        self.assertEqual(info["cpu"], 0)
        self.assertEqual(info["engines"], {})
        self.assertIsNone(info["client_id"])
        self.assertIsNone(info["shared"])  # no drm-shared-* keys: unknown, not 0

    def test_parse_fdinfo_prefers_resident_keys(self):
        info = drm.parse_fdinfo("drm-driver:\tamdgpu\ndrm-resident-vram:\t8 KiB\ndrm-memory-vram:\t9 KiB\n"
                                "drm-shared-vram:\t0\ndrm-shared-gtt:\t4 KiB\ndrm-shared-cpu:\t0\n")
        self.assertEqual(info["vram"], 8 * 1024)
        self.assertEqual(info["shared"], 4 * 1024)

    def test_parse_fdinfo_shared_needs_all_memory_regions(self):
        info = drm.parse_fdinfo("drm-driver:\tamdgpu\ndrm-memory-vram:\t300 MiB\ndrm-shared-vram:\t0\n")
        self.assertIsNone(info["shared"])  # no drm-shared-gtt: whether GTT buffers are shared is unknown
        info = drm.parse_fdinfo("drm-driver:\tamdgpu\ndrm-shared-vram:\t0\ndrm-shared-gtt:\t0\n")
        self.assertIsNone(info["shared"])  # no drm-shared-cpu: a shared CPU-preferred buffer may be resident in GTT

    def test_parse_fdinfo_shared_cpu_counts_toward_private_gate(self):
        info = drm.parse_fdinfo("drm-driver:\tamdgpu\ndrm-shared-vram:\t0\ndrm-shared-gtt:\t0\n"
                                "drm-shared-cpu:\t64 KiB\n")
        self.assertEqual(info["shared"], 64 * 1024)
        info = drm.parse_fdinfo("drm-driver:\tamdgpu\ndrm-shared-vram:\t0\ndrm-shared-gtt:\t0\n"
                                "drm-shared-cpu:\tlots\n")
        self.assertIsNone(info["shared"])

    def test_parse_fdinfo_unparseable_is_unknown(self):
        info = drm.parse_fdinfo("drm-driver:\tamdgpu\ndrm-resident-vram:\tlots\ndrm-shared-vram:\t0\n"
                                "drm-shared-gtt:\t?\ndrm-engine-gfx:\ttorn\n")
        self.assertIsNone(info["vram"])
        self.assertIsNone(info["gtt"])  # missing entirely
        self.assertIsNone(info["shared"])
        self.assertIsNone(info["engines"])


PCI_IDS = """# fake
1002  Advanced Micro Devices, Inc. [AMD/ATI]
\t7550  Navi 48 [Radeon RX 9070/9070 XT/9070 GRE]
\t\t7550 1234  Subsystem line that must be skipped
\t13c0  Granite Ridge [Radeon 610M]
1003  Somebody else
\t9999  Not this vendor
"""


class CardsTests(unittest.TestCase):
    def test_cards(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            # An APU with a large BIOS carve-out is still integrated: the flag doesn't depend on size.
            _make_card(root, "card0", "0000:05:00.0", 0x1002, 0x13C0, 8 * GiB, gtt=3 * GiB, apu=True)
            _make_card(root, "card1", "0000:03:00.0", 0x1002, 0x7550, 16 * GiB, gtt=32 * GiB)
            _make_card(root, "card2", "0000:0a:00.0", 0x10DE, 0x9999, 24 * GiB, driver="nvidia")
            for junk in ("card1-DP-1", "renderD128"):
                (root / "drm" / junk).mkdir(parents=True)
            pci_ids = root / "pci.ids"
            pci_ids.write_text(PCI_IDS)

            found = drm.cards(str(root / "drm"), str(pci_ids))

            self.assertEqual([c.pdev for c in found], ["0000:03:00.0", "0000:05:00.0"])
            self.assertEqual([c.name for c in found], [
                "AMD Radeon RX 9070/9070 XT/9070 GRE",
                "AMD Radeon 610M",
            ])
            self.assertFalse(found[0].integrated)
            self.assertTrue(found[1].integrated)
            self.assertEqual(found[0].vram_total, 16 * GiB)
            self.assertEqual(found[0].key, drm.pci_key("0000:03:00.0"))
            self.assertEqual(found[0].vendor_id, 0x1002)
            self.assertEqual(found[0].device_id, 0x7550)

            adapters = drm.adapters(found)
            self.assertEqual([a.luid for a in adapters], [c.key for c in found])
            self.assertEqual(adapters[0].dedicated_vram, 16 * GiB)
            self.assertEqual(adapters[0].shared_system, 32 * GiB)
            self.assertEqual(adapters[1].shared_system, 3 * GiB)
            self.assertFalse(any(a.software for a in adapters))


class ReadDeviceTests(unittest.TestCase):
    def test_read_device(self):
        with tempfile.TemporaryDirectory() as tmp:
            dev = Path(tmp) / "card0" / "device"
            dev.mkdir(parents=True)
            (dev / "mem_info_vram_total").write_text(str(16 * GiB))
            (dev / "mem_info_vram_used").write_text(str(3 * GiB))
            (dev / "gpu_busy_percent").write_text("42\n")
            (dev / "mem_busy_percent").write_text("7\n")
            (dev / "current_link_speed").write_text("32.0 GT/s PCIe\n")
            (dev / "current_link_width").write_text("16\n")
            (dev / "pp_dpm_sclk").write_text("0: 500Mhz\n1: 2100Mhz\n2: 2570Mhz *\nS: 44Mhz *\n")
            hwmon = dev / "hwmon" / "hwmon0"
            hwmon.mkdir(parents=True)
            (hwmon / "temp1_label").write_text("edge\n")
            (hwmon / "temp1_input").write_text("57000\n")
            (hwmon / "temp2_label").write_text("junction\n")
            (hwmon / "temp2_input").write_text("65000\n")
            (hwmon / "power1_average").write_text("47000000\n")
            (hwmon / "power1_label").write_text("PPT\n")
            (hwmon / "power1_cap").write_text("340000000\n")
            (hwmon / "freq1_label").write_text("sclk\n")
            (hwmon / "freq1_input").write_text("27000000\n")
            (hwmon / "freq2_label").write_text("mclk\n")
            (hwmon / "freq2_input").write_text("1258000000\n")
            (hwmon / "fan1_input").write_text("1000\n")
            (hwmon / "fan1_max").write_text("2000\n")

            card = _card(dev, "0000:03:00.0", name="AMD Radeon RX 9070")
            d = drm.read_device(card, 0)

            self.assertEqual(d.index, 0)
            self.assertEqual(d.name, "AMD Radeon RX 9070")
            self.assertEqual(d.pci_bus, "0000:03:00.0")
            self.assertEqual(d.driver_model, "amdgpu")
            self.assertEqual(d.mem_total, 16 * GiB)
            self.assertEqual(d.mem_used, 3 * GiB)
            self.assertEqual(d.util_gpu, 42)
            self.assertEqual(d.util_mem, 7)
            self.assertEqual(d.temp, 57)
            self.assertEqual(d.power_w, 47.0)
            self.assertEqual(d.power_limit_w, 340.0)
            self.assertEqual(d.fan, 50)
            self.assertEqual(d.gfx_clock, 27)
            self.assertEqual(d.source, "AMD sysfs")
            self.assertEqual(d.power_kind, "PPT")
            self.assertEqual(d.mem_clock, 1258)
            self.assertEqual(d.sm_clock_max, 2570)
            self.assertEqual(d.pcie_gen, 5)
            self.assertEqual(d.pcie_width, 16)
            self.assertIsNone(d.enc)
            self.assertIsNone(d.dec)


class DrmCountersTests(unittest.TestCase):
    def test_drm_counters(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            pdev = "0000:03:00.0"
            dev = tmp / "drm" / "card1" / "device"
            dev.mkdir(parents=True)
            (dev / "mem_info_vram_total").write_text(str(16 * GiB))
            (dev / "mem_info_vram_used").write_text(str(1 * GiB))
            (dev / "mem_info_gtt_total").write_text(str(8 * GiB))
            (dev / "mem_info_gtt_used").write_text(str(64 * 1024))
            card = _card(dev, pdev)

            proc = tmp / "proc"
            _write_fd(proc, 100, 10, _fdinfo(26, pdev, gfx=1_000_000_000, vram=3, gtt=2))
            _write_fd(proc, 300, 12, _fdinfo(27, "0000:ff:00.0", gfx=9_999_999_999, vram=4))
            # DRM-looking fdinfo behind a non-DRM fd: the /proc/<pid>/fd target rules it out.
            _write_fd(proc, 400, 13, _fdinfo(28, pdev, gfx=9_999_999_999, vram=4), target="socket:[123]")

            clock = Clock()
            counters = drm.DrmCounters(found=[card], proc=str(proc), clock=clock)
            counters.start()
            clock.tick()

            self.assertTrue(counters.ok)
            procs, adapters = counters.sample()
            self.assertEqual(list(procs), [card.key])
            self.assertIn(100, procs[card.key])
            self.assertNotIn(300, procs[card.key])  # unknown pdev
            self.assertNotIn(400, procs[card.key])  # non-DRM fd
            g = procs[card.key][100]
            self.assertEqual(g.dedicated, 3 * 1024)  # private client, no shared buffers: resident is its own
            self.assertEqual(g.resident_vram, 3 * 1024)
            self.assertEqual(g.shared, 2048)
            self.assertEqual(g.drm_reason, "")
            self.assertFalse(g.paged_available)  # no commit counter: unavailable, not 0
            self.assertEqual(g.committed, 0)
            self.assertEqual(g.engines.get("3d", 0.0), 0.0)  # idle since the baseline
            self.assertEqual(adapters[card.key].dedicated, 1 * GiB)
            self.assertEqual(adapters[card.key].shared, 64 * 1024)

            (dev / "mem_info_vram_used").write_text(str(2 * GiB))
            _write_fd(proc, 100, 10, _fdinfo(26, pdev, gfx=1_500_000_000, vram=1, gtt=2))
            clock.tick()
            procs, adapters = counters.sample()
            self.assertEqual(procs[card.key][100].engines["3d"], 50.0)
            self.assertIn("engine", procs[card.key][100].available)
            self.assertEqual(adapters[card.key].dedicated, 2 * GiB)
            counters.close()

    def test_constructor_does_not_scan(self):
        with tempfile.TemporaryDirectory() as tmp:
            dev = Path(tmp) / "drm" / "card1" / "device"
            dev.mkdir(parents=True)
            with patch.object(drm.DrmCounters, "sample") as sample:
                drm.DrmCounters(found=[_card(dev, "0000:03:00.0")], proc=str(Path(tmp) / "proc"))
            sample.assert_not_called()  # an NVIDIA GPU may be selected: no fd walk until start()

    def test_rocm_process_engine_load_is_unknown(self):
        # A HIP process (llama.cpp-ROCm) holds /dev/kfd next to its render node: its VRAM is in fdinfo, but
        # its compute runs on KFD user queues with no per-process busy time, so engines are unavailable.
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            pdev = "0000:03:00.0"
            dev = tmp / "drm" / "card1" / "device"
            dev.mkdir(parents=True)
            card = _card(dev, pdev)
            proc = tmp / "proc"
            _write_fd(proc, 100, 3, "pos:\t0\nflags:\t02\n", target="/dev/kfd")
            _write_fd(proc, 100, 7, _fdinfo(500, pdev, vram=3891436))
            _write_fd(proc, 200, 7, _fdinfo(501, pdev, gfx=5, vram=4))  # a graphics client, for contrast
            clock = Clock()
            counters = drm.DrmCounters(found=[card], proc=str(proc), clock=clock)
            counters.start()
            clock.tick()
            procs, _ = counters.sample()
            self.assertEqual(procs[card.key][100].dedicated, 3891436 * 1024)
            self.assertTrue(procs[card.key][100].kfd)
            self.assertNotIn("engine", procs[card.key][100].available)
            self.assertIn("engine", procs[card.key][200].available)


class DrmEdgeCaseTests(unittest.TestCase):
    def _counters(self, tmp, fds, unified_video=False, sysfs=True):
        pdev = "0000:03:00.0"
        dev = tmp / "drm" / "card1" / "device"
        dev.mkdir(parents=True)
        if sysfs:
            (dev / "mem_info_vram_used").write_text(str(GiB))
            (dev / "mem_info_gtt_used").write_text("0")
        card = _card(dev, pdev)
        card.unified_video = unified_video
        proc = tmp / "proc"
        for fd_spec in fds:
            pid, fd, text, *target = fd_spec
            _write_fd(proc, pid, fd, text.replace("PDEV", pdev), *target)
        self.clock = Clock()
        counters = drm.DrmCounters(found=[card], proc=str(proc), clock=self.clock)
        counters.start()
        self.clock.tick()
        return card, proc, counters

    def test_old_kernel_without_shared_keys_is_not_summed(self):
        old = "drm-driver:\tamdgpu\ndrm-client-id:\t5\ndrm-pdev:\tPDEV\ndrm-memory-vram:\t4096 KiB\n"
        with tempfile.TemporaryDirectory() as tmp:
            card, _, counters = self._counters(Path(tmp), [(100, 5, old)])
            g = counters.sample()[0][card.key][100]
            self.assertFalse(g.attributed)  # resident may include imports: would double count
            self.assertEqual(g.drm_reason, drm.REASON_LEGACY)
            self.assertEqual(g.dedicated, 0)
            self.assertNotIn("dedicated", g.available)
            self.assertNotIn("shared", g.available)
            self.assertEqual(g.resident_vram, 4096 * 1024)  # still shown as a reference

    def test_any_shared_buffer_leaves_the_process_unattributed(self):
        # drm-shared-* counts by preferred placement and drm-resident-* by current placement, so resident minus
        # shared is not this process's private memory. Any sharing at all: n/a, never a subtraction.
        for extra in ({"shared_vram": 2}, {"shared_gtt": 4}, {"shared_cpu": 6}):
            with self.subTest(extra), tempfile.TemporaryDirectory() as tmp:
                card, _, counters = self._counters(Path(tmp), [(100, 5, _fdinfo(7, "PDEV", vram=3, gtt=10, **extra))])
                g = counters.sample()[0][card.key][100]
                self.assertEqual(g.drm_reason, drm.REASON_SHARED_BUFFERS)
                self.assertIsNone(g.shown_vram)
                self.assertNotIn("shared", g.available)
                self.assertEqual(g.resident_vram, 3 * 1024)
                self.assertEqual(g.resident_gtt, 10 * 1024)
                self.assertIn("engine", g.available)  # sharing buffers doesn't share the client's engine time

    def test_primary_node_is_not_charged(self):
        # logind (root) holds a copy of a compositor's /dev/dri/card* fd that a normal user can't see.
        with tempfile.TemporaryDirectory() as tmp:
            card, _, counters = self._counters(Path(tmp), [(100, 5, _fdinfo(7, "PDEV", vram=3), "/dev/dri/card1")])
            g = counters.sample()[0][card.key][100]
            self.assertEqual(g.drm_reason, drm.REASON_PRIMARY)
            self.assertIsNone(g.shown_vram)

    def test_unparseable_memory_is_unknown(self):
        text = _fdinfo(7, "PDEV", vram=3).replace("drm-resident-vram:\t3 KiB", "drm-resident-vram:\tlots")
        with tempfile.TemporaryDirectory() as tmp:
            card, _, counters = self._counters(Path(tmp), [(100, 5, text)])
            g = counters.sample()[0][card.key][100]
            self.assertIsNone(g.resident_vram)
            self.assertIsNone(g.shown_vram)

    def test_torn_engine_counter_is_unknown(self):
        text = _fdinfo(7, "PDEV").replace("drm-engine-gfx:\t0 ns", "drm-engine-gfx:\t")
        with tempfile.TemporaryDirectory() as tmp:
            card, _, counters = self._counters(Path(tmp), [(100, 5, text)])
            self.assertNotIn("engine", counters.sample()[0][card.key][100].available)

    def test_engine_counter_stepping_back_is_not_counted_twice(self):
        with tempfile.TemporaryDirectory() as tmp:
            card, proc, counters = self._counters(Path(tmp), [(100, 5, _fdinfo(7, "PDEV", gfx=1_000_000_000))])
            path = proc / "100" / "fdinfo" / "5"
            path.write_text(_fdinfo(7, card.pdev, gfx=900_000_000))  # the DRM spec allows a brief step back
            self.assertEqual(counters.sample()[0][card.key][100].engines.get("3d", 0.0), 0.0)
            self.clock.tick()
            path.write_text(_fdinfo(7, card.pdev, gfx=1_500_000_000))
            self.assertEqual(counters.sample()[0][card.key][100].engines["3d"], 50.0)  # from the high-water mark

    def test_unified_vcn_reports_video(self):
        with tempfile.TemporaryDirectory() as tmp:
            card, proc, counters = self._counters(Path(tmp), [(100, 5, _fdinfo(9, "PDEV").replace(
                "drm-engine-gfx:\t0 ns", "drm-engine-enc:\t0 ns"))], unified_video=True)
            (proc / "100" / "fdinfo" / "5").write_text(_fdinfo(9, card.pdev).replace(
                "drm-engine-gfx:\t0 ns", "drm-engine-enc:\t1000000000 ns"))
            engines = counters.sample()[0][card.key][100].engines
            self.assertEqual(engines["video"], 100.0)
            self.assertNotIn("encode", engines)

    def test_missing_sysfs_memory_is_unknown(self):
        with tempfile.TemporaryDirectory() as tmp:
            card, _, counters = self._counters(Path(tmp), [], sysfs=False)
            a = counters.sample()[1][card.key]
            self.assertIsNone(a.dedicated)
            self.assertIsNone(a.shared)
            self.assertIsNone(drm.read_device(card, 0).mem_used)

    def test_power_average_of_zero_falls_back_to_input(self):
        with tempfile.TemporaryDirectory() as tmp:
            dev = Path(tmp) / "device"
            hwmon = dev / "hwmon" / "hwmon0"
            hwmon.mkdir(parents=True)
            (hwmon / "power1_average").write_text("0\n")
            (hwmon / "power1_input").write_text("42000000\n")
            self.assertEqual(drm.read_device(_card(dev, "0000:03:00.0"), 0).power_w, 42.0)
            (hwmon / "power1_average").unlink()
            (hwmon / "power1_input").write_text("0\n")
            self.assertEqual(drm.read_device(_card(dev, "0000:03:00.0"), 0).power_w, 0.0)  # measured 0 stays 0

    def test_vcn_major_from_ip_discovery(self):
        with tempfile.TemporaryDirectory() as tmp:
            uvd = Path(tmp) / "ip_discovery" / "die" / "0" / "UVD" / "0"
            uvd.mkdir(parents=True)
            (uvd / "major").write_text("5\n")
            self.assertEqual(drm._vcn_major(tmp), 5)
            self.assertEqual(drm._vcn_major(str(Path(tmp) / "none")), 0)


class DrmScanTests(unittest.TestCase):
    PDEV = "0000:03:00.0"

    def _setup(self, tmp, timer=None):
        dev = tmp / "drm" / "card1" / "device"
        dev.mkdir(parents=True)
        card = _card(dev, self.PDEV)
        proc = tmp / "proc"
        proc.mkdir()
        self.clock = Clock()
        kw = {"clock": self.clock}
        if timer is not None:
            kw["timer"] = timer
        return card, proc, kw

    def test_new_client_engine_load_is_unknown_until_two_samples(self):
        with tempfile.TemporaryDirectory() as tmp:
            card, proc, kw = self._setup(Path(tmp))
            _write_fd(proc, 100, 5, _fdinfo(1, self.PDEV, gfx=0, vram=4096))
            counters = drm.DrmCounters(found=[card], proc=str(proc), **kw)
            counters.start()
            self.clock.tick()
            _write_fd(proc, 200, 5, _fdinfo(2, self.PDEV, gfx=500_000_000, vram=4096))
            procs = counters.sample()[0][card.key]
            self.assertIn("engine", procs[100].available)  # in the baseline: a real, measured idle
            self.assertNotIn("engine", procs[200].available)  # first seen now: unknown, not 0
            self.assertEqual(procs[200].dedicated, 4096 * 1024)  # memory doesn't need two samples
            self.assertFalse(procs[200].kfd)
            self.clock.tick()
            _write_fd(proc, 200, 5, _fdinfo(2, self.PDEV, gfx=1_000_000_000, vram=4096))
            self.assertEqual(counters.sample()[0][card.key][200].engines["3d"], 50.0)

    def test_engine_first_reported_counts_from_zero(self):
        # amdgpu omits an engine until it has run, so its whole counter is new work.
        with tempfile.TemporaryDirectory() as tmp:
            card, proc, kw = self._setup(Path(tmp))
            _write_fd(proc, 100, 5, _fdinfo(1, self.PDEV).replace("drm-engine-gfx:\t0 ns\n", ""))
            counters = drm.DrmCounters(found=[card], proc=str(proc), **kw)
            counters.start()
            self.clock.tick()
            _write_fd(proc, 100, 5, _fdinfo(1, self.PDEV, gfx=250_000_000))
            self.assertEqual(counters.sample()[0][card.key][100].engines["3d"], 25.0)

    def test_client_held_by_several_processes_is_charged_to_none(self):
        with tempfile.TemporaryDirectory() as tmp:
            card, proc, kw = self._setup(Path(tmp))
            _write_fd(proc, 100, 5, _fdinfo(1, self.PDEV, gfx=0, vram=4096))
            _write_fd(proc, 200, 9, _fdinfo(1, self.PDEV, gfx=0, vram=4096))
            _write_fd(proc, 300, 5, _fdinfo(2, self.PDEV, gfx=0, vram=8192))
            counters = drm.DrmCounters(found=[card], proc=str(proc), **kw)
            counters.start()
            self.clock.tick()
            procs = counters.sample()[0][card.key]
            for pid in (100, 200):
                self.assertEqual(procs[pid].drm_reason, drm.REASON_SHARED_CLIENT)
                self.assertIsNone(procs[pid].shown_vram)
                self.assertNotIn("engine", procs[pid].available)
            self.assertEqual(procs[300].shown_vram, 8192 * 1024)
            # The other rows still add up: the shared client's bytes are left out, not the whole total.
            self.assertEqual(sum(g.dedicated for g in procs.values()), 8192 * 1024)

    def test_private_and_co_owned_clients_leave_the_whole_process_unattributed(self):
        # pid 100 alone holds client 1 and shares client 2 with pid 200: charging client 1 to it would depend on
        # which other processes happened to be visible, so neither process gets bytes.
        with tempfile.TemporaryDirectory() as tmp:
            card, proc, kw = self._setup(Path(tmp))
            _write_fd(proc, 100, 5, _fdinfo(1, self.PDEV, vram=1024))
            _write_fd(proc, 100, 6, _fdinfo(2, self.PDEV, vram=2048))
            _write_fd(proc, 200, 5, _fdinfo(2, self.PDEV, vram=2048))
            counters = drm.DrmCounters(found=[card], proc=str(proc), **kw)
            counters.start()
            procs = counters.sample()[0][card.key]
            for pid in (100, 200):
                self.assertEqual(procs[pid].drm_reason, drm.REASON_SHARED_CLIENT)
                self.assertEqual(procs[pid].dedicated, 0)
                self.assertNotIn("dedicated", procs[pid].available)
            self.assertEqual(procs[100].resident_vram, 3072 * 1024)

    def test_uninspectable_processes_are_reported(self):
        with tempfile.TemporaryDirectory() as tmp:
            card, proc, kw = self._setup(Path(tmp))
            _write_fd(proc, 100, 5, _fdinfo(1, self.PDEV, vram=4096))
            (proc / "200" / "fd").mkdir(parents=True)
            real = os.scandir

            def scandir(path):
                if str(path).endswith(os.path.join("200", "fd")):
                    raise PermissionError(path)
                return real(path)

            with patch.object(drm.os, "scandir", scandir):
                counters = drm.DrmCounters(found=[card], proc=str(proc), **kw)
                self.assertIn(100, counters.sample()[0][card.key])
            self.assertIn("1 processes not inspectable", counters.status)
            counters.sample()
            self.assertEqual(counters.status, "")  # readable now

    def test_unreadable_fd_after_listing_is_incomplete(self):
        with tempfile.TemporaryDirectory() as tmp:
            card, proc, kw = self._setup(Path(tmp))
            _write_fd(proc, 100, 5, _fdinfo(1, self.PDEV, vram=4096))
            _write_fd(proc, 100, 6, _fdinfo(2, self.PDEV, vram=4096))
            real = os.readlink

            def readlink(path):
                if str(path).endswith(os.path.join("100", "fd", "6")):
                    raise PermissionError(path)
                return real(path)

            counters = drm.DrmCounters(found=[card], proc=str(proc), **kw)
            with patch.object(drm.os, "readlink", readlink):
                g = counters.sample()[0][card.key][100]
            self.assertEqual(g.drm_reason, drm.REASON_INCOMPLETE)  # client 2 might be shared: don't charge 1
            self.assertIsNone(g.shown_vram)

    def test_unreadable_fdinfo_is_incomplete(self):
        with tempfile.TemporaryDirectory() as tmp:
            card, proc, kw = self._setup(Path(tmp))
            _write_fd(proc, 100, 5, _fdinfo(1, self.PDEV, vram=4096))
            _write_fd(proc, 100, 6, _fdinfo(2, self.PDEV, vram=4096))
            real = open

            def guarded(path, *a, **k):
                if str(path).endswith(os.path.join("fdinfo", "6")):
                    raise PermissionError(path)
                return real(path, *a, **k)

            counters = drm.DrmCounters(found=[card], proc=str(proc), **kw)
            with patch("builtins.open", guarded):
                g = counters.sample()[0][card.key][100]
            self.assertEqual(g.drm_reason, drm.REASON_INCOMPLETE)

    def test_reused_pid_drops_cached_fds(self):
        with tempfile.TemporaryDirectory() as tmp:
            card, proc, kw = self._setup(Path(tmp))
            _write_fd(proc, 100, 5, _fdinfo(1, self.PDEV, vram=4096))
            (proc / "100" / "stat").write_text("100 (old) S 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 1000 0\n")
            counters = drm.DrmCounters(found=[card], proc=str(proc), **kw)
            counters.sample()
            self.assertIn(100, counters._known)
            (proc / "100" / "stat").write_text("100 (new) S 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 2000 0\n")
            counters.scan_budget = 0.0  # nothing is rescanned: only the cache could keep pid 100
            self.assertNotIn(100, counters.sample()[0][card.key])
            self.assertNotIn(100, counters._known)

    def test_scan_budget_keeps_rows_and_resumes(self):
        with tempfile.TemporaryDirectory() as tmp:
            ticks = [0.0]

            def timer():
                ticks[0] += 1.0
                return ticks[0]

            card, proc, kw = self._setup(Path(tmp), timer)
            for pid in (100, 200, 300, 400):
                _write_fd(proc, pid, 5, _fdinfo(pid, self.PDEV, gfx=0, vram=4096))
            counters = drm.DrmCounters(found=[card], proc=str(proc), **kw)
            counters.start()  # the baseline is never limited
            self.assertEqual(sorted(counters._known), [100, 200, 300, 400])
            counters.scan_budget = 2.5  # one process per refresh with this timer
            _write_fd(proc, 500, 5, _fdinfo(500, self.PDEV, gfx=0, vram=4096))
            rows = counters.sample()[0][card.key]
            self.assertTrue({100, 200, 300, 400} <= set(rows))  # no row flickers while the scan is partial
            self.assertIn(500, rows)  # processes not seen before are scanned first
            self.assertIn("scan budget reached", counters.status)
            counters.scan_budget = None
            counters.sample()
            self.assertEqual(counters.status, "")

    def test_budget_is_checked_inside_one_process(self):
        with tempfile.TemporaryDirectory() as tmp:
            ticks = [0.0]

            def timer():
                ticks[0] += 1.0
                return ticks[0]

            card, proc, kw = self._setup(Path(tmp), timer)
            _write_fd(proc, 50, 5, _fdinfo(1, self.PDEV, vram=4096))
            for fd in range(200):  # a process with many fds
                _write_fd(proc, 100, fd, "pos:\t0\n", target=f"socket:[{fd}]")
            counters = drm.DrmCounters(found=[card], proc=str(proc), **kw)
            counters.scan_budget = 3.5  # pid 50 and the first 127 fds of pid 100
            counters.sample()
            self.assertIn(50, counters._known)  # the first process of a refresh is listed in full
            self.assertNotIn(100, counters._known)  # interrupted partway: deferred, not cached as fd-less
            self.assertIn("scan budget reached", counters.status)

    def test_one_huge_process_does_not_stall_the_rotation(self):
        with tempfile.TemporaryDirectory() as tmp:
            ticks = [0.0]

            def timer():
                ticks[0] += 1.0
                return ticks[0]

            card, proc, kw = self._setup(Path(tmp), timer)
            for fd in range(200):
                _write_fd(proc, 100, fd, "pos:\t0\n", target=f"socket:[{fd}]")
            _write_fd(proc, 200, 5, _fdinfo(2, self.PDEV, vram=4096))
            counters = drm.DrmCounters(found=[card], proc=str(proc), **kw)
            counters.scan_budget = 1.5
            for _ in range(3):
                counters.sample()
            self.assertEqual(sorted(counters._known), [100, 200])

    def test_partial_scan_charges_no_one(self):
        # A process the budget skipped may hold the same client, so single ownership isn't proven this refresh.
        with tempfile.TemporaryDirectory() as tmp:
            ticks = [0.0]

            def timer():
                ticks[0] += 1.0
                return ticks[0]

            card, proc, kw = self._setup(Path(tmp), timer)
            for pid in (100, 200):
                _write_fd(proc, pid, 5, _fdinfo(pid, self.PDEV, vram=4096))
            counters = drm.DrmCounters(found=[card], proc=str(proc), **kw)
            counters.start()
            counters.scan_budget = 1.5
            procs = counters.sample()[0][card.key]
            self.assertTrue(counters.partial)
            for pid in (100, 200):
                self.assertEqual(procs[pid].drm_reason, drm.REASON_PARTIAL_SCAN)
                self.assertIsNone(procs[pid].shown_vram)
                self.assertNotIn("engine", procs[pid].available)
                self.assertEqual(procs[pid].resident_vram, 4096 * 1024)

    def test_rotation_resumes_at_a_known_process(self):
        with tempfile.TemporaryDirectory() as tmp:
            ticks = [0.0]

            def timer():
                ticks[0] += 1.0
                return ticks[0]

            card, proc, kw = self._setup(Path(tmp), timer)
            for pid in (100, 200, 300):
                _write_fd(proc, pid, 5, _fdinfo(pid, self.PDEV, vram=4096))
            counters = drm.DrmCounters(found=[card], proc=str(proc), **kw)
            counters.start()
            for pid in (900, 901):  # new processes are scanned first and run out the budget
                _write_fd(proc, pid, 5, _fdinfo(pid, self.PDEV, vram=4096))
            counters._cursor = 200
            counters.scan_budget = 1.5
            counters.sample()
            self.assertEqual(counters._cursor, 200)  # not 901: known processes 200 and 300 aren't skipped

    def test_incomplete_process_has_no_partial_totals(self):
        with tempfile.TemporaryDirectory() as tmp:
            card, proc, kw = self._setup(Path(tmp))
            _write_fd(proc, 100, 5, _fdinfo(1, self.PDEV, gfx=0, vram=4096))
            _write_fd(proc, 100, 6, _fdinfo(2, self.PDEV, gfx=0, vram=4096))
            real = open

            def guarded(path, *a, **k):
                if str(path).endswith(os.path.join("fdinfo", "6")):
                    raise PermissionError(path)
                return real(path, *a, **k)

            counters = drm.DrmCounters(found=[card], proc=str(proc), **kw)
            with patch("builtins.open", guarded):
                counters.start()
                self.clock.tick()
                g = counters.sample()[0][card.key][100]
            self.assertIsNone(g.resident_vram)
            self.assertNotIn("engine", g.available)
            self.assertIn("1 processes whose fds or fdinfo", counters.status)

class DrmReportTests(unittest.TestCase):
    def test_json_and_text_keep_unknown_apart_from_zero(self):
        from gputaskman.collector import ProcRow, Snapshot
        from gputaskman.pdh import ProcGpu
        from gputaskman.procinfo import ProcDetail
        from gputaskman.report import DRM_NOTES, to_dict, to_text

        private = ProcGpu(dedicated=GiB, shared=0, available={"dedicated", "shared", "engine"},
                          resident_vram=float(GiB), resident_gtt=0.0)
        compositor = ProcGpu(attributed=False, available={"engine"}, resident_vram=float(2 * GiB),
                             resident_gtt=None, drm_reason=drm.REASON_SHARED_BUFFERS)
        rows = [ProcRow(100, "llama-server", "LLM", private, ProcDetail(100, "llama-server")),
                ProcRow(200, "kwin_wayland", "compositor", compositor, ProcDetail(200, "kwin_wayland"))]
        snap = Snapshot(taken=1, adapter=None, adapters=[], nv=None, driver="", cuda="", vram_total=16 * GiB,
                        vram_used=4 * GiB, shared_used=0, backend="drm", procs=rows, shared_is_spill=True)
        data = to_dict(snap)
        self.assertTrue(data["shared_is_spill"])
        llama, kwin = data["processes"]
        self.assertEqual((llama["vram_bytes"], llama["shared_bytes"], llama["vram_resident_bytes"]), (GiB, 0, GiB))
        self.assertIsNone(llama["vram_unattributed_reason"])
        self.assertEqual((kwin["vram_bytes"], kwin["shared_bytes"]), (None, None))
        self.assertEqual((kwin["vram_resident_bytes"], kwin["gtt_resident_bytes"]), (2 * GiB, None))
        self.assertEqual(kwin["vram_unattributed_reason"], "shared_buffers")
        text = to_text(snap)
        line = next(x for x in text.splitlines() if x.lstrip().startswith("200"))
        self.assertEqual(line.split()[3:5], ["n/a", "N/A"])  # VRAM and SPILL unknown, not blank or 0
        self.assertIn(DRM_NOTES["shared_buffers"], text)
        self.assertNotIn(DRM_NOTES["legacy"], text)


class EngineAndSelectionTests(unittest.TestCase):
    def test_engine_names(self):
        self.assertEqual(drm._engine_key("gfx"), "3d")
        self.assertEqual(drm._engine_key("compute"), "compute")
        self.assertEqual(drm._engine_key("dma"), "copy")
        self.assertEqual(drm._engine_key("enc_1"), "encode")
        self.assertEqual(drm._engine_key("jpeg"), "decode")
        self.assertEqual(drm._engine_key("vpe"), "other")
        self.assertEqual(drm._engine_key("enc", unified_video=True), "video")
        self.assertEqual(drm._engine_key("gfx", unified_video=True), "3d")
        self.assertEqual(drm._engine_key("rcs0", intel=True), "3d")
        self.assertEqual(drm._engine_key("ccs0", intel=True), "compute")
        self.assertEqual(drm._engine_key("bcs0", intel=True), "copy")
        self.assertEqual(drm._engine_key("vcs0", intel=True), "video")
        self.assertEqual(drm._engine_key("vecs0", intel=True), "encode")
        self.assertEqual(drm._engine_key("rcs0", intel=False), "other")  # Intel-only prefix on AMD

    def test_intel_selectable_next_to_nvidia_on_linux(self):
        from gputaskman import collector
        from gputaskman.dxgi import Adapter

        nvidia = Adapter(1, "NVIDIA GPU", 0x10DE, 8 * GiB, 0, False)
        intel = Adapter(2, "Intel Arc A770", 0x8086, 16 * GiB, 0, False)
        with patch.object(collector, "BACKEND", "nvml"):
            self.assertIs(collector.select_adapter([intel, nvidia], 0), nvidia)
            self.assertIs(collector.select_adapter([intel, nvidia], 1), intel)


class IntelCardTests(unittest.TestCase):
    """Intel (i915/xe) cards through the same DRM fdinfo backend, from a synthetic sysfs tree."""

    INTEL = drm.DrmCard(key=12288, pdev="0000:03:00.0", sysfs="/x/card2/device", driver="i915",
                        vendor_id=0x8086, device_id=0x56A0, name="Intel Arc A770",
                        vram_total=16 * GiB, integrated=False, intel=True)
    IGPU = drm.DrmCard(key=32, pdev="0000:00:02.0", sysfs="/x/card1/device", driver="i915",
                       vendor_id=0x8086, device_id=0xA782, name="Intel UHD Graphics",
                       vram_total=0, integrated=True, intel=True)
    AMD = drm.DrmCard(key=0, pdev="0000:43:00.0", sysfs="/x/amd/device", driver="amdgpu",
                      vendor_id=0x1002, device_id=0x7480, name="AMD GPU", vram_total=8 * GiB,
                      integrated=False)

    def make(self, cards):
        return drm.DrmCounters(found=cards)

    def test_intel_engine_names(self):
        self.assertEqual(drm._engine_key("rcs0", intel=True), "3d")
        self.assertEqual(drm._engine_key("ccs0", intel=True), "compute")
        self.assertEqual(drm._engine_key("bcs0", intel=True), "copy")
        self.assertEqual(drm._engine_key("vcs0", intel=True), "video")
        self.assertEqual(drm._engine_key("vecs0", intel=True), "encode")
        self.assertEqual(drm._engine_key("rcs0", intel=False), "other")

    def test_intel_card_is_recognized_and_amd_unchanged(self):
        counters = self.make([self.INTEL])
        self.assertTrue(counters.ok)
        self.assertEqual([c.pdev for c in counters.cards], ["0000:03:00.0"])
        self.assertTrue(counters.cards[0].intel)

    def test_intel_fdinfo_fields_parse(self):
        text = ("drm-driver:\ti915\n"
                "drm-client-id:\t99\n"
                "drm-pdev:\t0000:03:00.0\n"
                "drm-engine-rcs0:\t1200000000 ns\n"
                "drm-engine-bcs0:\t400000000 ns\n"
                "drm-engine-ccs0:\t800000000 ns\n"
                "drm-engine-vcs0:\t200000000 ns\n"
                "drm-engine-vecs0:\t100000000 ns\n"
                "drm-resident-local0:\t1073741824\n"
                "drm-resident-stolen-local0:\t1048576\n"
                "drm-resident-system0:\t268435456\n")
        fields = drm.parse_fdinfo(text)
        self.assertEqual(fields["driver"], "i915")
        self.assertEqual(fields["pdev"], "0000:03:00.0")
        self.assertEqual(fields["client_id"], 99)
        self.assertEqual(fields["engines"], {"rcs0": 1_200_000_000, "bcs0": 400_000_000,
                                             "ccs0": 800_000_000, "vcs0": 200_000_000,
                                             "vecs0": 100_000_000})
        self.assertEqual(fields["vram"], GiB + 1_048_576)  # local0 + stolen-local0
        self.assertEqual(fields["gtt"], 268_435_456)  # system0
        self.assertIsNone(fields["shared"])  # shared gate needs both Intel shared regions

    def test_intel_missing_regions_stay_unknown_not_zero(self):
        text = ("drm-driver:\txe\n"
                "drm-client-id:\t7\n"
                "drm-engine-rcs0:\t1000 ns\n")
        fields = drm.parse_fdinfo(text)
        self.assertIsNone(fields["vram"])  # an iGPU has no local0: unknown, not 0
        self.assertIsNone(fields["gtt"])

    def test_intel_adapters_trust_the_client_stamp(self):
        self.INTEL.vram_used = 15 * GiB
        counters = self.make([self.INTEL])
        out = counters._adapters()
        self.assertEqual(out[self.INTEL.key].dedicated, float(15 * GiB))
        self.assertIsNone(out[self.INTEL.key].shared)  # no device-wide shared counter on Intel
        self.IGPU.vram_used = None
        counters = self.make([self.IGPU])
        out = counters._adapters()
        self.assertIsNone(out[self.IGPU.key].dedicated)
        self.assertIsNone(out[self.IGPU.key].shared)


if __name__ == "__main__":
    unittest.main()

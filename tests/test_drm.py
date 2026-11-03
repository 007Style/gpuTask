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


def _fdinfo(client, pdev, gfx=0, vram=0, gtt=0, shared_vram=0) -> str:
    return (
        "pos:\t0\n"
        "drm-driver:\tamdgpu\n"
        f"drm-client-id:\t{client}\n"
        f"drm-pdev:\t{pdev}\n"
        f"drm-memory-vram:\t{vram} KiB\n"
        f"drm-shared-vram:\t{shared_vram} KiB\n"
        f"drm-memory-gtt: \t{gtt} KiB\n"
        "drm-memory-cpu: \t0 KiB\n"
        f"drm-engine-gfx:\t{gfx} ns\n"
        f"drm-engine-capacity-gfx:\t1\n"
    )


class ParseTests(unittest.TestCase):
    def test_parse_size(self):
        self.assertEqual(drm.parse_size("2060 KiB"), 2060 * 1024)
        self.assertEqual(drm.parse_size("2 MiB"), 2 * 1024**2)
        self.assertEqual(drm.parse_size("1 GiB"), 1024**3)
        self.assertEqual(drm.parse_size("12 KiB"), 12 * 1024)
        self.assertEqual(drm.parse_size("0"), 0)
        self.assertEqual(drm.parse_size("512"), 512)
        self.assertEqual(drm.parse_size("  4 KiB  "), 4096)
        self.assertEqual(drm.parse_size("bogus"), 0)
        self.assertEqual(drm.parse_size("12 bananas"), 0)

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
        self.assertIsNone(info["shared_vram"])  # no drm-shared-* keys: unknown, not 0

    def test_parse_fdinfo_prefers_resident_keys(self):
        info = drm.parse_fdinfo("drm-driver:\tamdgpu\ndrm-resident-vram:\t8 KiB\ndrm-memory-vram:\t9 KiB\n"
                                "drm-shared-vram:\t0\ndrm-shared-gtt:\t4 KiB\n")
        self.assertEqual(info["vram"], 8 * 1024)
        self.assertEqual(info["shared_vram"], 0)
        self.assertEqual(info["shared_gtt"], 4 * 1024)

    def test_parse_fdinfo_shared_vram(self):
        info = drm.parse_fdinfo("drm-driver:\tamdgpu\ndrm-memory-vram:\t300 MiB\ndrm-shared-vram:\t200 MiB\n")
        self.assertEqual(info["vram"], 300 * 1024**2)
        self.assertEqual(info["shared_vram"], 200 * 1024**2)  # part of vram, not on top of it


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
            _write_fd(proc, 100, 10, _fdinfo(26, pdev, gfx=1_000_000_000, vram=3, gtt=2, shared_vram=2))
            _write_fd(proc, 200, 11, _fdinfo(26, pdev, gfx=1_000_000_000, vram=1, gtt=2))
            _write_fd(proc, 300, 12, _fdinfo(27, "0000:ff:00.0", gfx=9_999_999_999, vram=4))
            # DRM-looking fdinfo behind a non-DRM fd: the /proc/<pid>/fd target rules it out.
            _write_fd(proc, 400, 13, _fdinfo(28, pdev, gfx=9_999_999_999, vram=4), target="socket:[123]")

            times = iter([0, 1_000_000_000, 2_000_000_000])  # the constructor takes the baseline at 0
            counters = drm.DrmCounters(found=[card], proc=str(proc), clock=lambda: next(times))

            self.assertTrue(counters.ok)
            procs, adapters = counters.sample()
            self.assertEqual(list(procs), [card.key])
            self.assertIn(100, procs[card.key])
            self.assertNotIn(200, procs[card.key])  # same client, counted once, on the lowest pid
            self.assertNotIn(300, procs[card.key])  # unknown pdev
            self.assertNotIn(400, procs[card.key])  # non-DRM fd
            self.assertEqual(procs[card.key][100].dedicated, 1024)  # private only: 3 KiB resident - 2 KiB shared
            self.assertEqual(procs[card.key][100].shared_vram, 2048)
            self.assertFalse(procs[card.key][100].paged_available)  # no commit counter: unavailable, not 0
            self.assertEqual(procs[card.key][100].shared, 2048)
            self.assertEqual(procs[card.key][100].committed, 0)
            self.assertEqual(procs[card.key][100].engines.get("3d", 0.0), 0.0)  # idle since the baseline
            self.assertEqual(adapters[card.key].dedicated, 1 * GiB)
            self.assertEqual(adapters[card.key].shared, 64 * 1024)

            (dev / "mem_info_vram_used").write_text(str(2 * GiB))
            _write_fd(proc, 100, 10, _fdinfo(26, pdev, gfx=1_500_000_000, vram=1, gtt=2))

            procs, adapters = counters.sample()
            self.assertEqual(procs[card.key][100].engines["3d"], 50.0)
            self.assertIn("engine", procs[card.key][100].available)
            self.assertEqual(adapters[card.key].dedicated, 2 * GiB)
            counters.close()

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
            times = iter([0, 1_000_000_000])
            counters = drm.DrmCounters(found=[card], proc=str(proc), clock=lambda: next(times))
            procs, _ = counters.sample()
            self.assertEqual(procs[card.key][100].dedicated, 3891436 * 1024)
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
        for pid, fd, text in fds:
            _write_fd(proc, pid, fd, text.replace("PDEV", pdev))
        times = iter([0, 1_000_000_000, 2_000_000_000])
        return card, proc, drm.DrmCounters(found=[card], proc=str(proc), clock=lambda: next(times))

    def test_old_kernel_without_shared_keys_is_not_summed(self):
        old = "drm-driver:\tamdgpu\ndrm-client-id:\t5\ndrm-pdev:\tPDEV\ndrm-memory-vram:\t4096 KiB\n"
        with tempfile.TemporaryDirectory() as tmp:
            card, _, counters = self._counters(Path(tmp), [(100, 5, old)])
            g = counters.sample()[0][card.key][100]
            self.assertFalse(g.attributed)  # resident includes imports: would double count
            self.assertEqual(g.dedicated, 0)
            self.assertNotIn("dedicated", g.available)

    def test_shared_gtt_left_out_of_spill(self):
        text = _fdinfo(7, "PDEV", gtt=10) + "drm-shared-gtt:\t4 KiB\n"
        with tempfile.TemporaryDirectory() as tmp:
            card, _, counters = self._counters(Path(tmp), [(100, 5, text)])
            self.assertEqual(counters.sample()[0][card.key][100].shared, 6 * 1024)

    def test_unified_vcn_reports_video(self):
        with tempfile.TemporaryDirectory() as tmp:
            text = "drm-driver:\tamdgpu\ndrm-client-id:\t9\ndrm-pdev:\tPDEV\ndrm-shared-vram:\t0\ndrm-engine-enc:\t{} ns\n"
            card, proc, counters = self._counters(Path(tmp), [(100, 5, text.format(0))], unified_video=True)
            (proc / "100" / "fdinfo" / "5").write_text(text.format(1_000_000_000).replace("PDEV", card.pdev))
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

    def test_vcn_major_from_ip_discovery(self):
        with tempfile.TemporaryDirectory() as tmp:
            uvd = Path(tmp) / "ip_discovery" / "die" / "0" / "UVD" / "0"
            uvd.mkdir(parents=True)
            (uvd / "major").write_text("5\n")
            self.assertEqual(drm._vcn_major(tmp), 5)
            self.assertEqual(drm._vcn_major(str(Path(tmp) / "none")), 0)


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

    def test_amd_selectable_next_to_nvidia_on_linux(self):
        from gputaskman import collector
        from gputaskman.dxgi import Adapter

        nvidia = Adapter(1, "NVIDIA GPU", 0x10DE, 8 * GiB, 0, False)
        amd = Adapter(2, "AMD Radeon 610M", 0x1002, 2 * GiB, 0, False)
        with patch.object(collector, "BACKEND", "nvml"):
            self.assertIs(collector.select_adapter([amd, nvidia], 0), nvidia)
            self.assertIs(collector.select_adapter([amd, nvidia], 1), amd)


if __name__ == "__main__":
    unittest.main()

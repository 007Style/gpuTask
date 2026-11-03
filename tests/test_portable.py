from pathlib import Path
import struct
import tempfile
import unittest
import zipfile
import hashlib

from tools.build_portable import architecture, check_executable, create_archive, nvidia_license, validate_version


class PortableTests(unittest.TestCase):
    def write_executable(self, path, machine):
        header = bytearray(64)
        header[:2] = b"MZ"
        struct.pack_into("<I", header, 60, 64)
        path.write_bytes(header + b"PE\0\0" + struct.pack("<H", machine))

    def test_interpreter_architecture_not_host_architecture(self):
        self.assertEqual(architecture("win-amd64"), "x64")
        self.assertEqual(architecture("win-arm64"), "arm64")
        with self.assertRaises(ValueError):
            architecture("win32")

    def test_version_and_tag(self):
        validate_version("0.1.0", "v0.1.0")
        with self.assertRaises(ValueError):
            validate_version("0.1.0", "v0.2.0")
        with self.assertRaises(ValueError):
            validate_version("..\\outside")

    def test_inline_nvidia_license_is_preserved(self):
        notice = "#####\n# Copyright Example\n# Redistribution and use are permitted.\n#####\n"
        self.assertEqual(nvidia_license(notice + "import ctypes\n"), notice)
        with self.assertRaises(ValueError):
            nvidia_license("import ctypes\n")

    def test_executable_machine(self):
        with tempfile.TemporaryDirectory() as temporary:
            exe = Path(temporary) / "gputaskman.exe"
            for arch, machine in [("x64", 0x8664), ("arm64", 0xAA64)]:
                self.write_executable(exe, machine)
                check_executable(exe, arch)
            with self.assertRaises(ValueError):
                check_executable(exe, "x64")
            exe.write_bytes(b"not an exe")
            with self.assertRaises(ValueError):
                check_executable(exe, "x64")

    def test_archive_includes_complete_runtime_and_checksum(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            bundle = root / "bundle"
            (bundle / "_internal").mkdir(parents=True)
            self.write_executable(bundle / "gputaskman.exe", 0x8664)
            (bundle / "_internal" / "data.css").write_text("resource", encoding="utf-8")
            install_metadata = bundle / "_internal" / "gputaskman-0.1.0.dist-info"
            install_metadata.mkdir()
            (install_metadata / "direct_url.json").write_text(
                '{"url": "file:///private-build-path"}', encoding="utf-8",
            )
            archive = create_archive(bundle, root / "output", "0.1.0", "x64")
            with zipfile.ZipFile(archive) as zipped:
                self.assertEqual(set(zipped.namelist()), {
                    "gputaskman-0.1.0-windows-x64/gputaskman.exe",
                    "gputaskman-0.1.0-windows-x64/_internal/data.css",
                })
            expected = hashlib.sha256(archive.read_bytes()).hexdigest()
            self.assertEqual(
                archive.with_suffix(".zip.sha256").read_text(encoding="utf-8"),
                f"{expected}  {archive.name}\n",
            )

    def test_archive_rejects_driver_dlls(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.write_executable(root / "gputaskman.exe", 0xAA64)
            (root / "NVML.dll").write_bytes(b"driver")
            with self.assertRaisesRegex(ValueError, "must not be distributed"):
                create_archive(root, root / "output", "0.1.0", "arm64")


if __name__ == "__main__":
    unittest.main()

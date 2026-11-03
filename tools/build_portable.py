"""Build a native console executable (Windows or Linux) and archive its complete runtime."""

from __future__ import annotations

import argparse
import hashlib
from importlib import metadata
from pathlib import Path
import re
import shutil
import struct
import subprocess
import sys
import sysconfig
import tarfile
import tempfile
import tomllib
import zipfile


ROOT = Path(__file__).resolve().parents[1]
MACHINES = {"x64": 0x8664, "arm64": 0xAA64}  # PE machine
ELF_MACHINES = {"x64": 0x3E, "arm64": 0xB7}  # ELF e_machine
# Provided by Windows or the installed GPU driver, never by the portable app.
FORBIDDEN = ("nvml.dll", "nvcuda.dll", "atiadlxx.dll", "dxgi.dll", "pdh.dll", "libnvidia-ml.so", "libcuda.so", "libdxcore.so")
PLATFORMS = {"win-amd64": ("windows", "x64"), "win-arm64": ("windows", "arm64"),
             "linux-x86_64": ("linux", "x64"), "linux-aarch64": ("linux", "arm64")}


def target(platform_tag: str) -> tuple[str, str]:
    if platform_tag not in PLATFORMS:
        raise ValueError(f"Unsupported build interpreter: {platform_tag}")
    return PLATFORMS[platform_tag]


def architecture(platform_tag: str) -> str:
    return target(platform_tag)[1]


def is_forbidden(name: str) -> bool:
    name = name.lower()
    return any(name == f or name.startswith(f + ".") for f in FORBIDDEN)


def validate_version(version: str, tag: str | None = None) -> None:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9.+-]*", version):
        raise ValueError(f"Unsafe package version: {version!r}")
    if tag is not None and tag != f"v{version}":
        raise ValueError(f"Release tag {tag!r} must match package version v{version}")


def check_executable(path: Path, arch: str) -> None:
    with path.open("rb") as binary:
        header = binary.read(64)
        if len(header) != 64 or header[:2] != b"MZ":
            raise ValueError(f"Not a Windows executable: {path}")
        binary.seek(struct.unpack_from("<I", header, 60)[0])
        pe = binary.read(6)
    if len(pe) != 6 or pe[:4] != b"PE\0\0":
        raise ValueError(f"Invalid PE header: {path}")
    machine = struct.unpack_from("<H", pe, 4)[0]
    if machine != MACHINES[arch]:
        raise ValueError(f"Expected {arch} PE machine {MACHINES[arch]:#06x}, got {machine:#06x}")


def check_elf(path: Path, arch: str) -> None:
    with path.open("rb") as binary:
        header = binary.read(20)
    if len(header) != 20 or header[:4] != b"\x7fELF":
        raise ValueError(f"Not an ELF executable: {path}")
    if header[4] != 2 or header[5] != 1:
        raise ValueError(f"Expected a 64-bit little-endian ELF: {path}")
    machine = struct.unpack_from("<H", header, 18)[0]
    if machine != ELF_MACHINES[arch]:
        raise ValueError(f"Expected {arch} ELF machine {ELF_MACHINES[arch]:#06x}, got {machine:#06x}")


def _bundle_files(bundle: Path) -> list[Path]:
    # Editable-install origins contain build-machine paths and are not needed at runtime.
    files = sorted(
        path for path in bundle.rglob("*")
        if path.is_file()
        and not (path.name == "direct_url.json" and path.parent.name.endswith(".dist-info"))
    )
    for path in files:
        if is_forbidden(path.name):
            raise ValueError(f"Driver/system library must not be distributed: {path}")
    return files


def _checksum(archive: Path) -> None:
    with archive.open("rb") as packaged:
        digest = hashlib.file_digest(packaged, "sha256").hexdigest()
    Path(f"{archive}.sha256").write_text(f"{digest}  {archive.name}\n", encoding="utf-8")


def create_linux_archive(bundle: Path, output: Path, version: str, arch: str) -> Path:
    """tar.gz keeps the executable bit, which a ZIP made with zipfile would drop."""
    validate_version(version)
    check_elf(bundle / "gputaskman", arch)
    files = _bundle_files(bundle)
    name = f"gputaskman-{version}-linux-{arch}"
    output.mkdir(parents=True, exist_ok=True)
    archive = output / f"{name}.tar.gz"

    def normalize(info: tarfile.TarInfo) -> tarfile.TarInfo:
        info.uid = info.gid = 0
        info.uname = info.gname = ""
        if info.name == f"{name}/gputaskman":
            info.mode |= 0o755  # filesystems without an executable bit (Windows) would otherwise drop it
        return info

    with tarfile.open(archive, "w:gz") as packed:
        for path in files:
            packed.add(path, str(Path(name) / path.relative_to(bundle)), recursive=False, filter=normalize)
    _checksum(archive)
    return archive


def create_archive(bundle: Path, output: Path, version: str, arch: str) -> Path:
    validate_version(version)
    check_executable(bundle / "gputaskman.exe", arch)
    files = _bundle_files(bundle)
    name = f"gputaskman-{version}-windows-{arch}"
    output.mkdir(parents=True, exist_ok=True)
    archive = output / f"{name}.zip"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as zipped:
        for path in files:
            zipped.write(path, Path(name) / path.relative_to(bundle))
    _checksum(archive)
    return archive


def nvidia_license(source: str) -> str:
    header, boundary, _ = source.partition("\n#####\n")
    if not boundary or "Redistribution and use" not in header or "Copyright" not in header:
        raise ValueError("NVIDIA Python bindings' inline license was not found")
    return header + boundary


def python_license() -> Path:
    base = Path(sys.base_prefix)
    candidates = [base / "LICENSE.txt", Path(sysconfig.get_paths()["stdlib"]) / "LICENSE.txt",
                  base / "share" / "doc" / f"python{sys.version_info.major}.{sys.version_info.minor}" / "LICENSE"]
    for path in candidates:
        if path.is_file():
            return path
    raise RuntimeError("Python's license text was not found")


def add_notices(bundle: Path, system: str = "windows") -> None:
    sep = "\\" if system == "windows" else "/"
    shutil.copy2(ROOT / "LICENSE", bundle / "LICENSE")
    shutil.copy2(python_license(), bundle / "PYTHON-LICENSE.txt")
    installer = metadata.distribution("pyinstaller")
    license_files = [
        path for path in installer.files or ()
        if path.name.lower() in {"copying.txt", "license", "license.txt"}
    ]
    if not license_files:
        raise RuntimeError("PyInstaller's bootloader license was not found")
    notices = bundle / "licenses" / "pyinstaller"
    notices.mkdir(parents=True)
    for path in license_files:
        shutil.copy2(installer.locate_file(path), notices / path.name)
    nvml = metadata.distribution("nvidia-ml-py")
    nvml_notices = bundle / "licenses" / "nvidia-ml-py"
    nvml_notices.mkdir(parents=True)
    (nvml_notices / "LICENSE.txt").write_text(
        nvidia_license(nvml.locate_file("pynvml.py").read_text(encoding="utf-8")),
        encoding="utf-8",
    )
    (bundle / "THIRD-PARTY-NOTICES.txt").write_text(
        "Python: see PYTHON-LICENSE.txt.\n"
        f"PyInstaller bootloader: see licenses{sep}pyinstaller (including its exception).\n"
        f"NVIDIA Python bindings: see licenses{sep}nvidia-ml-py.\n"
        f"Runtime package metadata and license texts are included in _internal{sep}*.dist-info.\n"
        "GPU libraries are provided by the installed driver, system libraries by the operating system.\n",
        encoding="utf-8",
    )


README_LINUX = """GPU Task Manager {version} - Linux {arch}

Extract the whole archive and keep _internal beside gputaskman:

  tar -xzf gputaskman-{version}-linux-{arch}.tar.gz
  cd gputaskman-{version}-linux-{arch}
  ./gputaskman
  ./gputaskman --once
  ./gputaskman --json

Press q to quit the interactive display.
Requires an NVIDIA GPU and driver (NVML, libnvidia-ml.so.1, comes with the driver) and glibc 2.35+
(Ubuntu 22.04, Debian 12, RHEL 10 or newer). Works natively, in containers, and in WSL; under WSL,
Windows host counters are read through interop when it is enabled (see the documentation).
No Python, CUDA Toolkit, models, or root is required; other users' processes need sudo to inspect.
No drivers or inference servers are bundled.
Snapshots can contain local paths and model names; redact before sharing.

Documentation: https://github.com/shanselman/gputaskman
"""


def build(output: Path, arch: str, tag: str | None = None) -> Path:
    system, actual = target(sysconfig.get_platform())
    if actual != arch:
        raise ValueError(f"Requested {arch}, but this Python interpreter is {actual}")
    version = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]["version"]
    validate_version(version, tag)
    with tempfile.TemporaryDirectory(prefix=f"gputaskman-{arch}-") as temporary:
        work = Path(temporary)
        subprocess.run(
            [
                sys.executable, "-m", "PyInstaller", "--noconfirm", "--clean",
                "--distpath", str(work / "dist"), "--workpath", str(work / "build"),
                str(ROOT / "tools" / "portable.spec"),
            ],
            cwd=ROOT,
            check=True,
        )
        bundle = work / "dist" / "gputaskman"
        add_notices(bundle, system)
        if system == "linux":
            (bundle / "README.txt").write_text(README_LINUX.format(version=version, arch=arch), encoding="utf-8")
            return create_linux_archive(bundle, output, version, arch)
        (bundle / "README.txt").write_text(
            f"GPU Task Manager {version} - Windows {arch}\n\n"
            "Extract the entire ZIP before running. Keep _internal beside gputaskman.exe.\n"
            "Open PowerShell or Windows Terminal in this folder:\n\n"
            "  .\\gputaskman.exe\n"
            "  .\\gputaskman.exe --once\n"
            "  .\\gputaskman.exe --json\n\n"
            "Press q to quit the interactive display.\n"
            "Requires Windows and an NVIDIA or AMD GPU with a current WDDM driver.\n"
            "No Python, uv, CUDA Toolkit, models, or administrator session is required.\n"
            "This build is unsigned. No drivers or inference servers are bundled.\n"
            "Snapshots can contain local paths and model names; redact before sharing.\n\n"
            "Documentation: https://github.com/shanselman/gputaskman\n",
            encoding="utf-8",
        )
        return create_archive(bundle, output, version, arch)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arch", choices=MACHINES, required=True)
    parser.add_argument("--output", type=Path, default=ROOT / "dist")
    parser.add_argument("--tag", help="Require a release tag matching the package version")
    args = parser.parse_args()
    print(build(args.output.resolve(), args.arch, args.tag))


if __name__ == "__main__":
    main()

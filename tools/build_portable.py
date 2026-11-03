"""Build a native Windows console executable and ZIP its complete runtime."""

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
import tempfile
import tomllib
import zipfile


ROOT = Path(__file__).resolve().parents[1]
MACHINES = {"x64": 0x8664, "arm64": 0xAA64}


def architecture(platform_tag: str) -> str:
    platforms = {"win-amd64": "x64", "win-arm64": "arm64"}
    if platform_tag not in platforms:
        raise ValueError(f"Unsupported build interpreter: {platform_tag}")
    return platforms[platform_tag]


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


def create_archive(bundle: Path, output: Path, version: str, arch: str) -> Path:
    validate_version(version)
    check_executable(bundle / "gputaskman.exe", arch)
    forbidden = {"nvml.dll", "nvcuda.dll", "dxgi.dll", "pdh.dll"}
    # Editable-install origins contain build-machine paths and are not needed at runtime.
    files = sorted(
        path for path in bundle.rglob("*")
        if path.is_file()
        and not (path.name == "direct_url.json" and path.parent.name.endswith(".dist-info"))
    )
    for path in files:
        if path.name.lower() in forbidden:
            raise ValueError(f"Driver/system DLL must not be distributed: {path}")
    name = f"gputaskman-{version}-windows-{arch}"
    output.mkdir(parents=True, exist_ok=True)
    archive = output / f"{name}.zip"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as zipped:
        for path in files:
            zipped.write(path, Path(name) / path.relative_to(bundle))
    with archive.open("rb") as packaged:
        digest = hashlib.file_digest(packaged, "sha256").hexdigest()
    archive.with_suffix(".zip.sha256").write_text(f"{digest}  {archive.name}\n", encoding="utf-8")
    return archive


def nvidia_license(source: str) -> str:
    header, boundary, _ = source.partition("\n#####\n")
    if not boundary or "Redistribution and use" not in header or "Copyright" not in header:
        raise ValueError("NVIDIA Python bindings' inline license was not found")
    return header + boundary


def add_notices(bundle: Path) -> None:
    shutil.copy2(ROOT / "LICENSE", bundle / "LICENSE")
    shutil.copy2(Path(sys.base_prefix) / "LICENSE.txt", bundle / "PYTHON-LICENSE.txt")
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
        "PyInstaller bootloader: see licenses\\pyinstaller (including its exception).\n"
        "NVIDIA Python bindings: see licenses\\nvidia-ml-py.\n"
        "Runtime package metadata and license texts are included in _internal\\*.dist-info.\n"
        "NVIDIA and Windows DLLs are provided by the installed driver and operating system.\n",
        encoding="utf-8",
    )


def build(output: Path, arch: str, tag: str | None = None) -> Path:
    actual = architecture(sysconfig.get_platform())
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
        add_notices(bundle)
        (bundle / "README.txt").write_text(
            f"GPU Task Manager {version} - Windows {arch}\n\n"
            "Extract the entire ZIP before running. Keep _internal beside gputaskman.exe.\n"
            "Open PowerShell or Windows Terminal in this folder:\n\n"
            "  .\\gputaskman.exe\n"
            "  .\\gputaskman.exe --once\n"
            "  .\\gputaskman.exe --json\n\n"
            "Press q to quit the interactive display.\n"
            "Requires Windows and an NVIDIA GPU with a current WDDM driver.\n"
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

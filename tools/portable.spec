from pathlib import Path

from PyInstaller.utils.hooks import collect_data_files, copy_metadata

root = Path(SPECPATH).parent
analysis = Analysis(
    [str(root / "tools" / "portable_entry.py")],
    pathex=[str(root / "src")],
    datas=collect_data_files("textual") + copy_metadata("gputaskman", recursive=True),
)

# These belong to Windows or the installed NVIDIA driver, not the portable app.
driver_dlls = {"nvml.dll", "nvcuda.dll", "dxgi.dll", "pdh.dll"}
analysis.binaries = [
    entry for entry in analysis.binaries
    if Path(entry[0]).name.lower() not in driver_dlls
]

pyz = PYZ(analysis.pure)
exe = EXE(
    pyz,
    analysis.scripts,
    [],
    exclude_binaries=True,
    name="gputaskman",
    console=True,
)
bundle = COLLECT(exe, analysis.binaries, analysis.datas, name="gputaskman")

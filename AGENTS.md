# AGENTS.md

Guide for coding agents and contributors changing gputaskman. The [README](README.md) explains
what users see and why each number can or cannot be trusted. This file covers the rules the code
must keep, where things live, and how to verify and ship changes.

## The one rule: provenance

gputaskman shows what is in GPU memory without overstating it. Every number is **measured**,
**inferred**, **estimated**, or **reported by a runtime**, and the UI and JSON must say which.
Never present an inference as a measurement. Most review findings so far came from breaking one of
these rules:

1. **Only measured bytes are summed.** `Snapshot.attributed` (`collector.py`) adds
   `ProcGpu.dedicated` only. `ProcGpu.shown_vram` (`pdh.py`: measured, else inferred, else `None`)
   is for display, sorting, model selection, and scan priority, never for totals. Inferred WSL rows
   sit *under* the `vmwp.exe` VM total and are never added to it again.
2. **Unknown is not zero.** Unavailable counters are `null` in JSON and `n/a` or blank in the UI.
   A measured zero stays `0`. Engine availability is independent of VRAM availability.
3. **Windows per-process memory uses `Local Usage` / `Non Local Usage`**, not `Dedicated Usage`,
   which counts shared surfaces against `dwm.exe`, `csrss.exe`, and the owning app. The adapter
   counter is the reference total.
4. **Model evidence has a ladder:** `api` > `mapped` > `cmdline` > `installed`. App settings mean
   *selected*. Log events are *timestamped history*, not current residency. Model-store folders only
   supply friendly names. Name-based hints (`kind_hint`, the `decision?` category) never change the
   evidence or the KV-cache math.
5. **Markers mean what they say.** `*` means only one explanation fits, or every explanation gives
   exactly the same bytes (near-equal is not enough). `?` is a heuristic owner. Caches, such as WSL
   foreign-context sizes in `~/.cache/gputaskman/`, only rank candidates and never prove ownership.
6. **Look, don't touch.** Never read another process's memory, dump weights, launch runtimes, send
   inference requests, or send telemetry. Local API probes are `GET` requests to ports the process
   already listens on.
7. **Stay cheap.** Never call `psutil.open_files()` (it takes seconds per process). Memory-map scans
   are bounded by `Collector.SCAN_BUDGET` (150 ms per refresh) and cached per process; GGUF headers
   and store manifests are cached. If you add a probe, measure its cost and add it to the README's
   cost table.
8. **Never bundle driver libraries.** Vendor libraries come from the installed driver. The
   `FORBIDDEN` list in `tools/build_portable.py` enforces this for NVIDIA libraries; extend it for
   any new vendor.
9. **Degrade with a reason.** A missing vendor library, a protected process, WSL interop being off,
   or an unpaired adapter reduces what is shown and says why. It must not crash or guess.
10. **Hardware claims need hardware.** Say which machine, driver, and workload validated a
    behavior, or label it as not validated. The README's live checks follow this pattern.

## Map

```
cli.py ──► Collector.sample()  (collector.py)
            ├─ adapters        dxgi.py (Windows) / linux.enumerate_adapters / macos.enumerate_adapters
            ├─ counters        pdh.py GpuCounters (Windows PDH) / linux.py LinuxCounters (NVML) /
            │                  macos.py MacCounters (IOKit Metal clients + footprint poller)
            ├─ device metrics  nvml.py (macOS: MacCounters.device from IOKit)
            ├─ WSL host        wslhost.py: HostStream (powershell PDH via interop), ContextTracker
            ├─ processes       procinfo.py: runtime DLLs, mapped files, cmdline, classify()
            ├─ model naming    stores.py (Ollama, LM Studio, Hugging Face), gguf.py (header, kind, sizes)
            ├─ LLM detail      llm.py: buffer logs, ServerProber (local APIs), build_view()
            └─ app adapters    handy.py (settings + bounded log tail)
         ──► Snapshot ──► app.py (Textual TUI) / report.py (text and JSON)
cli.py diff ──► diff.py: compares two saved --json snapshots (no collection)
```

The backend is chosen at import time in `collector.py`: `BACKEND` is `wddm` on Windows, `nvml` on
native Linux, `wsl` under WSL, and `metal` on macOS. One-shot output (`--once`, `--json`) uses `scan_budget=None` to
identify everything immediately; the TUI spreads scans across refreshes.

## Where to change things

| Change | Start in | Notes |
|---|---|---|
| Detect a new runtime | `procinfo.py` | DLL or shared-object names; detection is not proof of GPU work |
| Name models from a new store | `stores.py` | Friendly names only, never evidence |
| Probe a new local server API | `llm.py` `ServerProber` | GET only, short timeouts, the process's own ports |
| Support an app that reads and closes its model | `handy.py` as the template | Read-only settings plus a bounded, mtime-cached log tail; never read transcription content. Generalize into data-driven adapters once a second app needs one |
| Add a GPU vendor or backend | `dxgi.py`, `collector.py`, a new module | Pair devices by LUID or PCI bus ID, never by name or index; load vendor DLLs by absolute system path |
| Add a column or JSON field | `report.py` and `app.py` | Follow the null rules above; update the README tables. If it is a measured byte count, consider adding it to `diff.py` `MEASURED` |
| Change snapshot comparison | `diff.py` | Works only on saved `--json` files, never live state. Compare measured fields directly; keep runtime-reported and estimated values in their own sections; an absent row is `not listed`, not zero |
| Change WSL matching | `wslhost.py` `ContextTracker` | Keep `tests/test_wslhost.py` replays passing; they are measured sequences |

## Verify

```powershell
uv sync --locked
uv run --locked python -m unittest discover -s tests -q
uv run --locked gputaskman --once
uv run --locked gputaskman --json
```

- Tests are hardware-independent (mocks and replayed measurements). CI runs them on Windows, Ubuntu,
  and macOS with Python 3.11 and 3.13. UI tests that render NVIDIA snapshots pin
  `gputaskman.app.BACKEND`, so they behave the same on a Mac.
- A running `.venv\Scripts\gputaskman.exe` locks the script, so `uv run` fails with
  `Access is denied`. Quit the monitor first.
- Textual UI tests must `await pilot.pause()` before querying widgets.
- WSL: test in a fresh copy on the Linux filesystem with `uv run --locked`; do not reuse the
  Windows `.venv`. The project needs Python 3.11+, which uv downloads if the distro's is older.
- Live checks: run a bounded, known workload (for example a llama-server with a model you have, or
  a fixed-size CUDA allocation), then compare `--json` against Task Manager or `nvidia-smi`.
  `gputaskman diff before.json after.json` shows exactly which measured bucket moved.
  Record hardware, driver, and results in the PR.
- Screenshots: `tools/capture_screenshots.py` strips command lines and full paths, but review model
  and process names before publishing.

## Release

1. Bump `project.version` in `pyproject.toml`, run `uv lock`, then confirm with `uv lock --check`.
2. Run the tests, commit `Release vX.Y.Z`, and push `main`.
3. `git tag -a vX.Y.Z -m "GPU Task Manager vX.Y.Z"` and `git push origin refs/tags/vX.Y.Z`.

`.github/workflows/release.yml` builds Windows x64/ARM64 and Linux x64/ARM64 archives, smoke-tests
each extracted executable without Python, and publishes them together with checksums and generated
notes. The build fails if the tag does not match `project.version`. A manual `workflow_dispatch`
run builds artifacts without publishing.

## Contributor pull requests

- CI for fork PRs waits for approval:
  `gh api -X POST repos/shanselman/gputaskman/actions/runs/<run-id>/approve`.
- Maintainer fixes can be pushed to the contributor's branch when the PR allows maintainer edits.
  Merge with a merge commit (`gh pr merge <n> --merge --match-head-commit <sha>`) so the
  contributor's commits and credit are preserved.
- Review against the rules above, and ask for hardware evidence (or a "not validated" label) for
  new hardware claims.

## Known limits

Validated live: Windows 11 with an RTX 4080 SUPER (WDDM, plus WSL2), a Linux container with an
RTX 3060 Ti, and macOS 26.3.1 on an M4 Mac mini (Metal; own-user processes, not run as root).
Not yet validated on hardware: bare-metal Linux, multi-GPU systems, and GPU-backed WinML on
Copilot+ PCs. Planned work is listed under **Next** in the README.

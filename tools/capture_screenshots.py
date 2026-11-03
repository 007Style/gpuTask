"""Capture real local measurements through Textual, without publishing command lines or full paths."""

from __future__ import annotations

import argparse
import asyncio
import copy
import os
import time
from pathlib import Path
from unittest.mock import patch

from gputaskman.app import GpuTaskManager
from gputaskman.collector import Collector, Snapshot


def for_publication(snapshot: Snapshot) -> Snapshot:
    snapshot = copy.deepcopy(snapshot)
    for row in snapshot.procs:
        row.detail.cmdline = []
        row.detail.exe = ""
        for model in row.detail.models + row.detail.candidates:
            model.path = os.path.basename(model.path)
        if row.llm:
            for model in row.llm.models:
                model.path = os.path.basename(model.path)
            for model in row.llm.files + row.llm.other_files:
                model.path = os.path.basename(model.path)
            if row.llm.server and row.llm.server.model_path:
                row.llm.server.model_path = os.path.basename(row.llm.server.model_path)
            for server in (row.llm.server, row.llm.host):
                for m in server.models if server else []:
                    if m.get("path"):
                        m["path"] = os.path.basename(m["path"])
            if row.llm.log_source:
                row.llm.log_source = os.path.basename(row.llm.log_source)
    return snapshot


async def capture(snapshots: list[Snapshot], output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    app = GpuTaskManager()
    # Freeze a real sample for each view rather than inventing demonstration metrics.
    with patch.object(GpuTaskManager, "poll"):
        async with app.run_test(size=(160, 48)) as pilot:
            for snapshot in snapshots:
                app.apply(snapshot)
            await pilot.pause()
            final = snapshots[-1]
            llm = next((row for row in final.procs if row.llm and row.llm.models), None)
            if llm:
                app.selected_pid = llm.pid
            else:
                print("No loaded LLM found; overview will show the available GPU processes.")
            app.render_table(final)
            app.render_inspector()
            await pilot.pause()
            app.save_screenshot("overview.svg", path=str(output))

            dwm = next((row for row in final.procs if row.kind == "compositor"), None)
            if dwm:
                app.selected_pid = dwm.pid
                app.render_table(final)
                app.render_inspector()
                await pilot.pause()
                app.save_screenshot("compositor.svg", path=str(output))
            else:
                print("No compositor row available; compositor capture skipped.")

            app.sort_idx = next(i for i, (name, _) in enumerate(app.SORTS) if name == "spill")
            app.selected_pid = max(final.procs, key=lambda row: row.gpu.shared).pid if final.procs else None
            app.render_table(final)
            app.render_inspector()
            await pilot.pause()
            app.save_screenshot("system-memory.svg", path=str(output))

            await pilot.press("question_mark")
            await pilot.pause()
            app.save_screenshot("help.svg", path=str(output))
    print(f"Captured local measurements to {output}. Review all images before publishing.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("docs") / "screenshots")
    args = parser.parse_args()
    collector = Collector()
    try:
        snapshots = []
        for _ in range(8):
            time.sleep(1)
            snapshots.append(for_publication(collector.sample()))
    finally:
        collector.close()
    asyncio.run(capture(snapshots, args.output))


if __name__ == "__main__":
    main()

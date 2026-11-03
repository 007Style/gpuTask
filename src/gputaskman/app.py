"""The htop-style Textual UI."""

from __future__ import annotations

import os
import time
from collections import deque

from rich.console import Group
from rich.table import Table
from rich.text import Text
from textual import work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import DataTable, Footer, Markdown, Select, Static
from textual.worker import get_current_worker

from .collector import BACKEND, Collector, ProcRow, Snapshot
from .llm import gb
from .nvml import NvDevice, format_metric
from .report import METAL_NOTE, WSL_EMPTY, has_engine_counters, has_wddm_counters, inferred_note, vram_label, wsl_note

KIND_STYLE = {
    "LLM": "bold magenta", "decision": "bold orchid", "decision?": "orchid",
    "embedding": "plum2", "speech": "light_sky_blue1", "diffusion": "orchid",
    "AI": "medium_purple1", "compute": "orange3", "3D/game": "green3",
    "video": "yellow", "browser": "dodger_blue2", "app": "cyan", "compositor": "grey50",
    "kernel": "grey50", "system": "grey50", "wsl-vm": "bold spring_green2", "external": "grey50",
}
SEG_STYLE = {"weights": "magenta", "kv": "dark_orange", "compute": "yellow", "runtime": "grey58", "unknown": "grey42"}
SPARK = "▁▂▃▄▅▆▇█"

EXPLAIN = {
    "wsl-vm": "The WSL virtual machine as Windows sees it. Windows charges all GPU memory used by Linux processes "
              "in every WSL distro (and WSLg) to vmwp.exe, so this is the exact total. The split across Linux "
              "processes is not exposed.",
    "compositor": "Desktop Window Manager: its own back buffers and the composed image for each monitor. "
                  "Windows' 'Dedicated Usage' counter shows dwm as huge (it counts every app surface it can see); "
                  "this tool prefers Local Usage for attribution. The adapter counter is the total; "
                  "per-process samples can overlap or differ in timing.",
    "kernel": "Driver/kernel-owned allocations: paging buffers, display scan-out, firmware, scheduler state.",
    "system": "Session/system process holding display and cursor surfaces.",
    "browser": "Chromium GPU process: page tiles, images, video frames, canvas, WebGL/WebGPU buffers. "
               "Deeper: edge://gpu, or DevTools > Performance with the GPU track.",
    "3D/game": "Textures, meshes, render targets and descriptor heaps. Windows only sees opaque allocations; "
               "per-resource names need a graphics debugger (PIX on Windows, NVIDIA Nsight Graphics).",
    "video": "Decoded/encoded video frames and codec surfaces (NVDEC/NVENC).",
    "compute": "CUDA allocations are opaque to the OS. For more detail trace cudaMalloc with Nsight Systems.",
    "diffusion": "Model components are identified from launch arguments; GPU allocations cannot be split by component.",
    "AI": "ML runtime allocations (tensors, workspaces). PyTorch: torch.cuda.memory_summary(); "
          "ONNX Runtime/DirectML pools are opaque from outside.",
    "app": "UI surfaces: swap chains, window back-buffers, cached bitmaps/glyphs.",
}

HELP_MD = """\
# What can be known about GPU memory?

| Layer | Source | Works for |
|---|---|---|
| **Device** totals, load, bandwidth, PCIe, power | NVML (`nvml.dll`) | NVIDIA |
| **Per-process** VRAM, spill to RAM, per-engine load | Windows WDDM counters (dxgkrnl) | every API incl. **CUDA** |
| **Which runtime** (CUDA, Vulkan, D3D12, ggml, ONNX, PyTorch...) | DLLs mapped in the process | everything |
| **Which model file** | memory-mapped files + command line | llama.cpp, stable-diffusion.cpp, LM Studio, Ollama |
| **Model name & kind** (LLM, decision, embedding, speech) | GGUF header, Ollama/LM Studio/Hugging Face stores, local server APIs | GGUF family |
| **Weights / KV cache / compute** split | GGUF header math, llama.cpp/Ollama logs, local server APIs | GGUF/llama.cpp family |
| Per-texture / per-buffer names | needs a debugger (PIX, Nsight) attached to the app | games, 3D apps |

**Evidence** shows how a model was identified: `api` (the runtime says it is loaded), `mapped`
(Windows shows the file mapped into the process right now), `cmdline` (a launch argument only).
Two sources, including `api` or `mapped`, is shown as *confirmed*.

**Local workloads only:** inference, decision models, vision, speech, training, graphics and video.
Cloud-hosted models are outside this monitor's scope. DLLs identify runtimes, not models or tasks.
`decision?` / *likely decision model (name)* is a filename or model-folder hint when a specific
GGUF kind is missing. It does not confirm the kind, GPU residency, or execution. Installed
candidates remain unproven; engine activity is measured per process, not per model.

**Linux / WSL:** on native Linux, per-process VRAM and SM/encoder/decoder load come from NVML
(no spill or commit counters exist there, so those columns stay blank). Under **WSL**, Windows
charges all Linux GPU memory to the VM (`vmwp.exe`), so VRAM and load are device-wide only; rows
are Linux processes with `/dev/dxg` open and show `n/a` rather than a made-up share. When WSL
interop is enabled, the Windows host's counters are read through `powershell.exe`: **WSL VM**
(`vmwp.exe`) is then the exact total of Linux GPU memory, and Windows apps appear as `W:<pid>`
rows. `--no-host` turns this off. Linux rows get their VRAM by matching the VM's GPU contexts
(newest first) to Linux processes: `*` when candidates agree on the size, `?` for a heuristic owner.
Cached foreign sizes only help rank guesses; they never prove ownership across runs.

**Why does nvidia-smi say N/A?** GeForce cards on Windows run in *WDDM* mode: Windows, not the
NVIDIA driver, owns video memory, so NVML can't attribute it to processes. The Windows counters can.

**dwm.exe looks huge elsewhere?** Windows' per-process "Dedicated Usage" counts every surface a
process can *see*, so dwm can show nearly the whole card. VRAM here is what's charged to each
process ("Local Usage" when available). The adapter counter is the total, not the sum of rows.

**SPILL** = GPU-accessible system RAM. This can include pinned host buffers or intentional
allocations, not just overflow. Large usage under VRAM pressure can indicate fallback;
the counter alone does not prove it.

**PAGED** = max(Total Committed - Local Usage - Non Local Usage, 0). This is an estimate of
non-resident committed memory, not a measured eviction count or proof of pagefile activity.

**Inside** labels are inferred from runtimes and files. Log sizes describe allocated buffers,
not necessarily current residency. A game's textures cannot be enumerated here.

**GPU%** = busiest engine (like Task Manager). On GeForce/WDDM, CUDA kernels are usually reported
on the **3D** engine.

Keys: `↑/↓` select · `s` sort · `c` chart · `a` show all · `p` pause · `?` this help · `q` quit

All eight trend charts are visible by default. They stack in narrow terminals and
share rows when there is room. Click the **Charts** dropdown to show all charts or
one metric; `c` cycles the same choices. A blank sample means no reading; zero is
drawn at the baseline. Percentage charts use 0–100%, temperature uses 0–120°C,
and the power and SM clock scales are shown below the charts. In short terminals,
scroll the dashboard or inspector without losing the process list.
"""


def spark(values, width: int = 30, top: float = 100.0) -> str:
    vals = list(values)[-width:]
    return "".join(
        SPARK[min(max(int(v / top * (len(SPARK) - 1) + 0.5), 0), len(SPARK) - 1)]
        if v is not None else " "
        for v in vals
    )


def stacked_bar(parts: list[tuple[float, str]], total: float, width: int) -> Text:
    t = Text()
    used = 0
    for value, style in parts:
        n = int(round(value / total * width)) if total else 0
        n = min(n, width - used)
        if n > 0:
            t.append("█" * n, style=style)
            used += n
    t.append("░" * (width - used), style="grey27")
    return t


class HelpScreen(ModalScreen):
    BINDINGS = [Binding("escape,q,question_mark", "app.pop_screen", "Close")]
    DEFAULT_CSS = """
    HelpScreen { align: center middle; }
    HelpScreen > VerticalScroll { width: 140; max-width: 95%; height: 85%; border: thick $accent; background: $surface; padding: 1 2; }
    """

    def compose(self) -> ComposeResult:
        with VerticalScroll():
            yield Markdown(HELP_MD)


METAL_CHARTS = ("GPU", "VRAM")


class GpuTaskManager(App):
    TITLE = "gputaskman"
    CSS = """
    Screen { layout: vertical; }
    #dashboard { height: auto; max-height: 50%; }
    #device { height: auto; border: round $accent; padding: 0 1; }
    #history { height: auto; border: round $secondary; padding: 0 1; }
    #chart-controls { height: 1; }
    #chart-label { width: 8; text-style: bold; }
    #chart-mode { width: 24; }
    #chart-hint { width: 1fr; color: $text-muted; }
    #charts { height: auto; }
    #chart-note { height: auto; color: $text-muted; }
    #procs { height: 1fr; min-height: 4; }
    #inspector { height: auto; max-height: 30%; overflow-y: auto; border: round $secondary; padding: 0 1; }
    """
    BINDINGS = [
        Binding("q", "quit", "Quit"),
        Binding("s", "cycle_sort", "Sort"),
        Binding("c", "cycle_chart", "Chart"),
        Binding("a", "toggle_all", "All/Top"),
        Binding("p", "pause", "Pause"),
        Binding("question_mark", "help", "Explain"),
    ]
    SORTS = [("VRAM", lambda r: -(r.gpu.shown_vram or 0)), ("GPU%", lambda r: -r.gpu.gpu_pct),
             ("spill", lambda r: -r.gpu.shared), ("paged", lambda r: -r.gpu.paged_out),
             ("name", lambda r: r.name.lower())]
    CHART_COLORS = {
        "GPU": "#4ade80",
        "SM clock": "#fb923c",
        "VRAM": "#60a5fa",
        "Mem activity": "#a78bfa",
        "Temperature": "#ef4444",
        "Power": "#facc15",
        "Encoder": "#e2e8f0",
        "Decoder": "#22d3ee",
    }
    CHARTS = tuple(CHART_COLORS)

    def __init__(self, gpu_index: int = 0, interval: float = 1.0, host: bool = True) -> None:
        super().__init__()
        self.gpu_index = gpu_index
        self.interval = interval
        self.host = host
        self.snap: Snapshot | None = None
        self.selected_key: str | None = None  # ProcRow.key: Linux and Windows host PIDs can collide
        self.sort_idx = 0
        self.show_all = False
        self.paused = False
        self.hist_util: deque = deque(maxlen=60)
        self.hist_clock: deque = deque(maxlen=60)
        self.hist_bw: deque = deque(maxlen=60)
        self.hist_vram: deque = deque(maxlen=60)
        self.hist_temp: deque = deque(maxlen=60)
        self.hist_power: deque = deque(maxlen=60)
        self.hist_enc: deque = deque(maxlen=60)
        self.hist_dec: deque = deque(maxlen=60)
        self.chart_idx = -1
        self._auto_selected = False

    @property
    def selected_pid(self) -> int | None:
        """PID of the selected local process; None for Windows (W:) and outside (H:) rows."""
        key = self.selected_key
        return int(key) if key and key.isdigit() else None

    @selected_pid.setter
    def selected_pid(self, pid: int | None) -> None:
        self.selected_key = None if pid is None else str(pid)

    def compose(self) -> ComposeResult:
        with VerticalScroll(id="dashboard"):
            yield Static("Sampling GPU counters...", id="device")
            with Vertical(id="history"):
                with Horizontal(id="chart-controls"):
                    yield Static("Charts", id="chart-label")
                    yield Select(
                        [("All charts", -1), *((self.CHARTS[index], index) for index in self.available_charts)],
                        value=-1, allow_blank=False, compact=True, id="chart-mode",
                    )
                    yield Static("Click to choose · c cycles", id="chart-hint")
                yield Static(id="charts")
                yield Static(id="chart-note")
        table = DataTable(id="procs", cursor_type="row", zebra_stripes=True)
        yield table
        yield Static("", id="inspector")
        yield Footer()

    def on_mount(self) -> None:
        t = self.query_one("#procs", DataTable)
        for col in ("PID", "Process", "Kind", "Runtime", "VRAM", "Spill", "Paged", "GPU%", "3D", "Copy", "Video", "What's inside"):
            t.add_column(col, key=col)
        t.focus()
        self.poll()

    @work(thread=True, exclusive=True)
    def poll(self) -> None:
        worker = get_current_worker()
        collector = Collector(self.gpu_index, host=self.host)
        time.sleep(0.5)
        try:
            while not worker.is_cancelled:
                if not self.paused:
                    snap = collector.sample()
                    self.call_from_thread(self.apply, snap)
                time.sleep(self.interval)
        finally:
            collector.close()

    # ------------------------------------------------------------------ rendering

    def apply(self, snap: Snapshot) -> None:
        self.snap = snap
        nv = snap.nv
        if nv:
            self.hist_util.append(nv.util_gpu)
            self.hist_clock.append(nv.sm_clock)
            self.hist_bw.append(nv.util_mem)
            self.hist_temp.append(nv.temp)
            self.hist_power.append(nv.power_w)
            self.hist_enc.append(nv.enc)
            self.hist_dec.append(nv.dec)
        else:
            for history in (self.hist_util, self.hist_clock, self.hist_bw, self.hist_temp,
                            self.hist_power, self.hist_enc, self.hist_dec):
                history.append(None)
        self.hist_vram.append(100 * snap.vram_used / snap.vram_total if snap.vram_total else None)
        if not self._auto_selected:
            llm = next((r for r in sorted(snap.procs, key=self.SORTS[0][1]) if r.llm), None)
            if llm:
                self.selected_key = llm.key
            self._auto_selected = True
        self.query_one("#device", Static).update(self.render_device(snap))
        self.render_charts()
        self.render_table(snap)
        self.render_inspector()

    def render_device(self, s: Snapshot) -> Group:
        nv = s.nv
        name = s.adapter.name if s.adapter else (nv.name if nv else "GPU")
        if s.backend == "metal":
            head = Text.assemble((name, "bold"), f"   {s.driver}  ", ("Metal", "bold yellow"), "  unified memory")
        else:
            head = Text.assemble((name, "bold"), f"   driver {s.driver}  CUDA {s.cuda}  ",
                                 (nv.driver_model if nv else "", "bold yellow"), f"  {nv.pci_bus if nv else ''}")
        if self.paused:
            head.append("   [PAUSED]", style="bold red")

        # Fit category proportions to the adapter total; row samples need not sum exactly.
        by_kind: dict[str, float] = {}
        for r in s.procs:
            if r.gpu.attributed:  # unknown shares stay in "unattributed" instead of showing as 0
                by_kind[r.kind] = by_kind.get(r.kind, 0) + r.gpu.dedicated
        attributed = sum(by_kind.values())
        scale = min(1.0, s.vram_used / attributed) if attributed else 1.0
        order = sorted(by_kind.items(), key=lambda kv: -kv[1])
        parts = [(v * scale, KIND_STYLE.get(k, "white")) for k, v in order]
        other = max(s.vram_used - attributed * scale, 0)
        parts.append((other, "grey35"))
        width = max(min(self.size.width - 40, 90), 20)
        vram = Text("GPU mem " if s.backend == "metal" else "VRAM ")
        vram.append_text(stacked_bar(parts, s.vram_total, width))
        vram.append(f" {gb(s.vram_used)} / {gb(s.vram_total)}", style="bold")
        legend = Text("     ")
        for k, v in order[:7]:
            legend.append("■ ", style=KIND_STYLE.get(k, "white"))
            legend.append(f"{k} {gb(v)}  ")
        if other > 32 << 20:
            legend.append("■ ", style="grey35")
            legend.append(f"unattributed {gb(other)}  ")
        legend.append(f"free {gb(max(s.vram_total - s.vram_used, 0))}", style="grey62")
        if s.shared_used > 64 << 20:
            legend.append(f"   shared(RAM) {gb(s.shared_used)}", style="yellow")
        if scale < 1.0:
            legend.append("   [bar normalized; labels are raw]", style="grey62")
        if s.backend == "wsl":
            legend.append("\n     " + wsl_note(s), style="grey62")
        if s.backend == "metal":
            legend.append("\n     " + METAL_NOTE, style="grey62")

        lines = [head, vram, legend]
        if nv and s.backend == "metal":
            lines.append(Text(f"GPU {format_metric(nv.util_gpu, unit='%')}",
                              style=f"bold {self.CHART_COLORS['GPU']}"))
        elif nv:
            rx = nv.pcie_rx_kbs / 1024 if nv.pcie_rx_kbs is not None else None
            tx = nv.pcie_tx_kbs / 1024 if nv.pcie_tx_kbs is not None else None
            pstate = f"P{nv.pstate}" if nv.pstate is not None else "P N/A"
            m = Text()
            m.append(f"GPU {format_metric(nv.util_gpu, unit='%')}", style=f"bold {self.CHART_COLORS['GPU']}")
            m.append(f"  Mem activity {format_metric(nv.util_mem, unit='%')}", style=self.CHART_COLORS["Mem activity"])
            m.append(f"  PCIe↓{format_metric(rx, '.0f'):>5} ↑{format_metric(tx, '.0f'):>4} MB/s")
            if nv.enc or nv.dec:
                m.append(f"  enc {format_metric(nv.enc, unit='%')} dec {format_metric(nv.dec, unit='%')}")
            lines.append(m)
            st = Text(f"{format_metric(nv.temp, unit='°C')}  {format_metric(nv.power_w, '.0f')}/"
                      f"{format_metric(nv.power_limit_w, '.0f')} W  {pstate}  "
                      f"SM {format_metric(nv.sm_clock)}/{format_metric(nv.sm_clock_max)} MHz  "
                      f"mem {format_metric(nv.mem_clock)} MHz  video {format_metric(nv.video_clock)} MHz  "
                      f"PCIe gen{format_metric(nv.pcie_gen)} x{format_metric(nv.pcie_width)}  "
                      f"CC {nv.compute_cap or 'N/A'}", style="grey70")
            if nv.throttle:
                st.append("  throttle: " + ",".join(nv.throttle), style="yellow")
            lines.append(st)
        return Group(*lines)

    def chart_series(self, index: int, nv: NvDevice | None) -> tuple[deque, float, str, str]:
        chart = self.CHARTS[index]
        color = self.CHART_COLORS[chart]
        if chart == "GPU":
            return self.hist_util, 100, "%", color
        if chart == "SM clock":
            values = [value for value in self.hist_clock if value is not None]
            scale = max((nv.sm_clock_max if nv else None) or 0, max(values, default=0), 1)
            return self.hist_clock, scale, " MHz", color
        if chart == "VRAM":
            return self.hist_vram, 100, "%", color
        if chart == "Mem activity":
            return self.hist_bw, 100, "%", color
        if chart == "Temperature":
            return self.hist_temp, 120, "°C", color
        if chart == "Power":
            values = [value for value in self.hist_power if value is not None]
            scale = max((nv.power_limit_w if nv else None) or 0, max(values, default=0), 1)
            return self.hist_power, scale, " W", color
        if chart == "Encoder":
            return self.hist_enc, 100, "%", color
        return self.hist_dec, 100, "%", color

    def render_charts(self) -> None:
        if self.snap is None:
            return
        widget = self.query_one("#charts", Static)
        # Reserve the border, padding, and scrollbar even before child resize events arrive.
        width = max(1, self.size.width - 6)
        indices = list(self.available_charts) if self.chart_idx == -1 else [self.chart_idx]
        columns = min(len(indices), 3, max(1, (width + 2) // 46))
        cell_width = max(1, (width - 2 * (columns - 1)) // columns)
        grid = Table.grid(padding=(0, 2), pad_edge=False, expand=True)
        for _ in range(columns):
            grid.add_column(width=cell_width, no_wrap=True, overflow="crop")
        cells = []
        for index in indices:
            values, scale, unit, style = self.chart_series(index, self.snap.nv)
            current = values[-1] if values else None
            cell = Text(f"{self.CHARTS[index]:<12} ", style=style)
            cell.append(f"{format_metric(current, '.0f', unit):>8} ", style="bold")
            cell.append(spark(values, max(1, cell_width - 22), scale))
            cells.append(cell)
        for start in range(0, len(cells), columns):
            grid.add_row(*cells[start:start + columns])
        widget.update(grid)
        _, power_scale, _, _ = self.chart_series(self.CHARTS.index("Power"), self.snap.nv)
        _, clock_scale, _, _ = self.chart_series(self.CHARTS.index("SM clock"), self.snap.nv)
        self.query_one("#chart-note", Static).update(
            "Recent samples · blank = no reading" if self.snap.backend == "metal" else
            f"Recent samples · blank = no reading · power 0–{power_scale:.0f} W · "
            f"SM clock 0–{clock_scale:.0f} MHz"
        )

    def on_resize(self) -> None:
        if self.is_mounted and self.snap is not None:
            self.query_one("#device", Static).update(self.render_device(self.snap))
            self.call_after_refresh(self.render_charts)

    def visible_rows(self, s: Snapshot) -> list[ProcRow]:
        rows = s.procs
        if not self.show_all:
            rows = [r for r in rows if not r.gpu.attributed or r.gpu.dedicated >= 32 << 20 or r.gpu.gpu_pct >= 1 or r.gpu.shared >= 64 << 20
                    or r.gpu.paged_out >= 128 << 20 or r.llm]
        rows = sorted(rows, key=self.SORTS[self.sort_idx][1])
        vm = next((i for i, r in enumerate(rows) if r.kind == "wsl-vm"), None)
        if vm is not None and self.SORTS[self.sort_idx][0] == "VRAM":
            # Linux rows share the WSL VM row's total: keep them under it, ordered by their inferred sizes.
            linux = [r for r in rows if not r.gpu.attributed and not r.host]
            rest = [r for r in rows if r not in linux]
            vm = rest.index(rows[vm])
            rows = rest[:vm + 1] + linux + rest[vm + 1:]
        return rows

    def render_table(self, s: Snapshot) -> None:
        t = self.query_one("#procs", DataTable)
        rows = self.visible_rows(s)
        keep = self.selected_key
        t.clear()

        def pct(v: float | None) -> Text:
            if v is None:
                return Text("n/a", justify="right", style="grey62")
            return Text(f"{v:.0f}" if v >= 0.5 else "", justify="right", style="bold" if v >= 20 else "")

        for r in rows:
            e = r.gpu.engines
            wddm = has_wddm_counters(s, r)
            engines = has_engine_counters(s, r)
            split = engines and s.backend != "metal"  # Metal reports one GPU time, no per-engine split
            spill = Text(gb(r.gpu.shared) if wddm and r.gpu.shared >= 1 << 20 else "", justify="right",
                         style="bold red" if r.gpu.shared >= 256 << 20 and r.kind in ("LLM", "AI") else "yellow")
            paged = Text(gb(r.gpu.paged_out) if wddm and r.gpu.paged_out >= 1 << 20 else "", justify="right", style="grey62")
            t.add_row(
                Text(r.pid_label, justify="right", style="grey62" if r.host else ""),
                Text(r.name),
                Text(r.kind, style=KIND_STYLE.get(r.kind, "")),
                Text(" ".join(r.detail.runtimes[:3]), style="grey70"),
                Text(vram_label(r.gpu), justify="right", style="bold" if r.gpu.attributed else "grey62"), spill, paged,
                pct(r.gpu.gpu_pct if engines else None),
                pct(e.get("3d", 0) + e.get("compute", 0) if split else None),
                pct(e.get("copy", 0) if split else None),
                pct(e.get("decode", 0) + e.get("encode", 0) if split else None),
                Text(r.inside, style="magenta" if r.llm else "grey62", overflow="ellipsis", no_wrap=True),
                key=r.key,
            )
        if keep is not None:
            try:
                t.move_cursor(row=t.get_row_index(keep), animate=False, scroll=False)
            except Exception:
                pass
        t.border_subtitle = f"sort: {self.SORTS[self.sort_idx][0]}  ·  {'all' if self.show_all else 'top'} processes"
        if s.backend == "wsl" and not any(not r.host for r in s.procs):
            t.border_subtitle = WSL_EMPTY

    def render_inspector(self) -> None:
        panel = self.query_one("#inspector", Static)
        s = self.snap
        r = next((x for x in s.procs if x.key == self.selected_key), None) if s else None
        if r is None:
            panel.update(Text("Select a process (↑/↓) to see what's inside its GPU memory.  Press ? to learn what's knowable.", style="grey62"))
            return
        where = {"windows": "Windows PID ", "outside": "host PID "}.get(r.origin, "") if r.host else ""
        panel.border_title = f"{r.name} ({where}{r.pid}) - {r.kind}"
        out: list = []
        e = r.gpu.engines
        wddm = has_wddm_counters(s, r)
        engines = " ".join(f"{k} {v:.0f}%" for k, v in e.items() if v >= 0.5) or "idle"
        hdr = Text.assemble(("VRAM ", "bold"), (vram_label(r.gpu), "bold"), "   spill(RAM) ",
                            (gb(r.gpu.shared) if wddm else "n/a", "bold red" if r.gpu.shared > 256 << 20 else ""),
                            "   paged(est) ", gb(r.gpu.paged_out) if wddm else "n/a",
                            "   engines ", engines if has_engine_counters(s, r) else "n/a")
        if r.detail.runtimes:
            hdr.append("   runtime ", style="grey62")
            hdr.append(" ".join(r.detail.runtimes), style="cyan")
        out.append(hdr)
        if not r.host and r.gpu.attributed and r.gpu.inferred_by:
            out.append(Text(inferred_note(r.gpu), style="grey62"))
        if not r.host and not r.gpu.attributed:
            if r.gpu.inferred is not None:
                out.append(Text(inferred_note(r.gpu), style="grey62"))
            elif s.wsl_split:
                out.append(Text(f"No per-process VRAM: {s.wsl_split}.", style="grey62"))
            elif s.backend == "metal":
                out.append(Text("GPU memory n/a: footprint can't read other users' processes without root "
                                "(or hasn't measured this one yet). GPU% comes from Metal's per-process GPU time.",
                                style="grey62"))
        if wddm and r.gpu.paged_out >= 64 << 20:
            out.append(Text(f"{gb(r.gpu.paged_out)} of {gb(r.gpu.committed)} committed is not accounted for by local/non-local "
                            "usage. Estimated non-residency, not proof of eviction or disk paging.",
                            style="yellow"))

        if r.handy:
            if r.handy.selected_model:
                out.append(Text(f"Handy selected (settings; not load proof): {r.handy.selected_model}", style="grey70"))
            if r.handy.last_event:
                out.append(Text(f"Handy last event: {r.handy.last_event} {r.handy.last_model or ''} "
                                f"at {r.handy.last_event_local} (runtime-reported history, not current GPU residency)",
                                style="green"))

        v = r.llm
        if v:
            width = max(min(self.size.width - 8, 100), 20)
            total = max(r.gpu.shown_vram or 0, sum(x.bytes for x in v.segments), 1)
            out.append(stacked_bar([(x.bytes, SEG_STYLE[x.kind]) for x in v.segments], total, width))
            legend = Text()
            for x in v.segments:
                legend.append("■ ", style=SEG_STYLE[x.kind])
                legend.append(f"{x.label} {gb(x.bytes)}", style="bold" if x.exact else "")
                legend.append("  ")
            if v.log_source:
                legend.append(f"(allocated sizes from {os.path.basename(v.log_source)})", style="green")
            out.append(legend)
            if v.models:
                mt = Table(box=None, padding=(0, 1), show_edge=False, header_style="bold grey70")
                for col in ("", "model", "arch", "params", "quant", "layers", "train ctx", "on GPU (est)", "evidence", "file"):
                    mt.add_column(col)
                for i, g in enumerate(v.models):
                    f = v.files[i] if i < len(v.files) else None
                    mt.add_row(Text(v.kind_label(i), style="magenta"), v.label(i), g.arch, g.params_label, g.quant,
                               str(g.n_layers or ""), f"{g.ctx_train:,}" if g.ctx_train else "",
                               gb(g.gpu_weight_bytes()), Text(f.evidence if f else "", style="green"),
                               Text(os.path.basename(g.path), style="grey50", overflow="ellipsis"))
                out.append(mt)
            for f in v.other_files:
                kind = f", {f.kind_label}" if f.kind_label else ""
                out.append(Text(f"file: {f.store_name or f.path}  ({gb(f.size)}, {f.evidence}{kind})", style="grey62"))
            if v.server:
                sv = v.server
                st = Text.assemble(("server ", "bold"), (f"{sv.kind} {sv.url}", "cyan"))
                if sv.n_ctx:
                    st.append(f"  ctx {sv.n_ctx:,}")
                if sv.slots_total is not None:
                    st.append(f"  slots busy {sv.slots_busy}/{sv.slots_total}")
                if sv.gen_tok_s:
                    st.append(f"  gen {sv.gen_tok_s:.1f} tok/s", style="bold green")
                if sv.prompt_tok_s:
                    st.append(f"  prompt {sv.prompt_tok_s:.0f} tok/s")
                for m in sv.models[:4]:
                    if sv.kind == "lmstudio":
                        st.append(f"\n  loaded: {m['name']} {m.get('type') or ''} {m.get('quant') or ''} ctx {m.get('ctx') or '?'}")
                out.append(st)
            for n in v.notes:
                out.append(Text("• " + n, style="bold red" if n.startswith("!") else "grey70"))
        else:
            out.append(Text(EXPLAIN.get(r.kind, ""), style="grey70"))
            for m in r.detail.models:
                kind = f", {m.kind_label}" if m.kind_label else ""
                out.append(Text(f"file: {m.store_name or m.path}  ({gb(m.size)}, {m.evidence}{kind})", style="magenta"))
            if r.detail.candidates:
                out.append(Text("No mapped model file was observed. Some runtimes read then close model files. "
                                "Installed candidates (not proven loaded):", style="grey70"))
                for m in r.detail.candidates:
                    kind = f", {m.kind_label}" if m.kind_label else ""
                    out.append(Text(f"  {m.store_name or m.path}  ({gb(m.size)}{kind})", style="grey62"))
        if r.detail.ports:
            out.append(Text("listening: " + ", ".join(map(str, r.detail.ports[:8])), style="grey50"))
        if r.detail.cmdline:
            out.append(Text(" ".join(r.detail.cmdline)[:400], style="grey42", overflow="ellipsis"))
        elif r.detail.access_denied:
            out.append(Text("(protected process - run elevated for runtime/file details)", style="grey42"))
        panel.update(Group(*out))

    # ------------------------------------------------------------------ events/actions

    def on_data_table_row_highlighted(self, event: DataTable.RowHighlighted) -> None:
        if event.row_key is not None and event.row_key.value is not None:
            self.selected_key = str(event.row_key.value)
            self.render_inspector()

    def action_cycle_sort(self) -> None:
        self.sort_idx = (self.sort_idx + 1) % len(self.SORTS)
        if self.snap:
            self.render_table(self.snap)

    @property
    def available_charts(self) -> list[int]:
        """Chart indices this backend has data for: IOKit (macOS) has no clocks, power, temperature or codecs."""
        if BACKEND == "metal":
            return [i for i, name in enumerate(self.CHARTS) if name in METAL_CHARTS]
        return list(range(len(self.CHARTS)))

    def action_cycle_chart(self) -> None:
        order = [-1, *self.available_charts]
        nxt = order[(order.index(self.chart_idx) + 1) % len(order)] if self.chart_idx in order else -1
        self.query_one("#chart-mode", Select).value = nxt

    def on_select_changed(self, event: Select.Changed) -> None:
        if event.select.id == "chart-mode" and isinstance(event.value, int):
            self.chart_idx = event.value
            self.render_charts()

    def action_toggle_all(self) -> None:
        self.show_all = not self.show_all
        if self.snap:
            self.render_table(self.snap)

    def action_pause(self) -> None:
        self.paused = not self.paused
        if self.snap:
            self.query_one("#device", Static).update(self.render_device(self.snap))

    def action_help(self) -> None:
        self.push_screen(HelpScreen())

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
from textual.containers import VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import DataTable, Footer, Markdown, Static
from textual.worker import get_current_worker

from .collector import Collector, ProcRow, Snapshot
from .llm import gb

KIND_STYLE = {
    "LLM": "bold magenta", "AI": "medium_purple1", "compute": "orange3", "3D/game": "green3",
    "video": "yellow", "browser": "dodger_blue2", "app": "cyan", "compositor": "grey50",
    "kernel": "grey50", "system": "grey50",
}
SEG_STYLE = {"weights": "magenta", "kv": "dark_orange", "compute": "yellow", "runtime": "grey58", "unknown": "grey42"}
SPARK = "▁▂▃▄▅▆▇█"

EXPLAIN = {
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
| **Which model file** | memory-mapped files + command line | llama.cpp, LM Studio, Ollama, llama-cpp-python |
| **Weights / KV cache / compute** split | GGUF header math, llama.cpp/Ollama logs, local server APIs | GGUF/llama.cpp family |
| Per-texture / per-buffer names | needs a debugger (PIX, Nsight) attached to the app | games, 3D apps |

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

Keys: `↑/↓` select · `s` sort · `a` show all · `p` pause · `?` this help · `q` quit
"""


def spark(values, width: int = 30, top: float = 100.0) -> str:
    vals = list(values)[-width:]
    return "".join(SPARK[min(int(v / top * (len(SPARK) - 1) + 0.5), len(SPARK) - 1)] if v else " " for v in vals)


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


class GpuTaskManager(App):
    TITLE = "gputaskman"
    CSS = """
    Screen { layout: vertical; }
    #device { height: auto; border: round $accent; padding: 0 1; }
    #procs { height: 1fr; min-height: 8; }
    #inspector { height: auto; max-height: 55%; border: round $secondary; padding: 0 1; }
    """
    BINDINGS = [
        Binding("q", "quit", "Quit"),
        Binding("s", "cycle_sort", "Sort"),
        Binding("a", "toggle_all", "All/Top"),
        Binding("p", "pause", "Pause"),
        Binding("question_mark", "help", "Explain"),
    ]
    SORTS = [("VRAM", lambda r: -r.gpu.dedicated), ("GPU%", lambda r: -r.gpu.gpu_pct),
             ("spill", lambda r: -r.gpu.shared), ("paged", lambda r: -r.gpu.paged_out),
             ("name", lambda r: r.name.lower())]

    def __init__(self, gpu_index: int = 0, interval: float = 1.0) -> None:
        super().__init__()
        self.gpu_index = gpu_index
        self.interval = interval
        self.snap: Snapshot | None = None
        self.selected_pid: int | None = None
        self.sort_idx = 0
        self.show_all = False
        self.paused = False
        self.hist_util: deque = deque(maxlen=60)
        self.hist_bw: deque = deque(maxlen=60)
        self.hist_vram: deque = deque(maxlen=60)
        self._auto_selected = False

    def compose(self) -> ComposeResult:
        yield Static("Sampling GPU counters...", id="device")
        table = DataTable(id="procs", cursor_type="row", zebra_stripes=True)
        yield table
        yield Static("", id="inspector")
        yield Footer()

    def on_mount(self) -> None:
        t = self.query_one("#procs", DataTable)
        for col in ("PID", "Process", "Kind", "Runtime", "VRAM", "Spill", "Paged", "GPU%", "3D", "Copy", "Video", "What's inside"):
            t.add_column(col, key=col)
        self.poll()

    @work(thread=True, exclusive=True)
    def poll(self) -> None:
        worker = get_current_worker()
        collector = Collector(self.gpu_index)
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
            self.hist_util.append(nv.util_gpu or 0)
            self.hist_bw.append(nv.util_mem or 0)
        self.hist_vram.append(100 * snap.vram_used / snap.vram_total if snap.vram_total else 0)
        if not self._auto_selected:
            llm = next((r for r in sorted(snap.procs, key=lambda r: -r.gpu.dedicated) if r.llm), None)
            if llm:
                self.selected_pid = llm.pid
            self._auto_selected = True
        self.query_one("#device", Static).update(self.render_device(snap))
        self.render_table(snap)
        self.render_inspector()

    def render_device(self, s: Snapshot) -> Group:
        nv = s.nv
        name = s.adapter.name if s.adapter else (nv.name if nv else "GPU")
        head = Text.assemble((name, "bold"), f"   driver {s.driver}  CUDA {s.cuda}  ",
                             (nv.driver_model if nv else "", "bold yellow"), f"  {nv.pci_bus if nv else ''}")
        if self.paused:
            head.append("   [PAUSED]", style="bold red")

        # Fit category proportions to the adapter total; row samples need not sum exactly.
        by_kind: dict[str, float] = {}
        for r in s.procs:
            by_kind[r.kind] = by_kind.get(r.kind, 0) + r.gpu.dedicated
        attributed = sum(by_kind.values())
        scale = min(1.0, s.vram_used / attributed) if attributed else 1.0
        order = sorted(by_kind.items(), key=lambda kv: -kv[1])
        parts = [(v * scale, KIND_STYLE.get(k, "white")) for k, v in order]
        other = max(s.vram_used - attributed * scale, 0)
        parts.append((other, "grey35"))
        width = max(min(self.size.width - 40, 90), 20)
        vram = Text("VRAM ")
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

        lines = [head, vram, legend]
        if nv:
            m = Text()
            m.append("GPU ", style="bold")
            m.append(f"{nv.util_gpu:>3}% ", style="bold green")
            m.append(spark(self.hist_util, 24), style="green")
            m.append("  MemBW ", style="bold")
            m.append(f"{nv.util_mem:>3}% ", style="bold dark_orange")
            m.append(spark(self.hist_bw, 24), style="dark_orange")
            m.append(f"  PCIe↓{(nv.pcie_rx_kbs or 0) / 1024:>5.0f} ↑{(nv.pcie_tx_kbs or 0) / 1024:>4.0f} MB/s")
            if nv.enc or nv.dec:
                m.append(f"  enc {nv.enc}% dec {nv.dec}%")
            lines.append(m)
            st = Text(f"{nv.temp}°C  {nv.power_w or 0:.0f}/{nv.power_limit_w or 0:.0f} W  P{nv.pstate}  "
                      f"SM {nv.sm_clock}/{nv.sm_clock_max} MHz  mem {nv.mem_clock} MHz  "
                      f"PCIe gen{nv.pcie_gen} x{nv.pcie_width}  CC {nv.compute_cap}", style="grey70")
            if nv.throttle:
                st.append("  throttle: " + ",".join(nv.throttle), style="yellow")
            lines.append(st)
        return Group(*lines)

    def visible_rows(self, s: Snapshot) -> list[ProcRow]:
        rows = s.procs
        if not self.show_all:
            rows = [r for r in rows if r.gpu.dedicated >= 32 << 20 or r.gpu.gpu_pct >= 1 or r.gpu.shared >= 64 << 20
                    or r.gpu.paged_out >= 128 << 20 or r.llm]
        return sorted(rows, key=self.SORTS[self.sort_idx][1])

    def render_table(self, s: Snapshot) -> None:
        t = self.query_one("#procs", DataTable)
        rows = self.visible_rows(s)
        keep = self.selected_pid
        t.clear()

        def pct(v: float) -> Text:
            return Text(f"{v:.0f}" if v >= 0.5 else "", justify="right", style="bold" if v >= 20 else "")

        for r in rows:
            e = r.gpu.engines
            spill = Text(gb(r.gpu.shared) if r.gpu.shared >= 1 << 20 else "", justify="right",
                         style="bold red" if r.gpu.shared >= 256 << 20 and r.kind in ("LLM", "AI") else "yellow")
            paged = Text(gb(r.gpu.paged_out) if r.gpu.paged_out >= 1 << 20 else "", justify="right", style="grey62")
            t.add_row(
                Text(str(r.pid), justify="right"),
                Text(r.name),
                Text(r.kind, style=KIND_STYLE.get(r.kind, "")),
                Text(" ".join(r.detail.runtimes[:3]), style="grey70"),
                Text(gb(r.gpu.dedicated), justify="right", style="bold"), spill, paged, pct(r.gpu.gpu_pct),
                pct(e.get("3d", 0) + e.get("compute", 0)), pct(e.get("copy", 0)),
                pct(e.get("decode", 0) + e.get("encode", 0)),
                Text(r.inside, style="magenta" if r.llm else "grey62", overflow="ellipsis", no_wrap=True),
                key=str(r.pid),
            )
        if keep is not None:
            try:
                t.move_cursor(row=t.get_row_index(str(keep)), animate=False, scroll=False)
            except Exception:
                pass
        t.border_subtitle = f"sort: {self.SORTS[self.sort_idx][0]}  ·  {'all' if self.show_all else 'top'} processes"

    def render_inspector(self) -> None:
        panel = self.query_one("#inspector", Static)
        s = self.snap
        r = next((x for x in s.procs if x.pid == self.selected_pid), None) if s else None
        if r is None:
            panel.update(Text("Select a process (↑/↓) to see what's inside its GPU memory.  Press ? to learn what's knowable.", style="grey62"))
            return
        panel.border_title = f"{r.name} ({r.pid}) - {r.kind}"
        out: list = []
        e = r.gpu.engines
        hdr = Text.assemble(("VRAM ", "bold"), (gb(r.gpu.dedicated), "bold"), "   spill(RAM) ",
                            (gb(r.gpu.shared), "bold red" if r.gpu.shared > 256 << 20 else ""),
                            "   paged(est) ", gb(r.gpu.paged_out),
                            "   engines ", " ".join(f"{k} {v:.0f}%" for k, v in e.items() if v >= 0.5) or "idle")
        if r.detail.runtimes:
            hdr.append("   runtime ", style="grey62")
            hdr.append(" ".join(r.detail.runtimes), style="cyan")
        out.append(hdr)
        if r.gpu.paged_out >= 64 << 20:
            out.append(Text(f"{gb(r.gpu.paged_out)} of {gb(r.gpu.committed)} committed is not accounted for by local/non-local "
                            "usage. Estimated non-residency, not proof of eviction or disk paging.",
                            style="yellow"))

        v = r.llm
        if v:
            width = max(min(self.size.width - 8, 100), 20)
            total = max(r.gpu.dedicated, sum(x.bytes for x in v.segments), 1)
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
                for col in ("", "model", "arch", "params", "quant", "layers", "train ctx", "on GPU (est)", "file"):
                    mt.add_column(col)
                for g in v.models:
                    mt.add_row(Text(g.kind, style="magenta"), g.name, g.arch, g.params_label, g.quant,
                               str(g.n_layers or ""), f"{g.ctx_train:,}" if g.ctx_train else "",
                               gb(g.gpu_weight_bytes()), Text(os.path.basename(g.path), style="grey50", overflow="ellipsis"))
                out.append(mt)
            for f in v.other_files:
                out.append(Text(f"file: {f.path}  ({gb(f.size)})", style="grey62"))
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
                        st.append(f"\n  loaded: {m.get('id')} {m.get('quantization', '')} ctx {m.get('loaded_context_length') or m.get('max_context_length', '?')}")
                out.append(st)
            for n in v.notes:
                out.append(Text("• " + n, style="bold red" if n.startswith("!") else "grey70"))
        else:
            out.append(Text(EXPLAIN.get(r.kind, ""), style="grey70"))
            for m in r.detail.models:
                out.append(Text(f"file: {m.path}  ({gb(m.size)}, {m.source})", style="magenta"))
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
            self.selected_pid = int(event.row_key.value)
            self.render_inspector()

    def action_cycle_sort(self) -> None:
        self.sort_idx = (self.sort_idx + 1) % len(self.SORTS)
        if self.snap:
            self.render_table(self.snap)

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

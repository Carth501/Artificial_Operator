from __future__ import annotations

import tkinter as tk
from tkinter import ttk

from simulation.engine import SimulationEngine

from .panels import APP_BACKGROUND, ControlPanel, ModuleMapPanel, ModulePanel, MotionPanel, VariableGroupPanel


class SimulationApp:
    def __init__(self, engine: SimulationEngine) -> None:
        self._engine = engine
        self.root = tk.Tk()
        self.root.title("Artificial Operator Sandbox")
        self.root.geometry("1180x760")
        self.root.minsize(980, 640)
        self.root.configure(bg=APP_BACKGROUND)

        self._status_var = tk.StringVar(value="Initializing simulation...")
        self._tick_job: str | None = None
        self._group_panels: dict[str, VariableGroupPanel] = {}
        self._module_panels: dict[str, ModulePanel] = {}

        self._configure_style()
        initial_snapshot = self._engine.snapshot()
        self._build_layout(initial_snapshot)
        self.refresh(initial_snapshot)
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    def run(self) -> None:
        self._schedule_next_tick()
        self.root.mainloop()

    def _configure_style(self) -> None:
        style = ttk.Style(self.root)
        style.theme_use("clam")
        style.configure("App.TFrame", background=APP_BACKGROUND)
        style.configure("Panel.TFrame", background="#fffaf2")
        style.configure(
            "Card.TLabelframe",
            background="#fffaf2",
            foreground="#22313a",
            borderwidth=1,
            relief="solid",
        )
        style.configure(
            "Card.TLabelframe.Label",
            background="#fffaf2",
            foreground="#22313a",
            font=("Bahnschrift SemiBold", 11),
        )
        style.configure("Title.TLabel", background=APP_BACKGROUND, foreground="#22313a", font=("Bahnschrift SemiBold", 22))
        style.configure("Subtitle.TLabel", background=APP_BACKGROUND, foreground="#6d5f54", font=("Bahnschrift", 11))
        style.configure("MetricLabel.TLabel", background="#fffaf2", foreground="#22313a", font=("Bahnschrift SemiBold", 10))
        style.configure("MetricValue.TLabel", background="#fffaf2", foreground="#22313a", font=("Consolas", 13, "bold"))
        style.configure("MetricUnit.TLabel", background="#fffaf2", foreground="#6d5f54", font=("Bahnschrift", 9))
        style.configure("SectionTitle.TLabel", background="#fffaf2", foreground="#9c4721", font=("Bahnschrift SemiBold", 11))
        style.configure("AxisLabel.TLabel", background="#fffaf2", foreground="#6d5f54", font=("Bahnschrift SemiBold", 10))
        style.configure(
            "Command.TButton",
            font=("Bahnschrift SemiBold", 10),
            padding=(12, 9),
            background="#c56b32",
            foreground="#fffaf2",
            borderwidth=0,
            focusthickness=0,
        )
        style.map(
            "Command.TButton",
            background=[("active", "#a35221"), ("pressed", "#8e441a")],
            foreground=[("disabled", "#eddccb")],
        )
        style.configure(
            "Metric.Horizontal.TProgressbar",
            troughcolor="#e7dccd",
            background="#2f7d63",
            bordercolor="#e7dccd",
            lightcolor="#2f7d63",
            darkcolor="#2f7d63",
        )

    def _build_layout(self, snapshot: dict[str, object]) -> None:
        shell = ttk.Frame(self.root, padding=20, style="App.TFrame")
        shell.pack(fill="both", expand=True)
        shell.columnconfigure(0, weight=3)
        shell.columnconfigure(1, weight=2)
        shell.rowconfigure(1, weight=1)

        header = ttk.Frame(shell, style="App.TFrame")
        header.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 18))
        header.columnconfigure(0, weight=1)

        ttk.Label(header, text="Artificial Operator", style="Title.TLabel").grid(row=0, column=0, sticky="w")
        ttk.Label(
            header,
            text="Spaceship sandbox with module integrity, container systems, and manual flight controls.",
            style="Subtitle.TLabel",
        ).grid(row=1, column=0, sticky="w", pady=(4, 0))
        ttk.Label(header, textvariable=self._status_var, style="Subtitle.TLabel").grid(row=0, column=1, rowspan=2, sticky="e")

        self._module_column = ttk.Frame(shell, style="App.TFrame")
        self._module_column.grid(row=1, column=0, sticky="nsew", padx=(0, 16))
        self._module_column.columnconfigure(0, weight=1)

        right_column = ttk.Frame(shell, style="App.TFrame")
        right_column.grid(row=1, column=1, sticky="nsew")
        right_column.columnconfigure(0, weight=1)

        self._module_map_panel = ModuleMapPanel(right_column)
        self._module_map_panel.grid(row=0, column=0, sticky="new")

        self._motion_panel = MotionPanel(right_column)
        self._motion_panel.grid(row=1, column=0, sticky="new", pady=(16, 0))

        self._core_column = ttk.Frame(right_column, style="App.TFrame")
        self._core_column.grid(row=2, column=0, sticky="new", pady=(16, 0))
        self._core_column.columnconfigure(0, weight=1)

        self._control_panel = ControlPanel(
            right_column,
            on_pause=self._handle_pause,
            on_reset=self._handle_reset,
        )
        self._control_panel.grid(row=3, column=0, sticky="new", pady=(16, 0))

        modules = snapshot.get("modules", ())
        if isinstance(modules, (list, tuple)):
            self._render_modules(modules)

    def refresh(self, snapshot: dict[str, object]) -> None:
        modules = snapshot.get("modules", ())
        if isinstance(modules, (list, tuple)):
            self._render_modules(modules)
            self._module_map_panel.render(modules)

        groups = snapshot.get("core_groups", {})
        if isinstance(groups, dict):
            self._render_variable_groups(groups)

        position = snapshot.get("position", {})
        velocity = snapshot.get("velocity", {})
        if isinstance(position, dict) and isinstance(velocity, dict):
            self._motion_panel.render(position, velocity)

        active_actions = snapshot.get("active_actions", ())
        if isinstance(active_actions, tuple):
            status_suffix = ", ".join(active_actions) if active_actions else "Idle"
        else:
            status_suffix = "Idle"

        paused = bool(snapshot.get("paused", False))
        self._control_panel.set_paused(paused)
        elapsed_seconds = float(snapshot.get("elapsed_seconds", 0.0))
        state_label = "Paused" if paused else "Running"
        alerts = snapshot.get("alerts", ())
        failed_modules = 0
        module_count = 0
        if isinstance(modules, (list, tuple)):
            module_count = len(modules)
            failed_modules = sum(1 for module in modules if not bool(module.get("operational", False)))
        status_text = f"Mission time {elapsed_seconds:6.1f}s | {state_label} | Modules {module_count} total, {failed_modules} failed | {status_suffix}"
        if isinstance(alerts, tuple) and alerts:
            status_text = f"{status_text} | ALERT: {', '.join(alerts)}"
        self._status_var.set(status_text)

    def _render_variable_groups(self, groups: dict[str, object]) -> None:
        panel_row = 0
        active_groups: set[str] = set()
        for group_name, variables in groups.items():
            if group_name in {"position", "velocity"}:
                continue
            if not isinstance(variables, list):
                continue

            panel = self._group_panels.get(group_name)
            if panel is None:
                title = group_name.replace("_", " ").title()
                panel = VariableGroupPanel(self._core_column, title)
                self._group_panels[group_name] = panel
            panel.grid(row=panel_row, column=0, sticky="ew", pady=(0, 16))
            panel.render(variables)
            active_groups.add(group_name)
            panel_row += 1

        for group_name, panel in self._group_panels.items():
            if group_name not in active_groups:
                panel.grid_remove()

    def _render_modules(self, modules: list[object] | tuple[object, ...]) -> None:
        active_module_ids: set[str] = set()
        for row_index, module in enumerate(modules):
            if not isinstance(module, dict):
                continue

            module_id = str(module.get("id", ""))
            if not module_id:
                continue

            panel = self._module_panels.get(module_id)
            if panel is None:
                panel = ModulePanel(
                    self._module_column,
                    module=module,
                    on_thruster_start=self._handle_thruster_start,
                    on_thruster_stop=self._handle_thruster_stop,
                    on_conversion=self._handle_conversion,
                )
                self._module_panels[module_id] = panel

            panel.grid(row=row_index, column=0, sticky="ew", pady=(0, 16))
            panel.render(module)
            active_module_ids.add(module_id)

        for module_id, panel in self._module_panels.items():
            if module_id not in active_module_ids:
                panel.grid_remove()

    def _handle_thruster_start(self, action_id: str) -> None:
        self._engine.start_action(action_id)
        self.refresh(self._engine.snapshot())

    def _handle_thruster_stop(self, action_id: str) -> None:
        self._engine.stop_action(action_id)
        self.refresh(self._engine.snapshot())

    def _handle_conversion(self, action_id: str) -> None:
        self._engine.trigger_conversion(action_id)
        self.refresh(self._engine.snapshot())

    def _handle_pause(self) -> None:
        self._engine.toggle_pause()
        self.refresh(self._engine.snapshot())

    def _handle_reset(self) -> None:
        self._engine.reset()
        self.refresh(self._engine.snapshot())

    def _schedule_next_tick(self) -> None:
        snapshot = self._engine.step()
        self.refresh(snapshot)
        interval_ms = max(50, int(self._engine.tick_seconds * 1000))
        self._tick_job = self.root.after(interval_ms, self._schedule_next_tick)

    def _on_close(self) -> None:
        if self._tick_job is not None:
            self.root.after_cancel(self._tick_job)
            self._tick_job = None
        self.root.destroy()
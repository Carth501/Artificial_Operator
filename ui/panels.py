from __future__ import annotations

import tkinter as tk
from tkinter import ttk
from typing import Any, Callable


APP_BACKGROUND = "#f4efe6"
PANEL_BACKGROUND = "#fffaf2"
TEXT_PRIMARY = "#22313a"
TEXT_MUTED = "#6d5f54"
MODULE_MAP_FILL = "#e7dccd"
MODULE_MAP_OPERATIONAL = "#2f7d63"
MODULE_MAP_FAILED = "#a35221"
THRUSTER_IDLE = "#d3b893"
THRUSTER_ACTIVE = "#2f7d63"
THRUSTER_TEXT = "#fffaf2"
THRUSTER_DISABLED = "#ddd3c7"


class VariableRow(ttk.Frame):
    def __init__(self, parent: ttk.Widget) -> None:
        super().__init__(parent, style="Panel.TFrame")
        self._value_var = tk.StringVar(value="0.00")
        self._unit_var = tk.StringVar(value="")

        self.columnconfigure(0, weight=1)
        self.columnconfigure(1, weight=0)
        self.columnconfigure(2, weight=0)

        self._label = ttk.Label(self, style="MetricLabel.TLabel")
        self._value = ttk.Label(self, textvariable=self._value_var, style="MetricValue.TLabel")
        self._unit = ttk.Label(self, textvariable=self._unit_var, style="MetricUnit.TLabel")
        self._progress = ttk.Progressbar(self, mode="determinate", maximum=100.0, style="Metric.Horizontal.TProgressbar")

        self._label.grid(row=0, column=0, sticky="w")
        self._value.grid(row=0, column=1, sticky="e", padx=(12, 6))
        self._unit.grid(row=0, column=2, sticky="e")
        self._progress.grid(row=1, column=0, columnspan=3, sticky="ew", pady=(6, 0))

    def render(self, variable: dict[str, Any]) -> None:
        self._label.configure(text=variable["label"])
        precision = int(variable.get("precision", 2))
        self._value_var.set(f"{float(variable['value']):.{precision}f}")
        self._unit_var.set(str(variable.get("unit", "")))

        minimum = variable.get("minimum")
        maximum = variable.get("maximum")
        if minimum is None or maximum is None or maximum == minimum:
            self._progress.grid_remove()
            return

        span = float(maximum) - float(minimum)
        percentage = ((float(variable["value"]) - float(minimum)) / span) * 100.0
        self._progress.configure(value=max(0.0, min(100.0, percentage)))
        self._progress.grid()


class VariableGroupPanel(ttk.LabelFrame):
    def __init__(self, parent: ttk.Widget, title: str) -> None:
        super().__init__(parent, text=title, padding=14, style="Card.TLabelframe")
        self._rows: dict[str, VariableRow] = {}
        self.columnconfigure(0, weight=1)

    def render(self, variables: list[dict[str, Any]]) -> None:
        active_names = {variable["name"] for variable in variables}
        for row_index, variable in enumerate(variables):
            row = self._rows.get(variable["name"])
            if row is None:
                row = VariableRow(self)
                self._rows[variable["name"]] = row
            row.grid(row=row_index, column=0, sticky="ew", pady=(0, 12))
            row.render(variable)

        for name, row in self._rows.items():
            if name not in active_names:
                row.grid_remove()


class MotionPanel(ttk.LabelFrame):
    def __init__(self, parent: ttk.Widget) -> None:
        super().__init__(parent, text="Motion", padding=14, style="Card.TLabelframe")
        self.columnconfigure(0, weight=1)
        self.columnconfigure(1, weight=1)
        self._position_vars = {axis: tk.StringVar(value="0.00") for axis in ("x", "y", "z")}
        self._velocity_vars = {axis: tk.StringVar(value="0.00") for axis in ("x", "y", "z")}

        self._build_vector_section(0, "Position", self._position_vars, "m")
        self._build_vector_section(1, "Velocity", self._velocity_vars, "m/s")

    def _build_vector_section(
        self,
        column_index: int,
        title: str,
        variables: dict[str, tk.StringVar],
        unit: str,
    ) -> None:
        section = ttk.Frame(self, style="Panel.TFrame", padding=(8, 4))
        section.grid(row=0, column=column_index, sticky="nsew", padx=(0, 12 if column_index == 0 else 0))
        section.columnconfigure(1, weight=1)

        ttk.Label(section, text=title, style="SectionTitle.TLabel").grid(row=0, column=0, columnspan=3, sticky="w", pady=(0, 8))
        for row_index, axis in enumerate(("x", "y", "z"), start=1):
            ttk.Label(section, text=axis.upper(), style="AxisLabel.TLabel").grid(row=row_index, column=0, sticky="w", pady=3)
            ttk.Label(section, textvariable=variables[axis], style="MetricValue.TLabel").grid(row=row_index, column=1, sticky="e", padx=(12, 6))
            ttk.Label(section, text=unit, style="MetricUnit.TLabel").grid(row=row_index, column=2, sticky="e")

    def render(self, position: dict[str, float], velocity: dict[str, float]) -> None:
        for axis in ("x", "y", "z"):
            self._position_vars[axis].set(f"{position.get(axis, 0.0):.2f}")
            self._velocity_vars[axis].set(f"{velocity.get(axis, 0.0):.2f}")


class ModuleMapPanel(ttk.LabelFrame):
    def __init__(self, parent: ttk.Widget) -> None:
        super().__init__(parent, text="Module Map", padding=14, style="Card.TLabelframe")
        self.columnconfigure(0, weight=1)
        self._canvas = tk.Canvas(
            self,
            background=PANEL_BACKGROUND,
            highlightthickness=0,
            bd=0,
            height=220,
        )
        self._canvas.grid(row=0, column=0, sticky="nsew")

    def render(self, modules: list[dict[str, Any]] | tuple[dict[str, Any], ...]) -> None:
        self._canvas.delete("all")
        module_count = len(modules)
        if module_count == 0:
            self._canvas.configure(height=120)
            self._canvas.create_text(140, 60, text="No modules configured", fill=TEXT_MUTED, font=("Bahnschrift", 11))
            return

        columns = 2 if module_count > 1 else 1
        square_size = 96
        gap = 16
        padding = 12
        rows = (module_count + columns - 1) // columns
        canvas_width = (square_size * columns) + (gap * (columns - 1)) + (padding * 2)
        canvas_height = (square_size * rows) + (gap * (rows - 1)) + (padding * 2)
        self._canvas.configure(width=canvas_width, height=canvas_height)

        for index, module in enumerate(modules):
            column_index = index % columns
            row_index = index // columns
            x0 = padding + column_index * (square_size + gap)
            y0 = padding + row_index * (square_size + gap)
            x1 = x0 + square_size
            y1 = y0 + square_size

            number = int(module.get("number", index + 1))
            label = str(module.get("label", "Module"))
            operational = bool(module.get("operational", False))
            fill = MODULE_MAP_OPERATIONAL if operational else MODULE_MAP_FAILED

            self._canvas.create_rectangle(x0, y0, x1, y1, fill=MODULE_MAP_FILL, outline=fill, width=3)
            self._canvas.create_text(
                (x0 + x1) / 2,
                y0 + 26,
                text=f"M{number}",
                fill=fill,
                font=("Bahnschrift SemiBold", 18),
            )
            self._canvas.create_text(
                (x0 + x1) / 2,
                y0 + 58,
                text=label,
                fill=TEXT_PRIMARY,
                width=square_size - 16,
                justify="center",
                font=("Bahnschrift", 10),
            )


class ModulePanel(ttk.LabelFrame):
    def __init__(
        self,
        parent: ttk.Widget,
        module: dict[str, Any],
        on_thruster_start: Callable[[str], None],
        on_thruster_stop: Callable[[str], None],
        on_conversion: Callable[[str], None],
    ) -> None:
        super().__init__(parent, text=self._module_title(module), padding=14, style="Card.TLabelframe")
        self._thruster_buttons: dict[str, tk.Button] = {}
        self._conversion_buttons: dict[str, ttk.Button] = {}
        self._system_status_vars: dict[str, tk.StringVar] = {}
        self._variable_rows: dict[str, VariableRow] = {}
        self._integrity_var = tk.StringVar(value="Integrity 0.0")
        self._status_var = tk.StringVar(value="Unknown")

        self.columnconfigure(0, weight=1)
        header = ttk.Frame(self, style="Panel.TFrame")
        header.grid(row=0, column=0, sticky="ew", pady=(0, 12))
        header.columnconfigure(0, weight=1)

        ttk.Label(header, textvariable=self._integrity_var, style="SectionTitle.TLabel").grid(row=0, column=0, sticky="w")
        ttk.Label(header, textvariable=self._status_var, style="AxisLabel.TLabel").grid(row=0, column=1, sticky="e")

        self._integrity_bar = ttk.Progressbar(self, mode="determinate", maximum=100.0, style="Metric.Horizontal.TProgressbar")
        self._integrity_bar.grid(row=1, column=0, sticky="ew", pady=(0, 14))

        self._content = ttk.Frame(self, style="Panel.TFrame")
        self._content.grid(row=2, column=0, sticky="ew")
        self._content.columnconfigure(0, weight=1)

        self._build_systems(module, on_thruster_start, on_thruster_stop, on_conversion)
        self.render(module)

    def _module_title(self, module: dict[str, Any]) -> str:
        number = int(module.get("number", 0))
        label = str(module.get("label", "Module"))
        if number <= 0:
            return label
        return f"Module {number}: {label}"

    def _build_systems(
        self,
        module: dict[str, Any],
        on_thruster_start: Callable[[str], None],
        on_thruster_stop: Callable[[str], None],
        on_conversion: Callable[[str], None],
    ) -> None:
        systems = module.get("systems", [])
        if not isinstance(systems, list):
            return

        for system_index, system in enumerate(systems):
            system_id = str(system["id"])
            section = ttk.Frame(self._content, style="Panel.TFrame", padding=(0, 0, 0, 0))
            section.grid(row=system_index, column=0, sticky="ew", pady=(0, 16))
            section.columnconfigure(0, weight=1)

            header = ttk.Frame(section, style="Panel.TFrame")
            header.grid(row=0, column=0, sticky="ew", pady=(0, 8))
            header.columnconfigure(0, weight=1)
            ttk.Label(header, text=str(system["label"]), style="SectionTitle.TLabel").grid(row=0, column=0, sticky="w")
            status_var = tk.StringVar(value="Unknown")
            self._system_status_vars[system_id] = status_var
            ttk.Label(header, textvariable=status_var, style="AxisLabel.TLabel").grid(row=0, column=1, sticky="e")

            content_row = 1
            variables = system.get("variables", [])
            if isinstance(variables, list):
                for variable in variables:
                    row = VariableRow(section)
                    row.grid(row=content_row, column=0, sticky="ew", pady=(0, 10))
                    self._variable_rows[str(variable["name"])] = row
                    content_row += 1

            actions = system.get("actions", [])
            if isinstance(actions, list) and actions:
                action_frame = ttk.Frame(section, style="Panel.TFrame")
                action_frame.grid(row=content_row, column=0, sticky="ew")
                for column_index in (0, 1):
                    action_frame.columnconfigure(column_index, weight=1)

                for action_index, action in enumerate(actions):
                    column_index = action_index % 2
                    row_index = action_index // 2
                    action_id = str(action["id"])
                    pad = (0, 10 if column_index == 0 else 0)
                    if action.get("kind") == "thruster":
                        button = tk.Button(
                            action_frame,
                            text=str(action["label"]),
                            font=("Bahnschrift SemiBold", 10),
                            bg=THRUSTER_IDLE,
                            fg=TEXT_PRIMARY,
                            activebackground=THRUSTER_ACTIVE,
                            activeforeground=THRUSTER_TEXT,
                            disabledforeground=TEXT_MUTED,
                            relief=tk.FLAT,
                            bd=0,
                            padx=12,
                            pady=10,
                            cursor="hand2",
                            highlightthickness=0,
                        )
                        button.grid(row=row_index, column=column_index, sticky="ew", padx=pad, pady=6)
                        button.bind("<ButtonPress-1>", lambda _event, current_id=action_id: on_thruster_start(current_id))
                        button.bind("<ButtonRelease-1>", lambda _event, current_id=action_id: on_thruster_stop(current_id))
                        button.bind("<Leave>", lambda _event, current_id=action_id: on_thruster_stop(current_id))
                        self._thruster_buttons[action_id] = button
                    else:
                        button = ttk.Button(
                            action_frame,
                            text=str(action["label"]),
                            style="Command.TButton",
                            command=lambda current_id=action_id: on_conversion(current_id),
                        )
                        button.grid(row=row_index, column=column_index, sticky="ew", padx=pad, pady=6)
                        self._conversion_buttons[action_id] = button

    def render(self, module: dict[str, Any]) -> None:
        self.configure(text=self._module_title(module))
        integrity = float(module.get("integrity", 0.0))
        operational = bool(module.get("operational", False))
        self._integrity_var.set(f"Integrity {integrity:5.1f}")
        self._status_var.set("Operational" if operational else "Failed")
        self._integrity_bar.configure(value=max(0.0, min(100.0, integrity)))

        systems = module.get("systems", [])
        if not isinstance(systems, list):
            return

        for system in systems:
            system_id = str(system["id"])
            status_var = self._system_status_vars.get(system_id)
            if status_var is not None:
                status_var.set("Operational" if system.get("operational", False) else "Failed")

            variables = system.get("variables", [])
            if isinstance(variables, list):
                for variable in variables:
                    row = self._variable_rows.get(str(variable["name"]))
                    if row is not None:
                        row.render(variable)

            actions = system.get("actions", [])
            if isinstance(actions, list):
                for action in actions:
                    action_id = str(action["id"])
                    if action.get("kind") == "thruster":
                        button = self._thruster_buttons.get(action_id)
                        if button is None:
                            continue
                        if not action.get("operational", False):
                            button.configure(state=tk.DISABLED, bg=THRUSTER_DISABLED, fg=TEXT_MUTED)
                        elif action.get("active", False):
                            button.configure(state=tk.NORMAL, bg=THRUSTER_ACTIVE, fg=THRUSTER_TEXT)
                        else:
                            button.configure(state=tk.NORMAL, bg=THRUSTER_IDLE, fg=TEXT_PRIMARY)
                    else:
                        button = self._conversion_buttons.get(action_id)
                        if button is None:
                            continue
                        if action.get("operational", False):
                            button.state(["!disabled"])
                        else:
                            button.state(["disabled"])


class ControlPanel(ttk.LabelFrame):
    def __init__(
        self,
        parent: ttk.Widget,
        on_pause: Callable[[], None],
        on_reset: Callable[[], None],
    ) -> None:
        super().__init__(parent, text="Simulation", padding=14, style="Card.TLabelframe")
        self._pause_var = tk.StringVar(value="Pause Simulation")
        self.columnconfigure(0, weight=1)
        self.columnconfigure(1, weight=1)

        ttk.Button(self, textvariable=self._pause_var, style="Command.TButton", command=on_pause).grid(
            row=0,
            column=0,
            sticky="ew",
            padx=(0, 8),
        )
        ttk.Button(self, text="Reset State", style="Command.TButton", command=on_reset).grid(
            row=0,
            column=1,
            sticky="ew",
        )

    def set_paused(self, paused: bool) -> None:
        self._pause_var.set("Resume Simulation" if paused else "Pause Simulation")
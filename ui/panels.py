from __future__ import annotations

import tkinter as tk
from tkinter import ttk
from typing import Any, Callable


APP_BACKGROUND = "#f4efe6"
PANEL_BACKGROUND = "#fffaf2"
TEXT_PRIMARY = "#22313a"
TEXT_MUTED = "#6d5f54"
THRUSTER_IDLE = "#d3b893"
THRUSTER_ACTIVE = "#2f7d63"
THRUSTER_TEXT = "#fffaf2"


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


class ActionPanel(ttk.LabelFrame):
    def __init__(
        self,
        parent: ttk.Widget,
        thrusters: tuple[dict[str, Any], ...],
        conversions: tuple[dict[str, Any], ...],
        on_thruster_start: Callable[[str], None],
        on_thruster_stop: Callable[[str], None],
        on_conversion: Callable[[str], None],
        on_pause: Callable[[], None],
        on_reset: Callable[[], None],
    ) -> None:
        super().__init__(parent, text="Actions", padding=14, style="Card.TLabelframe")
        self._thruster_buttons: dict[str, tk.Button] = {}
        self._pause_var = tk.StringVar(value="Pause Simulation")

        thruster_frame = ttk.Frame(self, style="Panel.TFrame")
        thruster_frame.grid(row=0, column=0, sticky="ew")

        ttk.Label(thruster_frame, text="Thrusters", style="SectionTitle.TLabel").grid(row=0, column=0, columnspan=2, sticky="w", pady=(0, 8))
        for index, thruster in enumerate(thrusters, start=1):
            column_index = 0 if index % 2 == 1 else 1
            row_index = ((index - 1) // 2) + 1
            button = tk.Button(
                thruster_frame,
                text=thruster["label"],
                font=("Bahnschrift SemiBold", 10),
                bg=THRUSTER_IDLE,
                fg=TEXT_PRIMARY,
                activebackground=THRUSTER_ACTIVE,
                activeforeground=THRUSTER_TEXT,
                relief=tk.FLAT,
                bd=0,
                padx=12,
                pady=10,
                cursor="hand2",
                highlightthickness=0,
            )
            button.grid(row=row_index, column=column_index, sticky="ew", padx=(0, 10 if column_index == 0 else 0), pady=6)
            button.bind("<ButtonPress-1>", lambda _event, action_id=thruster["id"]: on_thruster_start(action_id))
            button.bind("<ButtonRelease-1>", lambda _event, action_id=thruster["id"]: on_thruster_stop(action_id))
            button.bind("<Leave>", lambda _event, action_id=thruster["id"]: on_thruster_stop(action_id))
            self._thruster_buttons[thruster["id"]] = button

        for column_index in (0, 1):
            thruster_frame.columnconfigure(column_index, weight=1)

        conversion_frame = ttk.Frame(self, style="Panel.TFrame")
        conversion_frame.grid(row=1, column=0, sticky="ew", pady=(18, 0))
        ttk.Label(conversion_frame, text="Conversion", style="SectionTitle.TLabel").grid(row=0, column=0, sticky="w", pady=(0, 8))

        for row_index, conversion in enumerate(conversions, start=1):
            ttk.Button(
                conversion_frame,
                text=conversion["label"],
                style="Command.TButton",
                command=lambda action_id=conversion["id"]: on_conversion(action_id),
            ).grid(row=row_index, column=0, sticky="ew", pady=6)

        control_frame = ttk.Frame(self, style="Panel.TFrame")
        control_frame.grid(row=2, column=0, sticky="ew", pady=(18, 0))
        ttk.Label(control_frame, text="Simulation", style="SectionTitle.TLabel").grid(row=0, column=0, columnspan=2, sticky="w", pady=(0, 8))
        ttk.Button(control_frame, textvariable=self._pause_var, style="Command.TButton", command=on_pause).grid(row=1, column=0, sticky="ew", padx=(0, 8))
        ttk.Button(control_frame, text="Reset State", style="Command.TButton", command=on_reset).grid(row=1, column=1, sticky="ew")
        control_frame.columnconfigure(0, weight=1)
        control_frame.columnconfigure(1, weight=1)
        self.columnconfigure(0, weight=1)

    def set_active_actions(self, active_actions: tuple[str, ...]) -> None:
        active_set = set(active_actions)
        for action_id, button in self._thruster_buttons.items():
            if action_id in active_set:
                button.configure(bg=THRUSTER_ACTIVE, fg=THRUSTER_TEXT)
            else:
                button.configure(bg=THRUSTER_IDLE, fg=TEXT_PRIMARY)

    def set_paused(self, paused: bool) -> None:
        self._pause_var.set("Resume Simulation" if paused else "Pause Simulation")
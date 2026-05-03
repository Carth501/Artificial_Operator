from __future__ import annotations

from pathlib import Path
import random
from typing import Any

from .actions import ActionCatalog, load_action_catalog
from .profiles import compute_net_delta
from .state import (
    SimulationConfig,
    SimulationState,
    build_initial_state,
    clamp_value,
    load_simulation_config,
    vector_from_state,
)


class SimulationEngine:
    def __init__(self, simulation_config: SimulationConfig, action_catalog: ActionCatalog) -> None:
        self._simulation_config = simulation_config
        self._action_catalog = action_catalog
        self._state = build_initial_state(simulation_config.variable_definitions)
        self._random = random.Random(simulation_config.random_seed)
        self._active_actions: set[str] = set()
        self._paused = False

    @classmethod
    def from_config_directory(cls, config_directory: str | Path) -> "SimulationEngine":
        config_root = Path(config_directory)
        return cls.from_paths(
            config_root / "variables.json",
            config_root / "actions.json",
        )

    @classmethod
    def from_paths(
        cls,
        variables_config_path: str | Path,
        actions_config_path: str | Path,
    ) -> "SimulationEngine":
        simulation_config = load_simulation_config(variables_config_path)
        action_catalog = load_action_catalog(actions_config_path)
        return cls(simulation_config, action_catalog)

    @property
    def tick_seconds(self) -> float:
        return self._simulation_config.tick_seconds

    @property
    def active_actions(self) -> tuple[str, ...]:
        return tuple(sorted(self._active_actions))

    def list_thrusters(self) -> tuple[dict[str, Any], ...]:
        return tuple(
            {
                "id": thruster.action_id,
                "label": thruster.label,
                "axis": thruster.axis,
                "direction": thruster.direction,
                "acceleration_per_second": thruster.acceleration_per_second,
            }
            for thruster in self._action_catalog.thrusters.values()
        )

    def list_conversions(self) -> tuple[dict[str, Any], ...]:
        return tuple(
            {
                "id": conversion.action_id,
                "label": conversion.label,
                "source_variable": conversion.source_variable,
                "source_amount": conversion.source_amount,
                "target_variable": conversion.target_variable,
                "target_amount": conversion.target_amount,
            }
            for conversion in self._action_catalog.conversions.values()
        )

    def start_action(self, action_id: str) -> None:
        if action_id in self._action_catalog.thrusters:
            self._active_actions.add(action_id)

    def stop_action(self, action_id: str) -> None:
        self._active_actions.discard(action_id)

    def trigger_conversion(self, action_id: str) -> bool:
        conversion = self._action_catalog.conversions.get(action_id)
        if conversion is None:
            return False

        source_value = self._state.values.get(conversion.source_variable, 0.0)
        if source_value < conversion.source_amount:
            return False

        self._set_variable(
            conversion.source_variable,
            source_value - conversion.source_amount,
        )
        target_value = self._state.values.get(conversion.target_variable, 0.0)
        self._set_variable(
            conversion.target_variable,
            target_value + conversion.target_amount,
        )
        return True

    def toggle_pause(self) -> bool:
        self._paused = not self._paused
        return self._paused

    def reset(self) -> None:
        self._state = build_initial_state(self._simulation_config.variable_definitions)
        self._random = random.Random(self._simulation_config.random_seed)
        self._active_actions.clear()
        self._paused = False

    def step(self, dt_seconds: float | None = None) -> dict[str, Any]:
        step_seconds = self.tick_seconds if dt_seconds is None else float(dt_seconds)
        if self._paused:
            return self.snapshot()

        effective_actions = self._effective_actions()
        self._apply_profiles(step_seconds, effective_actions)
        self._apply_thrusters(step_seconds, effective_actions)
        self._integrate_position(step_seconds)
        self._state.elapsed_seconds += step_seconds
        return self.snapshot()

    def snapshot(self) -> dict[str, Any]:
        groups: dict[str, list[dict[str, Any]]] = {}
        variables: dict[str, dict[str, Any]] = {}

        for name, definition in self._simulation_config.variable_definitions.items():
            payload = {
                "name": name,
                "label": definition.label,
                "group": definition.group,
                "unit": definition.unit,
                "value": self._state.values[name],
                "minimum": definition.minimum,
                "maximum": definition.maximum,
                "precision": definition.precision,
            }
            variables[name] = payload
            groups.setdefault(definition.group, []).append(payload)

        return {
            "elapsed_seconds": self._state.elapsed_seconds,
            "paused": self._paused,
            "active_actions": self.active_actions,
            "variables": variables,
            "groups": groups,
            "position": vector_from_state(self._state.values, "position"),
            "velocity": vector_from_state(self._state.values, "velocity"),
        }

    def _effective_actions(self) -> set[str]:
        if self._state.values.get("Fuel", 0.0) <= 0.0:
            return {
                action_id
                for action_id in self._active_actions
                if action_id not in self._action_catalog.thrusters
            }
        return set(self._active_actions)

    def _apply_profiles(self, dt_seconds: float, active_actions: set[str]) -> None:
        for name, definition in self._simulation_config.variable_definitions.items():
            if not definition.profiles:
                continue

            current_value = self._state.values[name]
            delta = compute_net_delta(definition.profiles, dt_seconds, active_actions, self._random)
            self._set_variable(name, current_value + delta)

    def _apply_thrusters(self, dt_seconds: float, active_actions: set[str]) -> None:
        for action_id in active_actions:
            thruster = self._action_catalog.thrusters.get(action_id)
            if thruster is None:
                continue

            velocity_name = f"velocity_{thruster.axis}"
            next_velocity = self._state.values.get(velocity_name, 0.0) + (
                thruster.acceleration_per_second * thruster.direction * dt_seconds
            )
            self._set_variable(velocity_name, next_velocity)

    def _integrate_position(self, dt_seconds: float) -> None:
        for axis in ("x", "y", "z"):
            position_name = f"position_{axis}"
            velocity_name = f"velocity_{axis}"
            current_position = self._state.values.get(position_name, 0.0)
            current_velocity = self._state.values.get(velocity_name, 0.0)
            self._set_variable(position_name, current_position + current_velocity * dt_seconds)

    def _set_variable(self, name: str, value: float) -> None:
        definition = self._simulation_config.variable_definitions.get(name)
        if definition is None:
            self._state.values[name] = value
            return

        self._state.values[name] = clamp_value(value, definition.minimum, definition.maximum)
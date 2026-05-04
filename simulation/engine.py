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
        (
            self._variable_modules,
            self._variable_systems,
            self._action_modules,
            self._action_systems,
        ) = self._build_ownership_indexes()
        self._validate_configuration()
        self._state = build_initial_state(simulation_config)
        self._random = random.Random(simulation_config.random_seed)
        self._active_actions: set[str] = set()
        self._paused = False

    @classmethod
    def from_config_directory(cls, config_directory: str | Path) -> "SimulationEngine":
        config_root = Path(config_directory)
        return cls.from_paths(
            config_root / "variables.json",
            config_root / "actions.json",
            config_root / "modules.json",
            config_root / "systems.json",
        )

    @classmethod
    def from_paths(
        cls,
        variables_config_path: str | Path,
        actions_config_path: str | Path,
        modules_config_path: str | Path | None = None,
        systems_config_path: str | Path | None = None,
    ) -> "SimulationEngine":
        simulation_config = load_simulation_config(
            variables_config_path,
            modules_config_path,
            systems_config_path,
        )
        action_catalog = load_action_catalog(actions_config_path)
        return cls(simulation_config, action_catalog)

    @property
    def tick_seconds(self) -> float:
        return self._simulation_config.tick_seconds

    @property
    def active_actions(self) -> tuple[str, ...]:
        power_context = self._evaluate_power_network(self.tick_seconds)
        return tuple(sorted(self._effective_actions(power_context)))

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
                "byproduct_variable": conversion.byproduct_variable,
                "byproduct_amount": conversion.byproduct_amount,
            }
            for conversion in self._action_catalog.conversions.values()
        )

    def list_modules(self) -> tuple[dict[str, Any], ...]:
        return tuple(
            {
                "id": module.module_id,
                "number": module_number,
                "label": module.label,
                "initial_integrity": module.initial_integrity,
                "connections": module.connections,
                "systems_ids": module.systems_ids,
                "systems": tuple(
                    {
                        "id": system.system_id,
                        "module_id": system.module_id,
                        "label": system.label,
                        "kind": system.kind,
                        "variable_names": system.variable_names,
                        "action_ids": system.action_ids,
                        "power_draw_per_second": system.power_draw_per_second,
                        "power_generation_per_second": system.power_generation_per_second,
                        "generated_variable": system.generated_variable,
                    }
                    for system in module.systems
                ),
            }
            for module_number, module in enumerate(self._simulation_config.module_definitions.values(), start=1)
        )

    def start_action(self, action_id: str) -> None:
        power_context = self._evaluate_power_network(self.tick_seconds)
        if action_id in self._action_catalog.thrusters and self._action_is_operational(action_id, power_context):
            self._active_actions.add(action_id)

    def stop_action(self, action_id: str) -> None:
        self._active_actions.discard(action_id)

    def trigger_conversion(self, action_id: str) -> bool:
        power_context = self._evaluate_power_network(self.tick_seconds)
        conversion = self._action_catalog.conversions.get(action_id)
        if conversion is None:
            return False
        if not self._action_is_operational(action_id, power_context):
            return False
        if not self._variable_is_operational(conversion.source_variable):
            return False
        if not self._variable_is_operational(conversion.target_variable):
            return False
        if conversion.byproduct_variable is not None and not self._variable_is_operational(conversion.byproduct_variable):
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
        if conversion.byproduct_variable is not None and conversion.byproduct_amount > 0.0:
            byproduct_value = self._state.values.get(conversion.byproduct_variable, 0.0)
            self._set_variable(
                conversion.byproduct_variable,
                byproduct_value + conversion.byproduct_amount,
            )
        return True

    def toggle_pause(self) -> bool:
        self._paused = not self._paused
        return self._paused

    def reset(self) -> None:
        self._state = build_initial_state(self._simulation_config)
        self._random = random.Random(self._simulation_config.random_seed)
        self._active_actions.clear()
        self._paused = False

    def set_module_integrity(self, module_id: str, integrity: float) -> float:
        module = self._simulation_config.module_definitions.get(module_id)
        if module is None:
            raise KeyError(module_id)

        previous_integrity = self._state.module_integrity[module_id]
        next_integrity = max(0.0, float(integrity))
        self._state.module_integrity[module_id] = next_integrity
        if previous_integrity > 0.0 and next_integrity <= 0.0:
            self._handle_module_failure(module)
        return next_integrity

    def step(self, dt_seconds: float | None = None) -> dict[str, Any]:
        step_seconds = self.tick_seconds if dt_seconds is None else float(dt_seconds)
        if self._paused:
            return self.snapshot()

        power_context = self._apply_power_network(step_seconds)
        effective_actions = self._effective_actions(power_context)
        self._apply_profiles(step_seconds, effective_actions)
        self._apply_thrusters(step_seconds, effective_actions)
        self._integrate_position(step_seconds)
        self._state.elapsed_seconds += step_seconds
        return self.snapshot()

    def snapshot(self) -> dict[str, Any]:
        power_context = self._evaluate_power_network(self.tick_seconds)
        effective_actions = tuple(sorted(self._effective_actions(power_context)))
        groups: dict[str, list[dict[str, Any]]] = {}
        core_groups: dict[str, list[dict[str, Any]]] = {}
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
            if name not in self._variable_modules:
                core_groups.setdefault(definition.group, []).append(payload)

        return {
            "elapsed_seconds": self._state.elapsed_seconds,
            "paused": self._paused,
            "active_actions": effective_actions,
            "alerts": power_context["alerts"],
            "variables": variables,
            "groups": groups,
            "core_groups": core_groups,
            "modules": tuple(self._build_module_snapshot(variables, power_context, set(effective_actions))),
            "position": vector_from_state(self._state.values, "position"),
            "velocity": vector_from_state(self._state.values, "velocity"),
        }

    def _effective_actions(self, power_context: dict[str, Any]) -> set[str]:
        effective_actions = {
            action_id for action_id in self._active_actions if self._action_is_operational(action_id, power_context)
        }
        if self._state.values.get("H2", 0.0) <= 0.0 or self._state.values.get("O2", 0.0) <= 0.0:
            return {
                action_id
                for action_id in effective_actions
                if action_id not in self._action_catalog.thrusters
            }
        return effective_actions

    def _apply_profiles(self, dt_seconds: float, active_actions: set[str]) -> None:
        for name, definition in self._simulation_config.variable_definitions.items():
            if not definition.profiles:
                continue
            if not self._variable_is_operational(name):
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
        if not self._variable_is_operational(name):
            return

        self._assign_variable(name, value)

    def _assign_variable(self, name: str, value: float) -> None:
        definition = self._simulation_config.variable_definitions.get(name)
        if definition is None:
            self._state.values[name] = value
            return

        self._state.values[name] = clamp_value(value, definition.minimum, definition.maximum)

    def _build_ownership_indexes(
        self,
    ) -> tuple[
        dict[str, Any],
        dict[str, Any],
        dict[str, Any],
        dict[str, Any],
    ]:
        variable_modules: dict[str, Any] = {}
        variable_systems: dict[str, Any] = {}
        action_modules: dict[str, Any] = {}
        action_systems: dict[str, Any] = {}

        for module in self._simulation_config.module_definitions.values():
            for system in module.systems:
                for variable_name in system.variable_names:
                    if variable_name in variable_modules:
                        raise ValueError(f"Variable is owned by multiple systems: {variable_name}")
                    variable_modules[variable_name] = module
                    variable_systems[variable_name] = system

                for action_id in system.action_ids:
                    if action_id in action_modules:
                        raise ValueError(f"Action is owned by multiple systems: {action_id}")
                    action_modules[action_id] = module
                    action_systems[action_id] = system

        return variable_modules, variable_systems, action_modules, action_systems

    def _validate_configuration(self) -> None:
        for variable_name in self._variable_modules:
            if variable_name not in self._simulation_config.variable_definitions:
                raise ValueError(f"Module references unknown variable: {variable_name}")

        for action_id in self._action_modules:
            if action_id not in self._action_catalog.action_ids:
                raise ValueError(f"Module references unknown action: {action_id}")

        for module in self._simulation_config.module_definitions.values():
            for system in module.systems:
                if system.generated_variable is None:
                    continue
                if system.generated_variable not in self._simulation_config.variable_definitions:
                    raise ValueError(f"System references unknown generated variable: {system.system_id}")
                if system.generated_variable not in self._variable_modules:
                    raise ValueError(f"Generated variable must belong to a container system: {system.generated_variable}")

        for name, definition in self._simulation_config.variable_definitions.items():
            if definition.group in {"ship", "position", "velocity"}:
                continue
            if name not in self._variable_modules:
                raise ValueError(f"Variable must belong to a container system: {name}")

        for action_id in self._action_catalog.action_ids:
            if action_id not in self._action_modules:
                raise ValueError(f"Action must belong to a mechanism system: {action_id}")

    def _build_module_snapshot(
        self,
        variables: dict[str, dict[str, Any]],
        power_context: dict[str, Any],
        active_actions: set[str],
    ) -> list[dict[str, Any]]:
        modules: list[dict[str, Any]] = []

        for module_number, module in enumerate(self._simulation_config.module_definitions.values(), start=1):
            systems: list[dict[str, Any]] = []
            owned_variables: list[dict[str, Any]] = []
            owned_actions: list[dict[str, Any]] = []
            operational = self._module_is_operational(module.module_id)
            insufficient_power = module.module_id in power_context["insufficient_module_ids"]

            for system in module.systems:
                system_variables = [variables[name] for name in system.variable_names]
                powered = self._system_is_powered(system.system_id, power_context)
                system_actions = [
                    self._build_action_payload(action_id, power_context, active_actions)
                    for action_id in system.action_ids
                ]
                owned_variables.extend(system_variables)
                owned_actions.extend(system_actions)
                system_status = "Failed"
                if operational:
                    system_status = "Operational" if powered else "Insufficient Power"
                systems.append(
                    {
                        "id": system.system_id,
                        "module_id": system.module_id,
                        "label": system.label,
                        "kind": system.kind,
                        "operational": operational and powered,
                        "powered": powered,
                        "status": system_status,
                        "variables": system_variables,
                        "actions": system_actions,
                        "power_draw_per_second": system.power_draw_per_second,
                        "power_generation_per_second": system.power_generation_per_second,
                        "generated_variable": system.generated_variable,
                    }
                )

            module_status = "Failed"
            if operational:
                module_status = "Insufficient Power" if insufficient_power else "Operational"
            modules.append(
                {
                    "id": module.module_id,
                    "number": module_number,
                    "label": module.label,
                    "integrity": self._state.module_integrity[module.module_id],
                    "operational": operational,
                    "status": module_status,
                    "insufficient_power": insufficient_power,
                    "connections": module.connections,
                    "systems_ids": module.systems_ids,
                    "systems": systems,
                    "variables": owned_variables,
                    "actions": owned_actions,
                }
            )

        return modules

    def _build_action_payload(
        self,
        action_id: str,
        power_context: dict[str, Any],
        active_actions: set[str],
    ) -> dict[str, Any]:
        thruster = self._action_catalog.thrusters.get(action_id)
        if thruster is not None:
            return {
                "id": thruster.action_id,
                "label": thruster.label,
                "kind": "thruster",
                "axis": thruster.axis,
                "direction": thruster.direction,
                "acceleration_per_second": thruster.acceleration_per_second,
                "active": thruster.action_id in active_actions,
                "operational": self._action_is_operational(thruster.action_id, power_context),
            }

        conversion = self._action_catalog.conversions[action_id]
        return {
            "id": conversion.action_id,
            "label": conversion.label,
            "kind": "conversion",
            "source_variable": conversion.source_variable,
            "source_amount": conversion.source_amount,
            "target_variable": conversion.target_variable,
            "target_amount": conversion.target_amount,
            "byproduct_variable": conversion.byproduct_variable,
            "byproduct_amount": conversion.byproduct_amount,
            "active": False,
            "operational": self._action_is_operational(conversion.action_id, power_context),
        }

    def _module_is_operational(self, module_id: str) -> bool:
        return self._state.module_integrity.get(module_id, 0.0) > 0.0

    def _action_is_operational(self, action_id: str, power_context: dict[str, Any] | None = None) -> bool:
        module = self._action_modules.get(action_id)
        if module is None:
            return True
        if not self._module_is_operational(module.module_id):
            return False

        power_state = self._evaluate_power_network(self.tick_seconds) if power_context is None else power_context
        system = self._action_systems.get(action_id)
        if system is None:
            return True
        return self._system_is_powered(system.system_id, power_state)

    def _variable_is_operational(self, name: str) -> bool:
        module = self._variable_modules.get(name)
        if module is None:
            return True
        return self._module_is_operational(module.module_id)

    def _system_is_powered(self, system_id: str, power_context: dict[str, Any]) -> bool:
        return system_id in power_context["powered_system_ids"]

    def _apply_power_network(self, dt_seconds: float) -> dict[str, Any]:
        power_context = self._evaluate_power_network(dt_seconds)
        for variable_name, delta in power_context["electricity_deltas"].items():
            current_value = self._state.values.get(variable_name, 0.0)
            self._set_variable(variable_name, current_value + delta)
        return power_context

    def _evaluate_power_network(self, dt_seconds: float) -> dict[str, Any]:
        operational_module_ids = {
            module_id
            for module_id in self._simulation_config.module_definitions
            if self._module_is_operational(module_id)
        }
        adjacency = self._build_power_adjacency(operational_module_ids)
        powered_system_ids: set[str] = set()
        insufficient_module_ids: set[str] = set()
        electricity_deltas: dict[str, float] = {}
        alerts: list[str] = []
        visited: set[str] = set()

        for module_id in self._simulation_config.module_definitions:
            if module_id not in operational_module_ids or module_id in visited:
                continue

            queue = [module_id]
            component: list[str] = []
            visited.add(module_id)
            while queue:
                current_module_id = queue.pop(0)
                component.append(current_module_id)
                for neighbor_id in adjacency[current_module_id]:
                    if neighbor_id in visited:
                        continue
                    visited.add(neighbor_id)
                    queue.append(neighbor_id)

            component_context = self._evaluate_power_component(tuple(component), dt_seconds)
            powered_system_ids.update(component_context["powered_system_ids"])
            insufficient_module_ids.update(component_context["insufficient_module_ids"])
            if component_context["alert"] is not None:
                alerts.append(component_context["alert"])
            for variable_name, delta in component_context["electricity_deltas"].items():
                electricity_deltas[variable_name] = electricity_deltas.get(variable_name, 0.0) + delta

        return {
            "powered_system_ids": powered_system_ids,
            "insufficient_module_ids": insufficient_module_ids,
            "electricity_deltas": electricity_deltas,
            "alerts": tuple(dict.fromkeys(alerts)),
        }

    def _build_power_adjacency(self, operational_module_ids: set[str]) -> dict[str, set[str]]:
        adjacency = {module_id: set() for module_id in operational_module_ids}
        for module_id in operational_module_ids:
            module = self._simulation_config.module_definitions[module_id]
            for neighbor_id in module.connections:
                if neighbor_id not in operational_module_ids:
                    continue
                adjacency[module_id].add(neighbor_id)
                adjacency[neighbor_id].add(module_id)
        return adjacency

    def _evaluate_power_component(self, module_ids: tuple[str, ...], dt_seconds: float) -> dict[str, Any]:
        storage_variable_names: list[str] = []
        mechanism_systems: list[Any] = []
        generation_deltas: dict[str, float] = {}
        powered_generator_ids: set[str] = set()

        for module_id in module_ids:
            module = self._simulation_config.module_definitions[module_id]
            for system in module.systems:
                if system.kind == "container":
                    for variable_name in system.variable_names:
                        if variable_name.lower() == "electricity" and self._variable_is_operational(variable_name):
                            storage_variable_names.append(variable_name)
                    continue

                mechanism_systems.append(system)
                if system.power_generation_per_second <= 0.0 or system.generated_variable is None:
                    continue

                target_module = self._variable_modules.get(system.generated_variable)
                if target_module is None or target_module.module_id not in module_ids:
                    continue
                if not self._variable_is_operational(system.generated_variable):
                    continue

                generation_deltas[system.generated_variable] = (
                    generation_deltas.get(system.generated_variable, 0.0)
                    + system.power_generation_per_second * dt_seconds
                )
                powered_generator_ids.add(system.system_id)

        starting_electricity = sum(self._state.values.get(name, 0.0) for name in storage_variable_names)
        total_generation = sum(generation_deltas.values())
        total_draw = sum(system.power_draw_per_second * dt_seconds for system in mechanism_systems)

        if total_draw <= starting_electricity + total_generation + 1e-9:
            electricity_deltas = self._allocate_power_draw(storage_variable_names, generation_deltas, total_draw)
            return {
                "powered_system_ids": {system.system_id for system in mechanism_systems},
                "insufficient_module_ids": set(),
                "electricity_deltas": electricity_deltas,
                "alert": None,
            }

        if total_draw > 0.0:
            return {
                "powered_system_ids": powered_generator_ids,
                "insufficient_module_ids": set(module_ids),
                "electricity_deltas": generation_deltas,
                "alert": "Insufficient power",
            }

        return {
            "powered_system_ids": powered_generator_ids,
            "insufficient_module_ids": set(),
            "electricity_deltas": generation_deltas,
            "alert": None,
        }

    def _allocate_power_draw(
        self,
        storage_variable_names: list[str],
        generation_deltas: dict[str, float],
        total_draw: float,
    ) -> dict[str, float]:
        electricity_deltas = dict(generation_deltas)
        remaining_draw = total_draw
        available_by_variable = {
            variable_name: self._state.values.get(variable_name, 0.0) + electricity_deltas.get(variable_name, 0.0)
            for variable_name in storage_variable_names
        }

        for variable_name in storage_variable_names:
            if remaining_draw <= 0.0:
                break
            draw_amount = min(remaining_draw, available_by_variable[variable_name])
            electricity_deltas[variable_name] = electricity_deltas.get(variable_name, 0.0) - draw_amount
            remaining_draw -= draw_amount

        return electricity_deltas

    def _handle_module_failure(self, module: Any) -> None:
        for system in module.systems:
            for action_id in system.action_ids:
                self._active_actions.discard(action_id)
            for variable_name in system.variable_names:
                self._assign_variable(variable_name, 0.0)
from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class VariableDefinition:
    name: str
    label: str
    unit: str = ""
    group: str = "resource"
    initial: float = 0.0
    minimum: float | None = None
    maximum: float | None = None
    precision: int = 2
    profiles: tuple[dict[str, Any], ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class SystemDefinition:
    system_id: str
    module_id: str
    label: str
    kind: str
    variable_names: tuple[str, ...] = field(default_factory=tuple)
    action_ids: tuple[str, ...] = field(default_factory=tuple)
    power_draw_per_second: float = 0.0
    power_generation_per_second: float = 0.0
    generated_variable: str | None = None


@dataclass(frozen=True)
class ModuleDefinition:
    module_id: str
    label: str
    initial_integrity: float = 100.0
    connections: tuple[str, ...] = field(default_factory=tuple)
    systems_ids: tuple[str, ...] = field(default_factory=tuple)
    systems: tuple[SystemDefinition, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class SimulationConfig:
    tick_seconds: float
    random_seed: int
    variable_definitions: dict[str, VariableDefinition]
    module_definitions: dict[str, ModuleDefinition]


@dataclass
class SimulationState:
    values: dict[str, float]
    module_integrity: dict[str, float]
    elapsed_seconds: float = 0.0

    def copy(self) -> "SimulationState":
        return SimulationState(
            values=dict(self.values),
            module_integrity=dict(self.module_integrity),
            elapsed_seconds=self.elapsed_seconds,
        )


def clamp_value(value: float, minimum: float | None, maximum: float | None) -> float:
    if minimum is not None:
        value = max(value, minimum)
    if maximum is not None:
        value = min(value, maximum)
    return value


def load_simulation_config(
    config_path: str | Path,
    modules_config_path: str | Path | None = None,
    systems_config_path: str | Path | None = None,
) -> SimulationConfig:
    path = Path(config_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    definitions: dict[str, VariableDefinition] = {}
    module_definitions = load_module_definitions(
        modules_config_path or path.with_name("modules.json"),
        systems_config_path or path.with_name("systems.json"),
    )

    for raw_variable in payload["variables"]:
        definition = VariableDefinition(
            name=raw_variable["name"],
            label=raw_variable.get("label", raw_variable["name"]),
            unit=raw_variable.get("unit", ""),
            group=raw_variable.get("group", "resource"),
            initial=float(raw_variable.get("initial", 0.0)),
            minimum=_optional_float(raw_variable.get("minimum")),
            maximum=_optional_float(raw_variable.get("maximum")),
            precision=int(raw_variable.get("precision", 2)),
            profiles=tuple(raw_variable.get("profiles", [])),
        )
        if definition.name in definitions:
            raise ValueError(f"Duplicate variable definition: {definition.name}")
        definitions[definition.name] = definition

    return SimulationConfig(
        tick_seconds=float(payload.get("tick_seconds", 0.1)),
        random_seed=int(payload.get("random_seed", 7)),
        variable_definitions=definitions,
        module_definitions=module_definitions,
    )


def load_module_definitions(
    modules_config_path: str | Path,
    systems_config_path: str | Path,
) -> dict[str, ModuleDefinition]:
    path = Path(modules_config_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    module_metadata: dict[str, dict[str, Any]] = {}
    system_definitions = load_system_definitions(systems_config_path)
    assigned_system_ids: set[str] = set()

    for raw_module in payload.get("modules", []):
        module_id = raw_module["id"]
        if module_id in module_metadata:
            raise ValueError(f"Duplicate module definition: {module_id}")

        initial_integrity = float(raw_module.get("initial_integrity", 100.0))
        if initial_integrity < 0.0:
            raise ValueError(f"Module integrity cannot be negative: {module_id}")

        module_metadata[module_id] = {
            "label": raw_module.get("label", module_id),
            "initial_integrity": initial_integrity,
            "connections": _string_tuple(raw_module.get("connections", [])),
            "systems_ids": _string_tuple(raw_module.get("systems_ids", [])),
        }

    for module_id, metadata in module_metadata.items():
        for connected_module_id in metadata["connections"]:
            if connected_module_id == module_id:
                raise ValueError(f"Module cannot connect to itself: {module_id}")
            if connected_module_id not in module_metadata:
                raise ValueError(f"Module references unknown connection: {module_id} -> {connected_module_id}")

        for system_id in metadata["systems_ids"]:
            if system_id not in system_definitions:
                raise ValueError(f"Module references unknown system: {module_id} -> {system_id}")
            if system_id in assigned_system_ids:
                raise ValueError(f"System is owned by multiple modules: {system_id}")
            assigned_system_ids.add(system_id)

    unassigned_system_ids = set(system_definitions) - assigned_system_ids
    if unassigned_system_ids:
        names = ", ".join(sorted(unassigned_system_ids))
        raise ValueError(f"Systems must belong to a module: {names}")

    return {
        module_id: ModuleDefinition(
            module_id=module_id,
            label=str(metadata["label"]),
            initial_integrity=float(metadata["initial_integrity"]),
            connections=tuple(str(connection) for connection in metadata["connections"]),
            systems_ids=tuple(str(system_id) for system_id in metadata["systems_ids"]),
            systems=tuple(
                _with_module_id(system_definitions[system_id], module_id)
                for system_id in metadata["systems_ids"]
            ),
        )
        for module_id, metadata in module_metadata.items()
    }


def load_system_definitions(
    config_path: str | Path,
 ) -> dict[str, SystemDefinition]:
    path = Path(config_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    systems_by_id: dict[str, SystemDefinition] = {}
    system_ids: set[str] = set()

    for raw_system in payload.get("systems", []):
        system_id = raw_system["id"]
        if system_id in system_ids:
            raise ValueError(f"Duplicate system definition: {system_id}")

        kind = str(raw_system["kind"]).lower()
        variable_names = _string_tuple(raw_system.get("variable_names", []))
        action_ids = _string_tuple(raw_system.get("action_ids", []))
        power_draw_per_second = float(raw_system.get("power_draw_per_second", 0.0))
        power_generation_per_second = float(raw_system.get("power_generation_per_second", 0.0))
        generated_variable = raw_system.get("generated_variable")
        if generated_variable is not None:
            generated_variable = str(generated_variable)

        if power_draw_per_second < 0.0:
            raise ValueError(f"System power draw cannot be negative: {system_id}")
        if power_generation_per_second < 0.0:
            raise ValueError(f"System power generation cannot be negative: {system_id}")

        if kind == "container":
            if not variable_names:
                raise ValueError(f"Container system requires variable_names: {system_id}")
            if action_ids:
                raise ValueError(f"Container system cannot declare action_ids: {system_id}")
            if power_draw_per_second > 0.0 or power_generation_per_second > 0.0 or generated_variable is not None:
                raise ValueError(f"Container system cannot declare power behavior: {system_id}")
        elif kind == "mechanism":
            if variable_names:
                raise ValueError(f"Mechanism system cannot declare variable_names: {system_id}")
            if not action_ids and power_generation_per_second <= 0.0:
                raise ValueError(f"Mechanism system requires action_ids or power generation: {system_id}")
            if power_generation_per_second > 0.0 and generated_variable is None:
                raise ValueError(f"Power-generating mechanism requires generated_variable: {system_id}")
            if power_generation_per_second <= 0.0 and generated_variable is not None:
                raise ValueError(f"generated_variable requires positive power generation: {system_id}")
        else:
            raise ValueError(f"Unsupported system kind: {kind}")

        systems_by_id[system_id] = SystemDefinition(
            system_id=system_id,
            module_id="",
            label=raw_system.get("label", system_id),
            kind=kind,
            variable_names=variable_names,
            action_ids=action_ids,
            power_draw_per_second=power_draw_per_second,
            power_generation_per_second=power_generation_per_second,
            generated_variable=generated_variable,
        )
        system_ids.add(system_id)

    return systems_by_id


def build_initial_state(simulation_config: SimulationConfig) -> SimulationState:
    return SimulationState(
        values={name: definition.initial for name, definition in simulation_config.variable_definitions.items()},
        module_integrity={
            module_id: definition.initial_integrity
            for module_id, definition in simulation_config.module_definitions.items()
        },
    )


def vector_from_state(values: dict[str, float], prefix: str) -> dict[str, float]:
    return {axis: values.get(f"{prefix}_{axis}", 0.0) for axis in ("x", "y", "z")}


def _optional_float(raw_value: Any) -> float | None:
    if raw_value is None:
        return None
    return float(raw_value)


def _string_tuple(raw_values: Any) -> tuple[str, ...]:
    if raw_values is None:
        return ()

    values = tuple(str(raw_value) for raw_value in raw_values)
    if len(set(values)) != len(values):
        raise ValueError(f"Duplicate values are not allowed: {values}")
    return values


def _with_module_id(system: SystemDefinition, module_id: str) -> SystemDefinition:
    return SystemDefinition(
        system_id=system.system_id,
        module_id=module_id,
        label=system.label,
        kind=system.kind,
        variable_names=system.variable_names,
        action_ids=system.action_ids,
        power_draw_per_second=system.power_draw_per_second,
        power_generation_per_second=system.power_generation_per_second,
        generated_variable=system.generated_variable,
    )
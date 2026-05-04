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
    label: str
    kind: str
    variable_names: tuple[str, ...] = field(default_factory=tuple)
    action_ids: tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class ModuleDefinition:
    module_id: str
    label: str
    initial_integrity: float = 100.0
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
) -> SimulationConfig:
    path = Path(config_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    definitions: dict[str, VariableDefinition] = {}
    module_definitions = load_module_definitions(modules_config_path or path.with_name("modules.json"))

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


def load_module_definitions(config_path: str | Path) -> dict[str, ModuleDefinition]:
    path = Path(config_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    module_definitions: dict[str, ModuleDefinition] = {}
    system_ids: set[str] = set()

    for raw_module in payload.get("modules", []):
        module_id = raw_module["id"]
        if module_id in module_definitions:
            raise ValueError(f"Duplicate module definition: {module_id}")

        systems: list[SystemDefinition] = []
        for raw_system in raw_module.get("systems", []):
            system_id = raw_system["id"]
            if system_id in system_ids:
                raise ValueError(f"Duplicate system definition: {system_id}")

            kind = str(raw_system["kind"]).lower()
            variable_names = _string_tuple(raw_system.get("variable_names", []))
            action_ids = _string_tuple(raw_system.get("action_ids", []))

            if kind == "container":
                if not variable_names:
                    raise ValueError(f"Container system requires variable_names: {system_id}")
                if action_ids:
                    raise ValueError(f"Container system cannot declare action_ids: {system_id}")
            elif kind == "mechanism":
                if not action_ids:
                    raise ValueError(f"Mechanism system requires action_ids: {system_id}")
                if variable_names:
                    raise ValueError(f"Mechanism system cannot declare variable_names: {system_id}")
            else:
                raise ValueError(f"Unsupported system kind: {kind}")

            systems.append(
                SystemDefinition(
                    system_id=system_id,
                    label=raw_system.get("label", system_id),
                    kind=kind,
                    variable_names=variable_names,
                    action_ids=action_ids,
                )
            )
            system_ids.add(system_id)

        initial_integrity = float(raw_module.get("initial_integrity", 100.0))
        if initial_integrity < 0.0:
            raise ValueError(f"Module integrity cannot be negative: {module_id}")

        module_definitions[module_id] = ModuleDefinition(
            module_id=module_id,
            label=raw_module.get("label", module_id),
            initial_integrity=initial_integrity,
            systems=tuple(systems),
        )

    return module_definitions


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
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
class SimulationConfig:
    tick_seconds: float
    random_seed: int
    variable_definitions: dict[str, VariableDefinition]


@dataclass
class SimulationState:
    values: dict[str, float]
    elapsed_seconds: float = 0.0

    def copy(self) -> "SimulationState":
        return SimulationState(values=dict(self.values), elapsed_seconds=self.elapsed_seconds)


def clamp_value(value: float, minimum: float | None, maximum: float | None) -> float:
    if minimum is not None:
        value = max(value, minimum)
    if maximum is not None:
        value = min(value, maximum)
    return value


def load_simulation_config(config_path: str | Path) -> SimulationConfig:
    path = Path(config_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    definitions: dict[str, VariableDefinition] = {}

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
        definitions[definition.name] = definition

    return SimulationConfig(
        tick_seconds=float(payload.get("tick_seconds", 0.1)),
        random_seed=int(payload.get("random_seed", 7)),
        variable_definitions=definitions,
    )


def build_initial_state(definitions: dict[str, VariableDefinition]) -> SimulationState:
    return SimulationState(values={name: definition.initial for name, definition in definitions.items()})


def vector_from_state(values: dict[str, float], prefix: str) -> dict[str, float]:
    return {axis: values.get(f"{prefix}_{axis}", 0.0) for axis in ("x", "y", "z")}


def _optional_float(raw_value: Any) -> float | None:
    if raw_value is None:
        return None
    return float(raw_value)
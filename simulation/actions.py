from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path


@dataclass(frozen=True)
class ThrusterAction:
    action_id: str
    label: str
    axis: str
    direction: float
    acceleration_per_second: float


@dataclass(frozen=True)
class ConversionAction:
    action_id: str
    label: str
    source_variable: str
    source_amount: float
    target_variable: str
    target_amount: float


@dataclass(frozen=True)
class ActionCatalog:
    thrusters: dict[str, ThrusterAction]
    conversions: dict[str, ConversionAction]

    @property
    def action_ids(self) -> set[str]:
        return set(self.thrusters) | set(self.conversions)

    @property
    def thruster_ids(self) -> set[str]:
        return set(self.thrusters)


def load_action_catalog(config_path: str | Path) -> ActionCatalog:
    payload = json.loads(Path(config_path).read_text(encoding="utf-8"))
    thrusters: dict[str, ThrusterAction] = {}
    conversions: dict[str, ConversionAction] = {}

    for raw_thruster in payload.get("thrusters", []):
        thruster = ThrusterAction(
            action_id=raw_thruster["id"],
            label=raw_thruster.get("label", raw_thruster["id"]),
            axis=str(raw_thruster["axis"]).lower(),
            direction=float(raw_thruster["direction"]),
            acceleration_per_second=float(raw_thruster["acceleration_per_second"]),
        )
        if thruster.action_id in thrusters or thruster.action_id in conversions:
            raise ValueError(f"Duplicate action definition: {thruster.action_id}")
        thrusters[thruster.action_id] = thruster

    for raw_conversion in payload.get("conversions", []):
        conversion = ConversionAction(
            action_id=raw_conversion["id"],
            label=raw_conversion.get("label", raw_conversion["id"]),
            source_variable=raw_conversion["source_variable"],
            source_amount=float(raw_conversion["source_amount"]),
            target_variable=raw_conversion["target_variable"],
            target_amount=float(raw_conversion["target_amount"]),
        )
        if conversion.action_id in conversions or conversion.action_id in thrusters:
            raise ValueError(f"Duplicate action definition: {conversion.action_id}")
        conversions[conversion.action_id] = conversion

    return ActionCatalog(thrusters=thrusters, conversions=conversions)
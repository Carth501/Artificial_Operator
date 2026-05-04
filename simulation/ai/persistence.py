from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .agents import TargetPolicyParameters


POLICY_FILE_VERSION = 1
TARGET_POLICY_KIND = "target_position_policy"


def save_target_policy_parameters(
    file_path: str | Path,
    parameters: TargetPolicyParameters,
) -> Path:
    path = Path(file_path)
    payload = {
        "version": POLICY_FILE_VERSION,
        "policy_kind": TARGET_POLICY_KIND,
        "parameters": {
            "brake_distance_multiplier": parameters.brake_distance_multiplier,
            "approach_distance_multiplier": parameters.approach_distance_multiplier,
            "approach_velocity_multiplier": parameters.approach_velocity_multiplier,
            "settle_velocity_multiplier": parameters.settle_velocity_multiplier,
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return path


def load_target_policy_parameters(file_path: str | Path) -> TargetPolicyParameters:
    payload = json.loads(Path(file_path).read_text(encoding="utf-8"))
    return parameters_from_payload(payload)


def parameters_from_payload(payload: dict[str, Any]) -> TargetPolicyParameters:
    if int(payload.get("version", 0)) != POLICY_FILE_VERSION:
        raise ValueError(f"Unsupported policy version: {payload.get('version')}")
    if str(payload.get("policy_kind", "")) != TARGET_POLICY_KIND:
        raise ValueError(f"Unsupported policy kind: {payload.get('policy_kind')}")

    raw_parameters = payload.get("parameters")
    if not isinstance(raw_parameters, dict):
        raise ValueError("Policy payload must contain a parameters object")

    return TargetPolicyParameters(
        brake_distance_multiplier=float(raw_parameters["brake_distance_multiplier"]),
        approach_distance_multiplier=float(raw_parameters["approach_distance_multiplier"]),
        approach_velocity_multiplier=float(raw_parameters["approach_velocity_multiplier"]),
        settle_velocity_multiplier=float(raw_parameters["settle_velocity_multiplier"]),
    )
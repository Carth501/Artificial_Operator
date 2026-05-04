from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

from .agents import TargetPolicyParameters


POLICY_FILE_VERSION = 1
TARGET_POLICY_KIND = "target_position_policy"


@dataclass(frozen=True)
class TargetPolicyMetadata:
    source: str | None = None
    training_rounds: int | None = None
    curriculum_size: int | None = None
    curriculum_seed: int | None = None
    anchor_target_position: tuple[float, float, float] | None = None
    target_ranges: tuple[float, float, float] | None = None
    best_total_reward: float | None = None
    best_average_reward: float | None = None
    success_rate: float | None = None
    anchor_stop_reason: str | None = None


def save_target_policy_parameters(
    file_path: str | Path,
    parameters: TargetPolicyParameters,
    metadata: TargetPolicyMetadata | None = None,
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
    if metadata is not None:
        payload["metadata"] = _metadata_to_payload(metadata)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return path


def load_target_policy_parameters(file_path: str | Path) -> TargetPolicyParameters:
    payload = json.loads(Path(file_path).read_text(encoding="utf-8"))
    return parameters_from_payload(payload)


def load_target_policy_metadata(file_path: str | Path) -> TargetPolicyMetadata | None:
    payload = json.loads(Path(file_path).read_text(encoding="utf-8"))
    return metadata_from_payload(payload)


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


def metadata_from_payload(payload: dict[str, Any]) -> TargetPolicyMetadata | None:
    raw_metadata = payload.get("metadata")
    if raw_metadata is None:
        return None
    if not isinstance(raw_metadata, dict):
        raise ValueError("Policy metadata must be an object when present")

    return TargetPolicyMetadata(
        source=_optional_str(raw_metadata.get("source")),
        training_rounds=_optional_int(raw_metadata.get("training_rounds")),
        curriculum_size=_optional_int(raw_metadata.get("curriculum_size")),
        curriculum_seed=_optional_int(raw_metadata.get("curriculum_seed")),
        anchor_target_position=_optional_vector3(raw_metadata.get("anchor_target_position")),
        target_ranges=_optional_vector3(raw_metadata.get("target_ranges")),
        best_total_reward=_optional_float(raw_metadata.get("best_total_reward")),
        best_average_reward=_optional_float(raw_metadata.get("best_average_reward")),
        success_rate=_optional_float(raw_metadata.get("success_rate")),
        anchor_stop_reason=_optional_str(raw_metadata.get("anchor_stop_reason")),
    )


def _metadata_to_payload(metadata: TargetPolicyMetadata) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    if metadata.source is not None:
        payload["source"] = metadata.source
    if metadata.training_rounds is not None:
        payload["training_rounds"] = metadata.training_rounds
    if metadata.curriculum_size is not None:
        payload["curriculum_size"] = metadata.curriculum_size
    if metadata.curriculum_seed is not None:
        payload["curriculum_seed"] = metadata.curriculum_seed
    if metadata.anchor_target_position is not None:
        payload["anchor_target_position"] = list(metadata.anchor_target_position)
    if metadata.target_ranges is not None:
        payload["target_ranges"] = list(metadata.target_ranges)
    if metadata.best_total_reward is not None:
        payload["best_total_reward"] = metadata.best_total_reward
    if metadata.best_average_reward is not None:
        payload["best_average_reward"] = metadata.best_average_reward
    if metadata.success_rate is not None:
        payload["success_rate"] = metadata.success_rate
    if metadata.anchor_stop_reason is not None:
        payload["anchor_stop_reason"] = metadata.anchor_stop_reason
    return payload


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    return int(value)


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    return str(value)


def _optional_vector3(value: Any) -> tuple[float, float, float] | None:
    if value is None:
        return None
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise ValueError("Vector metadata entries must be 3-item arrays")
    return (float(value[0]), float(value[1]), float(value[2]))
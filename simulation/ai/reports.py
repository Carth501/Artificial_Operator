from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .models import TargetPositionObjective
from .persistence import TargetPolicyMetadata
from .training import PolicyComparisonEntry, PolicyComparisonResult, TargetPositionCurriculum


COMPARISON_REPORT_VERSION = 1
COMPARISON_REPORT_KIND = "policy_comparison_report"


def save_policy_comparison_report(
    file_path: str | Path,
    objective: TargetPositionObjective,
    curriculum: TargetPositionCurriculum,
    comparison_result: PolicyComparisonResult,
    *,
    dt_seconds: float,
) -> Path:
    path = Path(file_path)
    payload = comparison_report_to_payload(
        objective,
        curriculum,
        comparison_result,
        dt_seconds=dt_seconds,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return path


def comparison_report_to_payload(
    objective: TargetPositionObjective,
    curriculum: TargetPositionCurriculum,
    comparison_result: PolicyComparisonResult,
    *,
    dt_seconds: float,
) -> dict[str, Any]:
    return {
        "version": COMPARISON_REPORT_VERSION,
        "report_kind": COMPARISON_REPORT_KIND,
        "target_position": list(objective.target_position),
        "tolerance": objective.tolerance,
        "settle_velocity": objective.settle_velocity,
        "max_steps": objective.max_steps,
        "dt_seconds": dt_seconds,
        "curriculum_size": comparison_result.curriculum_size,
        "curriculum_seed": curriculum.seed,
        "target_ranges": list(curriculum.target_ranges),
        "entries": [_comparison_entry_to_payload(entry) for entry in comparison_result.entries],
    }


def _comparison_entry_to_payload(entry: PolicyComparisonEntry) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "rank": entry.rank,
        "label": entry.label,
        "average_reward": entry.average_reward,
        "total_reward": entry.total_reward,
        "success_rate": entry.success_rate,
        "anchor_stop_reason": entry.anchor_stop_reason,
        "anchor_distance_to_target": entry.anchor_distance_to_target,
    }
    if entry.metadata is not None:
        payload["metadata"] = _metadata_to_payload(entry.metadata)
    return payload


def _metadata_to_payload(metadata: TargetPolicyMetadata) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for field_name, value in metadata.__dict__.items():
        if value is None:
            continue
        if isinstance(value, tuple):
            payload[field_name] = list(value)
            continue
        payload[field_name] = value
    return payload
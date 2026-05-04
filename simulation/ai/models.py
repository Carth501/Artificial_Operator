from __future__ import annotations

from dataclasses import dataclass


Vector3 = tuple[float, float, float]


@dataclass(frozen=True)
class Observation:
    elapsed_seconds: float
    position: Vector3
    velocity: Vector3
    active_actions: tuple[str, ...]
    alerts: tuple[str, ...]
    resources: dict[str, float]
    operational_thrusters: frozenset[str]


@dataclass(frozen=True)
class TargetPositionObjective:
    target_position: Vector3
    tolerance: float = 0.5
    settle_velocity: float = 0.25
    max_steps: int = 600
    progress_interval: int = 20


@dataclass(frozen=True)
class ObjectiveStatus:
    distance_to_target: float
    reached_target: bool
    settled: bool
    success: bool


@dataclass(frozen=True)
class AIRunResult:
    success: bool
    stop_reason: str
    steps_completed: int
    elapsed_seconds: float
    final_position: Vector3
    final_velocity: Vector3
    distance_to_target: float
    active_actions: tuple[str, ...]
    alerts: tuple[str, ...]
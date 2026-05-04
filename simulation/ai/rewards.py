from __future__ import annotations

from typing import Protocol

from .models import ObjectiveStatus, Observation, RewardBreakdown, TargetPositionObjective


class RewardModel(Protocol):
    def score_transition(
        self,
        previous_observation: Observation,
        previous_status: ObjectiveStatus,
        observation: Observation,
        status: ObjectiveStatus,
        objective: TargetPositionObjective,
        requested_actions: tuple[str, ...],
        stop_reason: str | None,
    ) -> RewardBreakdown:
        ...


class TargetPositionRewardModel:
    def __init__(
        self,
        *,
        progress_weight: float = 4.0,
        distance_weight: float = 0.05,
        action_penalty_per_thruster: float = 0.02,
        success_bonus: float = 50.0,
        failure_penalty: float = 20.0,
    ) -> None:
        self._progress_weight = progress_weight
        self._distance_weight = distance_weight
        self._action_penalty_per_thruster = action_penalty_per_thruster
        self._success_bonus = success_bonus
        self._failure_penalty = failure_penalty

    def score_transition(
        self,
        previous_observation: Observation,
        previous_status: ObjectiveStatus,
        observation: Observation,
        status: ObjectiveStatus,
        objective: TargetPositionObjective,
        requested_actions: tuple[str, ...],
        stop_reason: str | None,
    ) -> RewardBreakdown:
        del previous_observation
        del observation
        del objective

        progress_reward = (previous_status.distance_to_target - status.distance_to_target) * self._progress_weight
        distance_penalty = -status.distance_to_target * self._distance_weight
        action_penalty = -len(requested_actions) * self._action_penalty_per_thruster

        terminal_reward = 0.0
        if stop_reason == "target_reached":
            terminal_reward = self._success_bonus
        elif stop_reason is not None:
            terminal_reward = -self._failure_penalty

        total_reward = progress_reward + distance_penalty + action_penalty + terminal_reward
        return RewardBreakdown(
            progress_reward=progress_reward,
            distance_penalty=distance_penalty,
            action_penalty=action_penalty,
            terminal_reward=terminal_reward,
            total_reward=total_reward,
        )
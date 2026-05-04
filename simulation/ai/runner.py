from __future__ import annotations

from math import sqrt
from typing import Any, Callable

from simulation.engine import SimulationEngine

from .agents import Agent
from .models import AIRunResult, EpisodeStep, ObjectiveStatus, Observation, TargetPositionObjective, Vector3
from .rewards import RewardModel, TargetPositionRewardModel


ProgressCallback = Callable[[int, Observation, ObjectiveStatus], None]


def build_observation(snapshot: dict[str, Any]) -> Observation:
    variables = snapshot.get("variables", {})
    resources: dict[str, float] = {}
    if isinstance(variables, dict):
        for name, payload in variables.items():
            if not isinstance(payload, dict):
                continue
            group_name = str(payload.get("group", ""))
            if group_name in {"position", "velocity"}:
                continue
            resources[str(name)] = float(payload.get("value", 0.0))

    position_payload = snapshot.get("position", {})
    velocity_payload = snapshot.get("velocity", {})
    modules = snapshot.get("modules", ())
    active_actions = snapshot.get("active_actions", ())
    alerts = snapshot.get("alerts", ())

    return Observation(
        elapsed_seconds=float(snapshot.get("elapsed_seconds", 0.0)),
        position=_vector_from_payload(position_payload),
        velocity=_vector_from_payload(velocity_payload),
        active_actions=tuple(str(action_id) for action_id in active_actions),
        alerts=tuple(str(alert) for alert in alerts),
        resources=resources,
        operational_thrusters=_operational_thrusters_from_modules(modules),
    )


def evaluate_objective(observation: Observation, objective: TargetPositionObjective) -> ObjectiveStatus:
    displacement = tuple(
        target_value - current_value
        for current_value, target_value in zip(observation.position, objective.target_position, strict=True)
    )
    distance_to_target = sqrt(sum(component * component for component in displacement))
    reached_target = all(abs(component) <= objective.tolerance for component in displacement)
    settled = all(abs(component) <= objective.settle_velocity for component in observation.velocity)
    return ObjectiveStatus(
        distance_to_target=distance_to_target,
        reached_target=reached_target,
        settled=settled,
        success=reached_target and settled,
    )


class SimulationAIRunner:
    def __init__(
        self,
        engine: SimulationEngine,
        agent: Agent,
        reward_model: RewardModel | None = None,
    ) -> None:
        self._engine = engine
        self._agent = agent
        self._reward_model = TargetPositionRewardModel() if reward_model is None else reward_model

    def run(
        self,
        objective: TargetPositionObjective,
        *,
        dt_seconds: float | None = None,
        reset_engine: bool = True,
        progress_callback: ProgressCallback | None = None,
    ) -> AIRunResult:
        if reset_engine:
            self._engine.reset()

        snapshot = self._engine.snapshot()
        observation = build_observation(snapshot)
        status = evaluate_objective(observation, objective)
        if status.success:
            return self._build_result(snapshot, status, 0, "target_reached", 0.0, ())
        if not observation.operational_thrusters:
            return self._build_result(snapshot, status, 0, "no_operational_thrusters", 0.0, ())

        total_reward = 0.0
        episode_steps: list[EpisodeStep] = []

        for step_index in range(1, objective.max_steps + 1):
            previous_observation = observation
            previous_status = status
            requested_actions = self._agent.select_actions(observation, objective)
            self._engine.set_active_actions(requested_actions)
            snapshot = self._engine.step(dt_seconds)
            observation = build_observation(snapshot)
            status = evaluate_objective(observation, objective)

            stop_reason: str | None = None
            if status.success:
                stop_reason = "target_reached"
            else:
                stop_reason = _detect_blocker(snapshot, observation, requested_actions)
                if stop_reason is None and step_index == objective.max_steps:
                    stop_reason = "step_limit_reached"

            requested_actions_tuple = tuple(sorted(requested_actions))
            reward_breakdown = self._reward_model.score_transition(
                previous_observation,
                previous_status,
                observation,
                status,
                objective,
                requested_actions_tuple,
                stop_reason,
            )
            total_reward += reward_breakdown.total_reward
            episode_steps.append(
                EpisodeStep(
                    step_index=step_index,
                    elapsed_seconds=observation.elapsed_seconds,
                    requested_actions=requested_actions_tuple,
                    active_actions=observation.active_actions,
                    position=observation.position,
                    velocity=observation.velocity,
                    distance_to_target=status.distance_to_target,
                    stop_reason=stop_reason,
                    reward=reward_breakdown.total_reward,
                    reward_breakdown=reward_breakdown,
                )
            )

            if progress_callback is not None and (
                step_index == 1
                or status.success
                or step_index % max(1, objective.progress_interval) == 0
            ):
                progress_callback(step_index, observation, status)

            if stop_reason is not None:
                self._engine.set_active_actions(())
                snapshot = self._engine.snapshot()
                return self._build_result(
                    snapshot,
                    status,
                    step_index,
                    stop_reason,
                    total_reward,
                    tuple(episode_steps),
                )

        self._engine.set_active_actions(())
        snapshot = self._engine.snapshot()
        return self._build_result(
            snapshot,
            status,
            objective.max_steps,
            "step_limit_reached",
            total_reward,
            tuple(episode_steps),
        )

    def _build_result(
        self,
        snapshot: dict[str, Any],
        status: ObjectiveStatus,
        steps_completed: int,
        stop_reason: str,
        total_reward: float,
        episode_steps: tuple[EpisodeStep, ...],
    ) -> AIRunResult:
        observation = build_observation(snapshot)
        return AIRunResult(
            success=status.success,
            stop_reason=stop_reason,
            steps_completed=steps_completed,
            elapsed_seconds=observation.elapsed_seconds,
            final_position=observation.position,
            final_velocity=observation.velocity,
            distance_to_target=status.distance_to_target,
            active_actions=observation.active_actions,
            alerts=observation.alerts,
            total_reward=total_reward,
            episode_steps=episode_steps,
        )


def _vector_from_payload(payload: Any) -> Vector3:
    if not isinstance(payload, dict):
        return (0.0, 0.0, 0.0)

    return tuple(float(payload.get(axis, 0.0)) for axis in ("x", "y", "z"))


def _operational_thrusters_from_modules(modules: Any) -> frozenset[str]:
    if not isinstance(modules, (list, tuple)):
        return frozenset()

    thruster_ids: set[str] = set()
    for module in modules:
        if not isinstance(module, dict):
            continue
        for action in module.get("actions", ()):
            if not isinstance(action, dict):
                continue
            if action.get("kind") != "thruster":
                continue
            if not bool(action.get("operational", False)):
                continue
            thruster_ids.add(str(action.get("id", "")))
    return frozenset(action_id for action_id in thruster_ids if action_id)


def _detect_blocker(
    snapshot: dict[str, Any],
    observation: Observation,
    requested_actions: set[str],
) -> str | None:
    if not requested_actions:
        return None

    active_actions = snapshot.get("active_actions", ())
    if any(str(action_id) in requested_actions for action_id in active_actions):
        return None

    if "Insufficient power" in observation.alerts:
        return "insufficient_power"
    if observation.resources.get("H2", 1.0) <= 0.0 or observation.resources.get("O2", 1.0) <= 0.0:
        return "resource_exhausted"
    if not observation.operational_thrusters:
        return "no_operational_thrusters"
    return None
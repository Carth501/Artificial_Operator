from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Callable

from simulation.engine import SimulationEngine

from .agents import ParameterizedTargetPositionAgent, TargetPolicyParameters
from .models import AIRunResult, TargetPositionObjective
from .rewards import RewardModel
from .runner import SimulationAIRunner


EngineFactory = Callable[[], SimulationEngine]


@dataclass(frozen=True)
class TrainingEvaluation:
    round_index: int
    parameters: TargetPolicyParameters
    total_reward: float
    success: bool
    stop_reason: str
    distance_to_target: float


@dataclass(frozen=True)
class TrainingResult:
    training_rounds: int
    best_parameters: TargetPolicyParameters
    best_total_reward: float
    best_run_result: AIRunResult
    evaluations: tuple[TrainingEvaluation, ...]


class TargetPositionPolicyTrainer:
    def __init__(
        self,
        engine_factory: EngineFactory,
        reward_model: RewardModel | None = None,
    ) -> None:
        self._engine_factory = engine_factory
        self._reward_model = reward_model

    def train(
        self,
        objective: TargetPositionObjective,
        *,
        rounds: int = 6,
        dt_seconds: float | None = None,
        initial_parameters: TargetPolicyParameters | None = None,
    ) -> TrainingResult:
        if rounds <= 0:
            raise ValueError("rounds must be greater than zero")

        best_parameters = TargetPolicyParameters() if initial_parameters is None else initial_parameters
        best_run_result = self._evaluate(best_parameters, objective, dt_seconds)
        evaluations = [
            TrainingEvaluation(
                round_index=0,
                parameters=best_parameters,
                total_reward=best_run_result.total_reward,
                success=best_run_result.success,
                stop_reason=best_run_result.stop_reason,
                distance_to_target=best_run_result.distance_to_target,
            )
        ]

        for round_index in range(1, rounds + 1):
            candidate_parameters = self._build_candidates(best_parameters)
            round_best_parameters = best_parameters
            round_best_run_result = best_run_result

            for parameters in candidate_parameters:
                run_result = self._evaluate(parameters, objective, dt_seconds)
                evaluations.append(
                    TrainingEvaluation(
                        round_index=round_index,
                        parameters=parameters,
                        total_reward=run_result.total_reward,
                        success=run_result.success,
                        stop_reason=run_result.stop_reason,
                        distance_to_target=run_result.distance_to_target,
                    )
                )

                if run_result.total_reward > round_best_run_result.total_reward:
                    round_best_parameters = parameters
                    round_best_run_result = run_result

            best_parameters = round_best_parameters
            best_run_result = round_best_run_result

        return TrainingResult(
            training_rounds=rounds,
            best_parameters=best_parameters,
            best_total_reward=best_run_result.total_reward,
            best_run_result=best_run_result,
            evaluations=tuple(evaluations),
        )

    def _evaluate(
        self,
        parameters: TargetPolicyParameters,
        objective: TargetPositionObjective,
        dt_seconds: float | None,
    ) -> AIRunResult:
        engine = self._engine_factory()
        agent = ParameterizedTargetPositionAgent(engine.list_thrusters(), parameters=parameters)
        runner = SimulationAIRunner(engine, agent, reward_model=self._reward_model)
        return runner.run(objective, dt_seconds=dt_seconds)

    def _build_candidates(self, parameters: TargetPolicyParameters) -> tuple[TargetPolicyParameters, ...]:
        candidates = {
            TargetPolicyParameters(),
            parameters,
            self._scaled(parameters, "brake_distance_multiplier", 0.75),
            self._scaled(parameters, "brake_distance_multiplier", 1.25),
            self._scaled(parameters, "approach_distance_multiplier", 0.75),
            self._scaled(parameters, "approach_distance_multiplier", 1.25),
            self._scaled(parameters, "approach_velocity_multiplier", 0.75),
            self._scaled(parameters, "approach_velocity_multiplier", 1.25),
            self._scaled(parameters, "settle_velocity_multiplier", 0.75),
            self._scaled(parameters, "settle_velocity_multiplier", 1.25),
        }
        return tuple(sorted(candidates, key=_parameter_sort_key))

    def _scaled(
        self,
        parameters: TargetPolicyParameters,
        field_name: str,
        multiplier: float,
    ) -> TargetPolicyParameters:
        raw_value = getattr(parameters, field_name) * multiplier
        return replace(parameters, **{field_name: _clamp_parameter(field_name, raw_value)})


def _clamp_parameter(field_name: str, value: float) -> float:
    minimum, maximum = PARAMETER_BOUNDS[field_name]
    return max(minimum, min(maximum, value))


def _parameter_sort_key(parameters: TargetPolicyParameters) -> tuple[float, float, float, float]:
    return (
        parameters.brake_distance_multiplier,
        parameters.approach_distance_multiplier,
        parameters.approach_velocity_multiplier,
        parameters.settle_velocity_multiplier,
    )


PARAMETER_BOUNDS = {
    "brake_distance_multiplier": (0.25, 3.0),
    "approach_distance_multiplier": (0.5, 5.0),
    "approach_velocity_multiplier": (0.25, 3.0),
    "settle_velocity_multiplier": (0.25, 2.0),
}
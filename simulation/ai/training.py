from __future__ import annotations

from dataclasses import dataclass, replace
import random
from typing import Callable

from simulation.engine import SimulationEngine

from .agents import ParameterizedTargetPositionAgent, TargetPolicyParameters
from .models import AIRunResult, TargetPositionObjective
from .persistence import TargetPolicyMetadata
from .rewards import RewardModel
from .runner import SimulationAIRunner


EngineFactory = Callable[[], SimulationEngine]


@dataclass(frozen=True)
class TargetPositionCurriculum:
    objectives: tuple[TargetPositionObjective, ...]
    seed: int
    target_ranges: tuple[float, float, float]


@dataclass(frozen=True)
class TrainingEvaluation:
    round_index: int
    parameters: TargetPolicyParameters
    total_reward: float
    average_reward: float
    success: bool
    success_rate: float
    stop_reason: str
    distance_to_target: float
    curriculum_size: int


@dataclass(frozen=True)
class TrainingResult:
    training_rounds: int
    curriculum_size: int
    best_parameters: TargetPolicyParameters
    best_total_reward: float
    best_average_reward: float
    best_run_result: AIRunResult
    best_run_results: tuple[AIRunResult, ...]
    evaluations: tuple[TrainingEvaluation, ...]


@dataclass(frozen=True)
class PolicyEvaluationResult:
    curriculum_size: int
    total_reward: float
    average_reward: float
    success_rate: float
    anchor_success: bool
    anchor_stop_reason: str
    anchor_distance_to_target: float
    anchor_run_result: AIRunResult
    run_results: tuple[AIRunResult, ...]


@dataclass(frozen=True)
class PolicyComparisonCandidate:
    label: str
    parameters: TargetPolicyParameters
    metadata: TargetPolicyMetadata | None = None


@dataclass(frozen=True)
class PolicyComparisonEntry:
    rank: int
    label: str
    average_reward: float
    total_reward: float
    success_rate: float
    anchor_stop_reason: str
    anchor_distance_to_target: float
    metadata: TargetPolicyMetadata | None


@dataclass(frozen=True)
class PolicyComparisonResult:
    curriculum_size: int
    entries: tuple[PolicyComparisonEntry, ...]


class TargetPositionPolicyEvaluator:
    def __init__(
        self,
        engine_factory: EngineFactory,
        reward_model: RewardModel | None = None,
    ) -> None:
        self._engine_factory = engine_factory
        self._reward_model = reward_model

    def evaluate(
        self,
        parameters: TargetPolicyParameters,
        objective: TargetPositionObjective,
        *,
        dt_seconds: float | None = None,
    ) -> PolicyEvaluationResult:
        return self.evaluate_curriculum(parameters, (objective,), dt_seconds=dt_seconds)

    def evaluate_curriculum(
        self,
        parameters: TargetPolicyParameters,
        objectives: tuple[TargetPositionObjective, ...],
        *,
        dt_seconds: float | None = None,
    ) -> PolicyEvaluationResult:
        if not objectives:
            raise ValueError("at least one objective is required")
        run_results = _evaluate_policy_curriculum(
            self._engine_factory,
            self._reward_model,
            parameters,
            objectives,
            dt_seconds,
        )
        return _summarize_policy_evaluation(run_results)


class TargetPositionPolicyComparer:
    def __init__(
        self,
        engine_factory: EngineFactory,
        reward_model: RewardModel | None = None,
    ) -> None:
        self._evaluator = TargetPositionPolicyEvaluator(engine_factory, reward_model=reward_model)

    def compare_curriculum(
        self,
        candidates: tuple[PolicyComparisonCandidate, ...],
        objectives: tuple[TargetPositionObjective, ...],
        *,
        dt_seconds: float | None = None,
    ) -> PolicyComparisonResult:
        if len(candidates) < 2:
            raise ValueError("at least two policy candidates are required")
        if not objectives:
            raise ValueError("at least one objective is required")

        scored_candidates: list[tuple[PolicyComparisonCandidate, PolicyEvaluationResult]] = []
        for candidate in candidates:
            evaluation_result = self._evaluator.evaluate_curriculum(
                candidate.parameters,
                objectives,
                dt_seconds=dt_seconds,
            )
            scored_candidates.append((candidate, evaluation_result))

        ranked_candidates = sorted(
            scored_candidates,
            key=_comparison_sort_key,
            reverse=True,
        )
        entries = tuple(
            PolicyComparisonEntry(
                rank=index,
                label=candidate.label,
                average_reward=evaluation_result.average_reward,
                total_reward=evaluation_result.total_reward,
                success_rate=evaluation_result.success_rate,
                anchor_stop_reason=evaluation_result.anchor_stop_reason,
                anchor_distance_to_target=evaluation_result.anchor_distance_to_target,
                metadata=candidate.metadata,
            )
            for index, (candidate, evaluation_result) in enumerate(ranked_candidates, start=1)
        )
        return PolicyComparisonResult(
            curriculum_size=len(objectives),
            entries=entries,
        )


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
        return self.train_curriculum(
            (objective,),
            rounds=rounds,
            dt_seconds=dt_seconds,
            initial_parameters=initial_parameters,
        )

    def train_curriculum(
        self,
        objectives: tuple[TargetPositionObjective, ...],
        *,
        rounds: int = 6,
        dt_seconds: float | None = None,
        initial_parameters: TargetPolicyParameters | None = None,
    ) -> TrainingResult:
        if rounds <= 0:
            raise ValueError("rounds must be greater than zero")
        if not objectives:
            raise ValueError("at least one objective is required")

        best_parameters = TargetPolicyParameters() if initial_parameters is None else initial_parameters
        best_run_results = self._evaluate_curriculum(best_parameters, objectives, dt_seconds)
        best_evaluation = _summarize_training_evaluation(0, best_parameters, best_run_results)
        evaluations = [
            best_evaluation
        ]

        for round_index in range(1, rounds + 1):
            candidate_parameters = self._build_candidates(best_parameters)
            round_best_parameters = best_parameters
            round_best_run_results = best_run_results
            round_best_evaluation = best_evaluation

            for parameters in candidate_parameters:
                run_results = self._evaluate_curriculum(parameters, objectives, dt_seconds)
                evaluation = _summarize_training_evaluation(round_index, parameters, run_results)
                evaluations.append(evaluation)

                if evaluation.average_reward > round_best_evaluation.average_reward:
                    round_best_parameters = parameters
                    round_best_run_results = run_results
                    round_best_evaluation = evaluation

            best_parameters = round_best_parameters
            best_run_results = round_best_run_results
            best_evaluation = round_best_evaluation

        return TrainingResult(
            training_rounds=rounds,
            curriculum_size=len(objectives),
            best_parameters=best_parameters,
            best_total_reward=best_evaluation.total_reward,
            best_average_reward=best_evaluation.average_reward,
            best_run_result=best_run_results[0],
            best_run_results=best_run_results,
            evaluations=tuple(evaluations),
        )

    def _evaluate(
        self,
        parameters: TargetPolicyParameters,
        objective: TargetPositionObjective,
        dt_seconds: float | None,
    ) -> AIRunResult:
        return _evaluate_policy_curriculum(
            self._engine_factory,
            self._reward_model,
            parameters,
            (objective,),
            dt_seconds,
        )[0]

    def _evaluate_curriculum(
        self,
        parameters: TargetPolicyParameters,
        objectives: tuple[TargetPositionObjective, ...],
        dt_seconds: float | None,
    ) -> tuple[AIRunResult, ...]:
        return _evaluate_policy_curriculum(
            self._engine_factory,
            self._reward_model,
            parameters,
            objectives,
            dt_seconds,
        )

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


def build_target_position_curriculum(
    anchor_objective: TargetPositionObjective,
    *,
    count: int,
    seed: int,
    target_range_x: float,
    target_range_y: float,
    target_range_z: float,
) -> TargetPositionCurriculum:
    if count <= 0:
        raise ValueError("count must be greater than zero")
    if target_range_x < 0.0 or target_range_y < 0.0 or target_range_z < 0.0:
        raise ValueError("target ranges must be zero or greater")

    objectives = [anchor_objective]
    if count == 1:
        return TargetPositionCurriculum(
            objectives=(anchor_objective,),
            seed=seed,
            target_ranges=(target_range_x, target_range_y, target_range_z),
        )

    generator = random.Random(seed)
    seen_positions = {anchor_objective.target_position}
    while len(objectives) < count:
        candidate_position = (
            generator.uniform(-target_range_x, target_range_x),
            generator.uniform(-target_range_y, target_range_y),
            generator.uniform(-target_range_z, target_range_z),
        )
        if candidate_position in seen_positions:
            continue
        seen_positions.add(candidate_position)
        objectives.append(replace(anchor_objective, target_position=candidate_position))

    return TargetPositionCurriculum(
        objectives=tuple(objectives),
        seed=seed,
        target_ranges=(target_range_x, target_range_y, target_range_z),
    )


def _summarize_training_evaluation(
    round_index: int,
    parameters: TargetPolicyParameters,
    run_results: tuple[AIRunResult, ...],
) -> TrainingEvaluation:
    evaluation_result = _summarize_policy_evaluation(run_results)
    return TrainingEvaluation(
        round_index=round_index,
        parameters=parameters,
        total_reward=evaluation_result.total_reward,
        average_reward=evaluation_result.average_reward,
        success=evaluation_result.success_rate == 1.0,
        success_rate=evaluation_result.success_rate,
        stop_reason=evaluation_result.anchor_stop_reason,
        distance_to_target=evaluation_result.anchor_distance_to_target,
        curriculum_size=evaluation_result.curriculum_size,
    )


def _evaluate_policy_curriculum(
    engine_factory: EngineFactory,
    reward_model: RewardModel | None,
    parameters: TargetPolicyParameters,
    objectives: tuple[TargetPositionObjective, ...],
    dt_seconds: float | None,
) -> tuple[AIRunResult, ...]:
    run_results: list[AIRunResult] = []
    for objective in objectives:
        engine = engine_factory()
        agent = ParameterizedTargetPositionAgent(engine.list_thrusters(), parameters=parameters)
        runner = SimulationAIRunner(engine, agent, reward_model=reward_model)
        run_results.append(runner.run(objective, dt_seconds=dt_seconds))
    return tuple(run_results)


def _summarize_policy_evaluation(run_results: tuple[AIRunResult, ...]) -> PolicyEvaluationResult:
    total_reward = sum(run_result.total_reward for run_result in run_results)
    success_count = sum(1 for run_result in run_results if run_result.success)
    anchor_run_result = run_results[0]
    curriculum_size = len(run_results)
    return PolicyEvaluationResult(
        curriculum_size=curriculum_size,
        total_reward=total_reward,
        average_reward=total_reward / curriculum_size,
        success_rate=success_count / curriculum_size,
        anchor_success=anchor_run_result.success,
        anchor_stop_reason=anchor_run_result.stop_reason,
        anchor_distance_to_target=anchor_run_result.distance_to_target,
        anchor_run_result=anchor_run_result,
        run_results=run_results,
    )


def _comparison_sort_key(
    scored_candidate: tuple[PolicyComparisonCandidate, PolicyEvaluationResult],
) -> tuple[float, float, float]:
    _, evaluation_result = scored_candidate
    return (
        evaluation_result.average_reward,
        evaluation_result.success_rate,
        -evaluation_result.anchor_distance_to_target,
    )
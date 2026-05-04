from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from simulation import SimulationEngine
from simulation.ai import (
    ParameterizedTargetPositionAgent,
    PolicyComparisonCandidate,
    PolicyComparisonResult,
    PolicyEvaluationResult,
    SimulationAIRunner,
    TargetPolicyMetadata,
    TargetPolicyParameters,
    TargetPositionAgent,
    TargetPositionCurriculum,
    TargetPositionPolicyComparer,
    TargetPositionPolicyEvaluator,
    TargetPositionObjective,
    TargetPositionPolicyTrainer,
    build_target_position_curriculum,
    load_target_policy_metadata,
    load_target_policy_parameters,
    save_policy_comparison_report,
    save_target_policy_parameters,
    update_target_policy_metadata,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Artificial Operator Sandbox")
    parser.add_argument(
        "--mode",
        choices=("manual", "ai", "train", "evaluate", "compare"),
        default="manual",
        help="Choose the manual app, the AI runner, the policy trainer, a held-out evaluator, or a policy comparer.",
    )
    parser.add_argument(
        "--config-dir",
        type=Path,
        default=None,
        help="Optional config directory containing variables.json, actions.json, modules.json, and systems.json.",
    )
    parser.add_argument("--target-x", type=float, default=None, help="AI target position on the X axis.")
    parser.add_argument("--target-y", type=float, default=None, help="AI target position on the Y axis.")
    parser.add_argument("--target-z", type=float, default=None, help="AI target position on the Z axis.")
    parser.add_argument("--tolerance", type=float, default=0.5, help="Per-axis arrival tolerance in meters.")
    parser.add_argument(
        "--settle-velocity",
        type=float,
        default=0.25,
        help="Per-axis velocity threshold required to count the target as settled.",
    )
    parser.add_argument("--max-steps", type=int, default=600, help="Maximum AI simulation steps before timing out.")
    parser.add_argument("--dt", type=float, default=None, help="Optional AI step size override in seconds.")
    parser.add_argument(
        "--progress-every",
        type=int,
        default=20,
        help="Emit AI progress every N simulation steps.",
    )
    parser.add_argument(
        "--training-rounds",
        type=int,
        default=6,
        help="Number of policy-search rounds to run in training mode.",
    )
    parser.add_argument(
        "--policy-file",
        type=Path,
        default=None,
        help="Optional policy JSON file to load for AI mode or to use as the initial policy in training mode.",
    )
    parser.add_argument(
        "--compare-policy",
        type=Path,
        action="append",
        default=None,
        help="Path to a saved policy JSON to compare in compare mode. Repeat this flag for multiple policies.",
    )
    parser.add_argument(
        "--save-policy",
        type=Path,
        default=None,
        help="Optional destination path for saving the best trained policy in training mode.",
    )
    parser.add_argument(
        "--save-evaluation-metadata",
        action="store_true",
        help="When evaluating a saved policy, merge the held-out evaluation summary back into that same policy file.",
    )
    parser.add_argument(
        "--save-comparison-report",
        type=Path,
        default=None,
        help="Optional destination path for saving a JSON report from compare mode.",
    )
    parser.add_argument(
        "--curriculum-targets",
        type=int,
        default=1,
        help="Number of target positions to train across in training mode.",
    )
    parser.add_argument(
        "--curriculum-seed",
        type=int,
        default=17,
        help="Random seed used to generate additional curriculum targets in training mode.",
    )
    parser.add_argument(
        "--target-range-x",
        type=float,
        default=None,
        help="Optional absolute X range for generated curriculum targets.",
    )
    parser.add_argument(
        "--target-range-y",
        type=float,
        default=None,
        help="Optional absolute Y range for generated curriculum targets.",
    )
    parser.add_argument(
        "--target-range-z",
        type=float,
        default=None,
        help="Optional absolute Z range for generated curriculum targets.",
    )
    return parser


def run_manual_mode(config_directory: Path) -> int:
    from ui import SimulationApp

    engine = SimulationEngine.from_config_directory(config_directory)
    app = SimulationApp(engine)
    app.run()
    return 0


def run_ai_mode(config_directory: Path, args: argparse.Namespace) -> int:
    engine = SimulationEngine.from_config_directory(config_directory)
    policy_suffix = "default_policy"
    if args.policy_file is None:
        agent = TargetPositionAgent(engine.list_thrusters())
    else:
        parameters = load_target_policy_parameters(args.policy_file)
        metadata = load_target_policy_metadata(args.policy_file)
        agent = ParameterizedTargetPositionAgent(engine.list_thrusters(), parameters=parameters)
        policy_suffix = f"policy_file={args.policy_file}"
        if metadata is not None:
            policy_suffix = f"{policy_suffix} | {_format_policy_metadata(metadata)}"
    runner = SimulationAIRunner(engine, agent)
    objective = TargetPositionObjective(
        target_position=(args.target_x, args.target_y, args.target_z),
        tolerance=args.tolerance,
        settle_velocity=args.settle_velocity,
        max_steps=args.max_steps,
        progress_interval=args.progress_every,
    )

    step_seconds = engine.tick_seconds if args.dt is None else args.dt
    print(
        "AI mode objective | "
        f"target={_format_vector(objective.target_position)} | "
        f"tolerance={objective.tolerance:.2f} | "
        f"settle_velocity={objective.settle_velocity:.2f} | "
        f"max_steps={objective.max_steps} | "
        f"dt={step_seconds:.2f} | "
        f"{policy_suffix}"
    )

    result = runner.run(
        objective,
        dt_seconds=args.dt,
        progress_callback=_print_ai_progress,
    )

    print(
        "AI complete | "
        f"success={result.success} | "
        f"reason={result.stop_reason} | "
        f"steps={result.steps_completed} | "
        f"elapsed={result.elapsed_seconds:.2f}s | "
        f"reward={result.total_reward:.3f}"
    )
    print(
        "Final state | "
        f"position={_format_vector(result.final_position)} | "
        f"velocity={_format_vector(result.final_velocity)} | "
        f"distance={result.distance_to_target:.3f} | "
        f"episode_steps={len(result.episode_steps)}"
    )

    return 0 if result.success else 1


def run_training_mode(config_directory: Path, args: argparse.Namespace) -> int:
    initial_parameters = None if args.policy_file is None else load_target_policy_parameters(args.policy_file)
    objective = TargetPositionObjective(
        target_position=(args.target_x, args.target_y, args.target_z),
        tolerance=args.tolerance,
        settle_velocity=args.settle_velocity,
        max_steps=args.max_steps,
        progress_interval=args.progress_every,
    )
    curriculum = _build_objective_curriculum(objective, args)
    trainer = TargetPositionPolicyTrainer(
        lambda: SimulationEngine.from_config_directory(config_directory),
    )
    step_seconds = args.dt if args.dt is not None else SimulationEngine.from_config_directory(config_directory).tick_seconds
    print(
        "Training mode objective | "
        f"target={_format_vector(objective.target_position)} | "
        f"rounds={args.training_rounds} | "
        f"curriculum_targets={len(curriculum.objectives)} | "
        f"curriculum_seed={curriculum.seed} | "
        f"tolerance={objective.tolerance:.2f} | "
        f"settle_velocity={objective.settle_velocity:.2f} | "
        f"max_steps={objective.max_steps} | "
        f"dt={step_seconds:.2f}"
    )

    training_result = trainer.train_curriculum(
        curriculum.objectives,
        rounds=args.training_rounds,
        dt_seconds=args.dt,
        initial_parameters=initial_parameters,
    )
    best_run = training_result.best_run_result
    curriculum_success_rate = _success_rate(training_result.best_run_results)

    saved_policy_suffix = ""
    if args.save_policy is not None:
        save_target_policy_parameters(
            args.save_policy,
            training_result.best_parameters,
            metadata=_build_policy_metadata(objective, curriculum, training_result),
        )
        saved_policy_suffix = f" | saved_policy={args.save_policy}"

    print(
        "Training complete | "
        f"rounds={training_result.training_rounds} | "
        f"evaluations={len(training_result.evaluations)} | "
        f"best_total_reward={training_result.best_total_reward:.3f} | "
        f"best_avg_reward={training_result.best_average_reward:.3f} | "
        f"success_rate={curriculum_success_rate:.2f} | "
        f"anchor_success={best_run.success} | "
        f"reason={best_run.stop_reason}"
        f"{saved_policy_suffix}"
    )
    print(
        "Best policy | "
        f"brake_distance_multiplier={training_result.best_parameters.brake_distance_multiplier:.3f} | "
        f"approach_distance_multiplier={training_result.best_parameters.approach_distance_multiplier:.3f} | "
        f"approach_velocity_multiplier={training_result.best_parameters.approach_velocity_multiplier:.3f} | "
        f"settle_velocity_multiplier={training_result.best_parameters.settle_velocity_multiplier:.3f}"
    )
    print(
        "Best final state | "
        f"position={_format_vector(best_run.final_position)} | "
        f"velocity={_format_vector(best_run.final_velocity)} | "
        f"distance={best_run.distance_to_target:.3f} | "
        f"episode_steps={len(best_run.episode_steps)}"
    )

    return 0 if best_run.success else 1


def run_evaluation_mode(config_directory: Path, args: argparse.Namespace) -> int:
    objective = TargetPositionObjective(
        target_position=(args.target_x, args.target_y, args.target_z),
        tolerance=args.tolerance,
        settle_velocity=args.settle_velocity,
        max_steps=args.max_steps,
        progress_interval=args.progress_every,
    )
    curriculum = _build_objective_curriculum(objective, args)
    evaluator = TargetPositionPolicyEvaluator(
        lambda: SimulationEngine.from_config_directory(config_directory),
    )
    step_seconds = args.dt if args.dt is not None else SimulationEngine.from_config_directory(config_directory).tick_seconds
    parameters, policy_suffix = _load_policy_for_evaluation(args.policy_file)
    print(
        "Evaluation mode objective | "
        f"target={_format_vector(objective.target_position)} | "
        f"evaluation_targets={len(curriculum.objectives)} | "
        f"evaluation_seed={curriculum.seed} | "
        f"tolerance={objective.tolerance:.2f} | "
        f"settle_velocity={objective.settle_velocity:.2f} | "
        f"max_steps={objective.max_steps} | "
        f"dt={step_seconds:.2f} | "
        f"{policy_suffix}"
    )

    evaluation_result = evaluator.evaluate_curriculum(
        parameters,
        curriculum.objectives,
        dt_seconds=args.dt,
    )

    saved_evaluation_suffix = ""
    if args.save_evaluation_metadata:
        update_target_policy_metadata(
            args.policy_file,
            _build_evaluation_metadata(curriculum, evaluation_result),
        )
        saved_evaluation_suffix = f" | updated_policy={args.policy_file}"

    print(
        "Evaluation complete | "
        f"evaluated_targets={evaluation_result.curriculum_size} | "
        f"eval_total_reward={evaluation_result.total_reward:.3f} | "
        f"eval_avg_reward={evaluation_result.average_reward:.3f} | "
        f"eval_success_rate={evaluation_result.success_rate:.2f} | "
        f"anchor_success={evaluation_result.anchor_success} | "
        f"anchor_reason={evaluation_result.anchor_stop_reason}"
        f"{saved_evaluation_suffix}"
    )
    _print_anchor_final_state(evaluation_result)
    return 0


def run_compare_mode(config_directory: Path, args: argparse.Namespace) -> int:
    objective = TargetPositionObjective(
        target_position=(args.target_x, args.target_y, args.target_z),
        tolerance=args.tolerance,
        settle_velocity=args.settle_velocity,
        max_steps=args.max_steps,
        progress_interval=args.progress_every,
    )
    curriculum = _build_objective_curriculum(objective, args)
    comparer = TargetPositionPolicyComparer(
        lambda: SimulationEngine.from_config_directory(config_directory),
    )
    step_seconds = args.dt if args.dt is not None else SimulationEngine.from_config_directory(config_directory).tick_seconds
    candidates = _load_policy_comparison_candidates(args.compare_policy)
    print(
        "Compare mode objective | "
        f"target={_format_vector(objective.target_position)} | "
        f"evaluation_targets={len(curriculum.objectives)} | "
        f"evaluation_seed={curriculum.seed} | "
        f"compare_policies={len(candidates)} | "
        f"dt={step_seconds:.2f}"
    )

    comparison_result = comparer.compare_curriculum(
        candidates,
        curriculum.objectives,
        dt_seconds=args.dt,
    )

    saved_report_suffix = ""
    if args.save_comparison_report is not None:
        save_policy_comparison_report(
            args.save_comparison_report,
            objective,
            curriculum,
            comparison_result,
            dt_seconds=step_seconds,
        )
        saved_report_suffix = f" | saved_report={args.save_comparison_report}"

    print(
        "Comparison complete | "
        f"compared_policies={len(comparison_result.entries)} | "
        f"evaluation_targets={comparison_result.curriculum_size}"
        f"{saved_report_suffix}"
    )
    _print_comparison_entries(comparison_result)
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    project_root = Path(__file__).resolve().parent
    config_directory = project_root / "config" if args.config_dir is None else args.config_dir

    if args.mode == "manual":
        return run_manual_mode(config_directory)

    _validate_ai_args(parser, args)
    if args.mode == "ai":
        return run_ai_mode(config_directory, args)
    if args.mode == "evaluate":
        return run_evaluation_mode(config_directory, args)
    if args.mode == "compare":
        return run_compare_mode(config_directory, args)
    return run_training_mode(config_directory, args)


def _validate_ai_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    missing_targets = [
        flag_name
        for flag_name in ("target_x", "target_y", "target_z")
        if getattr(args, flag_name) is None
    ]
    if missing_targets:
        parser.error("AI mode requires --target-x, --target-y, and --target-z.")
    if args.tolerance < 0.0:
        parser.error("--tolerance must be zero or greater.")
    if args.settle_velocity < 0.0:
        parser.error("--settle-velocity must be zero or greater.")
    if args.max_steps <= 0:
        parser.error("--max-steps must be greater than zero.")
    if args.progress_every <= 0:
        parser.error("--progress-every must be greater than zero.")
    if args.training_rounds <= 0:
        parser.error("--training-rounds must be greater than zero.")
    if args.curriculum_targets <= 0:
        parser.error("--curriculum-targets must be greater than zero.")
    if args.mode == "evaluate" and args.save_evaluation_metadata and args.policy_file is None:
        parser.error("--save-evaluation-metadata requires --policy-file in evaluation mode.")
    if args.mode == "compare" and (args.compare_policy is None or len(args.compare_policy) < 2):
        parser.error("compare mode requires at least two --compare-policy values.")
    for field_name in ("target_range_x", "target_range_y", "target_range_z"):
        field_value = getattr(args, field_name)
        if field_value is not None and field_value < 0.0:
            parser.error(f"--{field_name.replace('_', '-')} must be zero or greater when provided.")
    if args.dt is not None and args.dt <= 0.0:
        parser.error("--dt must be greater than zero when provided.")


def _print_ai_progress(step_index: int, observation: object, status: object) -> None:
    position = getattr(observation, "position")
    velocity = getattr(observation, "velocity")
    elapsed_seconds = float(getattr(observation, "elapsed_seconds"))
    distance_to_target = float(getattr(status, "distance_to_target"))
    print(
        "AI progress | "
        f"step={step_index} | "
        f"t={elapsed_seconds:.2f}s | "
        f"position={_format_vector(position)} | "
        f"velocity={_format_vector(velocity)} | "
        f"distance={distance_to_target:.3f}"
    )


def _format_vector(vector: Sequence[float]) -> str:
    return f"({vector[0]:.2f}, {vector[1]:.2f}, {vector[2]:.2f})"


def _build_objective_curriculum(
    objective: TargetPositionObjective,
    args: argparse.Namespace,
) -> TargetPositionCurriculum:
    return build_target_position_curriculum(
        objective,
        count=args.curriculum_targets,
        seed=args.curriculum_seed,
        target_range_x=_resolve_target_range(args.target_range_x, args.target_x),
        target_range_y=_resolve_target_range(args.target_range_y, args.target_y),
        target_range_z=_resolve_target_range(args.target_range_z, args.target_z),
    )


def _resolve_target_range(explicit_range: float | None, anchor_value: float) -> float:
    if explicit_range is not None:
        return explicit_range
    return max(abs(anchor_value), 5.0)


def _success_rate(run_results: Sequence[object]) -> float:
    if not run_results:
        return 0.0
    success_count = sum(1 for run_result in run_results if bool(getattr(run_result, "success", False)))
    return success_count / len(run_results)


def _build_policy_metadata(
    objective: TargetPositionObjective,
    curriculum: TargetPositionCurriculum,
    training_result: object,
) -> TargetPolicyMetadata:
    return TargetPolicyMetadata(
        source="training",
        training_rounds=int(getattr(training_result, "training_rounds")),
        curriculum_size=int(getattr(training_result, "curriculum_size")),
        curriculum_seed=curriculum.seed,
        anchor_target_position=objective.target_position,
        target_ranges=curriculum.target_ranges,
        best_total_reward=float(getattr(training_result, "best_total_reward")),
        best_average_reward=float(getattr(training_result, "best_average_reward")),
        success_rate=_success_rate(getattr(training_result, "best_run_results")),
        anchor_stop_reason=str(getattr(getattr(training_result, "best_run_result"), "stop_reason")),
    )


def _format_policy_metadata(metadata: TargetPolicyMetadata) -> str:
    parts: list[str] = []
    if metadata.curriculum_size is not None:
        parts.append(f"curriculum_size={metadata.curriculum_size}")
    if metadata.curriculum_seed is not None:
        parts.append(f"curriculum_seed={metadata.curriculum_seed}")
    if metadata.best_average_reward is not None:
        parts.append(f"policy_avg_reward={metadata.best_average_reward:.3f}")
    if metadata.evaluation_curriculum_seed is not None:
        parts.append(f"eval_seed={metadata.evaluation_curriculum_seed}")
    if metadata.evaluation_average_reward is not None:
        parts.append(f"eval_avg_reward={metadata.evaluation_average_reward:.3f}")
    return " | ".join(parts)


def _load_policy_for_evaluation(
    policy_file: Path | None,
) -> tuple[TargetPolicyParameters, str]:
    if policy_file is None:
        return TargetPolicyParameters(), "default_policy"

    parameters = load_target_policy_parameters(policy_file)
    metadata = load_target_policy_metadata(policy_file)
    suffix = f"policy_file={policy_file}"
    if metadata is not None:
        suffix = f"{suffix} | {_format_policy_metadata(metadata)}"
    return parameters, suffix


def _print_anchor_final_state(evaluation_result: PolicyEvaluationResult) -> None:
    anchor_run_result = evaluation_result.anchor_run_result
    print(
        "Anchor final state | "
        f"position={_format_vector(anchor_run_result.final_position)} | "
        f"velocity={_format_vector(anchor_run_result.final_velocity)} | "
        f"distance={anchor_run_result.distance_to_target:.3f} | "
        f"episode_steps={len(anchor_run_result.episode_steps)}"
    )


def _build_evaluation_metadata(
    curriculum: TargetPositionCurriculum,
    evaluation_result: PolicyEvaluationResult,
) -> TargetPolicyMetadata:
    return TargetPolicyMetadata(
        evaluation_curriculum_size=evaluation_result.curriculum_size,
        evaluation_curriculum_seed=curriculum.seed,
        evaluation_target_ranges=curriculum.target_ranges,
        evaluation_total_reward=evaluation_result.total_reward,
        evaluation_average_reward=evaluation_result.average_reward,
        evaluation_success_rate=evaluation_result.success_rate,
        evaluation_anchor_stop_reason=evaluation_result.anchor_stop_reason,
    )


def _load_policy_comparison_candidates(
    policy_files: list[Path] | None,
) -> tuple[PolicyComparisonCandidate, ...]:
    if not policy_files:
        return ()

    candidates: list[PolicyComparisonCandidate] = []
    for policy_file in policy_files:
        candidates.append(
            PolicyComparisonCandidate(
                label=policy_file.stem,
                parameters=load_target_policy_parameters(policy_file),
                metadata=load_target_policy_metadata(policy_file),
            )
        )
    return tuple(candidates)


def _print_comparison_entries(comparison_result: PolicyComparisonResult) -> None:
    for entry in comparison_result.entries:
        metadata_suffix = ""
        if entry.metadata is not None:
            formatted_metadata = _format_policy_metadata(entry.metadata)
            if formatted_metadata:
                metadata_suffix = f" | {formatted_metadata}"
        print(
            "Comparison rank | "
            f"rank={entry.rank} | "
            f"label={entry.label} | "
            f"avg_reward={entry.average_reward:.3f} | "
            f"total_reward={entry.total_reward:.3f} | "
            f"success_rate={entry.success_rate:.2f} | "
            f"anchor_reason={entry.anchor_stop_reason} | "
            f"anchor_distance={entry.anchor_distance_to_target:.3f}"
            f"{metadata_suffix}"
        )


if __name__ == "__main__":
    raise SystemExit(main())
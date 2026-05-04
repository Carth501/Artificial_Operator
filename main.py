from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from simulation import SimulationEngine
from simulation.ai import SimulationAIRunner, TargetPositionAgent, TargetPositionObjective


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Artificial Operator Sandbox")
    parser.add_argument(
        "--mode",
        choices=("manual", "ai"),
        default="manual",
        help="Choose the manual Tkinter app or the headless AI runner.",
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
    return parser


def run_manual_mode(config_directory: Path) -> int:
    from ui import SimulationApp

    engine = SimulationEngine.from_config_directory(config_directory)
    app = SimulationApp(engine)
    app.run()
    return 0


def run_ai_mode(config_directory: Path, args: argparse.Namespace) -> int:
    engine = SimulationEngine.from_config_directory(config_directory)
    agent = TargetPositionAgent(engine.list_thrusters())
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
        f"dt={step_seconds:.2f}"
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
        f"elapsed={result.elapsed_seconds:.2f}s"
    )
    print(
        "Final state | "
        f"position={_format_vector(result.final_position)} | "
        f"velocity={_format_vector(result.final_velocity)} | "
        f"distance={result.distance_to_target:.3f}"
    )

    return 0 if result.success else 1


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    project_root = Path(__file__).resolve().parent
    config_directory = project_root / "config" if args.config_dir is None else args.config_dir

    if args.mode == "manual":
        return run_manual_mode(config_directory)

    _validate_ai_args(parser, args)
    return run_ai_mode(config_directory, args)


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


if __name__ == "__main__":
    raise SystemExit(main())
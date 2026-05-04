from __future__ import annotations

from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from main import main
from simulation.ai import (
    SimulationAIRunner,
    TargetPolicyMetadata,
    TargetPolicyParameters,
    TargetPositionCurriculum,
    TargetPositionAgent,
    TargetPositionObjective,
    TargetPositionPolicyTrainer,
    build_target_position_curriculum,
    load_target_policy_metadata,
    load_target_policy_parameters,
    save_target_policy_parameters,
)
from simulation.engine import SimulationEngine


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_ROOT = PROJECT_ROOT / "config"


class SimulationAIRunnerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.engine = SimulationEngine.from_config_directory(CONFIG_ROOT)
        self.agent = TargetPositionAgent(self.engine.list_thrusters())
        self.runner = SimulationAIRunner(self.engine, self.agent)

    def test_runner_reaches_positive_x_target(self) -> None:
        objective = TargetPositionObjective(
            target_position=(5.0, 0.0, 0.0),
            tolerance=0.6,
            settle_velocity=0.6,
            max_steps=80,
        )

        result = self.runner.run(objective, dt_seconds=0.5)

        self.assertTrue(result.success)
        self.assertEqual(result.stop_reason, "target_reached")
        self.assertAlmostEqual(result.final_position[0], 5.0, delta=objective.tolerance)
        self.assertLessEqual(abs(result.final_velocity[0]), objective.settle_velocity)
        self.assertEqual(len(result.episode_steps), result.steps_completed)
        self.assertGreater(result.total_reward, 0.0)
        self.assertEqual(result.episode_steps[-1].stop_reason, "target_reached")
        self.assertGreater(result.episode_steps[-1].reward_breakdown.terminal_reward, 0.0)

    def test_runner_reaches_mixed_axis_target(self) -> None:
        objective = TargetPositionObjective(
            target_position=(3.0, -2.0, 1.5),
            tolerance=0.6,
            settle_velocity=0.6,
            max_steps=120,
        )

        result = self.runner.run(objective, dt_seconds=0.5)

        self.assertTrue(result.success)
        self.assertEqual(result.stop_reason, "target_reached")
        self.assertAlmostEqual(result.final_position[0], 3.0, delta=objective.tolerance)
        self.assertAlmostEqual(result.final_position[1], -2.0, delta=objective.tolerance)
        self.assertAlmostEqual(result.final_position[2], 1.5, delta=objective.tolerance)
        self.assertLessEqual(abs(result.final_velocity[0]), objective.settle_velocity)
        self.assertLessEqual(abs(result.final_velocity[1]), objective.settle_velocity)
        self.assertLessEqual(abs(result.final_velocity[2]), objective.settle_velocity)

    def test_runner_reports_step_limit_reached(self) -> None:
        objective = TargetPositionObjective(
            target_position=(100.0, 0.0, 0.0),
            tolerance=0.5,
            settle_velocity=0.25,
            max_steps=1,
        )

        result = self.runner.run(objective, dt_seconds=0.5)

        self.assertFalse(result.success)
        self.assertEqual(result.stop_reason, "step_limit_reached")
        self.assertEqual(len(result.episode_steps), result.steps_completed)
        self.assertLess(result.total_reward, 0.0)
        self.assertEqual(result.episode_steps[-1].stop_reason, "step_limit_reached")

    def test_runner_reports_no_operational_thrusters(self) -> None:
        self.engine.set_module_integrity("propulsion", 0.0)
        objective = TargetPositionObjective(target_position=(5.0, 0.0, 0.0), max_steps=20)

        result = self.runner.run(objective, dt_seconds=0.5, reset_engine=False)

        self.assertFalse(result.success)
        self.assertEqual(result.stop_reason, "no_operational_thrusters")
        self.assertEqual(result.steps_completed, 0)
        self.assertEqual(result.episode_steps, ())
        self.assertEqual(result.total_reward, 0.0)


class AIModeCliTests(unittest.TestCase):
    def test_main_ai_mode_runs_headless(self) -> None:
        output = StringIO()

        with redirect_stdout(output):
            exit_code = main(
                [
                    "--mode",
                    "ai",
                    "--target-x",
                    "5",
                    "--target-y",
                    "0",
                    "--target-z",
                    "0",
                    "--dt",
                    "0.5",
                    "--max-steps",
                    "80",
                    "--progress-every",
                    "100",
                ]
            )

        stdout = output.getvalue()
        self.assertEqual(exit_code, 0)
        self.assertIn("AI mode objective", stdout)
        self.assertIn("reason=target_reached", stdout)
        self.assertIn("reward=", stdout)
        self.assertIn("Final state", stdout)

    def test_main_ai_mode_loads_saved_policy_file(self) -> None:
        with TemporaryDirectory() as temp_dir:
            policy_path = Path(temp_dir) / "saved-policy.json"
            save_target_policy_parameters(
                policy_path,
                TargetPolicyParameters(),
                metadata=TargetPolicyMetadata(
                    source="training",
                    training_rounds=2,
                    curriculum_size=3,
                    curriculum_seed=17,
                    best_average_reward=62.651,
                ),
            )
            output = StringIO()

            with redirect_stdout(output):
                exit_code = main(
                    [
                        "--mode",
                        "ai",
                        "--target-x",
                        "5",
                        "--target-y",
                        "0",
                        "--target-z",
                        "0",
                        "--dt",
                        "0.5",
                        "--max-steps",
                        "80",
                        "--progress-every",
                        "100",
                        "--policy-file",
                        str(policy_path),
                    ]
                )

        stdout = output.getvalue()
        self.assertEqual(exit_code, 0)
        self.assertIn("policy_file=", stdout)
        self.assertIn("curriculum_size=3", stdout)
        self.assertIn("policy_avg_reward=62.651", stdout)
        self.assertIn("reason=target_reached", stdout)


class TrainingModeTests(unittest.TestCase):
    def test_build_target_position_curriculum_is_deterministic(self) -> None:
        anchor = TargetPositionObjective(target_position=(5.0, 0.0, 0.0), tolerance=0.6, settle_velocity=0.6)

        first = build_target_position_curriculum(
            anchor,
            count=3,
            seed=17,
            target_range_x=6.0,
            target_range_y=4.0,
            target_range_z=2.0,
        )
        second = build_target_position_curriculum(
            anchor,
            count=3,
            seed=17,
            target_range_x=6.0,
            target_range_y=4.0,
            target_range_z=2.0,
        )

        self.assertIsInstance(first, TargetPositionCurriculum)
        self.assertEqual(first.objectives, second.objectives)
        self.assertEqual(first.objectives[0], anchor)
        self.assertEqual(len(first.objectives), 3)

    def test_policy_persistence_round_trip(self) -> None:
        parameters = TargetPolicyParameters(
            brake_distance_multiplier=1.25,
            approach_distance_multiplier=3.0,
            approach_velocity_multiplier=0.75,
            settle_velocity_multiplier=1.5,
        )
        metadata = TargetPolicyMetadata(
            source="training",
            training_rounds=3,
            curriculum_size=4,
            curriculum_seed=17,
            anchor_target_position=(5.0, 0.0, 0.0),
            target_ranges=(6.0, 4.0, 2.0),
            best_total_reward=187.953,
            best_average_reward=62.651,
            success_rate=1.0,
            anchor_stop_reason="target_reached",
        )

        with TemporaryDirectory() as temp_dir:
            policy_path = Path(temp_dir) / "policy.json"
            save_target_policy_parameters(policy_path, parameters, metadata=metadata)
            loaded = load_target_policy_parameters(policy_path)
            loaded_metadata = load_target_policy_metadata(policy_path)

        self.assertEqual(loaded, parameters)
        self.assertEqual(loaded_metadata, metadata)

    def test_trainer_improves_reward_from_poor_initial_policy(self) -> None:
        trainer = TargetPositionPolicyTrainer(lambda: SimulationEngine.from_config_directory(CONFIG_ROOT))
        objective = TargetPositionObjective(
            target_position=(5.0, 0.0, 0.0),
            tolerance=0.6,
            settle_velocity=0.6,
            max_steps=80,
        )
        initial_parameters = TargetPolicyParameters(
            brake_distance_multiplier=0.25,
            approach_distance_multiplier=0.5,
            approach_velocity_multiplier=2.5,
            settle_velocity_multiplier=2.0,
        )

        result = trainer.train(
            objective,
            rounds=3,
            dt_seconds=0.5,
            initial_parameters=initial_parameters,
        )

        self.assertGreater(result.best_total_reward, result.evaluations[0].total_reward)
        self.assertTrue(result.best_run_result.success)
        self.assertEqual(result.best_run_result.stop_reason, "target_reached")

    def test_trainer_curriculum_tracks_multiple_objectives(self) -> None:
        trainer = TargetPositionPolicyTrainer(lambda: SimulationEngine.from_config_directory(CONFIG_ROOT))
        anchor = TargetPositionObjective(
            target_position=(5.0, 0.0, 0.0),
            tolerance=0.6,
            settle_velocity=0.6,
            max_steps=80,
        )
        curriculum = build_target_position_curriculum(
            anchor,
            count=3,
            seed=17,
            target_range_x=6.0,
            target_range_y=4.0,
            target_range_z=2.0,
        )

        result = trainer.train_curriculum(curriculum.objectives, rounds=2, dt_seconds=0.5)

        self.assertEqual(result.curriculum_size, 3)
        self.assertEqual(len(result.best_run_results), 3)
        self.assertAlmostEqual(result.best_average_reward, result.best_total_reward / 3)
        self.assertEqual(result.evaluations[0].curriculum_size, 3)
        self.assertGreaterEqual(result.evaluations[0].success_rate, 0.0)

    def test_main_training_mode_runs_headless(self) -> None:
        output = StringIO()

        with redirect_stdout(output):
            exit_code = main(
                [
                    "--mode",
                    "train",
                    "--target-x",
                    "5",
                    "--target-y",
                    "0",
                    "--target-z",
                    "0",
                    "--dt",
                    "0.5",
                    "--max-steps",
                    "80",
                    "--training-rounds",
                    "3",
                ]
            )

        stdout = output.getvalue()
        self.assertEqual(exit_code, 0)
        self.assertIn("Training mode objective", stdout)
        self.assertIn("Training complete", stdout)
        self.assertIn("Best policy", stdout)

    def test_main_training_mode_runs_curriculum(self) -> None:
        output = StringIO()

        with redirect_stdout(output):
            exit_code = main(
                [
                    "--mode",
                    "train",
                    "--target-x",
                    "5",
                    "--target-y",
                    "0",
                    "--target-z",
                    "0",
                    "--dt",
                    "0.5",
                    "--max-steps",
                    "80",
                    "--training-rounds",
                    "2",
                    "--curriculum-targets",
                    "3",
                    "--curriculum-seed",
                    "17",
                    "--target-range-x",
                    "6",
                    "--target-range-y",
                    "4",
                    "--target-range-z",
                    "2",
                ]
            )

        stdout = output.getvalue()
        self.assertEqual(exit_code, 0)
        self.assertIn("curriculum_targets=3", stdout)
        self.assertIn("success_rate=", stdout)
        self.assertIn("best_avg_reward=", stdout)

    def test_main_training_mode_saves_policy_file(self) -> None:
        with TemporaryDirectory() as temp_dir:
            policy_path = Path(temp_dir) / "trained-policy.json"
            output = StringIO()

            with redirect_stdout(output):
                exit_code = main(
                    [
                        "--mode",
                        "train",
                        "--target-x",
                        "5",
                        "--target-y",
                        "0",
                        "--target-z",
                        "0",
                        "--dt",
                        "0.5",
                        "--max-steps",
                        "80",
                        "--training-rounds",
                        "2",
                        "--curriculum-targets",
                        "3",
                        "--curriculum-seed",
                        "17",
                        "--target-range-x",
                        "6",
                        "--target-range-y",
                        "4",
                        "--target-range-z",
                        "2",
                        "--save-policy",
                        str(policy_path),
                    ]
                )

            self.assertTrue(policy_path.exists())
            saved_parameters = load_target_policy_parameters(policy_path)
            saved_metadata = load_target_policy_metadata(policy_path)

        stdout = output.getvalue()
        self.assertEqual(exit_code, 0)
        self.assertEqual(
            saved_parameters,
            TargetPolicyParameters(
                brake_distance_multiplier=1.0,
                approach_distance_multiplier=3.125,
                approach_velocity_multiplier=1.0,
                settle_velocity_multiplier=1.0,
            ),
        )
        self.assertIsNotNone(saved_metadata)
        self.assertEqual(saved_metadata.curriculum_size, 3)
        self.assertEqual(saved_metadata.curriculum_seed, 17)
        self.assertEqual(saved_metadata.anchor_target_position, (5.0, 0.0, 0.0))
        self.assertEqual(saved_metadata.target_ranges, (6.0, 4.0, 2.0))
        self.assertEqual(saved_metadata.training_rounds, 2)
        self.assertIn("saved_policy=", stdout)


if __name__ == "__main__":
    unittest.main()
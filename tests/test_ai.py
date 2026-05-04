from __future__ import annotations

from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
import unittest

from main import main
from simulation.ai import SimulationAIRunner, TargetPositionAgent, TargetPositionObjective
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

    def test_runner_reports_no_operational_thrusters(self) -> None:
        self.engine.set_module_integrity("propulsion", 0.0)
        objective = TargetPositionObjective(target_position=(5.0, 0.0, 0.0), max_steps=20)

        result = self.runner.run(objective, dt_seconds=0.5, reset_engine=False)

        self.assertFalse(result.success)
        self.assertEqual(result.stop_reason, "no_operational_thrusters")
        self.assertEqual(result.steps_completed, 0)


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
        self.assertIn("Final state", stdout)


if __name__ == "__main__":
    unittest.main()
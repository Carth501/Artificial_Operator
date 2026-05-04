from __future__ import annotations

from pathlib import Path
import unittest

from simulation.engine import SimulationEngine
from simulation.learning import TargetNavigationLearner


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_ROOT = PROJECT_ROOT / "config"


class TargetNavigationLearnerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.engine = SimulationEngine.from_config_directory(CONFIG_ROOT)

    def test_training_reduces_best_distance_on_simple_x_target(self) -> None:
        learner = TargetNavigationLearner(self.engine, seed=17)
        learner.set_target(12.0, 0.0, 0.0)

        baseline_result = learner._evaluate_policy(learner._mean_parameters)
        status = learner.train_generations(generations=10, generation_size=10, elite_count=3)

        self.assertEqual(status["target"]["x"], 12.0)
        self.assertEqual(status["generations_completed"], 10)
        self.assertEqual(status["episodes_completed"], 100)
        self.assertGreater(status["best_reward"], baseline_result.total_reward)
        self.assertLessEqual(status["best_distance"], baseline_result.min_distance)

    def test_trained_policy_moves_ship_closer_to_target(self) -> None:
        learner = TargetNavigationLearner(self.engine, seed=17)
        learner.set_target(8.0, 0.0, 0.0)
        learner.train_generations(generations=12, generation_size=12, elite_count=4)

        demo_engine = self.engine.clone()
        starting_distance = learner.distance_to_target(demo_engine.snapshot())
        best_distance = starting_distance

        for _ in range(120):
            thrusters = learner.select_thrusters(demo_engine.snapshot())
            demo_engine.set_active_thrusters(thrusters)
            snapshot = demo_engine.step()
            best_distance = min(best_distance, learner.distance_to_target(snapshot))

        self.assertLess(best_distance, starting_distance)


if __name__ == "__main__":
    unittest.main()
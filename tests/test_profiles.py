from __future__ import annotations

import random
import unittest

from simulation.profiles import compute_net_delta


class ProfileBehaviorTests(unittest.TestCase):
    def test_constant_profile_applies_expected_delta(self) -> None:
        delta = compute_net_delta(
            [{"type": "constant", "rate_per_second": -0.5}],
            dt_seconds=2.0,
            active_actions=set(),
            rng=random.Random(3),
        )

        self.assertAlmostEqual(delta, -1.0)

    def test_action_profile_only_applies_when_action_is_active(self) -> None:
        profile = {
            "type": "action_rate",
            "rate_per_second": -1.25,
            "action_ids": ["thruster_x_positive"],
        }

        idle_delta = compute_net_delta([profile], 1.0, set(), random.Random(3))
        active_delta = compute_net_delta([profile], 1.0, {"thruster_x_positive"}, random.Random(3))

        self.assertEqual(idle_delta, 0.0)
        self.assertAlmostEqual(active_delta, -1.25)

    def test_stochastic_profile_is_repeatable_for_a_seed(self) -> None:
        profile = {
            "type": "stochastic",
            "min_rate_per_second": -0.03,
            "max_rate_per_second": -0.005,
        }
        first_rng = random.Random(11)
        second_rng = random.Random(11)

        first_run = [compute_net_delta([profile], 1.0, set(), first_rng) for _ in range(5)]
        second_run = [compute_net_delta([profile], 1.0, set(), second_rng) for _ in range(5)]

        self.assertEqual(first_run, second_run)
        self.assertGreater(len(set(first_run)), 1)


if __name__ == "__main__":
    unittest.main()
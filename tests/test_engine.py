from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from simulation.engine import SimulationEngine


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_ROOT = PROJECT_ROOT / "config"


class SimulationEngineTests(unittest.TestCase):
    def setUp(self) -> None:
        self.engine = SimulationEngine.from_config_directory(CONFIG_ROOT)

    def test_breathing_profiles_shift_o2_and_co2(self) -> None:
        self.engine.step(10.0)
        snapshot = self.engine.snapshot()

        self.assertAlmostEqual(snapshot["variables"]["O2"]["value"], 99.5)
        self.assertAlmostEqual(snapshot["variables"]["CO2"]["value"], 5.5)

    def test_thrusters_burn_fuel_and_move_ship(self) -> None:
        self.engine.start_action("thruster_x_positive")
        self.engine.step(2.0)
        self.engine.stop_action("thruster_x_positive")
        snapshot = self.engine.snapshot()

        self.assertAlmostEqual(snapshot["variables"]["Fuel"]["value"], 68.4)
        self.assertAlmostEqual(snapshot["variables"]["velocity_x"]["value"], 3.0)
        self.assertAlmostEqual(snapshot["variables"]["position_x"]["value"], 6.0)

    def test_conversion_moves_h2o_into_fuel(self) -> None:
        applied = self.engine.trigger_conversion("convert_h2o_to_fuel")
        snapshot = self.engine.snapshot()

        self.assertTrue(applied)
        self.assertAlmostEqual(snapshot["variables"]["H2O"]["value"], 78.0)
        self.assertAlmostEqual(snapshot["variables"]["Fuel"]["value"], 71.2)

    def test_engine_accepts_new_variable_from_config(self) -> None:
        variables_payload = json.loads((CONFIG_ROOT / "variables.json").read_text(encoding="utf-8"))
        variables_payload["variables"].append(
            {
                "name": "Power",
                "label": "Power",
                "group": "resource",
                "unit": "kWh",
                "initial": 42.0,
                "minimum": 0.0,
                "maximum": 100.0,
                "precision": 1,
                "profiles": [
                    {
                        "type": "constant",
                        "rate_per_second": -0.25
                    }
                ]
            }
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = Path(temp_dir)
            variables_path = temp_root / "variables.json"
            actions_path = temp_root / "actions.json"
            variables_path.write_text(json.dumps(variables_payload), encoding="utf-8")
            actions_path.write_text((CONFIG_ROOT / "actions.json").read_text(encoding="utf-8"), encoding="utf-8")

            engine = SimulationEngine.from_paths(variables_path, actions_path)
            engine.step(4.0)
            snapshot = engine.snapshot()

        self.assertIn("Power", snapshot["variables"])
        self.assertAlmostEqual(snapshot["variables"]["Power"]["value"], 41.0)


if __name__ == "__main__":
    unittest.main()
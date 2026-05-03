from __future__ import annotations

from pathlib import Path

from simulation import SimulationEngine
from ui import SimulationApp


def main() -> int:
    project_root = Path(__file__).resolve().parent
    engine = SimulationEngine.from_config_directory(project_root / "config")
    app = SimulationApp(engine)
    app.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
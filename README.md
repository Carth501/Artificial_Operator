# Artificial Operator

Artificial Operator is a Python sandbox for simulating a small spaceship with configurable life-support and motion variables. The first version is a Tkinter desktop app backed by a config-driven simulation engine, so you can change rates, actions, and variables without rewriting the core loop.

## Current Scope

- O2 is consumed at a constant rate to simulate breathing.
- CO2 is generated at a constant rate to mirror breathing output.
- Fuel is consumed only while thrusters are active.
- H2O depletes irregularly using a seeded stochastic profile.
- N2 is present as a passive variable and can be given behavior later.
- Position and velocity are tracked independently for x, y, and z.
- The UI includes live variable displays, hold-to-fire thruster buttons, H2O-to-Fuel conversion, pause, and reset controls.
- New variables can be added through configuration.

## Requirements

- Python 3.11 or newer
- Tkinter support in the Python installation

This project currently uses only the Python standard library.

## Run The App

From the project root:

```bash
python main.py
```

If your system has multiple Python installations, use the Python 3 interpreter you want for the app.

## Run Tests

```bash
python -m unittest tests.test_profiles tests.test_engine
```

## Controls

- Press and hold a thruster button to apply acceleration on that axis.
- Release the button to stop the thruster burn.
- Click `Convert H2O to Fuel` to spend water and generate fuel.
- Click `Pause Simulation` to freeze time updates.
- Click `Reset State` to restore the initial config-defined state.

## Project Layout

```text
Artificial_Operator/
|-- main.py
|-- README.md
|-- config/
|   |-- actions.json
|   `-- variables.json
|-- simulation/
|   |-- __init__.py
|   |-- actions.py
|   |-- engine.py
|   |-- profiles.py
|   `-- state.py
|-- tests/
|   |-- test_engine.py
|   `-- test_profiles.py
`-- ui/
    |-- __init__.py
    |-- app.py
    `-- panels.py
```

## Configuration

The sandbox is primarily driven by two JSON files.

### `config/variables.json`

This file defines:

- global simulation settings such as `tick_seconds` and `random_seed`
- each variable's name, label, unit, initial value, bounds, display precision, and update profiles

Each variable entry can include one or more profiles. The current built-in profile types are:

- `constant`: applies a fixed rate per second
- `action_rate`: applies a rate only when one or more named actions are active
- `stochastic`: applies a random rate between configured minimum and maximum values

Example variable definition:

```json
{
  "name": "O2",
  "label": "O2",
  "group": "resource",
  "unit": "kg",
  "initial": 100.0,
  "minimum": 0.0,
  "maximum": 100.0,
  "precision": 2,
  "profiles": [
    {
      "type": "constant",
      "rate_per_second": -0.05
    }
  ]
}
```

### `config/actions.json`

This file defines:

- thruster actions, including axis, direction, and acceleration
- conversion actions, including source and target variables plus amounts

The UI reads these definitions through the engine, so adding new thruster or conversion entries updates the control panel automatically.

Example conversion definition:

```json
{
  "id": "convert_h2o_to_fuel",
  "label": "Convert H2O to Fuel",
  "source_variable": "H2O",
  "source_amount": 2.0,
  "target_variable": "Fuel",
  "target_amount": 1.2
}
```

## Simulation Model

The engine uses a fixed real-time tick and simple Euler integration:

1. Apply profile-driven resource changes.
2. Apply any active thruster accelerations to velocity.
3. Integrate velocity into position.
4. Advance mission time.

This keeps the sandbox easy to tune and extend, but it is intentionally lightweight. It does not yet model ship mass, pressure, collisions, orbital mechanics, or persistence.

## Extending The Sandbox

### Add a new variable

Add a new object to `config/variables.json` with a unique `name`, display metadata, and any profiles you want. Most new variables will appear in the UI automatically without changing engine code.

Example:

```json
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
```

### Add a new action

For additional thrusters or conversions, update `config/actions.json`. The existing UI is already built to render the configured thrusters and conversion buttons.

### Add a new profile type

If you need behavior beyond `constant`, `action_rate`, or `stochastic`, add a new evaluator in `simulation/profiles.py` and reference it from variable profiles in `config/variables.json`.

## Notes

- The simulation seed is fixed by default so stochastic behavior is repeatable unless you change `random_seed`.
- Position and velocity are represented as named variables such as `position_x` and `velocity_z`, which keeps the state model generic.
- N2 is currently neutral by design; you can later give it leakage, balancing, or venting behavior through configuration or a new profile type.

## Next Useful Improvements

- Add in-app editing for variable profiles and action values.
- Add save and load support for simulation states.
- Add more ship systems such as power, pressure, heat, or tank capacity.
- Add richer physics constraints if you want more realism.

# Artificial Operator

Artificial Operator is a Python sandbox for simulating a small spaceship with configurable life-support and motion variables. The current version is a Tkinter desktop app backed by a config-driven simulation engine with explicit modules and systems, so you can change rates, actions, containers, and module ownership without rewriting the core loop.

## Current Scope

- O2 is consumed at a constant rate to simulate breathing.
- CO2 is generated at a constant rate to mirror breathing output.
- Fuel is consumed only while thrusters are active.
- H2O depletes irregularly using a seeded stochastic profile.
- N2 is present as a passive variable and can be given behavior later.
- Ship mass is tracked as a core ship property.
- Position and velocity are tracked independently for x, y, and z.
- Resource containers and action mechanisms live inside modules with integrity values.
- A module that reaches 0 integrity fails all of its systems, disables its mechanisms, and drains its containers.
- The UI includes live module displays, hold-to-fire thruster buttons, H2O-to-Fuel conversion, pause, and reset controls.
- New variables and actions can be added through configuration as long as they are assigned to systems in a module.

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
|   |-- modules.json
|   |-- systems.json
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

The sandbox is driven by four JSON files.

### `config/variables.json`

This file defines:

- global simulation settings such as `tick_seconds` and `random_seed`
- each variable's name, label, unit, initial value, bounds, display precision, and update profiles

Variables that are not referenced by container systems in `config/systems.json` are treated as core ship state. The shipped config keeps `Mass`, `position_*`, and `velocity_*` outside modules.

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

Every action must be referenced by a mechanism system in `config/systems.json` before it can operate.

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

### `config/modules.json`

This file defines:

- modules, including their labels and initial integrity values
- connections between modules on the module map
- the `systems_ids` each module owns

The shipped config includes a `resource_management` module and a `propulsion` module, with a connection drawn between them.

Example module definition:

```json
{
  "id": "resource_management",
  "label": "Resource Management",
  "initial_integrity": 100.0,
  "connections": ["propulsion"],
  "systems_ids": ["oxygen_tank", "fuel_conversion_machine"]
}
```

### `config/systems.json`

This file defines:

- whether a system is a `container` for variables or a `mechanism` for actions
- which variables or actions each system owns

The shipped config includes the Fuel, O2, N2, H2O, and CO2 container systems plus the fuel conversion machine under `resource_management`, and the thruster system under `propulsion`.

Example system definition:

```json
{
  "id": "oxygen_tank",
  "label": "Oxygen Tank",
  "kind": "container",
  "variable_names": ["O2"]
}
```

## Simulation Model

The engine uses a fixed real-time tick and simple Euler integration:

1. Apply profile-driven changes for variables whose container systems are still operational.
2. Apply any active thruster accelerations whose mechanism systems are still operational and have fuel available.
3. Integrate velocity into position.
4. Advance mission time.

If a module's integrity reaches 0, all of its systems fail immediately. Failed mechanism systems cannot operate, and failed container systems are drained to 0 and reject further writes until integrity is restored.

This keeps the sandbox easy to tune and extend, but it is intentionally lightweight. It does not yet use mass in thrust calculations, and it does not yet model pressure, collisions, orbital mechanics, or persistence.

## Extending The Sandbox

### Add a new variable

Add a new object to `config/variables.json` with a unique `name`, display metadata, and any profiles you want. If the variable represents a capacity or stored resource, also add it to a `container` system in `config/systems.json`. Core ship variables can remain unowned.

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

For additional thrusters or conversions, update `config/actions.json` and then assign the action ID to a `mechanism` system in `config/systems.json`. The existing UI renders actions from the module snapshot, so correctly owned actions appear automatically.

### Add a new profile type

If you need behavior beyond `constant`, `action_rate`, or `stochastic`, add a new evaluator in `simulation/profiles.py` and reference it from variable profiles in `config/variables.json`.

## Notes

- The simulation seed is fixed by default so stochastic behavior is repeatable unless you change `random_seed`.
- Position and velocity are represented as named variables such as `position_x` and `velocity_z`, which keeps the state model generic.
- Module integrity is runtime state and resets back to the configured `initial_integrity` values.
- N2 is currently neutral by design; you can later give it leakage, balancing, or venting behavior through configuration or a new profile type.

## Next Useful Improvements

- Add in-app editing for variable profiles and action values.
- Add save and load support for simulation states.
- Add more ship systems such as power, pressure, heat, or tank capacity.
- Add richer physics constraints if you want more realism.

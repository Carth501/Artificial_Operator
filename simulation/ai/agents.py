from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from .models import Observation, TargetPositionObjective


AXES = ("x", "y", "z")
AXIS_INDEX = {axis: index for index, axis in enumerate(AXES)}


class Agent(Protocol):
    def select_actions(self, observation: Observation, objective: TargetPositionObjective) -> set[str]:
        ...


@dataclass(frozen=True)
class ThrusterSpec:
    action_id: str
    axis: str
    direction: int
    acceleration_per_second: float


@dataclass(frozen=True)
class TargetPolicyParameters:
    brake_distance_multiplier: float = 1.0
    approach_distance_multiplier: float = 2.0
    approach_velocity_multiplier: float = 1.0
    settle_velocity_multiplier: float = 1.0


class ParameterizedTargetPositionAgent:
    def __init__(
        self,
        thrusters: tuple[dict[str, Any], ...],
        parameters: TargetPolicyParameters | None = None,
    ) -> None:
        self._thrusters_by_axis: dict[str, dict[int, ThrusterSpec]] = {axis: {} for axis in AXES}
        self._acceleration_by_axis: dict[str, float] = {axis: 0.0 for axis in AXES}
        self._parameters = TargetPolicyParameters() if parameters is None else parameters

        for thruster in thrusters:
            axis = str(thruster["axis"]).lower()
            if axis not in self._thrusters_by_axis:
                continue

            direction = 1 if float(thruster["direction"]) >= 0.0 else -1
            spec = ThrusterSpec(
                action_id=str(thruster["id"]),
                axis=axis,
                direction=direction,
                acceleration_per_second=abs(float(thruster["acceleration_per_second"])),
            )
            self._thrusters_by_axis[axis][direction] = spec
            self._acceleration_by_axis[axis] = max(
                self._acceleration_by_axis[axis],
                spec.acceleration_per_second,
            )

    def select_actions(self, observation: Observation, objective: TargetPositionObjective) -> set[str]:
        requested_actions: set[str] = set()

        for axis in AXES:
            action_id = self._select_axis_action(axis, observation, objective)
            if action_id is not None:
                requested_actions.add(action_id)

        return requested_actions

    def _select_axis_action(
        self,
        axis: str,
        observation: Observation,
        objective: TargetPositionObjective,
    ) -> str | None:
        axis_index = AXIS_INDEX[axis]
        current_position = observation.position[axis_index]
        current_velocity = observation.velocity[axis_index]
        target_position = objective.target_position[axis_index]
        displacement = target_position - current_position
        acceleration = self._acceleration_by_axis[axis]
        settle_velocity_threshold = objective.settle_velocity * self._parameters.settle_velocity_multiplier

        if acceleration <= 0.0:
            return None

        if abs(displacement) <= objective.tolerance:
            if abs(current_velocity) <= settle_velocity_threshold:
                return None
            return self._action_for_direction(axis, -current_velocity)

        target_direction = 1.0 if displacement > 0.0 else -1.0
        if current_velocity == 0.0 or current_velocity * target_direction <= 0.0:
            return self._action_for_direction(axis, target_direction)

        stopping_distance = (current_velocity * current_velocity) / (2.0 * acceleration)
        if stopping_distance * self._parameters.brake_distance_multiplier >= abs(displacement):
            return self._action_for_direction(axis, -current_velocity)

        if (
            abs(displacement) <= objective.tolerance * self._parameters.approach_distance_multiplier
            and abs(current_velocity) > settle_velocity_threshold * self._parameters.approach_velocity_multiplier
        ):
            return self._action_for_direction(axis, -current_velocity)

        return self._action_for_direction(axis, target_direction)

    def _action_for_direction(self, axis: str, direction: float) -> str | None:
        normalized_direction = 1 if direction >= 0.0 else -1
        spec = self._thrusters_by_axis[axis].get(normalized_direction)
        if spec is None:
            return None
        return spec.action_id


class TargetPositionAgent(ParameterizedTargetPositionAgent):
    def __init__(self, thrusters: tuple[dict[str, Any], ...]) -> None:
        super().__init__(thrusters, parameters=TargetPolicyParameters())
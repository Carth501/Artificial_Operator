from __future__ import annotations

from dataclasses import dataclass
import math
import random
from statistics import fmean, pstdev
from typing import Any

from .engine import SimulationEngine


IDLE_ACTION = "idle"
THRUSTER_ACTION_IDS = (
    "thruster_x_positive",
    "thruster_x_negative",
    "thruster_y_positive",
    "thruster_y_negative",
    "thruster_z_positive",
    "thruster_z_negative",
)


@dataclass(frozen=True)
class PolicyParameters:
    error_gain: float
    velocity_gain: float
    threshold: float


@dataclass(frozen=True)
class EpisodeResult:
    parameters: PolicyParameters
    total_reward: float
    min_distance: float
    success: bool


class TargetNavigationLearner:
    def __init__(self, engine: SimulationEngine, seed: int = 17) -> None:
        self._prototype_engine = engine.clone()
        self._random = random.Random(seed)
        self._max_acceleration = max(
            (thruster["acceleration_per_second"] for thruster in self._prototype_engine.list_thrusters()),
            default=1.0,
        )
        self._target = (10.0, 0.0, 0.0)
        self._generation_size = 12
        self._elite_count = 4
        self._reset_optimizer()

    @property
    def target(self) -> tuple[float, float, float]:
        return self._target

    def set_target(self, x: float, y: float, z: float) -> None:
        self._target = (float(x), float(y), float(z))
        self._reset_optimizer()

    def distance_to_target(self, snapshot: dict[str, Any]) -> float:
        position = snapshot.get("position", {})
        x = float(position.get("x", 0.0))
        y = float(position.get("y", 0.0))
        z = float(position.get("z", 0.0))
        return math.dist((x, y, z), self._target)

    def speed(self, snapshot: dict[str, Any]) -> float:
        velocity = snapshot.get("velocity", {})
        vx = float(velocity.get("x", 0.0))
        vy = float(velocity.get("y", 0.0))
        vz = float(velocity.get("z", 0.0))
        return math.sqrt((vx * vx) + (vy * vy) + (vz * vz))

    def select_thrusters(self, snapshot: dict[str, Any]) -> tuple[str, ...]:
        action_id = self._select_action(self._best_parameters, snapshot)
        self._last_action_id = action_id
        if action_id == IDLE_ACTION:
            return ()
        return (action_id,)

    def train_generations(
        self,
        generations: int = 1,
        generation_size: int | None = None,
        elite_count: int | None = None,
    ) -> dict[str, Any]:
        sample_count = self._generation_size if generation_size is None else max(3, int(generation_size))
        selected_elite_count = self._elite_count if elite_count is None else max(1, int(elite_count))
        selected_elite_count = min(selected_elite_count, sample_count)

        for _ in range(max(1, int(generations))):
            candidates = [self._best_parameters, self._mean_parameters]
            while len(candidates) < sample_count:
                candidates.append(self._sample_parameters())

            results = [self._evaluate_policy(parameters) for parameters in candidates]
            results.sort(key=lambda result: result.total_reward, reverse=True)
            elites = results[:selected_elite_count]

            self._mean_parameters = PolicyParameters(
                error_gain=fmean(result.parameters.error_gain for result in elites),
                velocity_gain=fmean(result.parameters.velocity_gain for result in elites),
                threshold=fmean(result.parameters.threshold for result in elites),
            )
            self._std_parameters = PolicyParameters(
                error_gain=max(0.08, pstdev(result.parameters.error_gain for result in elites)),
                velocity_gain=max(0.08, pstdev(result.parameters.velocity_gain for result in elites)),
                threshold=max(0.04, pstdev(result.parameters.threshold for result in elites)),
            )

            best_result = elites[0]
            if best_result.total_reward >= self._best_reward:
                self._best_reward = best_result.total_reward
                self._best_parameters = best_result.parameters
            self._best_distance = min(self._best_distance, best_result.min_distance)

            self._generations_completed += 1
            self._episodes_completed += len(results)
            self._successful_episodes += sum(1 for result in results if result.success)
            self._last_average_reward = fmean(result.total_reward for result in results)
            self._last_best_distance = best_result.min_distance

        return self.status()

    def status(self) -> dict[str, Any]:
        return {
            "target": {
                "x": self._target[0],
                "y": self._target[1],
                "z": self._target[2],
            },
            "best_distance": self._best_distance,
            "best_reward": self._best_reward,
            "episodes_completed": self._episodes_completed,
            "generations_completed": self._generations_completed,
            "has_policy": self._best_reward > float("-inf"),
            "last_action_id": self._last_action_id,
            "last_average_reward": self._last_average_reward,
            "last_best_distance": self._last_best_distance,
            "successful_episodes": self._successful_episodes,
        }

    def _reset_optimizer(self) -> None:
        self._mean_parameters = PolicyParameters(error_gain=0.8, velocity_gain=-0.8, threshold=0.3)
        self._std_parameters = PolicyParameters(error_gain=0.45, velocity_gain=0.45, threshold=0.12)
        self._best_parameters = self._mean_parameters
        self._best_reward = float("-inf")
        self._best_distance = float("inf")
        self._episodes_completed = 0
        self._generations_completed = 0
        self._last_action_id = IDLE_ACTION
        self._last_average_reward = 0.0
        self._last_best_distance = float("inf")
        self._successful_episodes = 0

    def _sample_parameters(self) -> PolicyParameters:
        return PolicyParameters(
            error_gain=self._random.gauss(self._mean_parameters.error_gain, self._std_parameters.error_gain),
            velocity_gain=self._random.gauss(self._mean_parameters.velocity_gain, self._std_parameters.velocity_gain),
            threshold=min(
                1.8,
                max(
                    0.05,
                    self._random.gauss(self._mean_parameters.threshold, self._std_parameters.threshold),
                ),
            ),
        )

    def _evaluate_policy(self, parameters: PolicyParameters) -> EpisodeResult:
        engine = self._prototype_engine.clone()
        previous_snapshot = engine.snapshot()
        previous_distance = self.distance_to_target(previous_snapshot)
        min_distance = previous_distance
        total_reward = 0.0

        for _ in range(self._episode_step_limit()):
            action_id = self._select_action(parameters, previous_snapshot)
            if action_id == IDLE_ACTION:
                engine.set_active_thrusters(())
            else:
                engine.set_active_thrusters((action_id,))

            snapshot = engine.step()
            current_distance = self.distance_to_target(snapshot)
            current_speed = self.speed(snapshot)
            min_distance = min(min_distance, current_distance)
            total_reward += (previous_distance - current_distance) * 10.0
            total_reward -= current_distance * 0.06
            total_reward -= current_speed * 0.08
            total_reward -= self._fuel_burn(previous_snapshot, snapshot) * 0.4

            if "Insufficient power" in snapshot.get("alerts", ()): 
                total_reward -= 3.0

            if any(not bool(module.get("operational", False)) for module in snapshot.get("modules", ())):
                total_reward -= 10.0
                return EpisodeResult(parameters, total_reward, min_distance, False)

            if self._is_success(snapshot):
                total_reward += 120.0
                return EpisodeResult(parameters, total_reward, min_distance, True)

            variables = snapshot.get("variables", {})
            if float(variables.get("H2", {}).get("value", 0.0)) <= 0.0:
                total_reward -= 15.0
                return EpisodeResult(parameters, total_reward, min_distance, False)
            if float(variables.get("O2", {}).get("value", 0.0)) <= 0.0:
                total_reward -= 15.0
                return EpisodeResult(parameters, total_reward, min_distance, False)

            previous_snapshot = snapshot
            previous_distance = current_distance

        total_reward -= previous_distance * 0.5
        return EpisodeResult(parameters, total_reward, min_distance, False)

    def _select_action(self, parameters: PolicyParameters, snapshot: dict[str, Any]) -> str:
        best_axis = ""
        best_margin = 0.0
        best_score = 0.0

        for axis in ("x", "y", "z"):
            normalized_error = self._normalized_error(snapshot, axis)
            normalized_velocity = self._normalized_velocity(snapshot, axis)
            score = (parameters.error_gain * normalized_error) + (parameters.velocity_gain * normalized_velocity)
            margin = abs(score) - parameters.threshold
            if margin > best_margin:
                best_axis = axis
                best_margin = margin
                best_score = score

        if not best_axis:
            return IDLE_ACTION
        if best_score > 0.0:
            return f"thruster_{best_axis}_positive"
        return f"thruster_{best_axis}_negative"

    def _normalized_error(self, snapshot: dict[str, Any], axis: str) -> float:
        position = snapshot.get("position", {})
        target_value = self._target["xyz".index(axis)]
        current_value = float(position.get(axis, 0.0))
        error_scale = max(10.0, self._target_distance())
        return max(-2.5, min(2.5, (target_value - current_value) / error_scale))

    def _normalized_velocity(self, snapshot: dict[str, Any], axis: str) -> float:
        velocity = snapshot.get("velocity", {})
        current_value = float(velocity.get(axis, 0.0))
        speed_scale = max(2.0, math.sqrt(max(1.0, self._target_distance()) * self._max_acceleration))
        return max(-2.5, min(2.5, current_value / speed_scale))

    def _episode_step_limit(self) -> int:
        return max(100, min(360, int(55.0 * math.sqrt(max(1.0, self._target_distance()) / self._max_acceleration))))

    def _target_distance(self) -> float:
        return math.dist((0.0, 0.0, 0.0), self._target)

    def _position_tolerance(self) -> float:
        return max(0.75, self._target_distance() * 0.02)

    def _is_success(self, snapshot: dict[str, Any]) -> bool:
        return self.distance_to_target(snapshot) <= self._position_tolerance() and self.speed(snapshot) <= 0.9

    def _fuel_burn(self, previous_snapshot: dict[str, Any], current_snapshot: dict[str, Any]) -> float:
        previous_variables = previous_snapshot.get("variables", {})
        current_variables = current_snapshot.get("variables", {})
        previous_h2 = float(previous_variables.get("H2", {}).get("value", 0.0))
        current_h2 = float(current_variables.get("H2", {}).get("value", 0.0))
        previous_o2 = float(previous_variables.get("O2", {}).get("value", 0.0))
        current_o2 = float(current_variables.get("O2", {}).get("value", 0.0))
        return max(0.0, previous_h2 - current_h2) + max(0.0, previous_o2 - current_o2)
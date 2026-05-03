from __future__ import annotations

from collections.abc import Iterable, Mapping
import random


def compute_net_delta(
    profiles: Iterable[Mapping[str, object]],
    dt_seconds: float,
    active_actions: set[str],
    rng: random.Random,
) -> float:
    return sum(
        compute_profile_delta(profile, dt_seconds, active_actions, rng)
        for profile in profiles
    )


def compute_profile_delta(
    profile: Mapping[str, object],
    dt_seconds: float,
    active_actions: set[str],
    rng: random.Random,
) -> float:
    profile_type = str(profile.get("type", "constant"))

    if profile_type == "constant":
        rate_per_second = _float_value(profile, "rate_per_second")
        return rate_per_second * dt_seconds

    if profile_type == "action_rate":
        rate_per_second = _float_value(profile, "rate_per_second")
        configured_actions = set(_string_values(profile, "action_ids"))
        activation = str(profile.get("activation", "any"))
        is_active = False
        if configured_actions:
            if activation == "all":
                is_active = configured_actions.issubset(active_actions)
            else:
                is_active = bool(configured_actions & active_actions)
        return rate_per_second * dt_seconds if is_active else 0.0

    if profile_type == "stochastic":
        minimum_rate = _float_value(profile, "min_rate_per_second")
        maximum_rate = _float_value(profile, "max_rate_per_second")
        low, high = sorted((minimum_rate, maximum_rate))
        return rng.uniform(low, high) * dt_seconds

    raise ValueError(f"Unsupported profile type: {profile_type}")


def _float_value(profile: Mapping[str, object], key: str) -> float:
    raw_value = profile.get(key)
    if raw_value is None:
        raise ValueError(f"Profile is missing required key: {key}")
    if isinstance(raw_value, bool):
        raise ValueError(f"Profile key must be numeric, not boolean: {key}")
    if isinstance(raw_value, (int, float, str)):
        return float(raw_value)
    raise ValueError(f"Profile key must be numeric: {key}")


def _string_values(profile: Mapping[str, object], key: str) -> tuple[str, ...]:
    raw_value = profile.get(key, ())
    if raw_value is None:
        return ()
    if isinstance(raw_value, (str, bytes)):
        raise ValueError(f"Profile key must be an iterable of strings: {key}")
    if not isinstance(raw_value, Iterable):
        raise ValueError(f"Profile key must be an iterable of strings: {key}")
    return tuple(str(item) for item in raw_value)
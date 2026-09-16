"""Input validation and conservative simulation safety limits."""

from __future__ import annotations

from dataclasses import dataclass
from math import hypot, isfinite


@dataclass(frozen=True)
class FlightLimits:
    min_altitude_m: float = 1.0
    max_altitude_m: float = 120.0
    max_distance_m: float = 2000.0
    min_speed_mps: float = 0.2
    max_speed_mps: float = 20.0


DEFAULT_LIMITS = FlightLimits()


def minimum_braking_distance_m(
    speed_mps: float,
    reaction_time_s: float = 0.55,
    deceleration_mps2: float = 3.5,
    margin_m: float = 2.5,
) -> float:
    """Return a conservative obstacle braking distance for the simulator."""
    speed = max(0.0, float(speed_mps))
    deceleration = max(0.5, float(deceleration_mps2))
    return max(
        3.5,
        speed * max(0.0, float(reaction_time_s))
        + speed * speed / (2.0 * deceleration)
        + max(0.0, float(margin_m)),
    )


def validate_destination(
    x_m: float,
    y_m: float,
    altitude_m: float,
    speed_mps: float,
    limits: FlightLimits = DEFAULT_LIMITS,
) -> None:
    values = (x_m, y_m, altitude_m, speed_mps)
    if not all(isfinite(float(value)) for value in values):
        raise ValueError("좌표와 속도는 유한한 숫자여야 합니다.")
    if hypot(x_m, y_m) > limits.max_distance_m:
        raise ValueError(f"목적지는 원점에서 {limits.max_distance_m:.0f}m 이내여야 합니다.")
    if not limits.min_altitude_m <= altitude_m <= limits.max_altitude_m:
        raise ValueError(
            f"고도는 {limits.min_altitude_m:.0f}~{limits.max_altitude_m:.0f}m 범위여야 합니다."
        )
    if not limits.min_speed_mps <= speed_mps <= limits.max_speed_mps:
        raise ValueError(
            f"속도는 {limits.min_speed_mps:.1f}~{limits.max_speed_mps:.0f}m/s 범위여야 합니다."
        )

"""Tests for simulated enemy-drone detections.

시뮬레이션 적 드론 탐지 변환 테스트입니다.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from perception.target_detector import TargetDetector


def _vector(x: float, y: float, z: float = 0.0) -> SimpleNamespace:
    return SimpleNamespace(x_val=x, y_val=y, z_val=z)


def _raw_detection() -> SimpleNamespace:
    return SimpleNamespace(
        name="BP_EnemyDrone_C_0",
        box2D=SimpleNamespace(
            min=_vector(100, 80),
            max=_vector(220, 180),
        ),
        box3D=SimpleNamespace(
            min=_vector(8.0, -1.0, -6.0),
            max=_vector(10.0, 1.0, -4.0),
        ),
        relative_pose=SimpleNamespace(position=_vector(9.0, 0.5, 0.2)),
    )


def test_normalize_and_build_obstacle_points() -> None:
    detection = TargetDetector.normalize([_raw_detection()], 640, 480)

    assert len(detection) == 1
    assert detection[0]["name"] == "BP_EnemyDrone_C_0"
    assert detection[0]["distance_m"] > 9.0
    points = TargetDetector.obstacle_points(detection, padding_m=2.0)
    assert points.shape == (64, 3)
    assert np.all(np.isfinite(points))


def test_duplicate_boxes_are_removed() -> None:
    raw = _raw_detection()
    detections = TargetDetector.normalize([raw, raw], 640, 480)
    assert len(detections) == 1


def test_forward_target_is_a_collision_threat() -> None:
    detection = TargetDetector.normalize([_raw_detection()], 640, 480)[0]
    assert TargetDetector.is_collision_threat(detection)


def test_distant_target_is_not_a_collision_threat() -> None:
    detection = TargetDetector.normalize([_raw_detection()], 640, 480)[0]
    detection["distance_m"] = 40.0
    assert not TargetDetector.is_collision_threat(detection)


def test_human_blueprint_is_classified_as_human() -> None:
    raw = _raw_detection()
    raw.name = "BP_AINormalPeople_Drone_C_2"

    detection = TargetDetector.normalize([raw], 640, 480)[0]

    assert detection["target_kind"] == "human"

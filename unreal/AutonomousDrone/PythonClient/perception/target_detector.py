"""AirSim detection normalization and dynamic-obstacle conversion.

AirSim 객체 탐지 결과를 정규화하고 동적 장애물 점군으로 변환합니다.
"""

from __future__ import annotations

import math
from typing import Iterable

import numpy as np


class TargetDetector:
    """Convert simulator detections into UI and planner friendly data.

    시뮬레이터 탐지 결과를 UI 및 경로계획기에서 사용할 수 있게 변환합니다.
    """

    HUMAN_NAME_TOKENS = (
        "ainormalpeople",
        "humantarget",
        "human",
        "person",
        "pedestrian",
        "ch01",
        "ch02",
    )

    @classmethod
    def target_kind(cls, name: object) -> str:
        """Classify a simulator object name for UI labels and logging.

        UI 라벨과 로깅에 사용할 시뮬레이터 객체 종류를 판별합니다.
        """
        lowered = str(name).casefold().replace("_", "")
        if any(token.replace("_", "") in lowered for token in cls.HUMAN_NAME_TOKENS):
            return "human"
        return "enemy_drone"

    @staticmethod
    def normalize(
        raw_detections: Iterable[object],
        image_width: int,
        image_height: int,
    ) -> list[dict[str, float | int | str]]:
        results: list[dict[str, float | int | str]] = []
        seen: set[tuple[str, int, int, int, int]] = set()

        for raw in raw_detections:
            box_2d = getattr(raw, "box2D", None)
            box_3d = getattr(raw, "box3D", None)
            relative_pose = getattr(raw, "relative_pose", None)
            relative_position = getattr(relative_pose, "position", None)
            if box_2d is None or relative_position is None:
                continue

            minimum_2d = getattr(box_2d, "min", None)
            maximum_2d = getattr(box_2d, "max", None)
            if minimum_2d is None or maximum_2d is None:
                continue

            x_min = max(0, min(int(image_width), int(minimum_2d.x_val)))
            y_min = max(0, min(int(image_height), int(minimum_2d.y_val)))
            x_max = max(0, min(int(image_width), int(maximum_2d.x_val)))
            y_max = max(0, min(int(image_height), int(maximum_2d.y_val)))
            if x_max <= x_min or y_max <= y_min:
                continue

            name = str(getattr(raw, "name", "EnemyDrone"))
            key = (name, x_min, y_min, x_max, y_max)
            if key in seen:
                continue
            seen.add(key)

            relative_x = float(relative_position.x_val)
            relative_y = float(relative_position.y_val)
            relative_z = float(relative_position.z_val)
            item: dict[str, float | int | str] = {
                "name": name,
                "target_kind": TargetDetector.target_kind(name),
                "x_min": x_min,
                "y_min": y_min,
                "x_max": x_max,
                "y_max": y_max,
                "image_width": int(image_width),
                "image_height": int(image_height),
                "relative_x": relative_x,
                "relative_y": relative_y,
                "relative_z": relative_z,
                "distance_m": math.sqrt(
                    relative_x * relative_x
                    + relative_y * relative_y
                    + relative_z * relative_z
                ),
            }

            if box_3d is not None:
                minimum_3d = getattr(box_3d, "min", None)
                maximum_3d = getattr(box_3d, "max", None)
                if minimum_3d is not None and maximum_3d is not None:
                    item.update(
                        {
                            "world_x_min": float(minimum_3d.x_val),
                            "world_y_min": float(minimum_3d.y_val),
                            "world_z_min": float(minimum_3d.z_val),
                            "world_x_max": float(maximum_3d.x_val),
                            "world_y_max": float(maximum_3d.y_val),
                            "world_z_max": float(maximum_3d.z_val),
                        }
                    )
            results.append(item)

        return results

    @staticmethod
    def obstacle_points(
        detections: Iterable[dict[str, float | int | str]],
        padding_m: float = 2.5,
    ) -> np.ndarray:
        """Build sparse padded 3D boxes in AirSim world NED coordinates.

        AirSim 월드 NED 좌표에서 안전 여유가 포함된 희소 3D 박스를 만듭니다.
        """
        clouds: list[np.ndarray] = []
        for detection in detections:
            required = (
                "world_x_min",
                "world_y_min",
                "world_z_min",
                "world_x_max",
                "world_y_max",
                "world_z_max",
            )
            if any(key not in detection for key in required):
                continue

            minimum = np.asarray(
                [
                    float(detection["world_x_min"]),
                    float(detection["world_y_min"]),
                    float(detection["world_z_min"]),
                ],
                dtype=np.float32,
            )
            maximum = np.asarray(
                [
                    float(detection["world_x_max"]),
                    float(detection["world_y_max"]),
                    float(detection["world_z_max"]),
                ],
                dtype=np.float32,
            )
            if (
                np.linalg.norm(maximum - minimum) < 0.01
                and np.linalg.norm(maximum) < 0.01
            ):
                continue
            lower = np.minimum(minimum, maximum) - float(padding_m)
            upper = np.maximum(minimum, maximum) + float(padding_m)
            if not np.all(np.isfinite(lower)) or not np.all(np.isfinite(upper)):
                continue

            axes = [
                np.linspace(lower[index], upper[index], 4, dtype=np.float32)
                for index in range(3)
            ]
            grid = np.meshgrid(*axes, indexing="ij")
            clouds.append(np.column_stack([axis.ravel() for axis in grid]))

        if not clouds:
            return np.empty((0, 3), dtype=np.float32)
        return np.vstack(clouds).astype(np.float32, copy=False)

    @staticmethod
    def is_collision_threat(
        detection: dict[str, float | int | str],
        maximum_distance_m: float = 20.0,
    ) -> bool:
        """Return true when a visible target overlaps the forward flight corridor.

        보이는 표적이 전방 비행 통로와 겹치면 참을 반환합니다.
        """
        distance = float(detection.get("distance_m", float("inf")))
        if distance > maximum_distance_m:
            return False

        relative_x = float(detection.get("relative_x", 0.0))
        relative_y = abs(float(detection.get("relative_y", 0.0)))
        relative_z = abs(float(detection.get("relative_z", 0.0)))
        if relative_x > 0.0 and relative_y <= 4.5 and relative_z <= 3.5:
            return True

        # Some simulator builds do not populate relative pose reliably. Use
        # the central image corridor as a conservative visual fallback.
        # 일부 시뮬레이터 버전은 상대 자세가 부정확하므로 화면 중앙 통로를
        # 보수적인 시각 기반 대체 판정으로 사용합니다.
        image_width = max(1, int(detection.get("image_width", 1)))
        image_height = max(1, int(detection.get("image_height", 1)))
        center_x = (
            float(detection.get("x_min", 0))
            + float(detection.get("x_max", 0))
        ) * 0.5
        center_y = (
            float(detection.get("y_min", 0))
            + float(detection.get("y_max", 0))
        ) * 0.5
        return (
            image_width * 0.25 <= center_x <= image_width * 0.75
            and image_height * 0.2 <= center_y <= image_height * 0.8
        )

"""Cosys-AirSim Echo/Radar point-cloud normalization and sensor fusion.

Cosys-AirSim Echo/Radar 점군 정규화 및 센서 융합 기능입니다.
"""

from __future__ import annotations

import json
from collections import defaultdict
from typing import Iterable

import numpy as np


class RadarProcessor:
    """Convert active Echo samples into enemy tracks and planner obstacles.

    Echo 활성 반사 샘플을 적 표적과 경로계획 장애물로 변환합니다.
    """

    ACTIVE_SAMPLE_WIDTH = 6
    ENEMY_NAME_TOKENS = ("enemy", "hostile", "bp_enemydrone")
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
    def parse_active_echo(cls, values: object) -> dict[str, np.ndarray]:
        """Split flat active Echo data into coordinates and reflection fields.

        평면형 Echo 활성 데이터를 좌표와 반사 속성으로 분리합니다.
        """
        raw = np.asarray(values, dtype=np.float32).reshape(-1)
        usable = raw.size - (raw.size % cls.ACTIVE_SAMPLE_WIDTH)
        if usable <= 0:
            empty_points = np.empty((0, 3), dtype=np.float32)
            empty_values = np.empty((0,), dtype=np.float32)
            return {
                "points": empty_points,
                "attenuation": empty_values,
                "total_distance": empty_values.copy(),
                "reflection_count": empty_values.copy(),
            }

        samples = raw[:usable].reshape((-1, cls.ACTIVE_SAMPLE_WIDTH))
        return {
            "points": samples[:, :3].astype(np.float32, copy=False),
            "attenuation": samples[:, 3].astype(np.float32, copy=False),
            "total_distance": samples[:, 4].astype(np.float32, copy=False),
            "reflection_count": samples[:, 5].astype(np.float32, copy=False),
        }

    @staticmethod
    def normalize_labels(raw_labels: object, point_count: int) -> list[str]:
        """Return one decoded ground-truth label for every Echo point.

        각 Echo 점에 대응하는 정답 객체 이름을 문자열로 정규화합니다.
        """
        if point_count <= 0:
            return []

        labels: list[object]
        if raw_labels is None:
            labels = []
        elif isinstance(raw_labels, bytes):
            labels = [raw_labels.decode("utf-8", errors="replace")]
        elif isinstance(raw_labels, str):
            candidate = raw_labels.strip()
            if candidate.startswith("["):
                try:
                    decoded = json.loads(candidate)
                    labels = list(decoded) if isinstance(decoded, list) else [candidate]
                except json.JSONDecodeError:
                    labels = [candidate]
            else:
                labels = [candidate] if candidate else []
        else:
            try:
                labels = list(raw_labels)  # type: ignore[arg-type]
            except TypeError:
                labels = [raw_labels]

        normalized = [
            value.decode("utf-8", errors="replace")
            if isinstance(value, bytes)
            else str(value)
            for value in labels
        ]
        if len(normalized) == 1 and point_count > 1:
            normalized *= point_count
        if len(normalized) < point_count:
            normalized.extend([""] * (point_count - len(normalized)))
        return normalized[:point_count]

    @classmethod
    def is_enemy_label(cls, label: str) -> bool:
        """Return true for simulator labels assigned to hostile drone actors.

        적 드론 Actor에 해당하는 시뮬레이터 객체 이름이면 참입니다.
        """
        lowered = str(label).casefold().replace("_", "")
        return any(token.replace("_", "") in lowered for token in cls.ENEMY_NAME_TOKENS)

    @classmethod
    def target_kind(cls, label: str) -> str | None:
        """Return the supported target class encoded in an Echo label.

        Echo 라벨에 포함된 지원 대상 종류를 반환합니다.
        """
        lowered = str(label).casefold().replace("_", "")
        if any(token.replace("_", "") in lowered for token in cls.HUMAN_NAME_TOKENS):
            return "human"
        if any(token.replace("_", "") in lowered for token in cls.ENEMY_NAME_TOKENS):
            return "enemy_drone"
        return None

    @classmethod
    def build_enemy_tracks(
        cls,
        world_points: np.ndarray,
        labels: Iterable[str],
        total_distance: np.ndarray,
        attenuation: np.ndarray,
        reflection_count: np.ndarray,
    ) -> list[dict[str, object]]:
        """Group labeled Radar returns into one world-space track per actor.

        객체명이 있는 Radar 반사점을 Actor별 월드 좌표 표적으로 묶습니다.
        """
        if not world_points.size:
            return []

        groups: dict[str, list[int]] = defaultdict(list)
        for index, label in enumerate(labels):
            if index >= len(world_points):
                break
            if cls.target_kind(label) is not None:
                groups[str(label)].append(index)

        tracks: list[dict[str, object]] = []
        for label, indices in groups.items():
            selected = world_points[np.asarray(indices, dtype=np.int64)]
            finite = np.all(np.isfinite(selected), axis=1)
            selected = selected[finite]
            if not selected.size:
                continue

            valid_indices = np.asarray(indices, dtype=np.int64)[finite]
            center = np.median(selected, axis=0).astype(np.float32)
            distances = total_distance[valid_indices]
            distances = distances[np.isfinite(distances) & (distances >= 0.0)]
            attenuations = attenuation[valid_indices]
            attenuations = attenuations[np.isfinite(attenuations)]
            reflections = reflection_count[valid_indices]
            reflections = reflections[np.isfinite(reflections)]
            tracks.append(
                {
                    "name": label,
                    "target_kind": cls.target_kind(label),
                    "center": center,
                    "minimum": np.min(selected, axis=0).astype(np.float32),
                    "maximum": np.max(selected, axis=0).astype(np.float32),
                    "distance_m": float(np.min(distances)) if distances.size else 0.0,
                    "attenuation_db": float(np.max(attenuations)) if attenuations.size else 0.0,
                    "reflection_count": int(np.max(reflections)) if reflections.size else 0,
                    "point_count": int(len(selected)),
                }
            )
        return tracks

    @staticmethod
    def obstacle_points(
        tracks: Iterable[dict[str, object]],
        padding_m: float = 3.0,
    ) -> np.ndarray:
        """Build padded sparse boxes around Radar-confirmed enemy tracks.

        Radar로 확인한 적 표적 주위에 여유가 포함된 희소 박스를 만듭니다.
        """
        clouds: list[np.ndarray] = []
        for track in tracks:
            center = np.asarray(track.get("center", []), dtype=np.float32)
            minimum = np.asarray(track.get("minimum", center), dtype=np.float32)
            maximum = np.asarray(track.get("maximum", center), dtype=np.float32)
            if center.shape != (3,) or minimum.shape != (3,) or maximum.shape != (3,):
                continue
            lower = np.minimum(minimum, maximum) - float(padding_m)
            upper = np.maximum(minimum, maximum) + float(padding_m)
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
    def associate_camera_detections(
        detections: Iterable[dict],
        tracks: Iterable[dict[str, object]],
        maximum_separation_m: float = 10.0,
    ) -> list[dict]:
        """Mark camera boxes whose world position agrees with a Radar track.

        카메라 객체 위치와 Radar 표적 위치가 일치하면 센서 융합 표식을 추가합니다.
        """
        track_list = list(tracks)
        results: list[dict] = []
        for original in detections:
            detection = dict(original)
            required = (
                "world_x_min",
                "world_y_min",
                "world_z_min",
                "world_x_max",
                "world_y_max",
                "world_z_max",
            )
            camera_center: np.ndarray | None = None
            if all(key in detection for key in required):
                camera_center = np.asarray(
                    [
                        (float(detection["world_x_min"]) + float(detection["world_x_max"])) * 0.5,
                        (float(detection["world_y_min"]) + float(detection["world_y_max"])) * 0.5,
                        (float(detection["world_z_min"]) + float(detection["world_z_max"])) * 0.5,
                    ],
                    dtype=np.float32,
                )

            best_track: dict[str, object] | None = None
            best_separation = float("inf")
            for track in track_list:
                center = np.asarray(track.get("center", []), dtype=np.float32)
                if center.shape != (3,):
                    continue
                if camera_center is None:
                    if len(track_list) == 1:
                        best_track = track
                    continue
                separation = float(np.linalg.norm(center - camera_center))
                if separation < best_separation:
                    best_separation = separation
                    best_track = track

            if best_track is not None and (
                camera_center is None or best_separation <= maximum_separation_m
            ):
                detection["radar_confirmed"] = True
                detection["radar_name"] = str(best_track.get("name", "Radar"))
                detection["radar_distance_m"] = float(best_track.get("distance_m", 0.0))
                detection["radar_attenuation_db"] = float(
                    best_track.get("attenuation_db", 0.0)
                )
            results.append(detection)
        return results

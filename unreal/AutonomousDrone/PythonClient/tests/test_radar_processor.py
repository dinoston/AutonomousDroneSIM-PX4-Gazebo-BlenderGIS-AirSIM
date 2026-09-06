"""Tests for Echo/Radar parsing, tracking, and camera association."""

from __future__ import annotations

import numpy as np

from perception.radar_processor import RadarProcessor


def test_parse_active_echo() -> None:
    parsed = RadarProcessor.parse_active_echo(
        [1.0, 2.0, 3.0, -4.0, 5.0, 1.0, 4.0, 5.0, 6.0, -7.0, 8.0, 2.0]
    )
    assert parsed["points"].shape == (2, 3)
    assert parsed["total_distance"].tolist() == [5.0, 8.0]
    assert parsed["reflection_count"].tolist() == [1.0, 2.0]


def test_enemy_tracks_and_obstacle_box() -> None:
    world = np.asarray(
        [[10.0, 1.0, -5.0], [10.5, 1.5, -4.5], [3.0, 9.0, 0.0]],
        dtype=np.float32,
    )
    tracks = RadarProcessor.build_enemy_tracks(
        world,
        ["BP_EnemyDrone_C_0", "BP_EnemyDrone_C_0", "GroundPlane"],
        np.asarray([11.0, 11.5, 9.5], dtype=np.float32),
        np.asarray([-3.0, -4.0, -1.0], dtype=np.float32),
        np.asarray([1.0, 2.0, 1.0], dtype=np.float32),
    )
    assert len(tracks) == 1
    assert tracks[0]["name"] == "BP_EnemyDrone_C_0"
    assert tracks[0]["point_count"] == 2
    assert tracks[0]["distance_m"] == 11.0
    obstacles = RadarProcessor.obstacle_points(tracks, padding_m=3.0)
    assert obstacles.shape == (64, 3)


def test_radar_camera_association() -> None:
    detections = [
        {
            "name": "BP_EnemyDrone_C_0",
            "world_x_min": 9.0,
            "world_y_min": 0.0,
            "world_z_min": -6.0,
            "world_x_max": 11.0,
            "world_y_max": 2.0,
            "world_z_max": -4.0,
        }
    ]
    tracks = [
        {
            "name": "BP_EnemyDrone_C_0",
            "center": np.asarray([10.0, 1.0, -5.0], dtype=np.float32),
            "distance_m": 10.5,
            "attenuation_db": -3.0,
        }
    ]
    fused = RadarProcessor.associate_camera_detections(detections, tracks)
    assert fused[0]["radar_confirmed"] is True
    assert fused[0]["radar_distance_m"] == 10.5


def test_non_enemy_label_is_not_a_track() -> None:
    tracks = RadarProcessor.build_enemy_tracks(
        np.asarray([[1.0, 2.0, 3.0]], dtype=np.float32),
        ["SM_Hangar2"],
        np.asarray([4.0], dtype=np.float32),
        np.asarray([-1.0], dtype=np.float32),
        np.asarray([1.0], dtype=np.float32),
    )
    assert tracks == []


def test_human_radar_returns_build_a_human_track() -> None:
    tracks = RadarProcessor.build_enemy_tracks(
        np.asarray([[12.0, 1.0, -2.0], [12.2, 1.1, -1.4]], dtype=np.float32),
        ["BP_AINormalPeople_Drone_C_0", "BP_AINormalPeople_Drone_C_0"],
        np.asarray([12.2, 12.4], dtype=np.float32),
        np.asarray([-5.0, -5.5], dtype=np.float32),
        np.asarray([1.0, 1.0], dtype=np.float32),
    )

    assert len(tracks) == 1
    assert tracks[0]["target_kind"] == "human"


def test_flock_radar_returns_build_a_bird_track() -> None:
    tracks = RadarProcessor.build_enemy_tracks(
        np.asarray([[90.0, 3.0, -20.0], [90.2, 3.1, -19.8]], dtype=np.float32),
        ["FlockCharacter_4", "FlockCharacter_4"],
        np.asarray([92.0, 92.2], dtype=np.float32),
        np.asarray([-12.0, -12.5], dtype=np.float32),
        np.asarray([1.0, 1.0], dtype=np.float32),
    )

    assert len(tracks) == 1
    assert tracks[0]["target_kind"] == "bird"

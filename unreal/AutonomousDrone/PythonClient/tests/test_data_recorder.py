"""Tests for synchronized dataset recording."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from data_collection.recorder import DataRecorder, RecordingConfig


def _config(output_root: Path) -> RecordingConfig:
    return RecordingConfig(
        output_root=output_root,
        dataset_name="Korea Drone Dataset",
        city="Seoul",
        region="Gangnam",
        terrain_type="dense_urban",
        sensors=(
            "rgb",
            "depth",
            "segmentation",
            "lidar",
            "radar",
            "telemetry",
            "annotations",
        ),
        sample_rate_hz=2.0,
    )


def test_recorder_writes_synchronized_frame(tmp_path: Path) -> None:
    recorder = DataRecorder(queue_capacity=4)
    session_dir = recorder.start(_config(tmp_path))
    recorder.update_telemetry({"timestamp": 101, "x": 1.0, "y": 2.0})
    recorder.update_lidar(
        np.asarray([[1.0, 2.0, 3.0]], dtype=np.float32),
        {"timestamp": 102, "x": 1.0, "y": 2.0, "z": -5.0},
    )
    recorder.update_radar(
        {
            "timestamp": 103,
            "points": np.asarray([[4.0, 5.0, 6.0]], dtype=np.float32),
            "labels": ["BP_AINormalPeople_Drone_C_1"],
        }
    )

    assert recorder.capture(
        {
            "RGB": b"rgb-png",
            "Depth": b"depth-png",
            "Segmentation": b"seg-png",
            "_camera_timestamps": {"RGB": 100},
            "_camera_sizes": {"RGB": {"width": 1280, "height": 720}},
        },
        [{"label": "human", "box_2d": [1, 2, 3, 4]}],
        {"type": "central_patrol", "route_index": 2},
    )
    recorder.stop()

    assert (session_dir / "rgb" / "000000.png").read_bytes() == b"rgb-png"
    assert (session_dir / "depth" / "000000.png").read_bytes() == b"depth-png"
    assert (session_dir / "segmentation" / "000000.png").read_bytes() == b"seg-png"
    assert (session_dir / "lidar" / "000000.npz").exists()
    assert (session_dir / "radar" / "000000.npz").exists()
    assert (session_dir / "class_map.json").exists()

    frame = json.loads((session_dir / "frames.jsonl").read_text(encoding="utf-8"))
    assert frame["frame_id"] == 0
    assert frame["telemetry"]["timestamp"] == 101
    assert frame["lidar"]["pose"]["timestamp"] == 102
    assert frame["radar"]["timestamp"] == 103
    assert frame["detections"][0]["label"] == "human"
    assert frame["mission"]["type"] == "central_patrol"

    session = json.loads((session_dir / "session.json").read_text(encoding="utf-8"))
    assert session["status"] == "complete"
    assert session["written_frames"] == 1
    assert session["dropped_frames"] == 0


def test_paused_recorder_does_not_enqueue_frames(tmp_path: Path) -> None:
    recorder = DataRecorder()
    recorder.start(_config(tmp_path))
    recorder.pause()
    assert not recorder.capture({"RGB": b"ignored"}, [], {})
    recorder.stop()
    assert recorder.stats()["written_frames"] == 0

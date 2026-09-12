"""Non-blocking synchronized sensor recorder for simulated drone datasets."""

from __future__ import annotations

import json
import queue
import re
import shutil
import threading
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from perception.segmentation_labels import DEFAULT_CLASS_MAP_PATH


IMAGE_SENSOR_KEYS = {
    "rgb": "RGB",
    "depth": "Depth",
    "segmentation": "Segmentation",
}


def _safe_name(value: str, fallback: str) -> str:
    cleaned = re.sub(r"[^0-9A-Za-z가-힣_-]+", "_", str(value).strip())
    return cleaned.strip("_") or fallback


def _json_safe(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


@dataclass(frozen=True)
class RecordingConfig:
    output_root: Path
    dataset_name: str
    city: str
    region: str
    terrain_type: str
    sensors: tuple[str, ...]
    sample_rate_hz: float = 2.0
    season: str = "summer"
    time_of_day: str = "noon"
    visibility: str = "clear"
    precipitation: str = "none"
    precipitation_intensity: float = 0.0
    wind_north_mps: float = 0.0
    wind_east_mps: float = 0.0

    def normalized(self) -> "RecordingConfig":
        allowed = {*IMAGE_SENSOR_KEYS, "lidar", "radar", "telemetry", "annotations"}
        selected = tuple(sensor for sensor in self.sensors if sensor in allowed)
        if not selected:
            raise ValueError("수집할 센서를 하나 이상 선택하세요.")
        return RecordingConfig(
            output_root=Path(self.output_root).expanduser(),
            dataset_name=_safe_name(self.dataset_name, "drone_dataset"),
            city=_safe_name(self.city, "unknown_city"),
            region=_safe_name(self.region, "unknown_region"),
            terrain_type=_safe_name(self.terrain_type, "unknown_terrain"),
            sensors=selected,
            sample_rate_hz=min(10.0, max(0.5, float(self.sample_rate_hz))),
            season=(
                self.season
                if self.season in {"spring", "summer", "autumn", "winter"}
                else "summer"
            ),
            time_of_day=(
                self.time_of_day
                if self.time_of_day
                in {"morning", "noon", "evening", "midnight", "day", "night"}
                else "noon"
            ),
            visibility=(
                self.visibility
                if self.visibility in {"clear", "cloudy", "fog"}
                else "clear"
            ),
            precipitation=(
                self.precipitation
                if self.precipitation in {"none", "rain", "snow"}
                else "none"
            ),
            precipitation_intensity=min(
                1.0,
                max(
                    0.0,
                    float(self.precipitation_intensity)
                    if self.precipitation in {"rain", "snow"}
                    else 0.0,
                ),
            ),
            wind_north_mps=min(30.0, max(-30.0, float(self.wind_north_mps))),
            wind_east_mps=min(30.0, max(-30.0, float(self.wind_east_mps))),
        )


class DataRecorder:
    """Capture the latest sensor samples on camera-frame boundaries.

    File writes happen on a dedicated thread so a slow disk cannot stall the
    Qt UI or flight-safety callbacks.
    """

    def __init__(self, queue_capacity: int = 64) -> None:
        self._lock = threading.Lock()
        self._queue: queue.Queue[dict[str, Any] | None] = queue.Queue(
            maxsize=max(4, int(queue_capacity))
        )
        self._writer: threading.Thread | None = None
        self._state = "idle"
        self._config: RecordingConfig | None = None
        self._session_dir: Path | None = None
        self._session_metadata: dict[str, Any] = {}
        self._next_frame_id = 0
        self._written_frames = 0
        self._dropped_frames = 0
        self._bytes_written = 0
        self._started_monotonic = 0.0
        self._ended_elapsed = 0.0
        self._last_error = ""
        self._latest_telemetry: dict[str, Any] = {}
        self._latest_lidar: tuple[np.ndarray, dict[str, Any]] | None = None
        self._latest_radar: dict[str, Any] | None = None

    @property
    def state(self) -> str:
        with self._lock:
            return self._state

    @property
    def session_dir(self) -> Path | None:
        with self._lock:
            return self._session_dir

    def start(self, config: RecordingConfig) -> Path:
        normalized = config.normalized()
        with self._lock:
            if self._state in {"recording", "paused", "stopping"}:
                raise RuntimeError("이미 데이터 수집 세션이 진행 중입니다.")

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        session_name = f"{normalized.city}_{normalized.region}_{timestamp}"
        session_dir = normalized.output_root / normalized.dataset_name / session_name
        suffix = 1
        while session_dir.exists():
            session_dir = normalized.output_root / normalized.dataset_name / (
                f"{session_name}_{suffix:02d}"
            )
            suffix += 1
        session_dir.mkdir(parents=True, exist_ok=False)
        for sensor in normalized.sensors:
            if sensor in {*IMAGE_SENSOR_KEYS, "lidar", "radar"}:
                (session_dir / sensor).mkdir()
        shutil.copy2(DEFAULT_CLASS_MAP_PATH, session_dir / "class_map.json")

        started_at = datetime.now(timezone.utc).isoformat()
        metadata = {
            "schema_version": 1,
            "status": "recording",
            "started_at_utc": started_at,
            "ended_at_utc": None,
            "config": _json_safe(asdict(normalized)),
            "coordinate_system": "AirSim local NED",
            "frame_manifest": "frames.jsonl",
            "class_map": "class_map.json",
        }
        (session_dir / "session.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        with self._lock:
            self._queue = queue.Queue(maxsize=self._queue.maxsize)
            self._config = normalized
            self._session_dir = session_dir
            self._session_metadata = metadata
            self._next_frame_id = 0
            self._written_frames = 0
            self._dropped_frames = 0
            self._bytes_written = 0
            self._started_monotonic = time.monotonic()
            self._ended_elapsed = 0.0
            self._last_error = ""
            self._state = "recording"
            self._writer = threading.Thread(
                target=self._writer_loop,
                name="DroneDatasetWriter",
                daemon=True,
            )
            self._writer.start()
        return session_dir

    def pause(self) -> None:
        with self._lock:
            if self._state != "recording":
                raise RuntimeError("진행 중인 데이터 수집이 없습니다.")
            self._state = "paused"

    def resume(self) -> None:
        with self._lock:
            if self._state != "paused":
                raise RuntimeError("일시정지된 데이터 수집이 없습니다.")
            self._state = "recording"

    def stop(self, timeout_seconds: float = 8.0) -> None:
        with self._lock:
            if self._state not in {"recording", "paused", "error"}:
                return
            self._state = "stopping"
            writer = self._writer
        try:
            self._queue.put(None, timeout=1.0)
        except queue.Full:
            with self._lock:
                self._last_error = "저장 큐가 가득 차 종료 신호를 보내지 못했습니다."
                self._state = "error"
            return
        if writer is not None:
            writer.join(timeout=max(0.1, float(timeout_seconds)))
        with self._lock:
            self._ended_elapsed = max(
                self._ended_elapsed,
                time.monotonic() - self._started_monotonic,
            )
            if writer is not None and writer.is_alive():
                self._last_error = "저장 작업이 제한 시간 안에 끝나지 않았습니다."
                self._state = "error"
            elif self._last_error:
                self._state = "error"
            elif self._state != "error":
                self._state = "stopped"

    def update_telemetry(self, data: dict[str, Any]) -> None:
        with self._lock:
            self._latest_telemetry = dict(data)

    def update_lidar(self, points: np.ndarray, pose: dict[str, Any]) -> None:
        with self._lock:
            self._latest_lidar = (
                np.asarray(points, dtype=np.float32).copy(),
                dict(pose),
            )

    def update_radar(self, snapshot: dict[str, Any]) -> None:
        copied: dict[str, Any] = {}
        for key, value in snapshot.items():
            copied[key] = value.copy() if isinstance(value, np.ndarray) else _json_safe(value)
        with self._lock:
            self._latest_radar = copied

    def capture(
        self,
        images: dict[str, Any],
        detections: list[dict[str, Any]],
        mission: dict[str, Any],
    ) -> bool:
        with self._lock:
            if self._state != "recording" or self._config is None:
                return False
            config = self._config
            frame_id = self._next_frame_id
            self._next_frame_id += 1
            telemetry = dict(self._latest_telemetry)
            lidar = self._latest_lidar
            radar = self._latest_radar

        image_payload = {
            sensor: bytes(images[source_key])
            for sensor, source_key in IMAGE_SENSOR_KEYS.items()
            if sensor in config.sensors and isinstance(images.get(source_key), bytes)
        }
        packet = {
            "frame_id": frame_id,
            "captured_at_utc": datetime.now(timezone.utc).isoformat(),
            "camera_timestamps": _json_safe(images.get("_camera_timestamps", {})),
            "camera_sizes": _json_safe(images.get("_camera_sizes", {})),
            "images": image_payload,
            "detections": _json_safe(detections) if "annotations" in config.sensors else [],
            "mission": _json_safe(mission),
            "telemetry": _json_safe(telemetry) if "telemetry" in config.sensors else {},
            "lidar": (
                (lidar[0].copy(), dict(lidar[1]))
                if "lidar" in config.sensors and lidar is not None
                else None
            ),
            "radar": (
                {
                    key: value.copy() if isinstance(value, np.ndarray) else _json_safe(value)
                    for key, value in radar.items()
                }
                if "radar" in config.sensors and radar is not None
                else None
            ),
        }
        try:
            self._queue.put_nowait(packet)
            return True
        except queue.Full:
            with self._lock:
                self._dropped_frames += 1
            return False

    def stats(self) -> dict[str, Any]:
        with self._lock:
            active = self._state in {"recording", "paused", "stopping"}
            elapsed = (
                max(0.0, time.monotonic() - self._started_monotonic)
                if self._started_monotonic and active
                else self._ended_elapsed
            )
            return {
                "state": self._state,
                "session_dir": str(self._session_dir or ""),
                "written_frames": self._written_frames,
                "queued_frames": self._queue.qsize(),
                "dropped_frames": self._dropped_frames,
                "bytes_written": self._bytes_written,
                "elapsed_seconds": elapsed,
                "last_error": self._last_error,
            }

    def _writer_loop(self) -> None:
        session_dir = self.session_dir
        if session_dir is None:
            return
        manifest_path = session_dir / "frames.jsonl"
        try:
            with manifest_path.open("a", encoding="utf-8", newline="\n") as manifest:
                while True:
                    packet = self._queue.get()
                    if packet is None:
                        break
                    annotation, written_bytes = self._write_packet(session_dir, packet)
                    line = json.dumps(annotation, ensure_ascii=False, separators=(",", ":"))
                    manifest.write(line + "\n")
                    manifest.flush()
                    written_bytes += len((line + "\n").encode("utf-8"))
                    with self._lock:
                        self._written_frames += 1
                        self._bytes_written += written_bytes
        except Exception as exc:
            with self._lock:
                self._last_error = str(exc)
                self._state = "error"
        finally:
            self._finish_session_file()

    def _write_packet(
        self,
        session_dir: Path,
        packet: dict[str, Any],
    ) -> tuple[dict[str, Any], int]:
        frame_name = f"{int(packet['frame_id']):06d}"
        files: dict[str, str] = {}
        written_bytes = 0
        for sensor, data in packet["images"].items():
            relative = Path(sensor) / f"{frame_name}.png"
            destination = session_dir / relative
            destination.write_bytes(data)
            files[sensor] = relative.as_posix()
            written_bytes += destination.stat().st_size

        lidar = packet.get("lidar")
        lidar_metadata: dict[str, Any] = {}
        if lidar is not None:
            points, pose = lidar
            relative = Path("lidar") / f"{frame_name}.npz"
            destination = session_dir / relative
            np.savez_compressed(destination, points=np.asarray(points, dtype=np.float32))
            files["lidar"] = relative.as_posix()
            lidar_metadata = {"pose": _json_safe(pose), "point_count": int(len(points))}
            written_bytes += destination.stat().st_size

        radar = packet.get("radar")
        radar_metadata: dict[str, Any] = {}
        if radar is not None:
            relative = Path("radar") / f"{frame_name}.npz"
            destination = session_dir / relative
            arrays = {
                key: value
                for key, value in radar.items()
                if isinstance(value, np.ndarray)
            }
            np.savez_compressed(destination, **arrays)
            files["radar"] = relative.as_posix()
            radar_metadata = {
                key: _json_safe(value)
                for key, value in radar.items()
                if not isinstance(value, np.ndarray)
            }
            radar_metadata["point_count"] = int(
                len(np.asarray(radar.get("points", [])))
            )
            written_bytes += destination.stat().st_size

        annotation = {
            "frame_id": int(packet["frame_id"]),
            "captured_at_utc": packet["captured_at_utc"],
            "camera_timestamps": packet["camera_timestamps"],
            "camera_sizes": packet["camera_sizes"],
            "files": files,
            "telemetry": packet["telemetry"],
            "mission": packet["mission"],
            "detections": packet["detections"],
            "lidar": lidar_metadata,
            "radar": radar_metadata,
        }
        return annotation, written_bytes

    def _finish_session_file(self) -> None:
        with self._lock:
            session_dir = self._session_dir
            metadata = dict(self._session_metadata)
            metadata.update(
                {
                    "status": "error" if self._last_error else "complete",
                    "ended_at_utc": datetime.now(timezone.utc).isoformat(),
                    "written_frames": self._written_frames,
                    "dropped_frames": self._dropped_frames,
                    "bytes_written": self._bytes_written,
                    "error": self._last_error or None,
                }
            )
        if session_dir is not None:
            (session_dir / "session.json").write_text(
                json.dumps(metadata, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

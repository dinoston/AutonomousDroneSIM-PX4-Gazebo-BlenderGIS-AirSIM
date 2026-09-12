"""Cosys-AirSim mission-control desktop application."""

from __future__ import annotations

import itertools
import json
import math
import queue
import sys
import threading
import time
from pathlib import Path

PYTHON_CLIENT_ROOT = Path(__file__).resolve().parents[1]
if str(PYTHON_CLIENT_ROOT) not in sys.path:
    sys.path.insert(0, str(PYTHON_CLIENT_ROOT))

import numpy as np
from PySide6.QtCore import QSettings, QThread, Qt, Signal
from PySide6.QtGui import QCloseEvent
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QPlainTextEdit,
    QScrollArea,
    QSplitter,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from common.airsim_client import AirSimController
from data_collection.recorder import DataRecorder, RecordingConfig
from data_collection.report import generate_session_report
from navigation.grid_planner import (
    AltitudeGridPlanner,
    PlannerConfig,
    build_vertical_barrier,
    split_terminal_vertical_leg,
)
from perception.camera_viewer import CameraViewer
from perception.lidar_viewer import LidarViewer
from perception.radar_processor import RadarProcessor
from perception.target_detector import TargetDetector
from ui.minimap import MiniMapWidget


class AirSimWorker(QThread):
    connection_changed = Signal(bool, str)
    status_changed = Signal(str)
    semantic_setup_completed = Signal(str)
    semantic_setup_failed = Signal(str)
    telemetry_updated = Signal(dict)
    images_updated = Signal(dict)
    lidar_updated = Signal(object)
    radar_updated = Signal(object)
    environment_applied = Signal(dict)
    command_completed = Signal(str)
    error_occurred = Signal(str)

    def __init__(self) -> None:
        super().__init__()
        self.controller = AirSimController()
        self._commands: queue.PriorityQueue = queue.PriorityQueue()
        self._sequence = itertools.count()
        self._running = True
        self._stream_sensors = True
        self._lidar_enabled = True
        self._radar_enabled = True
        self._sensor_ray_debug_enabled = False
        self._camera_interval_s = 0.5
        self._last_sensor_error = 0.0
        self._suspend_polling = False
        self._semantic_setup_running = False
        self._environment_apply_running = False
        self._queued_environment_args: tuple | None = None
        self._environment_apply_lock = threading.Lock()

    def submit(self, name: str, *args, priority: int = 10) -> None:
        self._commands.put((priority, next(self._sequence), name, args))

    def request_connect(self) -> None:
        self.submit("connect", priority=-10)

    def request_disconnect(self) -> None:
        # Prevent another sensor RPC from starting after the user requests a
        # disconnect. The current RPC finishes, then the priority command runs.
        # 연결 해제 요청 뒤에는 새 센서 RPC를 시작하지 않고, 현재 호출이
        # 끝나는 즉시 우선순위가 높은 해제 명령을 처리합니다.
        self._suspend_polling = True
        self.submit("disconnect", priority=-10)

    def _start_semantic_setup(self) -> None:
        """Apply the level class map without blocking flight commands."""
        if self._semantic_setup_running:
            self.status_changed.emit("Segmentation 클래스 적용이 이미 진행 중입니다.")
            return
        self._semantic_setup_running = True
        host = self.controller.host
        port = self.controller.port

        def configure() -> None:
            try:
                report = AirSimController.build_segmentation_report(host, port)
                if self.controller.connected:
                    self.controller.apply_segmentation_report(report)
                    self.semantic_setup_completed.emit(
                        self.controller.segmentation_status
                    )
            except Exception as exc:
                if self.controller.connected:
                    self.controller.apply_segmentation_error(str(exc))
                    self.semantic_setup_failed.emit(str(exc))
            finally:
                self._semantic_setup_running = False

        threading.Thread(
            target=configure,
            name="AirSimSemanticSetup",
            daemon=True,
        ).start()

    def _start_environment_apply(self, args: tuple) -> None:
        """Apply only the newest requested preset without blocking flight RPCs."""
        with self._environment_apply_lock:
            if self._environment_apply_running:
                self._queued_environment_args = args
                self.status_changed.emit(
                    "현재 환경 적용 뒤에 방금 선택한 환경을 이어서 적용합니다."
                )
                return
            self._environment_apply_running = True
        host = self.controller.host
        port = self.controller.port

        def configure(first_args: tuple) -> None:
            current_args = first_args
            while self.controller.connected:
                try:
                    values = AirSimController.apply_environment_once(
                        host, port, *current_args
                    )
                    if self.controller.connected:
                        self.environment_applied.emit(values)
                except Exception as exc:
                    if self.controller.connected:
                        self.error_occurred.emit(f"environment: {exc}")
                with self._environment_apply_lock:
                    next_args = self._queued_environment_args
                    self._queued_environment_args = None
                    if next_args is None:
                        self._environment_apply_running = False
                        return
                    current_args = next_args
            with self._environment_apply_lock:
                self._queued_environment_args = None
                self._environment_apply_running = False

        threading.Thread(
            target=configure,
            args=(args,),
            name="AirSimEnvironmentApply",
            daemon=True,
        ).start()

    def stop(self) -> None:
        self._running = False

    def set_sensor_streaming(self, enabled: bool) -> None:
        self._stream_sensors = enabled

    def set_camera_rate_hz(self, rate_hz: float) -> None:
        self._camera_interval_s = 1.0 / min(10.0, max(0.5, float(rate_hz)))

    def set_lidar_enabled(self, enabled: bool) -> None:
        self._lidar_enabled = bool(enabled)
        if self.controller.connected:
            self.submit("sensor_debug", "lidar", self._lidar_enabled, priority=2)

    def set_radar_enabled(self, enabled: bool) -> None:
        self._radar_enabled = bool(enabled)
        if self.controller.connected:
            self.submit("sensor_debug", "radar", self._radar_enabled, priority=2)

    def set_sensor_ray_debug_enabled(self, enabled: bool) -> None:
        self._sensor_ray_debug_enabled = bool(enabled)
        if self.controller.connected:
            self.submit(
                "sensor_debug",
                "sensor_ray",
                self._sensor_ray_debug_enabled,
                priority=2,
            )

    def discard_pending_navigation(self) -> None:
        """Remove queued movement commands before a stop or landing command."""
        retained: list[tuple] = []
        navigation_commands = {
            "move",
            "path",
            "recovery_path",
            "takeoff",
            "land",
        }
        while True:
            try:
                command = self._commands.get_nowait()
            except queue.Empty:
                break
            if command[2] not in navigation_commands:
                retained.append(command)
        for command in retained:
            self._commands.put(command)

    def run(self) -> None:
        next_telemetry = 0.0
        next_images = 0.0
        next_lidar = 0.0
        next_radar = 0.0
        while self._running:
            self._process_pending_commands()
            now = time.monotonic()
            if (
                self.controller.connected
                and not self._suspend_polling
                and now >= next_telemetry
            ):
                self._poll_telemetry()
                next_telemetry = now + 0.1
            # LiDAR is a flight-safety input and must keep running even when
            # the optional camera preview stream is disabled.
            # LiDAR는 비행 안전에 필요한 입력이므로 선택형 카메라 미리보기
            # 스트리밍이 꺼져 있어도 계속 작동해야 합니다.
            if (
                self.controller.connected
                and not self._suspend_polling
                and self._lidar_enabled
                and now >= next_lidar
            ):
                self._poll_lidar()
                next_lidar = now + 0.15
            # Radar is polled independently so either ranging sensor can be
            # enabled without forcing the other sensor to run.
            # 두 거리 센서를 독립적으로 켜고 끌 수 있도록 Radar는 별도
            # 주기로 수신합니다.
            if (
                self.controller.connected
                and not self._suspend_polling
                and self._radar_enabled
                and now >= next_radar
            ):
                self._poll_radar()
                next_radar = now + 0.1
            if (
                self.controller.connected
                and not self._suspend_polling
                and self._stream_sensors
                and now >= next_images
            ):
                self._poll_images()
                next_images = now + self._camera_interval_s
            self.msleep(20)
        self.controller.disconnect()

    def _process_pending_commands(self) -> None:
        for _ in range(5):
            try:
                _, _, name, args = self._commands.get_nowait()
            except queue.Empty:
                return
            try:
                if name == "connect":
                    self._suspend_polling = True
                    self.controller.connect()
                    self.status_changed.emit("AirSim RPC 연결됨 · 센서 설정 중…")
                    # Reapply both default-on UI states after each connection.
                    # 연결할 때마다 기본 ON인 두 UI 상태를 언리얼 표시에 동기화합니다.
                    self.controller.set_sensor_debug_visualization(
                        "lidar", self._lidar_enabled
                    )
                    self.controller.set_sensor_debug_visualization(
                        "radar", self._radar_enabled
                    )
                    self.controller.set_sensor_debug_visualization(
                        "sensor_ray", self._sensor_ray_debug_enabled
                    )
                    self._suspend_polling = False
                    self.connection_changed.emit(
                        True,
                        "연결됨 · 인식 클래스 준비 중…",
                    )
                    self._start_semantic_setup()
                elif name == "disconnect":
                    self.controller.disconnect()
                    self._suspend_polling = False
                    self.connection_changed.emit(False, "연결 해제")
                elif name == "arm":
                    self.controller.arm(bool(args[0]))
                elif name == "spawn":
                    self.controller.set_spawn(float(args[0]), float(args[1]))
                elif name == "takeoff":
                    self.controller.takeoff(float(args[0]))
                elif name == "hover":
                    self.controller.hover()
                elif name == "move":
                    self.controller.move_to(*args)
                elif name == "path":
                    self.controller.move_path(*args)
                elif name == "recovery_path":
                    self.controller.recover_and_move_path(*args)
                elif name == "land":
                    self.controller.land(*args)
                elif name == "mission_stop":
                    self.controller.emergency_stop()
                elif name == "emergency":
                    self.controller.emergency_stop()
                elif name == "sensor_debug":
                    self.controller.set_sensor_debug_visualization(
                        str(args[0]), bool(args[1])
                    )
                elif name == "environment":
                    self._start_environment_apply(args)
                elif name == "segmentation":
                    self._start_semantic_setup()
                else:
                    raise ValueError(f"알 수 없는 명령: {name}")
                if name not in {
                    "connect",
                    "disconnect",
                    "segmentation",
                    "environment",
                }:
                    self.command_completed.emit(name)
            except Exception as exc:
                if name in {"connect", "disconnect"}:
                    self._suspend_polling = False
                if name == "connect":
                    self.connection_changed.emit(False, "연결 실패")
                elif name == "disconnect":
                    self.connection_changed.emit(
                        self.controller.connected,
                        "연결 해제 실패",
                    )
                self.error_occurred.emit(f"{name}: {exc}")

    def _poll_telemetry(self) -> None:
        try:
            self.telemetry_updated.emit(self.controller.telemetry())
        except Exception as exc:
            self.connection_changed.emit(False, "통신 오류")
            self.error_occurred.emit(f"텔레메트리: {exc}")
            self.controller.disconnect()

    def _poll_images(self) -> None:
        try:
            self.images_updated.emit(self.controller.camera_images())
        except Exception as exc:
            now = time.monotonic()
            if now - self._last_sensor_error > 5.0:
                self.error_occurred.emit(f"카메라: {exc}")
                self._last_sensor_error = now

    def _poll_lidar(self) -> None:
        try:
            self.lidar_updated.emit(self.controller.lidar_snapshot())
        except Exception as exc:
            now = time.monotonic()
            if now - self._last_sensor_error > 5.0:
                self.error_occurred.emit(f"LiDAR: {exc}")
                self._last_sensor_error = now

    def _poll_radar(self) -> None:
        try:
            self.radar_updated.emit(self.controller.radar_snapshot())
        except Exception as exc:
            now = time.monotonic()
            if now - self._last_sensor_error > 5.0:
                self.error_occurred.emit(f"Radar: {exc}")
                self._last_sensor_error = now


class ReportWorker(QThread):
    """Build CSV summaries and a PDF without blocking the flight UI."""

    completed = Signal(str, object)
    failed = Signal(str)

    def __init__(self, session_dir: Path) -> None:
        super().__init__()
        self.session_dir = Path(session_dir)

    def run(self) -> None:
        try:
            result = generate_session_report(self.session_dir)
            self.completed.emit(str(result.pdf_path), result.summary)
        except Exception as exc:
            self.failed.emit(str(exc))


class MissionControlWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle(
            "Autonomous Drone Mission Control · Semantic/Data v1 · Avoidance v9"
        )
        self.resize(1500, 920)
        self.settings = QSettings("AutonomousDrone", "MissionControl")
        self.recorder = DataRecorder()
        self._report_worker: ReportWorker | None = None
        self._connected = False
        self._connection_transition: str | None = None
        self._semantic_ready = False
        self._takeoff_pending = False
        self._environment_apply_pending = False
        self._environment_requested: dict[str, object] = {}
        self._environment_state: dict[str, object] = {
            "season": "summer",
            "time_of_day": "noon",
            "visibility": "clear",
            "precipitation": "none",
            "precipitation_intensity": 0.0,
            "wind_north_mps": 0.0,
            "wind_east_mps": 0.0,
        }
        # The AirBase capture covers 1400 m. A 2.5 m planning grid keeps
        # full-map A* practical while retaining useful obstacle clearance.
        # AirBase 캡처 범위는 1400m이며, 2.5m 격자를 사용해 전체 지도 A*의
        # 계산량을 줄이면서 필요한 장애물 안전거리를 유지합니다.
        self.planner = AltitudeGridPlanner(
            PlannerConfig(
                half_extent_m=900.0,
                resolution_m=2.5,
                drone_radius_m=2.5,
                vertical_clearance_m=1.5,
                # LiDAR may select a higher layer in 2 m steps. The 8 m limit
                # allows a gentle climb without permitting an excessive escape.
                # LiDAR가 2m 간격의 상위 고도층을 선택할 수 있습니다. 최대 8m로
                # 제한하여 과도하게 상승하지 않고 완만하게 장애물을 넘습니다.
                altitude_step_m=2.0,
                max_extra_altitude_m=8.0,
            )
        )
        self._telemetry: dict | None = None
        self._planned_path: list[tuple[float, float, float]] = []
        self._active_target: tuple[float, float, float] | None = None
        self._route_waypoints: list[tuple[float, float, float]] = []
        self._route_queue: list[tuple[float, float, float]] = []
        self._route_running = False
        self._active_route_index: int | None = None
        self._autonomous_patrol_running = False
        self._autonomous_patrol_end_time = 0.0
        self._autonomous_patrol_visited = 0
        self._patrol_target_queue: list[tuple[float, float, float]] = []
        self._patrol_cycle_targets: list[tuple[float, float, float]] = []
        self._patrol_cycle_number = 0
        self._patrol_rng = np.random.default_rng()
        self._pending_descent_altitude: float | None = None
        self._pending_descent_safe_altitude: float | None = None
        self._pending_descent_commanded = False
        self._last_replan = 0.0
        self._last_emergency_stop = 0.0
        self._mission_stall_started = 0.0
        self._last_stall_recovery = 0.0
        self._obstacle_detection_count = 0
        self._avoidance_grace_until = 0.0
        self._last_collision_timestamp = 0.0
        self._avoidance_altitude_floor_m = 1.0
        self._avoidance_altitude_ceiling_m: float | None = None
        self._spawn_xy = (0.0, 0.0)
        self._latest_lidar_world = np.empty((0, 3), dtype=np.float32)
        self._radar_obstacle_points = np.empty((0, 3), dtype=np.float32)
        self._latest_radar_tracks: list[dict[str, object]] = []
        self._enemy_obstacle_points = np.empty((0, 3), dtype=np.float32)
        self._collision_obstacle_points = np.empty((0, 3), dtype=np.float32)
        self._last_enemy_seen = 0.0
        self._last_enemy_replan = 0.0
        self._enemy_detection_count = 0
        self._detection_error_reported = False
        self._last_collection_error = ""

        self.worker = AirSimWorker()
        self.worker.connection_changed.connect(self._on_connection_changed)
        self.worker.status_changed.connect(self._on_worker_status)
        self.worker.semantic_setup_completed.connect(
            self._on_semantic_setup_completed
        )
        self.worker.semantic_setup_failed.connect(self._on_semantic_setup_failed)
        self.worker.telemetry_updated.connect(self._on_telemetry)
        self.worker.images_updated.connect(self._on_images)
        self.worker.lidar_updated.connect(self._on_lidar)
        self.worker.radar_updated.connect(self._on_radar)
        self.worker.environment_applied.connect(self._on_environment_applied)
        self.worker.command_completed.connect(self._on_command_completed)
        self.worker.error_occurred.connect(self._on_error)
        self.worker.start()

        self._build_ui()
        self._apply_style()
        self._restore_settings()
        self._set_controls_enabled(False)

    def _build_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)

        header = QHBoxLayout()
        title = QLabel("AUTONOMOUS DRONE · MISSION CONTROL")
        title.setObjectName("title")
        self.status_indicator = QLabel("● 연결 안 됨")
        self.status_indicator.setObjectName("disconnected")
        self.connect_button = QPushButton("AirSim 연결")
        self.connect_button.clicked.connect(self._toggle_connection)
        self.sensor_checkbox = QCheckBox("카메라 화면 스트리밍")
        self.sensor_checkbox.setChecked(True)
        self.sensor_checkbox.toggled.connect(self.worker.set_sensor_streaming)
        self.lidar_checkbox = QCheckBox("LiDAR 사용")
        self.lidar_checkbox.setChecked(True)
        self.lidar_checkbox.toggled.connect(self.worker.set_lidar_enabled)
        self.lidar_checkbox.toggled.connect(self._on_lidar_toggled)
        self.radar_checkbox = QCheckBox("Radar 사용")
        self.radar_checkbox.setChecked(True)
        self.radar_checkbox.toggled.connect(self.worker.set_radar_enabled)
        self.radar_checkbox.toggled.connect(self._on_radar_toggled)
        self.sensor_ray_debug_checkbox = QCheckBox("센서 디버그 선")
        self.sensor_ray_debug_checkbox.setChecked(False)
        self.sensor_ray_debug_checkbox.toggled.connect(
            self.worker.set_sensor_ray_debug_enabled
        )
        header.addWidget(title)
        header.addStretch()
        header.addWidget(self.sensor_checkbox)
        header.addWidget(self.lidar_checkbox)
        header.addWidget(self.radar_checkbox)
        header.addWidget(self.sensor_ray_debug_checkbox)
        header.addWidget(self.status_indicator)
        header.addWidget(self.connect_button)
        root.addLayout(header)

        splitter = QSplitter()
        splitter.addWidget(self._build_control_panel())
        splitter.addWidget(self._build_sensor_panel())
        splitter.setSizes([380, 1120])
        root.addWidget(splitter)

        self.message_label = QLabel("Unreal에서 Play를 실행한 뒤 AirSim 연결을 누르세요.")
        self.message_label.setObjectName("message")
        root.addWidget(self.message_label)

    def _build_control_panel(self) -> QWidget:
        panel = QFrame()
        layout = QVBoxLayout(panel)

        flight_group = QGroupBox("비행 제어")
        flight_layout = QGridLayout(flight_group)
        self.arm_button = QPushButton("ARM")
        self.disarm_button = QPushButton("DISARM")
        self.takeoff_button = QPushButton("이륙")
        self.takeoff_button.setToolTip(
            "기존 이동 명령을 취소하고 API 제어·ARM을 확인한 뒤 설정 고도까지 이륙합니다."
        )
        self.hover_button = QPushButton("미션 중지·호버링")
        self.cancel_route_button = QPushButton("예약 목록 취소")
        self.land_button = QPushButton("착륙")
        self.emergency_button = QPushButton("긴급 정지")
        self.emergency_button.setObjectName("emergency")
        self.arm_button.clicked.connect(lambda: self.worker.submit("arm", True))
        self.disarm_button.clicked.connect(lambda: self.worker.submit("arm", False))
        self.takeoff_button.clicked.connect(self._takeoff)
        self.hover_button.clicked.connect(self._stop_active_mission)
        self.cancel_route_button.clicked.connect(self._clear_waypoints)
        self.land_button.clicked.connect(self._land)
        self.emergency_button.clicked.connect(self._emergency_stop)
        flight_layout.addWidget(self.arm_button, 0, 0)
        flight_layout.addWidget(self.disarm_button, 0, 1)
        flight_layout.addWidget(self.takeoff_button, 1, 0)
        flight_layout.addWidget(self.land_button, 1, 1)
        flight_layout.addWidget(self.hover_button, 2, 0)
        flight_layout.addWidget(self.cancel_route_button, 2, 1)
        flight_layout.addWidget(self.emergency_button, 3, 0, 1, 2)
        layout.addWidget(flight_group)

        destination_group = QGroupBox("목적지 · NED 기준")
        destination_form = QFormLayout(destination_group)
        self.takeoff_altitude = self._spinbox(1.0, 120.0, 5.0, " m")
        self.destination_x = self._spinbox(-2000.0, 2000.0, 10.0, " m")
        self.destination_y = self._spinbox(-2000.0, 2000.0, 0.0, " m")
        self.destination_altitude = self._spinbox(1.0, 120.0, 5.0, " m")
        self.speed = self._spinbox(0.2, 20.0, 3.0, " m/s")
        destination_form.addRow("이륙 고도", self.takeoff_altitude)
        destination_form.addRow("X · 전방/북쪽", self.destination_x)
        destination_form.addRow("Y · 오른쪽/동쪽", self.destination_y)
        destination_form.addRow("목적지 고도", self.destination_altitude)
        destination_form.addRow("이동 속도", self.speed)
        map_mode_row = QHBoxLayout()
        self.spawn_select_button = QPushButton("스폰 A 선택")
        self.target_select_button = QPushButton("경유지 선택")
        self.spawn_select_button.setCheckable(True)
        self.target_select_button.setCheckable(True)
        self.target_select_button.setChecked(True)
        self.spawn_select_button.clicked.connect(lambda: self._set_map_mode("spawn"))
        self.target_select_button.clicked.connect(lambda: self._set_map_mode("target"))
        map_mode_row.addWidget(self.spawn_select_button)
        map_mode_row.addWidget(self.target_select_button)
        destination_form.addRow("지도 클릭 모드", map_mode_row)
        self.apply_spawn_button = QPushButton("선택한 위치에 스폰 적용")
        self.apply_spawn_button.clicked.connect(self._apply_spawn)
        destination_form.addRow(self.apply_spawn_button)
        self.route_list = QListWidget()
        self.route_list.setMaximumHeight(86)
        self.route_list.setToolTip("드론이 B부터 표시된 순서대로 방문합니다.")
        destination_form.addRow("예약 경유지", self.route_list)
        route_edit_row = QHBoxLayout()
        self.add_waypoint_button = QPushButton("경유지 추가")
        self.remove_waypoint_button = QPushButton("선택 삭제")
        self.add_waypoint_button.clicked.connect(self._add_waypoint)
        self.remove_waypoint_button.clicked.connect(self._remove_selected_waypoint)
        route_edit_row.addWidget(self.add_waypoint_button)
        route_edit_row.addWidget(self.remove_waypoint_button)
        destination_form.addRow(route_edit_row)
        self.auto_replan_checkbox = QCheckBox("LiDAR 장애물 감지 시 자동 재탐색")
        self.auto_replan_checkbox.setChecked(True)
        destination_form.addRow(self.auto_replan_checkbox)
        self.enemy_avoidance_checkbox = QCheckBox("적 드론·사람 인식 및 동적 회피")
        self.enemy_avoidance_checkbox.setChecked(True)
        destination_form.addRow(self.enemy_avoidance_checkbox)
        self.move_button = QPushButton("A* 경로로 목표 B 이동")
        self.move_button.clicked.connect(self._move)
        destination_form.addRow(self.move_button)
        self.route_move_button = QPushButton("A→B→C 예약 경로 비행")
        self.route_move_button.clicked.connect(self._start_reserved_route)
        destination_form.addRow(self.route_move_button)
        self.patrol_duration_minutes = self._spinbox(0.5, 60.0, 5.0, " 분")
        self.patrol_duration_minutes.setSingleStep(0.5)
        destination_form.addRow("자율 정찰 시간", self.patrol_duration_minutes)
        patrol_row = QHBoxLayout()
        self.start_patrol_button = QPushButton("중앙 7구역 정찰 시작")
        self.stop_patrol_button = QPushButton("정찰 중지")
        self.start_patrol_button.clicked.connect(self._start_autonomous_patrol)
        self.stop_patrol_button.clicked.connect(self._stop_autonomous_patrol)
        patrol_row.addWidget(self.start_patrol_button)
        patrol_row.addWidget(self.stop_patrol_button)
        destination_form.addRow(patrol_row)
        layout.addWidget(destination_group)

        telemetry_group = QGroupBox("실시간 텔레메트리")
        telemetry_layout = QFormLayout(telemetry_group)
        self.telemetry_labels: dict[str, QLabel] = {}
        fields = [
            ("상태", "landed"),
            ("위치 X / Y", "position"),
            ("고도", "altitude"),
            ("속도", "speed"),
            ("속도 벡터", "velocity"),
            ("Roll / Pitch / Yaw", "attitude"),
            ("LiDAR 감지", "lidar"),
            ("Radar 감지", "radar"),
            ("탐지 표적", "enemy"),
            ("자율 정찰", "patrol"),
        ]
        for label, key in fields:
            value = QLabel("—")
            value.setTextInteractionFlags(value.textInteractionFlags())
            self.telemetry_labels[key] = value
            telemetry_layout.addRow(label, value)
        layout.addWidget(telemetry_group)
        layout.addStretch()

        scroll = QScrollArea()
        scroll.setObjectName("controlScroll")
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setMinimumWidth(350)
        scroll.setMaximumWidth(450)
        scroll.setWidget(panel)
        return scroll

    def _build_sensor_panel(self) -> QWidget:
        tabs = QTabWidget()
        self.camera_viewer = CameraViewer()
        self.lidar_viewer = LidarViewer()
        # Echo/Radar returns are intentionally sparse; larger blue points keep
        # a scan visible even when only a few rays are reflected.
        # Echo/Radar 반사점은 본래 희소하므로 소수의 점도 보이도록 크게 표시합니다.
        self.radar_viewer = LidarViewer(
            fixed_color=(0.14, 0.55, 1.0, 0.95),
            point_size=9.0,
        )
        minimap_image = PYTHON_CLIENT_ROOT / "assets" / "Minimap_AirBase.PNG"
        self.minimap = MiniMapWidget(
            half_extent_m=700.0,
            center_xy_m=(53.31, 159.39),
            background_path=str(minimap_image),
        )
        self.minimap.spawn_selected.connect(self._on_spawn_selected)
        self.minimap.target_selected.connect(self._on_target_selected)
        tabs.addTab(self.minimap, "미니맵 · A* 경로")
        tabs.addTab(self._build_environment_panel(), "환경 · 날씨")
        # Keep collection next to the minimap so it remains visible even when
        # the sensor pane is narrow.
        # 센서 패널이 좁아도 수집 탭이 가려지지 않도록 두 번째에 둡니다.
        tabs.addTab(self._build_data_collection_panel(), "데이터 수집")
        tabs.addTab(self.camera_viewer, "RGB · Depth · Segmentation")
        tabs.addTab(self.lidar_viewer, "LiDAR 3D 점군")
        tabs.addTab(self.radar_viewer, "Radar 3D 점군")
        return tabs

    def _build_environment_panel(self) -> QWidget:
        """Build deterministic weather controls used by collection sessions."""
        contents = QWidget()
        root = QVBoxLayout(contents)

        guide_group = QGroupBox("환경 설정 사용 방법")
        guide_layout = QVBoxLayout(guide_group)
        guide = QLabel(
            "1. 계절·시간대·하늘·강수를 선택  →  2. 강수량과 바람을 조절  →  "
            "3. <b>환경 적용</b><br>"
            "적용된 값은 다음 데이터 수집 세션의 session.json, 요약 CSV와 "
            "PDF 보고서에 자동으로 기록됩니다. 비교 실험에서는 경로와 환경값을 "
            "각 세션 동안 고정하는 것을 권장합니다."
        )
        guide.setWordWrap(True)
        guide.setStyleSheet(
            "background:#101722; border:1px solid #31547a; "
            "border-radius:5px; padding:10px; color:#d8edff;"
        )
        guide_layout.addWidget(guide)
        root.addWidget(guide_group)

        preset_group = QGroupBox("시간·기상 선택")
        preset_form = QFormLayout(preset_group)
        self.season_combo = QComboBox()
        self.season_combo.addItem("봄", "spring")
        self.season_combo.addItem("여름", "summer")
        self.season_combo.addItem("가을", "autumn")
        self.season_combo.addItem("겨울", "winter")
        self.time_of_day_combo = QComboBox()
        self.time_of_day_combo.addItem("아침", "morning")
        self.time_of_day_combo.addItem("점심", "noon")
        self.time_of_day_combo.addItem("저녁", "evening")
        self.time_of_day_combo.addItem("한밤", "midnight")
        self.visibility_combo = QComboBox()
        self.visibility_combo.addItem("맑음", "clear")
        self.visibility_combo.addItem("흐림", "cloudy")
        self.visibility_combo.addItem("안개", "fog")
        self.precipitation_combo = QComboBox()
        self.precipitation_combo.addItem("없음", "none")
        self.precipitation_combo.addItem("비", "rain")
        self.precipitation_combo.addItem("눈", "snow")
        self.precipitation_combo.currentIndexChanged.connect(
            self._update_environment_input_state
        )
        preset_form.addRow("계절", self.season_combo)
        preset_form.addRow("시간대", self.time_of_day_combo)
        preset_form.addRow("하늘/시정", self.visibility_combo)
        preset_form.addRow("강수", self.precipitation_combo)
        root.addWidget(preset_group)

        detail_group = QGroupBox("강도·바람")
        detail_form = QFormLayout(detail_group)
        self.precipitation_intensity = self._spinbox(0.0, 1.0, 0.6, "")
        self.precipitation_intensity.setSingleStep(0.1)
        self.precipitation_intensity.setToolTip(
            "비 또는 눈의 강도입니다. 0은 없음, 1은 최대 강도입니다."
        )
        self.wind_north = self._spinbox(-30.0, 30.0, 0.0, " m/s")
        self.wind_north.setSingleStep(1.0)
        self.wind_north.setToolTip("NED 기준: +는 북쪽, -는 남쪽 방향 바람입니다.")
        self.wind_east = self._spinbox(-30.0, 30.0, 0.0, " m/s")
        self.wind_east.setSingleStep(1.0)
        self.wind_east.setToolTip("NED 기준: +는 동쪽, -는 서쪽 방향 바람입니다.")
        detail_form.addRow("비/눈 강도", self.precipitation_intensity)
        detail_form.addRow("바람 N(북+)", self.wind_north)
        detail_form.addRow("바람 E(동+)", self.wind_east)
        root.addWidget(detail_group)

        action_group = QGroupBox("환경 제어")
        action_layout = QGridLayout(action_group)
        self.environment_apply_button = QPushButton("환경 적용")
        self.environment_reset_button = QPushButton("기본값으로 선택")
        self.environment_apply_button.clicked.connect(self._apply_environment)
        self.environment_reset_button.clicked.connect(self._reset_environment_inputs)
        self.environment_status_label = QLabel(
            "현재 기록값: 여름 · 점심 · 맑음 · 강수 없음 · 무풍"
        )
        self.environment_status_label.setWordWrap(True)
        self.environment_status_label.setStyleSheet(
            "color:#8fb9dc; padding-top:4px;"
        )
        action_layout.addWidget(self.environment_apply_button, 0, 0)
        action_layout.addWidget(self.environment_reset_button, 0, 1)
        action_layout.addWidget(self.environment_status_label, 1, 0, 1, 2)
        root.addWidget(action_group)

        log_group = QGroupBox("환경 적용 로그")
        log_layout = QVBoxLayout(log_group)
        self.environment_log = QPlainTextEdit()
        self.environment_log.setReadOnly(True)
        self.environment_log.setMaximumBlockCount(100)
        self.environment_log.setMinimumHeight(120)
        self.environment_log.setPlaceholderText(
            "환경 적용을 누르면 요청값과 Good SKY 프리셋 적용 결과가 표시됩니다."
        )
        log_layout.addWidget(self.environment_log)
        root.addWidget(log_group)

        note = QLabel(
            "※ 흐림은 레벨의 Volumetric Cloud 액터를 표시합니다. 액터가 없으면 "
            "구름 모양은 변하지 않습니다. Road Wetness/Road Snow는 AirSim용 "
            "노면 머티리얼 설정이 필요합니다."
        )
        note.setWordWrap(True)
        note.setStyleSheet("color:#9eabba; padding:6px;")
        root.addWidget(note)
        root.addStretch()
        self._update_environment_input_state()

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setWidget(contents)
        return scroll

    def _selected_environment(self) -> dict[str, object]:
        precipitation = str(self.precipitation_combo.currentData())
        return {
            "season": str(self.season_combo.currentData()),
            "time_of_day": str(self.time_of_day_combo.currentData()),
            "visibility": str(self.visibility_combo.currentData()),
            "precipitation": precipitation,
            "precipitation_intensity": (
                float(self.precipitation_intensity.value())
                if precipitation != "none"
                else 0.0
            ),
            "wind_north_mps": float(self.wind_north.value()),
            "wind_east_mps": float(self.wind_east.value()),
        }

    def _update_environment_input_state(self, *_args: object) -> None:
        has_precipitation = (
            str(self.precipitation_combo.currentData()) != "none"
        )
        self.precipitation_intensity.setEnabled(has_precipitation)

    def _reset_environment_inputs(self) -> None:
        self.season_combo.setCurrentIndex(
            self.season_combo.findData("summer")
        )
        self.time_of_day_combo.setCurrentIndex(
            self.time_of_day_combo.findData("noon")
        )
        self.visibility_combo.setCurrentIndex(
            self.visibility_combo.findData("clear")
        )
        self.precipitation_combo.setCurrentIndex(
            self.precipitation_combo.findData("none")
        )
        self.precipitation_intensity.setValue(0.6)
        self.wind_north.setValue(0.0)
        self.wind_east.setValue(0.0)
        self._update_environment_input_state()
        self.message_label.setText(
            "환경 선택을 기본값으로 돌렸습니다. 실제 적용은 환경 적용을 누르세요."
        )

    def _apply_environment(self) -> None:
        if not self._connected:
            self._on_error("환경 설정: AirSim에 먼저 연결하세요.")
            return
        selected = self._selected_environment()
        requested_time = {
            "morning": "아침 / SunRise",
            "noon": "점심 / Noon Clear Sky",
            "evening": "저녁 / SunSet",
            "midnight": "한밤 / Midnight Moon",
        }.get(str(selected["time_of_day"]), str(selected["time_of_day"]))
        self._append_environment_log(
            f"요청 → {requested_time}, 하늘={selected['visibility']}, "
            f"강수={selected['precipitation']}"
        )
        self._environment_apply_pending = True
        self._environment_requested = dict(selected)
        self.environment_apply_button.setEnabled(True)
        self.environment_apply_button.setText("적용 중 · 최신 선택 다시 적용")
        self.environment_status_label.setText("AirSim에 환경을 적용하는 중…")
        self.message_label.setText("시간대·날씨·바람을 적용하고 있습니다…")
        self.worker.submit(
            "environment",
            selected["season"],
            selected["time_of_day"],
            selected["visibility"],
            selected["precipitation"],
            selected["precipitation_intensity"],
            selected["wind_north_mps"],
            selected["wind_east_mps"],
            priority=1,
        )

    def _on_environment_applied(self, values: dict) -> None:
        self._environment_state = dict(values)
        self.environment_apply_button.setEnabled(self._connected)
        comparable_keys = {
            "season",
            "time_of_day",
            "visibility",
            "precipitation",
            "precipitation_intensity",
            "wind_north_mps",
            "wind_east_mps",
        }
        newest_applied = all(
            values.get(key) == self._environment_requested.get(key)
            for key in comparable_keys
        )
        self._environment_apply_pending = not newest_applied
        self.environment_apply_button.setText(
            "변경 다시 적용" if self._environment_apply_pending else "환경 적용"
        )
        labels = {
            "spring": "봄",
            "summer": "여름",
            "autumn": "가을",
            "winter": "겨울",
            "morning": "아침",
            "noon": "점심",
            "evening": "저녁",
            "midnight": "한밤",
            "day": "점심",
            "night": "한밤",
            "clear": "맑음",
            "cloudy": "흐림",
            "fog": "안개",
            "none": "없음",
            "rain": "비",
            "snow": "눈",
        }
        season_label = labels.get(str(values.get("season")), "-")
        time_label = labels.get(str(values.get("time_of_day")), "-")
        visibility_label = labels.get(str(values.get("visibility")), "-")
        precipitation_label = labels.get(
            str(values.get("precipitation")), "-"
        )
        intensity = float(values.get("precipitation_intensity", 0.0))
        north = float(values.get("wind_north_mps", 0.0))
        east = float(values.get("wind_east_mps", 0.0))
        description = (
            f"{season_label} · {time_label} · {visibility_label} · 강수 {precipitation_label} "
            f"{intensity:.1f} · 바람 N {north:.1f}, E {east:.1f} m/s"
        )
        self.environment_status_label.setText(f"현재 적용값: {description}")
        self.message_label.setText(f"환경 적용 완료: {description}")
        requested = bool(values.get("good_sky_requested", False))
        preset = str(values.get("good_sky_preset", "알 수 없음"))
        command = str(values.get("good_sky_command", ""))
        self._append_environment_log(
            f"{'완료' if requested else '실패'} → Good SKY={preset} · {command}"
        )

    def _append_environment_log(self, message: str) -> None:
        if not hasattr(self, "environment_log"):
            return
        timestamp = time.strftime("%H:%M:%S")
        self.environment_log.appendPlainText(f"[{timestamp}] {message}")

    def _build_data_collection_panel(self) -> QWidget:
        panel = QWidget()
        root = QVBoxLayout(panel)

        guide_group = QGroupBox("처음 사용하는 방법")
        guide_layout = QVBoxLayout(guide_group)
        guide = QLabel(
            "1. 도시·구역·지형을 기록  →  "
            "2. 저장할 센서를 선택  →  "
            "3. <b>● 수집 시작</b>  →  "
            "4. 드론 미션 실행  →  "
            "5. <b>■ 수집 종료</b>  →  "
            "6. 요약 CSV·그래프 PDF 자동 생성<br>"
            "RGB 프레임을 기준으로 Depth·Segmentation·LiDAR·Radar·"
            "비행 상태와 바운딩 박스를 함께 저장합니다. "
            "처음에는 기본값 2 Hz를 권장합니다."
        )
        guide.setWordWrap(True)
        guide.setStyleSheet(
            "background:#101722; border:1px solid #31547a; "
            "border-radius:5px; padding:10px; color:#d8edff;"
        )
        guide_layout.addWidget(guide)
        root.addWidget(guide_group)

        session_group = QGroupBox("데이터셋 세션")
        session_form = QFormLayout(session_group)
        self.dataset_name_edit = QLineEdit("KoreaDroneDataset")
        self.dataset_name_edit.setToolTip("실험 전체를 묶는 최상위 폴더 이름입니다.")
        self.city_edit = QLineEdit("AirBase")
        self.city_edit.setToolTip("현재 사용하는 도시 또는 Unreal 맵 이름입니다.")
        self.region_edit = QLineEdit("default")
        self.region_edit.setToolTip("같은 도시 안의 촬영 구역·시나리오 이름입니다.")
        self.terrain_combo = QComboBox()
        self.terrain_combo.setEditable(True)
        self.terrain_combo.addItems(
            [
                "산업단지",
                "고층 도심",
                "저층 주택가",
                "공원·광장",
                "산지",
                "해안",
                "교량·하천",
            ]
        )
        self.collection_root_edit = QLineEdit(
            str(Path.home() / "Documents" / "AutonomousDroneDatasets")
        )
        self.collection_root_edit.setToolTip("실제 PNG·NPZ·JSON 데이터가 저장될 상위 폴더입니다.")
        self.collection_browse_button = QPushButton("저장 폴더 선택")
        self.collection_browse_button.clicked.connect(self._browse_collection_root)
        output_row = QHBoxLayout()
        output_row.addWidget(self.collection_root_edit)
        output_row.addWidget(self.collection_browse_button)
        session_form.addRow("실험/데이터셋 이름", self.dataset_name_edit)
        session_form.addRow("도시 또는 맵", self.city_edit)
        session_form.addRow("구역/시나리오", self.region_edit)
        session_form.addRow("비교용 지형 분류", self.terrain_combo)
        session_form.addRow("저장 위치", output_row)
        session_help = QLabel(
            "※ 위 네 항목은 사용자가 실험 조건에 맞게 입력합니다. "
            "한 번 입력한 값은 다음 실행에도 유지됩니다."
        )
        session_help.setWordWrap(True)
        session_help.setStyleSheet("color:#8fb9dc; padding-top:4px;")
        session_form.addRow(session_help)
        root.addWidget(session_group)

        sensor_group = QGroupBox("동기화 수집 항목")
        sensor_layout = QGridLayout(sensor_group)
        sensor_labels = [
            ("rgb", "RGB"),
            ("depth", "Depth"),
            ("segmentation", "Segmentation"),
            ("lidar", "LiDAR"),
            ("radar", "Radar"),
            ("telemetry", "비행 상태"),
            ("annotations", "객체 정답·박스"),
        ]
        self.collection_sensor_checks: dict[str, QCheckBox] = {}
        for index, (key, label) in enumerate(sensor_labels):
            checkbox = QCheckBox(label)
            checkbox.setChecked(True)
            self.collection_sensor_checks[key] = checkbox
            sensor_layout.addWidget(checkbox, index // 3, index % 3)
        self.collection_rate = self._spinbox(0.5, 10.0, 2.0, " Hz")
        self.collection_rate.setSingleStep(0.5)
        sensor_layout.addWidget(QLabel("저장 주기"), 3, 0)
        sensor_layout.addWidget(self.collection_rate, 3, 1)
        root.addWidget(sensor_group)

        control_group = QGroupBox("수집 제어")
        control_layout = QGridLayout(control_group)
        self.collection_start_button = QPushButton("● 수집 시작")
        self.collection_pause_button = QPushButton("일시정지")
        self.collection_stop_button = QPushButton("■ 수집 종료")
        self.segmentation_apply_button = QPushButton("Segmentation 다시 적용")
        self.segmentation_apply_button.setToolTip(
            "연결 후 새 객체를 추가한 경우에만 누르세요. "
            "평상시에는 재연결 시 자동 적용됩니다."
        )
        self.collection_start_button.clicked.connect(self._start_collection)
        self.collection_pause_button.clicked.connect(self._toggle_collection_pause)
        self.collection_stop_button.clicked.connect(self._stop_collection)
        self.segmentation_apply_button.clicked.connect(self._reapply_segmentation)
        control_layout.addWidget(self.collection_start_button, 0, 0)
        control_layout.addWidget(self.collection_pause_button, 0, 1)
        control_layout.addWidget(self.collection_stop_button, 0, 2)
        control_layout.addWidget(self.segmentation_apply_button, 1, 0, 1, 3)
        root.addWidget(control_group)

        status_group = QGroupBox("수집 상태")
        status_form = QFormLayout(status_group)
        self.collection_status_labels: dict[str, QLabel] = {}
        for label, key in (
            ("상태", "state"),
            ("수집 시간", "elapsed"),
            ("저장 프레임", "frames"),
            ("대기/누락", "queue"),
            ("저장 용량", "bytes"),
            ("세션 폴더", "path"),
        ):
            value = QLabel("—")
            value.setWordWrap(True)
            self.collection_status_labels[key] = value
            status_form.addRow(label, value)
        root.addWidget(status_group)

        report_group = QGroupBox("수집 결과 분석")
        report_layout = QGridLayout(report_group)
        self.auto_report_checkbox = QCheckBox("수집 종료 후 요약·그래프 PDF 자동 생성")
        self.auto_report_checkbox.setChecked(True)
        self.current_report_button = QPushButton("현재 세션 PDF 다시 생성")
        self.existing_report_button = QPushButton("기존 세션 폴더 분석")
        self.current_report_button.clicked.connect(self._generate_current_report)
        self.existing_report_button.clicked.connect(self._select_existing_session_report)
        self.report_status_label = QLabel(
            "수집을 종료하면 analysis 폴더에 CSV 3개와 PDF 보고서가 생성됩니다."
        )
        self.report_status_label.setWordWrap(True)
        self.report_status_label.setStyleSheet("color:#8fb9dc; padding-top:4px;")
        report_layout.addWidget(self.auto_report_checkbox, 0, 0, 1, 2)
        report_layout.addWidget(self.current_report_button, 1, 0)
        report_layout.addWidget(self.existing_report_button, 1, 1)
        report_layout.addWidget(self.report_status_label, 2, 0, 1, 2)
        root.addWidget(report_group)
        root.addStretch()
        self._refresh_collection_status()
        return panel

    @staticmethod
    def _spinbox(minimum: float, maximum: float, value: float, suffix: str) -> QDoubleSpinBox:
        widget = QDoubleSpinBox()
        widget.setRange(minimum, maximum)
        widget.setDecimals(1)
        widget.setSingleStep(1.0)
        widget.setValue(value)
        widget.setSuffix(suffix)
        return widget

    def _browse_collection_root(self) -> None:
        selected = QFileDialog.getExistingDirectory(
            self,
            "데이터셋 저장 폴더 선택",
            self.collection_root_edit.text(),
        )
        if selected:
            self.collection_root_edit.setText(selected)

    def _selected_collection_sensors(self) -> tuple[str, ...]:
        return tuple(
            key
            for key, checkbox in self.collection_sensor_checks.items()
            if checkbox.isChecked()
        )

    def _start_collection(self) -> None:
        if not self._connected:
            self._on_error("데이터 수집: AirSim에 먼저 연결하세요.")
            return
        if not self._semantic_ready:
            self._on_error(
                "데이터 수집: Segmentation 클래스 준비가 끝난 뒤 시작하세요."
            )
            return
        try:
            sensors = self._selected_collection_sensors()
            config = RecordingConfig(
                output_root=Path(self.collection_root_edit.text()),
                dataset_name=self.dataset_name_edit.text(),
                city=self.city_edit.text(),
                region=self.region_edit.text(),
                terrain_type=self.terrain_combo.currentText(),
                sensors=sensors,
                sample_rate_hz=float(self.collection_rate.value()),
                season=str(self._environment_state["season"]),
                time_of_day=str(self._environment_state["time_of_day"]),
                visibility=str(self._environment_state["visibility"]),
                precipitation=str(self._environment_state["precipitation"]),
                precipitation_intensity=float(
                    self._environment_state["precipitation_intensity"]
                ),
                wind_north_mps=float(self._environment_state["wind_north_mps"]),
                wind_east_mps=float(self._environment_state["wind_east_mps"]),
            )
            # Camera frames are the synchronization boundary. Requested range
            # sensors are also enabled so their nearest samples are available.
            self.sensor_checkbox.setChecked(True)
            if "lidar" in sensors:
                self.lidar_checkbox.setChecked(True)
            if "radar" in sensors:
                self.radar_checkbox.setChecked(True)
            self.worker.set_camera_rate_hz(config.sample_rate_hz)
            session_dir = self.recorder.start(config)
            self.message_label.setText(f"데이터 수집 시작: {session_dir}")
        except Exception as exc:
            self._on_error(f"데이터 수집 시작 실패: {exc}")
        self._refresh_collection_status()

    def _toggle_collection_pause(self) -> None:
        try:
            if self.recorder.state == "recording":
                self.recorder.pause()
                self.message_label.setText("데이터 수집을 일시정지했습니다.")
            elif self.recorder.state == "paused":
                self.recorder.resume()
                self.message_label.setText("데이터 수집을 다시 시작했습니다.")
            else:
                raise RuntimeError("진행 중인 데이터 수집이 없습니다.")
        except Exception as exc:
            self._on_error(f"데이터 수집: {exc}")
        self._refresh_collection_status()

    def _stop_collection(self) -> None:
        self.recorder.stop()
        stats = self.recorder.stats()
        self.message_label.setText(
            f"데이터 수집 종료 · {stats['written_frames']}프레임 · "
            f"{self._format_bytes(int(stats['bytes_written']))}"
        )
        self._refresh_collection_status()
        session_dir = self.recorder.session_dir
        if (
            self.auto_report_checkbox.isChecked()
            and session_dir is not None
            and stats["state"] == "stopped"
        ):
            self._start_report_generation(session_dir)

    def _generate_current_report(self) -> None:
        session_dir = self.recorder.session_dir
        if session_dir is None:
            self._on_error("분석할 현재 데이터 수집 세션이 없습니다.")
            return
        self._start_report_generation(session_dir)

    def _select_existing_session_report(self) -> None:
        selected = QFileDialog.getExistingDirectory(
            self,
            "session.json과 frames.jsonl이 있는 세션 폴더 선택",
            self.collection_root_edit.text(),
        )
        if selected:
            self._start_report_generation(Path(selected))

    def _start_report_generation(self, session_dir: Path) -> None:
        if self._report_worker is not None and self._report_worker.isRunning():
            self.message_label.setText("데이터 분석 보고서를 이미 생성 중입니다.")
            return
        if self.recorder.state in {"recording", "paused", "stopping"}:
            self._on_error("데이터 수집을 종료한 뒤 보고서를 생성하세요.")
            return
        self._report_worker = ReportWorker(Path(session_dir))
        self._report_worker.completed.connect(self._on_report_completed)
        self._report_worker.failed.connect(self._on_report_failed)
        self._report_worker.finished.connect(self._refresh_collection_status)
        self.current_report_button.setEnabled(False)
        self.existing_report_button.setEnabled(False)
        self.report_status_label.setText("세션 요약·CSV·그래프 PDF 생성 중…")
        self.message_label.setText("데이터 수집 결과를 분석하고 있습니다…")
        self._report_worker.start()

    def _on_report_completed(self, pdf_path: str, summary: object) -> None:
        values = summary if isinstance(summary, dict) else {}
        frames = int(values.get("written_frames", 0))
        detections = int(values.get("detection_rows", 0))
        self.report_status_label.setText(
            f"완료 · {frames}프레임 · 객체 탐지 {detections}건\n{pdf_path}"
        )
        self.message_label.setText(f"데이터 분석 PDF 생성 완료: {pdf_path}")
        self._refresh_collection_status()

    def _on_report_failed(self, error: str) -> None:
        self.report_status_label.setText(f"보고서 생성 실패: {error}")
        self.message_label.setText(f"데이터 분석 보고서 생성 실패: {error}")
        self._refresh_collection_status()

    def _reapply_segmentation(self) -> None:
        if not self._connected:
            self._on_error("Segmentation: AirSim에 먼저 연결하세요.")
            return
        self._semantic_ready = False
        self._refresh_collection_status()
        self.worker.submit("segmentation", priority=1)
        self.message_label.setText(
            "현재 레벨의 Segmentation 클래스를 백그라운드에서 재적용 중…"
        )

    def _collection_mission_metadata(self) -> dict[str, object]:
        mission_type = (
            "central_patrol"
            if self._autonomous_patrol_running
            else "reserved_route"
            if self._route_running
            else "manual_or_hover"
        )
        return {
            "type": mission_type,
            "active_target": self._active_target,
            "route_index": self._active_route_index,
            "reserved_waypoints": self._route_waypoints,
            "planned_path": self._planned_path,
            "patrol_cycle": self._patrol_cycle_number,
            "patrol_visited": self._autonomous_patrol_visited,
        }

    @staticmethod
    def _format_bytes(value: int) -> str:
        size = float(max(0, value))
        for unit in ("B", "KB", "MB", "GB", "TB"):
            if size < 1024.0 or unit == "TB":
                return f"{size:.1f} {unit}"
            size /= 1024.0
        return f"{size:.1f} TB"

    def _refresh_collection_status(self) -> None:
        if not hasattr(self, "collection_status_labels"):
            return
        stats = self.recorder.stats()
        state_names = {
            "idle": "대기",
            "recording": "● 수집 중",
            "paused": "일시정지",
            "stopping": "저장 마무리 중",
            "stopped": "완료",
            "error": "오류",
        }
        self.collection_status_labels["state"].setText(
            state_names.get(str(stats["state"]), str(stats["state"]))
        )
        elapsed = int(float(stats["elapsed_seconds"]))
        self.collection_status_labels["elapsed"].setText(
            f"{elapsed // 60:02d}:{elapsed % 60:02d}"
        )
        self.collection_status_labels["frames"].setText(
            f"{stats['written_frames']}"
        )
        self.collection_status_labels["queue"].setText(
            f"{stats['queued_frames']} / 누락 {stats['dropped_frames']}"
        )
        self.collection_status_labels["bytes"].setText(
            self._format_bytes(int(stats["bytes_written"]))
        )
        self.collection_status_labels["path"].setText(
            str(stats["session_dir"] or "—")
        )
        state = str(stats["state"])
        self.collection_start_button.setEnabled(
            self._connected
            and self._semantic_ready
            and state not in {"recording", "paused", "stopping"}
        )
        self.collection_pause_button.setEnabled(state in {"recording", "paused"})
        self.collection_pause_button.setText(
            "수집 재개" if state == "paused" else "일시정지"
        )
        self.collection_stop_button.setEnabled(state in {"recording", "paused"})
        report_busy = self._report_worker is not None and self._report_worker.isRunning()
        has_session = bool(stats["session_dir"])
        self.current_report_button.setEnabled(
            has_session
            and state not in {"recording", "paused", "stopping"}
            and not report_busy
        )
        self.existing_report_button.setEnabled(not report_busy)
        error = str(stats.get("last_error", ""))
        if error and error != self._last_collection_error:
            self._last_collection_error = error
            self.message_label.setText(f"데이터 저장 오류: {error}")

    def _toggle_connection(self) -> None:
        if self._connection_transition is not None:
            return
        if self._connected:
            self._connection_transition = "disconnect"
            self.connect_button.setEnabled(False)
            self.connect_button.setText("연결 해제 중…")
            self._set_controls_enabled(False)
            self.message_label.setText(
                "AirSim 연결 해제 중… 현재 센서 호출이 끝나면 자동으로 해제됩니다."
            )
            self.worker.request_disconnect()
        else:
            self._connection_transition = "connect"
            self.connect_button.setEnabled(False)
            self.connect_button.setText("연결 중…")
            self.message_label.setText("AirSim 연결 중…")
            self.worker.request_connect()

    def _on_worker_status(self, message: str) -> None:
        self.message_label.setText(message)

    def _on_semantic_setup_completed(self, status: str) -> None:
        if not self._connected:
            return
        self._semantic_ready = True
        message = f"연결됨 · {status}"
        self.status_indicator.setText(f"● {message}")
        self.message_label.setText(message)
        self._set_controls_enabled(True)
        self._refresh_collection_status()

    def _on_semantic_setup_failed(self, error: str) -> None:
        if not self._connected:
            return
        self._semantic_ready = False
        self.status_indicator.setText("● 연결됨 · Segmentation 규칙 미적용")
        self.message_label.setText(
            f"비행 연결은 정상입니다. Segmentation 적용 실패: {error}"
        )
        # Semantic labels may fail independently of the flight RPC. Keep the
        # aircraft controls available, while data collection remains blocked.
        # Semantic 라벨 적용이 실패해도 비행 RPC는 사용할 수 있습니다.
        self._set_controls_enabled(True)
        self._refresh_collection_status()

    def _on_lidar_toggled(self, enabled: bool) -> None:
        """Clear LiDAR-only state when its default-on toggle is disabled.

        기본 ON인 LiDAR 토글을 끄면 LiDAR 전용 상태를 제거합니다.
        """
        if enabled:
            self.telemetry_labels["lidar"].setText("수신 대기")
            self.message_label.setText("LiDAR 수신과 근거리 회피를 켰습니다.")
            return
        self._latest_lidar_world = np.empty((0, 3), dtype=np.float32)
        self._obstacle_detection_count = 0
        self.lidar_viewer.update_points(np.empty((0, 3), dtype=np.float32))
        self.telemetry_labels["lidar"].setText("꺼짐")
        self._refresh_obstacle_map()
        self.message_label.setText("LiDAR 수신과 근거리 회피를 껐습니다.")

    def _on_radar_toggled(self, enabled: bool) -> None:
        """Clear Radar fusion state when its default-on toggle is disabled.

        기본 ON인 Radar 토글을 끄면 Radar 센서 융합 상태를 제거합니다.
        """
        if enabled:
            self.telemetry_labels["radar"].setText("수신 대기")
            self.message_label.setText("Radar 수신과 장거리 표적 탐지를 켰습니다.")
            return
        self._radar_obstacle_points = np.empty((0, 3), dtype=np.float32)
        self._latest_radar_tracks = []
        self.radar_viewer.update_points(np.empty((0, 3), dtype=np.float32))
        self.telemetry_labels["radar"].setText("꺼짐")
        self._refresh_obstacle_map()
        self.message_label.setText("Radar 수신과 센서 융합을 껐습니다.")

    def _takeoff(self) -> None:
        if self._takeoff_pending:
            return
        self._clear_active_mission_state()
        self.worker.discard_pending_navigation()
        self._takeoff_pending = True
        self.takeoff_button.setEnabled(False)
        self.message_label.setText("이륙 준비 · API 제어와 ARM 상태 확인 중…")
        self.worker.submit("takeoff", self.takeoff_altitude.value(), priority=2)

    def _clear_active_mission_state(self) -> None:
        """Cancel route state without deleting the user's reserved points."""
        patrol_was_running = self._autonomous_patrol_running
        visited = self._autonomous_patrol_visited
        self._cancel_autonomous_patrol()
        self._route_running = False
        self._route_queue = []
        self._active_route_index = None
        self._active_target = None
        self._planned_path = []
        self._pending_descent_altitude = None
        self._pending_descent_safe_altitude = None
        self._pending_descent_commanded = False
        self._obstacle_detection_count = 0
        self._enemy_detection_count = 0
        self._mission_stall_started = 0.0
        self.minimap.set_path([])
        if patrol_was_running:
            self.telemetry_labels["patrol"].setText(
                f"중지 · 방문 {visited}곳"
            )

    def _stop_active_mission(self) -> None:
        self._clear_active_mission_state()
        self.worker.discard_pending_navigation()
        self.worker.submit("mission_stop", priority=0)
        self.message_label.setText(
            "현재 미션을 중지하고 이 위치에서 호버링합니다. "
            "다른 미션을 바로 시작할 수 있습니다."
        )

    def _landing_approach_altitude(self) -> float | None:
        """Estimate a fast-descent endpoint about one metre above the surface."""
        if self._telemetry is None or not self._latest_lidar_world.size:
            return None
        current_altitude = float(self._telemetry["altitude"])
        current_x = float(self._telemetry["x"])
        current_y = float(self._telemetry["y"])
        points = self._latest_lidar_world
        finite = np.all(np.isfinite(points), axis=1)
        horizontal = np.hypot(points[:, 0] - current_x, points[:, 1] - current_y)
        surface_altitudes = -points[:, 2]
        underneath = (
            finite
            & (horizontal <= self.planner.config.drone_radius_m + 0.75)
            & (surface_altitudes <= current_altitude - 0.5)
            & (surface_altitudes >= -1.0)
        )
        if not np.any(underneath):
            return None
        surface_altitude = float(np.max(surface_altitudes[underneath]))
        approach_altitude = max(0.8, surface_altitude + 1.0)
        return min(current_altitude, approach_altitude)

    def _land(self) -> None:
        approach_altitude = self._landing_approach_altitude()
        self._clear_active_mission_state()
        self.worker.discard_pending_navigation()
        self.worker.submit("land", approach_altitude, 1.5, priority=0)
        if approach_altitude is None:
            self.message_label.setText(
                "하부 LiDAR 표면을 확인할 수 없어 기본 안전 착륙을 시작합니다."
            )
        else:
            self.message_label.setText(
                f"고도 {approach_altitude:.1f}m까지 1.5m/s로 접근한 뒤 착륙합니다."
            )

    def _emergency_stop(self) -> None:
        self._clear_active_mission_state()
        self.worker.discard_pending_navigation()
        self.worker.submit("emergency", priority=0)
        self.message_label.setText("모든 미션을 취소하고 긴급 호버링합니다.")

    def _move(self) -> None:
        try:
            self._cancel_autonomous_patrol()
            self._route_running = False
            self._route_queue = []
            self._active_route_index = None
            self._plan_and_fly(replan=False)
        except Exception as exc:
            self._on_error(f"경로계획: {exc}")

    def _set_map_mode(self, mode: str) -> None:
        self.minimap.set_selection_mode(mode)
        is_spawn = mode == "spawn"
        self.spawn_select_button.setChecked(is_spawn)
        self.target_select_button.setChecked(not is_spawn)
        self.message_label.setText(
            "미니맵에서 스폰 A를 클릭하세요."
            if is_spawn
            else "미니맵에서 추가할 경유지를 클릭하세요."
        )

    def _on_spawn_selected(self, x_m: float, y_m: float) -> None:
        self._spawn_xy = (x_m, y_m)
        self.message_label.setText(
            f"스폰 A 선택: X={x_m:.1f}, Y={y_m:.1f}m · 적용 버튼을 누르세요."
        )

    def _on_target_selected(self, x_m: float, y_m: float) -> None:
        self.destination_x.setValue(x_m)
        self.destination_y.setValue(y_m)
        next_label = self._route_label(len(self._route_waypoints))
        self.message_label.setText(
            f"경유지 {next_label} 후보: X={x_m:.1f}, Y={y_m:.1f}m · "
            "고도를 확인하고 경유지 추가를 누르세요."
        )

    @staticmethod
    def _route_label(index: int) -> str:
        value = max(0, int(index)) + 2
        label = ""
        while value:
            value, remainder = divmod(value - 1, 26)
            label = chr(ord("A") + remainder) + label
        return label

    def _refresh_route_list(self) -> None:
        self.route_list.clear()
        for index, (x_m, y_m, altitude_m) in enumerate(self._route_waypoints):
            label = self._route_label(index)
            self.route_list.addItem(
                f"{label}  X {x_m:.1f} · Y {y_m:.1f} · 고도 {altitude_m:.1f} m"
            )
        self.minimap.set_route_waypoints(self._route_waypoints)

    def _add_waypoint(self) -> None:
        waypoint = (
            float(self.destination_x.value()),
            float(self.destination_y.value()),
            float(self.destination_altitude.value()),
        )
        if self._route_waypoints:
            previous = self._route_waypoints[-1]
            if math.dist(previous, waypoint) < 0.1:
                self.message_label.setText("마지막 경유지와 같은 위치입니다.")
                return
        self._route_waypoints.append(waypoint)
        self._refresh_route_list()
        label = self._route_label(len(self._route_waypoints) - 1)
        self.message_label.setText(
            f"경유지 {label} 예약: X={waypoint[0]:.1f}, "
            f"Y={waypoint[1]:.1f}, 고도={waypoint[2]:.1f}m"
        )

    def _remove_selected_waypoint(self) -> None:
        row = self.route_list.currentRow()
        if row < 0 or row >= len(self._route_waypoints):
            self.message_label.setText("삭제할 경유지를 목록에서 선택하세요.")
            return
        self._route_waypoints.pop(row)
        self._refresh_route_list()
        self.message_label.setText("선택한 경유지를 삭제하고 순서를 다시 정리했습니다.")

    def _clear_waypoints(self) -> None:
        route_was_running = self._route_running
        if route_was_running:
            self._clear_active_mission_state()
            self.worker.discard_pending_navigation()
            self.worker.submit("mission_stop", priority=0)
        self._route_waypoints = []
        self._route_queue = []
        self._route_running = False
        self._active_route_index = None
        self._refresh_route_list()
        self.message_label.setText(
            "예약 경로를 취소하고 현재 위치에서 호버링합니다."
            if route_was_running
            else "B/C/D 예약 목록을 모두 취소했습니다."
        )

    def _start_reserved_route(self) -> None:
        if self._telemetry is None:
            self._on_error("예약 경로: 드론 위치를 아직 받지 못했습니다.")
            return
        if not self._route_waypoints:
            self._on_error("예약 경로: 경유지를 하나 이상 추가하세요.")
            return
        self._cancel_autonomous_patrol()
        self._route_queue = list(self._route_waypoints)
        self._route_running = True
        self._active_route_index = -1
        try:
            self._start_next_route_waypoint(reset_avoidance=True)
        except Exception as exc:
            self._route_running = False
            self._on_error(f"예약 경로 생성: {exc}")

    def _start_next_route_waypoint(self, reset_avoidance: bool = False) -> None:
        if not self._route_queue:
            self._route_running = False
            self._active_route_index = None
            self._active_target = None
            self._planned_path = []
            self.minimap.set_path([])
            self.message_label.setText("예약 경로의 모든 경유지에 도착했습니다.")
            return
        target = self._route_queue.pop(0)
        self._active_route_index = (
            0 if self._active_route_index is None else self._active_route_index + 1
        )
        self.destination_x.setValue(target[0])
        self.destination_y.setValue(target[1])
        self.destination_altitude.setValue(target[2])
        self._plan_and_fly(
            replan=False,
            target_override=target,
            reset_avoidance=reset_avoidance,
        )
        label = self._route_label(self._active_route_index)
        self.message_label.setText(
            f"예약 경로 {label}로 이동 중 · 남은 경유지 {len(self._route_queue)}개"
        )

    def _start_autonomous_patrol(self) -> None:
        if self._telemetry is None:
            self._on_error("자율 정찰: 드론 위치를 아직 받지 못했습니다.")
            return
        self._route_running = False
        self._route_queue = []
        self._active_route_index = None
        self._autonomous_patrol_running = True
        self._autonomous_patrol_visited = 0
        self._patrol_target_queue = []
        self._patrol_cycle_targets = []
        self._patrol_cycle_number = 0
        duration_seconds = float(self.patrol_duration_minutes.value()) * 60.0
        self._autonomous_patrol_end_time = time.monotonic() + duration_seconds
        self.telemetry_labels["patrol"].setText(
            f"진행 중 · {duration_seconds / 60.0:.1f}분 남음 · 방문 0곳"
        )
        try:
            self._start_next_patrol_target(reset_avoidance=True)
        except Exception as exc:
            self._cancel_autonomous_patrol()
            self._on_error(f"자율 정찰 시작: {exc}")

    def _stop_autonomous_patrol(self) -> None:
        if not self._autonomous_patrol_running:
            self.message_label.setText("진행 중인 중앙 순회 정찰이 없습니다.")
            return
        visited = self._autonomous_patrol_visited
        self._cancel_autonomous_patrol()
        self._active_target = None
        self._planned_path = []
        self._pending_descent_altitude = None
        self._pending_descent_safe_altitude = None
        self._pending_descent_commanded = False
        self.minimap.set_path([])
        self.worker.submit("hover", priority=1)
        self.telemetry_labels["patrol"].setText(f"중지 · 방문 {visited}곳")
        self.message_label.setText(
            f"중앙 순회 정찰을 중지했습니다 · 방문 지역 {visited}곳"
        )

    def _cancel_autonomous_patrol(self) -> None:
        self._autonomous_patrol_running = False
        self._autonomous_patrol_end_time = 0.0
        self._patrol_target_queue = []
        self._patrol_cycle_targets = []
        if hasattr(self, "minimap"):
            self.minimap.set_patrol_waypoints([])

    def _build_patrol_sweep_targets(self) -> list[tuple[float, float, float]]:
        """Build a repeatable central sweep with at least six distinct zones.

        중앙 권역을 기준으로 서로 다른 여섯 구역 이상을 훑는 순회점을 만듭니다.
        """
        if self._telemetry is None:
            return []
        center = np.asarray(self.minimap._base_center_xy_m, dtype=np.float64)
        current = np.asarray(
            [float(self._telemetry["x"]), float(self._telemetry["y"])],
            dtype=np.float64,
        )
        altitude = float(self.destination_altitude.value())
        # A 105 m ring covers the central facilities while remaining feasible
        # within a five-minute patrol at the default 3 m/s mission speed.
        # 반경 105m 순회는 중앙 시설을 넓게 훑으면서 기본 3m/s 기준 5분 안에
        # 최소 다섯 구역을 방문할 수 있는 길이입니다.
        radius = min(105.0, float(self.minimap._base_half_extent_m) * 0.22)
        sector_count = 6
        phase = float(self._patrol_rng.uniform(0.0, 2.0 * math.pi))
        ring = []
        for index in range(sector_count):
            angle = phase + 2.0 * math.pi * index / sector_count
            radial_jitter = float(self._patrol_rng.uniform(-10.0, 10.0))
            angular_jitter = float(self._patrol_rng.uniform(-0.06, 0.06))
            distance = radius + radial_jitter
            ring.append(
                center
                + distance
                * np.asarray(
                    [math.cos(angle + angular_jitter), math.sin(angle + angular_jitter)],
                    dtype=np.float64,
                )
            )

        # Start at the nearest sector, then sweep around the ring instead of
        # repeatedly crossing the whole map in a random order.
        # 현재 위치에서 가장 가까운 구역부터 원형으로 순회하여 무작위 장거리
        # 왕복 대신 중앙 지역을 연속적으로 훑습니다.
        nearest = min(
            range(len(ring)),
            key=lambda index: float(np.linalg.norm(ring[index] - current)),
        )
        ordered = ring[nearest:] + ring[:nearest]
        targets = [
            (float(point[0]), float(point[1]), altitude)
            for point in ordered
        ]
        # Finish each cycle through the center, producing seven visibly
        # different scan regions before a newly jittered cycle begins.
        targets.append((float(center[0]), float(center[1]), altitude))
        return targets

    def _start_next_patrol_target(self, reset_avoidance: bool = False) -> None:
        if not self._autonomous_patrol_running or self._telemetry is None:
            return
        last_error: Exception | None = None
        # If a point is temporarily blocked, skip that one and continue with
        # the rest of the coverage route. Generate at most one replacement
        # cycle in this call so an unhealthy map cannot loop forever.
        for cycle_attempt in range(2):
            if not self._patrol_target_queue:
                self._patrol_cycle_number += 1
                self._patrol_cycle_targets = self._build_patrol_sweep_targets()
                self._patrol_target_queue = list(self._patrol_cycle_targets)
                self.minimap.set_patrol_waypoints(self._patrol_cycle_targets)
            while self._patrol_target_queue:
                target = self._patrol_target_queue.pop(0)
                try:
                    self._plan_and_fly(
                        replan=False,
                        target_override=target,
                        reset_avoidance=reset_avoidance,
                    )
                    remaining_seconds = max(
                        0.0,
                        self._autonomous_patrol_end_time - time.monotonic(),
                    )
                    current_index = (
                        len(self._patrol_cycle_targets)
                        - len(self._patrol_target_queue)
                    )
                    self.message_label.setText(
                        f"중앙 순회 정찰 {self._patrol_cycle_number}회차 · "
                        f"구역 {current_index}/{len(self._patrol_cycle_targets)} · "
                        f"남은 시간 {remaining_seconds / 60.0:.1f}분 · "
                        f"방문 {self._autonomous_patrol_visited}곳"
                    )
                    return
                except (RuntimeError, ValueError) as exc:
                    last_error = exc
                    reset_avoidance = False
                    continue
        raise RuntimeError(
            f"중앙 순회 후보 14곳에서 이동 가능한 목표를 찾지 못했습니다: {last_error}"
        )

    def _update_autonomous_patrol(self) -> None:
        if (
            not self._autonomous_patrol_running
            or time.monotonic() < self._autonomous_patrol_end_time
        ):
            return
        visited = self._autonomous_patrol_visited
        self._cancel_autonomous_patrol()
        self._active_target = None
        self._planned_path = []
        self._pending_descent_altitude = None
        self._pending_descent_safe_altitude = None
        self._pending_descent_commanded = False
        self.minimap.set_path([])
        self.worker.submit("hover", priority=1)
        self.telemetry_labels["patrol"].setText(f"완료 · 방문 {visited}곳")
        self.message_label.setText(
            f"자율 정찰 시간이 끝나 자동 호버링합니다 · 방문 지역 {visited}곳"
        )

    def _continue_autonomous_patrol(self) -> None:
        try:
            self._start_next_patrol_target()
        except Exception as exc:
            visited = self._autonomous_patrol_visited
            self._cancel_autonomous_patrol()
            self._active_target = None
            self._planned_path = []
            self.minimap.set_path([])
            self.worker.submit("hover", priority=1)
            self.telemetry_labels["patrol"].setText(
                f"경로 생성 실패 · 방문 {visited}곳"
            )
            self._on_error(f"자율 정찰 경로 생성 실패 · 호버링: {exc}")

    def _apply_spawn(self) -> None:
        self.worker.submit("spawn", self._spawn_xy[0], self._spawn_xy[1], priority=2)

    def _plan_and_fly(
        self,
        replan: bool,
        start_override: tuple[float, float] | None = None,
        altitude_override: float | None = None,
        collision_escape: tuple[float, float, float, float, float] | None = None,
        target_override: tuple[float, float, float] | None = None,
        reset_avoidance: bool = True,
    ) -> None:
        if self._telemetry is None:
            raise RuntimeError("드론 위치를 아직 받지 못했습니다.")
        if not replan and reset_avoidance:
            # A manually started mission clears constraints learned by the
            # previous route. New collisions will establish fresh limits.
            # 사용자가 새 임무를 시작하면 이전 경로에서 학습한 고도 제한을
            # 초기화합니다. 새 충돌이 발생하면 제한을 다시 설정합니다.
            self._avoidance_altitude_floor_m = 1.0
            self._avoidance_altitude_ceiling_m = None
            self._collision_obstacle_points = np.empty(
                (0, 3),
                dtype=np.float32,
            )
        target = target_override or (
            self._active_target
            if replan and self._active_target is not None
            else (
                self.destination_x.value(),
                self.destination_y.value(),
                self.destination_altitude.value(),
            )
        )
        safe_target_altitude = max(
            target[2],
            self._avoidance_altitude_floor_m,
        )
        if self._avoidance_altitude_ceiling_m is not None:
            safe_target_altitude = min(
                safe_target_altitude,
                self._avoidance_altitude_ceiling_m,
            )
        start = start_override or (
            float(self._telemetry["x"]),
            float(self._telemetry["y"]),
        )
        # Refresh the local planning subset immediately before every A* run.
        # Radar can still display distant people/birds, but they do not seal
        # the navigation grid until they are close enough to matter.
        planning_points = self._planning_obstacle_points()
        current_altitude = (
            float(altitude_override)
            if altitude_override is not None
            else float(self._telemetry["altitude"])
        )
        self.planner.set_obstacle_points(planning_points)
        sensor_map_relaxed = False
        try:
            path = self.planner.plan(
                start,
                (target[0], target[1]),
                safe_target_altitude,
                current_altitude,
                max_altitude_m=self._avoidance_altitude_ceiling_m,
            )
        except RuntimeError:
            # Raw point clouds are instantaneous and can occasionally form a
            # false closed ring. Retry with only collision-confirmed surfaces;
            # the forward LiDAR guard remains active and will insert a verified
            # detour well before any real wall is reached.
            # 순간 점군이 가짜 폐곡선을 만들면 충돌로 확인된 면만 사용해 한 번
            # 재시도합니다. 실제 벽은 전방 LiDAR가 미리 확인해 우회벽을 추가합니다.
            self.planner.set_obstacle_points(self._collision_obstacle_points)
            path = self.planner.plan(
                start,
                (target[0], target[1]),
                safe_target_altitude,
                current_altitude,
                max_altitude_m=self._avoidance_altitude_ceiling_m,
            )
            sensor_map_relaxed = True
        finally:
            self.planner.set_obstacle_points(planning_points)
        # Keep the terminal descent out of moveOnPathAsync. Its lookahead can
        # start descending while the drone is still above the building that it
        # is passing, causing repeated roof detections and hesitation.
        # 마지막 하강 구간을 moveOnPathAsync에서 제외합니다. 선행 제어 때문에
        # 건물을 지나는 중 지붕 위에서 미리 내려가며 머뭇거리는 현상을 막습니다.
        flight_path, pending_descent = split_terminal_vertical_leg(path)
        self._pending_descent_altitude = pending_descent
        self._pending_descent_safe_altitude = None
        self._pending_descent_commanded = False
        self._planned_path = flight_path
        self._active_target = target
        self._mission_stall_started = 0.0
        self.minimap.set_target(target[0], target[1])
        self.minimap.set_path(flight_path)
        # A replacement path already supersedes the previous AirSim command.
        # Sending hover before every ordinary replan produced stop-and-go motion.
        # 새 경로 명령 자체가 기존 이동 명령을 대체하므로 일반 재탐색마다
        # 호버링을 먼저 보내지 않습니다. 실제 충돌 위험 때만 별도로 정지합니다.
        if collision_escape is None:
            self.worker.submit("path", flight_path, self.speed.value(), priority=3)
        else:
            self.worker.submit(
                "recovery_path",
                flight_path,
                self.speed.value(),
                collision_escape[0],
                collision_escape[1],
                collision_escape[2],
                collision_escape[3],
                collision_escape[4],
                priority=0,
            )
        cruise = max(point[2] for point in flight_path)
        self.message_label.setText(
            f"{'재탐색' if replan else '경로 생성'} 완료"
            f"{' · 순간 센서 폐곡선 제외' if sensor_map_relaxed else ''}: "
            f"웨이포인트 {len(path)}개, 최고 {cruise:.1f}m"
        )

    def _on_connection_changed(self, connected: bool, message: str) -> None:
        self._connected = connected
        self._semantic_ready = False
        self._connection_transition = None
        self.connect_button.setEnabled(True)
        if not connected:
            self._takeoff_pending = False
            self._environment_apply_pending = False
        if not connected and self.recorder.state in {"recording", "paused"}:
            self.recorder.stop()
        self.status_indicator.setText(f"● {message}")
        self.status_indicator.setObjectName("connected" if connected else "disconnected")
        self.status_indicator.style().unpolish(self.status_indicator)
        self.status_indicator.style().polish(self.status_indicator)
        self.connect_button.setText("연결 해제" if connected else "AirSim 연결")
        # Applying thousands of segmentation labels can briefly stall the
        # Unreal/AirSim RPC server. Starting flight during that window can trip
        # SimpleFlight's API watchdog and leave the drone hovering.
        # Segmentation 초기화가 끝나기 전에 비행을 시작하지 않도록 합니다.
        # Flight control is ready as soon as the primary AirSim RPC connects.
        # Semantic setup continues in the background and gates only collection.
        # 비행 버튼은 즉시 활성화하고, 데이터 수집만 Semantic 준비를 기다립니다.
        self._set_controls_enabled(connected)
        self.message_label.setText(message)
        self._refresh_collection_status()
        if connected:
            # Restore and apply the last selected environment on every new PIE
            # connection so recorded metadata always matches the simulator.
            # PIE 재연결마다 마지막 환경 선택을 다시 적용하여 저장 조건과
            # 실제 시뮬레이터 상태가 어긋나지 않도록 합니다.
            self._apply_environment()

    def _set_controls_enabled(self, enabled: bool) -> None:
        for widget in (
            self.arm_button,
            self.disarm_button,
            self.takeoff_button,
            self.hover_button,
            self.land_button,
            self.emergency_button,
            self.apply_spawn_button,
            self.move_button,
            self.route_move_button,
            self.start_patrol_button,
            self.stop_patrol_button,
            self.segmentation_apply_button,
            self.environment_apply_button,
        ):
            widget.setEnabled(enabled)
        self._refresh_collection_status()

    def _on_telemetry(self, data: dict) -> None:
        self._telemetry = data
        self.recorder.update_telemetry(data)
        self._refresh_collection_status()
        self._update_autonomous_patrol()
        collision_timestamp = float(data.get("collision_timestamp", 0.0))
        if (
            bool(data.get("has_collided", False))
            and collision_timestamp > self._last_collision_timestamp
        ):
            self._last_collision_timestamp = collision_timestamp
            self._recover_from_collision(data)

        # Drop waypoints that were reached or clearly passed so a later
        # low-speed safety check never points back toward an old waypoint.
        # 통과한 웨이포인트를 제거하여 저속 상태의 안전 검사가 이미 지나온
        # 지점을 다시 바라보며 좌우로 왕복하지 않도록 합니다.
        while len(self._planned_path) > 1:
            current_x = float(data["x"])
            current_y = float(data["y"])
            first_distance = math.hypot(
                self._planned_path[0][0] - current_x,
                self._planned_path[0][1] - current_y,
            )
            next_distance = math.hypot(
                self._planned_path[1][0] - current_x,
                self._planned_path[1][1] - current_y,
            )
            if first_distance <= 3.0 or next_distance < first_distance:
                self._planned_path.pop(0)
                self.minimap.set_path(self._planned_path)
            else:
                break
        self.minimap.set_drone(float(data["x"]), float(data["y"]))
        self.telemetry_labels["landed"].setText(str(data["landed"]))
        self.telemetry_labels["position"].setText(f'{data["x"]:.2f} / {data["y"]:.2f} m')
        self.telemetry_labels["altitude"].setText(f'{data["altitude"]:.2f} m')
        self.telemetry_labels["speed"].setText(f'{data["speed"]:.2f} m/s')
        self.telemetry_labels["velocity"].setText(
            f'{data["vx"]:.2f}, {data["vy"]:.2f}, {data["vz"]:.2f} m/s'
        )
        self.telemetry_labels["attitude"].setText(
            f'{data["roll"]:.1f}° / {data["pitch"]:.1f}° / {data["yaw"]:.1f}°'
        )
        if self._autonomous_patrol_running:
            remaining_seconds = max(
                0.0,
                self._autonomous_patrol_end_time - time.monotonic(),
            )
            self.telemetry_labels["patrol"].setText(
                f"진행 중 · {remaining_seconds / 60.0:.1f}분 남음 · "
                f"방문 {self._autonomous_patrol_visited}곳"
            )
        self._advance_terminal_descent(data)
        self._recover_stalled_mission(data)

    def _recover_stalled_mission(self, data: dict) -> None:
        """Replan when an active mission silently falls back to hover.

        SimpleFlight enters hover when its API goal watchdog expires. In a
        sensor-heavy scene that can happen without a real obstacle, leaving the
        UI route active but the vehicle stationary. After a short confirmation
        window, calculate a fresh safe path instead of blindly resending the
        stale command.

        API 워치독으로 임무가 중간에 호버링으로 바뀌었을 때, 일시적
        감속과 구분하여 안전 경로를 자동 재계산합니다.
        """
        if not self._connected or self._active_target is None:
            self._mission_stall_started = 0.0
            return

        current_x = float(data.get("x", 0.0))
        current_y = float(data.get("y", 0.0))
        remaining_xy = math.hypot(
            self._active_target[0] - current_x,
            self._active_target[1] - current_y,
        )
        speed = float(data.get("speed", 0.0))
        if remaining_xy <= 2.0 or speed > 0.35:
            self._mission_stall_started = 0.0
            return

        now = time.monotonic()
        if self._mission_stall_started <= 0.0:
            self._mission_stall_started = now
            return
        if now - self._mission_stall_started < 2.5:
            return
        if now - self._last_stall_recovery < 4.0:
            return

        self._last_stall_recovery = now
        self._mission_stall_started = 0.0
        try:
            self.worker.discard_pending_navigation()
            self._plan_and_fly(replan=True, reset_avoidance=False)
            self._avoidance_grace_until = now + 2.0
            self.message_label.setText(
                f"비행 정지 감지 · 남은 거리 {remaining_xy:.1f}m · "
                "안전 경로를 자동 재전송했습니다."
            )
        except Exception as exc:
            self.message_label.setText(
                f"비행 정지 감지 · 안전 경로 재확인 중: {exc}"
            )

    def _safe_terminal_descent_altitude(
        self,
        data: dict,
        requested_altitude_m: float,
    ) -> tuple[float, float | None]:
        """Return a descent altitude that clears any surface directly below.

        기체 바로 아래 표면과 안전거리를 확보하는 최종 하강 고도를 반환합니다.
        """
        points = self._combined_obstacle_points()
        if not points.size:
            return float(requested_altitude_m), None
        target_x = float(self._active_target[0]) if self._active_target else float(data["x"])
        target_y = float(self._active_target[1]) if self._active_target else float(data["y"])
        horizontal = np.hypot(points[:, 0] - target_x, points[:, 1] - target_y)
        obstacle_altitudes = -points[:, 2]
        current_altitude = float(data["altitude"])
        underneath = (
            (horizontal <= self.planner.config.drone_radius_m + 0.75)
            & (obstacle_altitudes <= current_altitude + 0.5)
            & (obstacle_altitudes >= -1.0)
        )
        if not np.any(underneath):
            return float(requested_altitude_m), None
        surface_altitude = float(np.max(obstacle_altitudes[underneath]))
        clearance = self.planner.config.vertical_clearance_m + 0.75
        safe_altitude = max(float(requested_altitude_m), surface_altitude + clearance)
        return min(current_altitude, safe_altitude), surface_altitude

    def _advance_terminal_descent(self, data: dict) -> None:
        """Descend only after reaching the target XY and checking below.

        목표 X/Y 도착과 하부 안전 확인이 끝난 뒤에만 수직 하강합니다.
        """
        if self._active_target is None:
            return
        remaining_xy = math.hypot(
            self._active_target[0] - float(data["x"]),
            self._active_target[1] - float(data["y"]),
        )
        if remaining_xy >= 1.5:
            return

        if self._pending_descent_altitude is None:
            self._active_target = None
            self._planned_path = []
            self.minimap.set_path([])
            if self._autonomous_patrol_running:
                self._autonomous_patrol_visited += 1
                self._continue_autonomous_patrol()
            elif self._route_running:
                self._start_next_route_waypoint()
            else:
                self.message_label.setText("목표 B에 도착했습니다.")
            return

        if not self._pending_descent_commanded:
            safe_altitude, surface_altitude = self._safe_terminal_descent_altitude(
                data,
                self._pending_descent_altitude,
            )
            self._pending_descent_safe_altitude = safe_altitude
            current_altitude = float(data["altitude"])
            if current_altitude <= safe_altitude + 0.35:
                self._pending_descent_commanded = True
            else:
                # A one-point command performs a vertical descent at the final
                # XY instead of rounding the preceding horizontal corner.
                # 단일 지점 명령으로 마지막 X/Y에서 수직 하강하여 앞선 수평
                # 경로의 모서리를 대각선으로 잘라 내려가지 않게 합니다.
                descent_speed = min(max(float(self.speed.value()), 1.0), 2.0)
                self.worker.submit(
                    "move",
                    self._active_target[0],
                    self._active_target[1],
                    safe_altitude,
                    descent_speed,
                    priority=2,
                )
                self._planned_path = [
                    (self._active_target[0], self._active_target[1], safe_altitude)
                ]
                self.minimap.set_path(self._planned_path)
                self._pending_descent_commanded = True
                if surface_altitude is None or safe_altitude <= self._pending_descent_altitude + 0.1:
                    self.message_label.setText("목표 X/Y 도착 · 안전 확인 후 수직 하강")
                else:
                    self.message_label.setText(
                        f"목표 아래 장애물 {surface_altitude:.1f}m · "
                        f"안전 고도 {safe_altitude:.1f}m까지 하강"
                    )
                return

        safe_altitude = self._pending_descent_safe_altitude
        if safe_altitude is None:
            safe_altitude = self._pending_descent_altitude
        if abs(float(data["altitude"]) - safe_altitude) <= 0.45:
            blocked_descent = safe_altitude > self._pending_descent_altitude + 0.1
            self._active_target = None
            self._planned_path = []
            self.minimap.set_path([])
            self._pending_descent_altitude = None
            self._pending_descent_safe_altitude = None
            self._pending_descent_commanded = False
            if self._autonomous_patrol_running:
                self._autonomous_patrol_visited += 1
                self._continue_autonomous_patrol()
            elif self._route_running:
                self._start_next_route_waypoint()
            else:
                self.message_label.setText(
                    "목표 아래 장애물 때문에 안전 고도에서 호버링합니다."
                    if blocked_descent
                    else "목표 B에 도착했습니다."
                )

    def _recover_from_collision(self, data: dict) -> None:
        """Stop pushing and add the touched surface to the obstacle map.

        충돌한 물체를 계속 밀지 않고 충돌면을 임시 장애물로 등록합니다.
        """
        object_name = str(data.get("collision_object", "알 수 없는 물체"))
        # Stop the active move command before doing any A* work. Planning can
        # take long enough for the old command to keep pushing into the wall.
        # A* 계산 전에 기존 이동 명령부터 취소합니다. 경로를 계산하는 동안
        # 이전 명령이 벽을 계속 미는 상황을 막습니다.
        self.worker.submit("emergency", priority=0)
        if not self._active_target or not self.auto_replan_checkbox.isChecked():
            self.message_label.setText(f"충돌 감지: {object_name} · 즉시 정지")
            return

        normal_x = float(data.get("collision_normal_x", 0.0))
        normal_y = float(data.get("collision_normal_y", 0.0))
        normal_z = float(data.get("collision_normal_z", 0.0))
        impact_x = float(data.get("collision_x", data["x"]))
        impact_y = float(data.get("collision_y", data["y"]))
        impact_z = float(
            data.get("collision_z", -float(data["altitude"]))
        )
        vehicle_z = -float(data["altitude"])

        # Complex meshes occasionally report an unreliable surface normal.
        # Infer a ceiling or floor from the impact-point direction as a backup.
        # 복잡한 메시에서는 충돌면 법선이 부정확할 수 있으므로 충돌 지점의
        # 방향을 보조 정보로 사용해 천장과 바닥을 판별합니다.
        impact_dx = impact_x - float(data["x"])
        impact_dy = impact_y - float(data["y"])
        impact_dz = impact_z - vehicle_z
        impact_horizontal = math.hypot(impact_dx, impact_dy)
        impact_is_vertical = abs(impact_dz) >= max(
            0.2,
            impact_horizontal * 0.65,
        )
        if abs(normal_z) < 0.55 and impact_is_vertical:
            # In NED, a negative impact delta is above the vehicle (ceiling),
            # so the safe escape direction has a positive/downward Z normal.
            # NED에서 음수 충돌 높이 차이는 기체 위쪽(천장)이므로 안전한
            # 회피 방향은 양수/아래쪽 Z 법선입니다.
            normal_x = 0.0
            normal_y = 0.0
            normal_z = 1.0 if impact_dz < 0.0 else -1.0
        normal_length = math.sqrt(
            normal_x * normal_x + normal_y * normal_y + normal_z * normal_z
        )
        if normal_length < 0.1:
            target_x = self._active_target[0] - float(data["x"])
            target_y = self._active_target[1] - float(data["y"])
            target_length = max(math.hypot(target_x, target_y), 1e-6)
            normal_x = -target_x / target_length
            normal_y = -target_y / target_length
            normal_z = 0.0
        else:
            normal_x /= normal_length
            normal_y /= normal_length
            normal_z /= normal_length

        # Some complex meshes return the contact normal toward the obstacle.
        # Reorient it toward the vehicle so recovery always moves away.
        # 복잡한 메시가 장애물 쪽 법선을 반환하는 경우가 있으므로, 복구 이동이
        # 항상 충돌면 반대쪽을 향하도록 기체 방향으로 법선을 보정합니다.
        away_x = float(data["x"]) - impact_x
        away_y = float(data["y"]) - impact_y
        away_z = vehicle_z - impact_z
        if normal_x * away_x + normal_y * away_y + normal_z * away_z < 0.0:
            normal_x = -normal_x
            normal_y = -normal_y
            normal_z = -normal_z

        obstacle_z = impact_z
        vertical_collision = abs(normal_z) >= 0.55
        if vertical_collision:
            # Register a local ceiling/floor patch instead of a vertical wall.
            # 수직 벽이 아니라 천장/바닥의 국소 평면으로 장애물을 등록합니다.
            patch_offsets = np.arange(-5.0, 5.1, 1.0, dtype=np.float32)
            patch_x, patch_y = np.meshgrid(patch_offsets, patch_offsets)
            collision_wall = np.column_stack(
                (
                    impact_x + patch_x.ravel(),
                    impact_y + patch_y.ravel(),
                    np.full(patch_x.size, obstacle_z, dtype=np.float32),
                )
            )
        else:
            horizontal_length = max(math.hypot(normal_x, normal_y), 1e-6)
            wall_normal_x = normal_x / horizontal_length
            wall_normal_y = normal_y / horizontal_length
            tangent_x, tangent_y = -wall_normal_y, wall_normal_x
            tangent_offsets = np.arange(-15.0, 15.1, 1.0, dtype=np.float32)
            # A single-height collision line made the planner assume the same
            # facade was open only 2 m higher. Register a conservative vertical
            # surface through every altitude layer available to this mission.
            # 한 고도의 충돌선만 등록하면 A*가 2m 위를 열린 공간으로 오판합니다.
            # 현재 임무에서 사용할 수 있는 모든 고도층에 보수적인 수직면을 만듭니다.
            vertical_offsets = np.arange(
                -self.planner.config.max_extra_altitude_m - 2.0,
                3.1,
                1.0,
                dtype=np.float32,
            )
            tangent_grid, vertical_grid = np.meshgrid(
                tangent_offsets,
                vertical_offsets,
            )
            collision_wall = np.column_stack(
                (
                    impact_x + tangent_x * tangent_grid.ravel(),
                    impact_y + tangent_y * tangent_grid.ravel(),
                    obstacle_z + vertical_grid.ravel(),
                )
            )

        # Keep learned collision surfaces until the user starts a new mission.
        # Otherwise the next LiDAR frame erases the wall and sends the vehicle
        # back into the same facade.
        # 학습한 충돌면은 새 임무 전까지 유지합니다. 다음 LiDAR 프레임이 벽을
        # 지워 기체를 같은 외벽으로 다시 보내는 현상을 방지합니다.
        if self._collision_obstacle_points.size:
            self._collision_obstacle_points = np.vstack(
                (self._collision_obstacle_points, collision_wall)
            )[-10000:]
        else:
            self._collision_obstacle_points = collision_wall
        self._refresh_obstacle_map()

        try:
            # Move beyond the planner's 2.5 m inflated safety radius before
            # starting the replacement route.
            # 새 경로를 시작하기 전에 경로계획기의 2.5m 안전 팽창 반경 밖으로
            # 충분히 후퇴합니다.
            retreat_distance = 4.5
            current_altitude = float(data["altitude"])
            if vertical_collision:
                retreat_start = (float(data["x"]), float(data["y"]))
                escape_altitude = max(
                    1.0,
                    current_altitude - normal_z * retreat_distance,
                )
                if normal_z > 0.0:
                    # A downward NED normal identifies a ceiling. Stay below
                    # this height for the rest of the current mission.
                    # NED 아래 방향 법선은 천장을 의미합니다. 현재 임무가 끝날
                    # 때까지 이 높이보다 낮게 비행합니다.
                    self._avoidance_altitude_ceiling_m = (
                        escape_altitude
                        if self._avoidance_altitude_ceiling_m is None
                        else min(
                            self._avoidance_altitude_ceiling_m,
                            escape_altitude,
                        )
                    )
                else:
                    # An upward NED normal identifies a floor. Stay above it.
                    # NED 위 방향 법선은 바닥을 의미하므로 안전 높이 이상을 유지합니다.
                    self._avoidance_altitude_floor_m = max(
                        self._avoidance_altitude_floor_m,
                        escape_altitude,
                    )
            else:
                retreat_start = (
                    float(data["x"]) + normal_x * retreat_distance,
                    float(data["y"]) + normal_y * retreat_distance,
                )
                escape_altitude = current_altitude
            self._last_replan = time.monotonic()
            # Plan from the expected retreat point. The worker physically backs
            # away first, climbs in place, and only then starts horizontal flight.
            # 예상 후퇴 지점에서 경로를 계산합니다. 작업 스레드는 실제로 먼저
            # 후퇴하고 제자리 상승을 마친 뒤에만 수평 비행을 시작합니다.
            self._plan_and_fly(
                replan=True,
                start_override=retreat_start,
                altitude_override=escape_altitude,
                collision_escape=(
                    normal_x,
                    normal_y,
                    normal_z,
                    current_altitude,
                    escape_altitude,
                ),
            )
            self._avoidance_grace_until = time.monotonic() + 0.75
            self.message_label.setText(
                f"충돌 복구: {object_name} · "
                f"{'하강' if normal_z > 0.55 else '상승' if normal_z < -0.55 else '후퇴'} 후 우회"
            )
        except Exception as exc:
            self._on_error(f"충돌 후 우회 경로 생성 실패: {exc}")

    def _on_images(self, images: dict) -> None:
        detections = images.get("_detections", [])
        if self.radar_checkbox.isChecked() and self._latest_radar_tracks:
            detections = RadarProcessor.associate_camera_detections(
                detections,
                self._latest_radar_tracks,
            )
        detections = [
            {**detection, "lidar_visible": self.lidar_checkbox.isChecked()}
            for detection in detections
        ]
        detection_error = str(images.get("_detection_error", ""))
        camera_images = {
            name: data
            for name, data in images.items()
            if not name.startswith("_") and isinstance(data, bytes)
        }
        self.camera_viewer.update_images(camera_images, detections)
        self.recorder.capture(
            images,
            detections,
            self._collection_mission_metadata(),
        )
        self._refresh_collection_status()
        if detection_error and not self._detection_error_reported:
            self._detection_error_reported = True
            self.message_label.setText(
                f"드론·사람 탐지 API 비활성: {detection_error}"
            )
        self._on_enemy_detections(detections)

    def _combined_obstacle_points(self) -> np.ndarray:
        """Combine enabled ranging sensors with persistent collision memory.

        활성 거리 센서 점군과 충돌로 학습한 장애물 정보를 합칩니다.
        """
        clouds = [
            cloud
            for cloud in (
                self._latest_lidar_world,
                self._radar_obstacle_points,
                self._enemy_obstacle_points,
                self._collision_obstacle_points,
            )
            if cloud.size
        ]
        if not clouds:
            return np.empty((0, 3), dtype=np.float32)
        return np.vstack(clouds).astype(np.float32, copy=False)

    def _points_near_vehicle(
        self,
        points: np.ndarray,
        maximum_distance_m: float,
    ) -> np.ndarray:
        """Return obstacle points close enough to affect the current flight."""
        if self._telemetry is None or not points.size:
            return np.empty((0, 3), dtype=np.float32)
        cloud = np.asarray(points, dtype=np.float32).reshape((-1, 3))
        finite = np.all(np.isfinite(cloud), axis=1)
        horizontal = np.hypot(
            cloud[:, 0] - float(self._telemetry["x"]),
            cloud[:, 1] - float(self._telemetry["y"]),
        )
        return cloud[finite & (horizontal <= float(maximum_distance_m))]

    def _planning_obstacle_points(self) -> np.ndarray:
        """Build a local collision map without turning remote targets into walls.

        표시용 장거리 Radar/카메라 표적은 유지하되, 실제 A* 회피 지도에는
        현재 비행에 영향을 줄 수 있는 근거리 점만 넣습니다.
        """
        clouds = [
            cloud
            for cloud in (
                self._points_near_vehicle(self._latest_lidar_world, 140.0),
                self._points_near_vehicle(self._radar_obstacle_points, 40.0),
                self._points_near_vehicle(self._enemy_obstacle_points, 40.0),
                self._collision_obstacle_points,
            )
            if cloud.size
        ]
        if not clouds:
            return np.empty((0, 3), dtype=np.float32)
        return np.vstack(clouds).astype(np.float32, copy=False)

    def _refresh_obstacle_map(self) -> None:
        """Push the currently enabled sensor clouds to planner and minimap.

        현재 활성화된 센서 점군을 경로계획기와 미니맵에 반영합니다.
        """
        displayed = self._combined_obstacle_points()
        self.planner.set_obstacle_points(self._planning_obstacle_points())
        self.minimap.set_obstacles(displayed)

    def _on_enemy_detections(self, detections: list[dict]) -> None:
        """Display enemy tracks and replan around an approaching drone.

        적 드론 추적 상태를 표시하고 접근하는 기체 주변으로 경로를 재탐색합니다.
        """
        now = time.monotonic()
        if detections:
            self._last_enemy_seen = now
            self._enemy_obstacle_points = TargetDetector.obstacle_points(
                detections,
                padding_m=2.5,
            )
            closest = min(
                detections,
                key=lambda detection: float(
                    detection.get("distance_m", float("inf"))
                ),
            )
            closest_distance = float(closest.get("distance_m", 0.0))
            human_count = sum(
                detection.get("target_kind") == "human"
                for detection in detections
            )
            bird_count = sum(
                detection.get("target_kind") == "bird"
                for detection in detections
            )
            drone_count = len(detections) - human_count - bird_count
            self.telemetry_labels["enemy"].setText(
                f"사람 {human_count}명 · 새 {bird_count}마리 · 드론 {drone_count}대 · "
                f"최근접 {closest_distance:.1f} m"
            )
        else:
            if now - self._last_enemy_seen > 1.5:
                self._enemy_obstacle_points = np.empty(
                    (0, 3),
                    dtype=np.float32,
                )
                self.telemetry_labels["enemy"].setText("탐지 없음")
            self._enemy_detection_count = 0
            self._refresh_obstacle_map()
            return

        self._refresh_obstacle_map()

        threats = [
            detection
            for detection in detections
            if TargetDetector.is_collision_threat(detection)
        ]
        if not threats or not self._active_target:
            self._enemy_detection_count = 0
            return

        self._enemy_detection_count += 1
        if self._enemy_detection_count < 2:
            return

        closest_threat = min(
            threats,
            key=lambda detection: float(detection["distance_m"]),
        )
        distance = float(closest_threat["distance_m"])
        if not self.enemy_avoidance_checkbox.isChecked():
            if distance <= 5.0:
                self.worker.submit("emergency", priority=0)
                self.message_label.setText(
                    f"적 드론 {distance:.1f}m 접근 · 안전 호버링"
                )
            return

        if (
            now < self._avoidance_grace_until
            or now - self._last_enemy_replan < 3.0
        ):
            return

        self._last_enemy_replan = now
        self._enemy_detection_count = 0
        speed = 0.0 if self._telemetry is None else float(self._telemetry["speed"])
        emergency_distance = max(4.0, speed * 2.0 + 2.0)
        try:
            # Stop only when contact is imminent; otherwise replace the route
            # without producing unnecessary stop-and-go motion.
            # 충돌이 임박한 경우에만 정지하고, 그 외에는 불필요한 끊김 없이
            # 현재 경로를 동적 장애물 우회 경로로 교체합니다.
            if distance <= emergency_distance:
                self.worker.submit("emergency", priority=0)
            self._plan_and_fly(replan=True)
            self._avoidance_grace_until = now + 2.0
            self.message_label.setText(
                f"적 드론 {distance:.1f}m 전방 감지 · 동적 회피 경로 생성"
            )
        except Exception as exc:
            self.worker.submit("emergency", priority=0)
            self._on_error(f"적 드론 회피 실패 · 호버링: {exc}")

    @staticmethod
    def _sensor_points_to_world(points: np.ndarray, pose: dict) -> np.ndarray:
        """Transform vehicle-local NED points into AirSim world NED.

        기체 로컬 NED 센서 점을 AirSim 월드 NED 좌표로 변환합니다.
        """
        if not points.size:
            return np.empty((0, 3), dtype=np.float32)
        quaternion = np.asarray(
            [
                float(pose.get("qw", 1.0)),
                float(pose.get("qx", 0.0)),
                float(pose.get("qy", 0.0)),
                float(pose.get("qz", 0.0)),
            ],
            dtype=np.float64,
        )
        quaternion /= max(float(np.linalg.norm(quaternion)), 1e-9)
        qw, qx, qy, qz = quaternion
        rotation = np.asarray(
            [
                [
                    1.0 - 2.0 * (qy * qy + qz * qz),
                    2.0 * (qx * qy - qz * qw),
                    2.0 * (qx * qz + qy * qw),
                ],
                [
                    2.0 * (qx * qy + qz * qw),
                    1.0 - 2.0 * (qx * qx + qz * qz),
                    2.0 * (qy * qz - qx * qw),
                ],
                [
                    2.0 * (qx * qz - qy * qw),
                    2.0 * (qy * qz + qx * qw),
                    1.0 - 2.0 * (qx * qx + qy * qy),
                ],
            ],
            dtype=np.float32,
        )
        world = np.asarray(points, dtype=np.float32) @ rotation.T
        world[:, 0] += float(pose["x"])
        world[:, 1] += float(pose["y"])
        world[:, 2] += float(pose["z"])
        return world

    def _on_radar(self, snapshot: object) -> None:
        """Display Echo returns and fuse labeled enemy tracks with avoidance.

        Echo 반사점을 표시하고 객체명이 확인된 적 표적을 회피 지도에 융합합니다.
        """
        data = dict(snapshot)
        self.recorder.update_radar(data)
        if not self.radar_checkbox.isChecked():
            return
        points = np.asarray(data.get("points", []), dtype=np.float32).reshape((-1, 3))
        if not points.size:
            self._latest_radar_tracks = []
            self._radar_obstacle_points = np.empty((0, 3), dtype=np.float32)
            self.radar_viewer.update_points(np.empty((0, 3), dtype=np.float32))
            self.telemetry_labels["radar"].setText("반사점 없음")
            self._refresh_obstacle_map()
            return

        finite = np.all(np.isfinite(points), axis=1)
        finite &= np.linalg.norm(points, axis=1) > 0.75
        indices = np.flatnonzero(finite)
        local = points[finite]
        self.radar_viewer.update_points(local)
        if not local.size:
            self._latest_radar_tracks = []
            self._radar_obstacle_points = np.empty((0, 3), dtype=np.float32)
            self.telemetry_labels["radar"].setText("유효 반사점 없음")
            self._refresh_obstacle_map()
            return

        pose = dict(data.get("pose", {}))
        world = self._sensor_points_to_world(local, pose)
        raw_labels = list(data.get("labels", []))
        labels = [raw_labels[index] if index < len(raw_labels) else "" for index in indices]
        attenuation = np.asarray(data.get("attenuation", []), dtype=np.float32)
        total_distance = np.asarray(data.get("total_distance", []), dtype=np.float32)
        reflection_count = np.asarray(
            data.get("reflection_count", []),
            dtype=np.float32,
        )
        attenuation = attenuation[indices]
        total_distance = total_distance[indices]
        reflection_count = reflection_count[indices]
        tracks = RadarProcessor.build_enemy_tracks(
            world,
            labels,
            total_distance,
            attenuation,
            reflection_count,
        )
        self._latest_radar_tracks = tracks
        self._radar_obstacle_points = RadarProcessor.obstacle_points(
            tracks,
            padding_m=3.0,
        )
        self._refresh_obstacle_map()

        if tracks:
            closest = min(tracks, key=lambda track: float(track["distance_m"]))
            human_count = sum(
                track.get("target_kind") == "human"
                for track in tracks
            )
            bird_count = sum(
                track.get("target_kind") == "bird"
                for track in tracks
            )
            drone_count = len(tracks) - human_count - bird_count
            self.telemetry_labels["radar"].setText(
                f'{len(local):,} returns · 사람 {human_count}명 · '
                f'새 {bird_count}마리 · 드론 {drone_count}대 · '
                f'{float(closest["distance_m"]):.1f} m'
            )
        else:
            self.telemetry_labels["radar"].setText(
                f"{len(local):,} returns · 표적 없음"
            )

    def _nearest_obstacle_ahead(
        self,
        world: np.ndarray,
        pose: dict,
    ) -> tuple[float, tuple[float, float, float], tuple[float, float], float] | None:
        """Return details for the nearest LiDAR hit in the motion corridor.

        드론의 이동 통로 안에서 가장 가까운 LiDAR 장애물의 거리·위치·진행
        방향·벽 폭 추정값을 반환합니다.
        """
        if self._telemetry is None or not world.size:
            return None

        velocity_x = float(self._telemetry["vx"])
        velocity_y = float(self._telemetry["vy"])
        horizontal_speed = math.hypot(velocity_x, velocity_y)
        if self._planned_path:
            # Always inspect the next commanded segment before using velocity.
            # Immediately after replanning, inertia still points at the old
            # wall and would otherwise cancel the new lateral detour repeatedly.
            # 항상 현재 속도보다 새로 명령한 다음 경로 구간을 먼저 검사합니다.
            # 재탐색 직후 관성은 여전히 기존 벽을 향하므로, 속도를 기준으로 하면
            # 새 측면 우회 명령을 계속 취소하는 문제가 생깁니다.
            waypoint = next(
                (
                    point
                    for point in self._planned_path
                    if math.hypot(
                        point[0] - float(pose["x"]),
                        point[1] - float(pose["y"]),
                    )
                    > 1.5
                ),
                None,
            )
            if waypoint is None:
                return None
            target_x = waypoint[0] - float(pose["x"])
            target_y = waypoint[1] - float(pose["y"])
            target_distance = math.hypot(target_x, target_y)
            direction_x = target_x / target_distance
            direction_y = target_y / target_distance
        elif horizontal_speed >= 0.35:
            direction_x = velocity_x / horizontal_speed
            direction_y = velocity_y / horizontal_speed
        elif self._active_target is not None:
            target_x = self._active_target[0] - float(pose["x"])
            target_y = self._active_target[1] - float(pose["y"])
            target_distance = math.hypot(target_x, target_y)
            if target_distance < 0.1:
                return None
            direction_x = target_x / target_distance
            direction_y = target_y / target_distance
        else:
            return None

        delta_x = world[:, 0] - float(pose["x"])
        delta_y = world[:, 1] - float(pose["y"])
        forward = delta_x * direction_x + delta_y * direction_y
        signed_lateral = -delta_x * direction_y + delta_y * direction_x
        lateral = np.abs(signed_lateral)
        vertical = np.abs(world[:, 2] - float(pose["z"]))

        # Give the planner enough distance to stop and choose a new route.
        # 드론이 정지한 뒤 새 경로를 선택할 수 있도록 충분한 탐지 거리를 둡니다.
        # At 8 m/s this looks roughly 40 m ahead. The previous three-second
        # cooldown could let the vehicle cover 24 m before checking again.
        # 8m/s에서는 약 40m 전방을 검사합니다. 기존 3초 유예는 재검사 전에
        # 최대 24m를 진행하게 만들어 벽에 닿을 수 있었습니다.
        detection_distance = max(15.0, horizontal_speed * 4.0 + 8.0)
        corridor_half_width = self.planner.config.drone_radius_m + 0.75
        mask = (
            (forward >= 0.5)
            & (forward <= detection_distance)
            & (lateral <= corridor_half_width)
            & (
                vertical
                <= self.planner.config.vertical_clearance_m + 0.75
            )
        )
        if not np.any(mask):
            return None
        candidates = np.flatnonzero(mask)
        candidates = candidates[np.argsort(forward[candidates])]
        nearest_index: int | None = None
        # Reject isolated rays. A real wall/large obstacle produces a compact
        # group of returns; one or two points are commonly vegetation edges,
        # particles, or residual self-reflections.
        # 실제 벽은 가까운 반사점 묶음을 만들지만 1~2개 점은 식생 가장자리,
        # 파티클 또는 기체 잔여 반사일 가능성이 높으므로 장애물로 확정하지 않습니다.
        for candidate in candidates[:96]:
            neighborhood = mask & (
                (np.abs(forward - forward[candidate]) <= 1.75)
                & (np.abs(signed_lateral - signed_lateral[candidate]) <= 1.5)
                & (np.abs(world[:, 2] - world[candidate, 2]) <= 1.5)
            )
            if int(np.count_nonzero(neighborhood)) >= 4:
                nearest_index = int(candidate)
                break
        if nearest_index is None:
            return None
        nearest_distance = float(forward[nearest_index])
        # Estimate the visible facade width from returns near the first hit,
        # then add clearance for sparse scans and the vehicle body.
        # 최초 반사점 주변 점으로 보이는 외벽 폭을 추정하고 희소한 스캔과
        # 기체 크기를 고려한 여유 폭을 더합니다.
        wall_band = mask & (forward <= nearest_distance + 6.0)
        visible_half_span = (
            float(np.percentile(lateral[wall_band], 90))
            if np.any(wall_band)
            else 0.0
        )
        barrier_half_span = min(25.0, max(6.0, visible_half_span + 4.0))
        hit = tuple(float(value) for value in world[nearest_index])
        return (
            nearest_distance,
            hit,
            (direction_x, direction_y),
            barrier_half_span,
        )

    def _remember_detected_wall(
        self,
        hit_xyz: tuple[float, float, float],
        travel_direction_xy: tuple[float, float],
        half_span_m: float,
    ) -> int:
        """Persist a detected facade across all usable flight layers.

        감지한 외벽을 현재 임무의 모든 사용 가능 고도층에 보존합니다.
        """
        requested_altitude = (
            float(self._active_target[2])
            if self._active_target is not None
            else float(self.destination_altitude.value())
        )
        maximum_altitude = requested_altitude + self.planner.config.max_extra_altitude_m
        if self._avoidance_altitude_ceiling_m is not None:
            maximum_altitude = min(maximum_altitude, self._avoidance_altitude_ceiling_m)
        barrier = build_vertical_barrier(
            hit_xyz,
            travel_direction_xy,
            half_span_m,
            self._avoidance_altitude_floor_m,
            max(self._avoidance_altitude_floor_m, maximum_altitude),
        )
        previous_count = len(self._collision_obstacle_points)
        if self._collision_obstacle_points.size:
            self._collision_obstacle_points = np.vstack(
                (self._collision_obstacle_points, barrier)
            )[-20000:]
        else:
            self._collision_obstacle_points = barrier
        self._refresh_obstacle_map()
        return previous_count

    def _rollback_detected_wall(self, previous_count: int) -> None:
        """Remove the most recent speculative LiDAR wall after a failed plan."""
        if previous_count <= 0:
            self._collision_obstacle_points = np.empty((0, 3), dtype=np.float32)
        elif previous_count <= len(self._collision_obstacle_points):
            self._collision_obstacle_points = self._collision_obstacle_points[
                :previous_count
            ].copy()
        self._refresh_obstacle_map()

    def _on_lidar(self, snapshot: object) -> None:
        points, pose = snapshot
        self.recorder.update_lidar(points, pose)
        if not self.lidar_checkbox.isChecked():
            return
        self.lidar_viewer.update_points(points)
        if not points.size:
            self._latest_lidar_world = np.empty((0, 3), dtype=np.float32)
            self._obstacle_detection_count = 0
            self.telemetry_labels["lidar"].setText("반사점 없음")
            self._refresh_obstacle_map()
            return
        # Ignore very short returns from the vehicle body and attached camera
        # meshes. They are not external obstacles.
        # 기체 본체와 부착 카메라 메시에서 생기는 근거리 반사점은 외부
        # 장애물이 아니므로 제외합니다.
        valid = np.linalg.norm(points, axis=1) > 0.75
        local = points[valid]
        if not local.size:
            self._latest_lidar_world = np.empty((0, 3), dtype=np.float32)
            self._obstacle_detection_count = 0
            self.telemetry_labels["lidar"].setText("유효 반사점 없음")
            self._refresh_obstacle_map()
            return
        # SensorLocalFrame follows vehicle roll and pitch as well as yaw.
        # Applying yaw alone turned ground returns into phantom obstacles while
        # the multirotor was tilted during acceleration.
        # SensorLocalFrame은 Yaw뿐 아니라 Roll/Pitch도 함께 움직입니다.
        # 모든 축 회전을 반영해 가속 중 지면 점이 가짜 장애물이 되지 않게 합니다.
        world = self._sensor_points_to_world(local, pose)
        self._latest_lidar_world = world
        self._refresh_obstacle_map()

        obstacle_ahead = self._nearest_obstacle_ahead(world, pose)
        preview_distance = obstacle_ahead[0] if obstacle_ahead is not None else None
        if "lidar" in self.telemetry_labels:
            detection_text = (
                f"전방 {preview_distance:.1f} m"
                if preview_distance is not None
                else "전방 장애물 없음"
            )
            self.telemetry_labels["lidar"].setText(
                f"{len(local):,} points · {detection_text}"
            )

        if not self._active_target:
            return
        now = time.monotonic()
        if now < self._avoidance_grace_until:
            self._obstacle_detection_count = 0
            return

        if obstacle_ahead is not None:
            obstacle_distance, hit_xyz, travel_direction, barrier_half_span = obstacle_ahead
            self._obstacle_detection_count += 1
            speed = 0.0 if self._telemetry is None else float(self._telemetry["speed"])
            escape_distance = max(7.0, speed * 1.25 + 3.0)

            # Require a short three-frame confirmation in addition to the
            # four-point spatial cluster above. At normal sensor rates this is
            # still early enough for the 15-40 m preview corridor.
            # 위의 4점 공간 군집에 더해 3프레임 연속 확인합니다. 일반 센서
            # 주기에서는 15~40m 사전 탐지 범위 안에서 충분히 빠르게 반응합니다.
            if self._obstacle_detection_count < 3:
                return

            if not self.auto_replan_checkbox.isChecked():
                if (
                    now - self._last_emergency_stop > 0.75
                ):
                    self._last_emergency_stop = now
                    self.worker.submit("emergency", priority=0)
                self.message_label.setText(
                    f"전방 {obstacle_distance:.1f}m 장애물 감지 · 안전 호버링"
                )
                return

            if now - self._last_replan > 0.75:
                self._last_replan = now
                self._obstacle_detection_count = 0
                previous_wall_count = len(self._collision_obstacle_points)
                try:
                    previous_wall_count = self._remember_detected_wall(
                        hit_xyz,
                        travel_direction,
                        barrier_half_span,
                    )
                    close_escape = (
                        obstacle_distance <= escape_distance
                        and self._telemetry is not None
                    )
                    if close_escape:
                        # Only a genuinely close obstacle needs braking and a
                        # short reverse escape. Far detections stay in motion.
                        # 정말 가까운 장애물에서만 제동 후 짧게 후퇴합니다.
                        # 멀리서 감지한 경우에는 이동을 유지한 채 경로만 바꿉니다.
                        self._last_emergency_stop = now
                        self.worker.submit("emergency", priority=0)
                        # If already close, back away before following the new
                        # side route. This is proactive escape, not collision recovery.
                        # 이미 가까우면 새 측면 경로를 따르기 전에 먼저 후퇴합니다.
                        # 실제 충돌 후 복구가 아니라 충돌 전 선제 회피입니다.
                        retreat_distance = 4.5
                        retreat_start = (
                            float(self._telemetry["x"]) - travel_direction[0] * retreat_distance,
                            float(self._telemetry["y"]) - travel_direction[1] * retreat_distance,
                        )
                        current_altitude = float(self._telemetry["altitude"])
                        self._plan_and_fly(
                            replan=True,
                            start_override=retreat_start,
                            altitude_override=current_altitude,
                            collision_escape=(
                                -travel_direction[0],
                                -travel_direction[1],
                                0.0,
                                current_altitude,
                                current_altitude,
                            ),
                        )
                    else:
                        # moveOnPathAsync replaces the old straight path with
                        # the detour without inserting a hover command.
                        # hover 명령을 끼우지 않고 기존 직선 경로를 우회 경로로
                        # 교체하여 비행을 계속합니다.
                        self._plan_and_fly(replan=True)
                    # Give the first lateral waypoint time to establish motion.
                    # During this lock the new path is not cancelled by the
                    # vehicle's remaining forward inertia.
                    # 첫 측면 웨이포인트로 이동할 시간을 주어 남아 있는 전진
                    # 관성이 새 우회 경로를 취소하지 않게 합니다.
                    self._avoidance_grace_until = time.monotonic() + 2.0
                    avoidance_mode = (
                        "안전 후퇴 후 우회"
                        if close_escape
                        else "이동 유지하며 우회"
                    )
                    self.message_label.setText(
                        f"전방 {obstacle_distance:.1f}m 벽 사전 감지 · {avoidance_mode}"
                    )
                except Exception as exc:
                    # A speculative wall must never remain in the map when it
                    # made A* impossible. Roll it back, retry with a narrower
                    # local wall, and keep the old flight command for a distant
                    # false positive instead of stopping in empty space.
                    # 새로 만든 추정 벽 때문에 A*가 실패하면 해당 벽을 되돌리고
                    # 더 좁은 국소 벽으로 재시도합니다. 먼 거리 오탐이면 기존
                    # 비행 명령을 유지하여 빈 공간에서 멈추지 않습니다.
                    self._rollback_detected_wall(previous_wall_count)
                    narrow_wall_count = len(self._collision_obstacle_points)
                    try:
                        narrow_wall_count = self._remember_detected_wall(
                            hit_xyz,
                            travel_direction,
                            max(4.0, barrier_half_span * 0.55),
                        )
                        self._plan_and_fly(replan=True)
                        self._avoidance_grace_until = time.monotonic() + 2.0
                        self.message_label.setText(
                            f"전방 {obstacle_distance:.1f}m 장애물 · 좁은 측면 우회로 계속 진행"
                        )
                    except Exception as narrow_exc:
                        self._rollback_detected_wall(narrow_wall_count)
                        if obstacle_distance <= escape_distance:
                            self.worker.submit("emergency", priority=0)
                            self.message_label.setText(
                                f"전방 {obstacle_distance:.1f}m 근접 장애물 · "
                                f"안전 경로 재확인 중 ({narrow_exc})"
                            )
                        else:
                            self._avoidance_grace_until = time.monotonic() + 1.5
                            self.message_label.setText(
                                f"전방 {obstacle_distance:.1f}m 희소 반사 제외 · 기존 경로 계속 진행"
                            )
                return
        else:
            self._obstacle_detection_count = 0

    def _on_command_completed(self, command: str) -> None:
        if command == "takeoff":
            self._takeoff_pending = False
            self.takeoff_button.setEnabled(self._connected)
        names = {
            "arm": "ARM/DISARM 명령 전송",
            "spawn": "선택한 스폰 A 위치를 적용했습니다.",
            "takeoff": "이륙 완료 · 설정한 고도에 도달했습니다.",
            "hover": "호버링 명령 전송",
            "mission_stop": "미션 중지 · 현재 위치 호버링",
            "move": "목적지 이동 시작",
            "path": "경로 비행 시작",
            "land": "빠른 접근 착륙 명령 전송",
            "emergency": "긴급 정지 명령 전송",
            "segmentation": "Segmentation 클래스를 현재 레벨에 다시 적용했습니다.",
        }
        self.message_label.setText(names.get(command, command))

    def _on_error(self, message: str) -> None:
        if message.startswith(("connect:", "disconnect:")):
            self._connection_transition = None
            self.connect_button.setEnabled(True)
            self.connect_button.setText(
                "연결 해제" if self._connected else "AirSim 연결"
            )
        if message.startswith("takeoff:"):
            self._takeoff_pending = False
            self.takeoff_button.setEnabled(self._connected)
        if message.startswith("environment:"):
            self._environment_apply_pending = False
            self.environment_apply_button.setEnabled(self._connected)
            self.environment_apply_button.setText("환경 적용")
            self.environment_status_label.setText(
                "환경 적용 실패 · 이전 적용값을 계속 기록합니다."
            )
            self._append_environment_log(f"오류 → {message}")
        self.message_label.setText(message)
        if not message.startswith(("센서:", "카메라:", "LiDAR:", "Radar:")):
            QMessageBox.warning(self, "Mission Control", message)

    def _restore_settings(self) -> None:
        self.takeoff_altitude.setValue(float(self.settings.value("takeoff_altitude", 5.0)))
        self.destination_x.setValue(float(self.settings.value("destination_x", 10.0)))
        self.destination_y.setValue(float(self.settings.value("destination_y", 0.0)))
        self.destination_altitude.setValue(float(self.settings.value("destination_altitude", 5.0)))
        self.speed.setValue(float(self.settings.value("speed", 3.0)))
        self.patrol_duration_minutes.setValue(
            float(self.settings.value("patrol_duration_minutes", 5.0))
        )
        try:
            stored_route = json.loads(str(self.settings.value("route_waypoints", "[]")))
            self._route_waypoints = [
                (float(point[0]), float(point[1]), float(point[2]))
                for point in stored_route
                if isinstance(point, (list, tuple)) and len(point) == 3
            ]
        except (TypeError, ValueError, json.JSONDecodeError):
            self._route_waypoints = []
        self._refresh_route_list()
        self.dataset_name_edit.setText(
            str(self.settings.value("collection_dataset_name", "KoreaDroneDataset"))
        )
        self.city_edit.setText(str(self.settings.value("collection_city", "AirBase")))
        self.region_edit.setText(
            str(self.settings.value("collection_region", "default"))
        )
        self.terrain_combo.setCurrentText(
            str(self.settings.value("collection_terrain", "산업·공항"))
        )
        self.collection_root_edit.setText(
            str(
                self.settings.value(
                    "collection_root",
                    str(Path.home() / "Documents" / "AutonomousDroneDatasets"),
                )
            )
        )
        self.collection_rate.setValue(
            float(self.settings.value("collection_rate_hz", 2.0))
        )
        for key, checkbox in self.collection_sensor_checks.items():
            stored = str(
                self.settings.value(f"collection_sensor_{key}", "true")
            ).strip().lower()
            checkbox.setChecked(stored not in {"false", "0", "no", "off"})
        auto_report = str(
            self.settings.value("collection_auto_report", "true")
        ).strip().lower()
        self.auto_report_checkbox.setChecked(auto_report not in {"false", "0", "no", "off"})
        # Every Mission Control run starts from a known, reproducible clear-day
        # baseline. Restoring the last test (often fog or rain) made a fresh run
        # appear to have the wrong default environment.
        environment_widgets = (
            (self.season_combo, "environment_season", "summer"),
            (self.time_of_day_combo, "environment_time_of_day", "noon"),
            (self.visibility_combo, "environment_visibility", "clear"),
            (self.precipitation_combo, "environment_precipitation", "none"),
        )
        for combo, _key, default in environment_widgets:
            index = combo.findData(default)
            combo.setCurrentIndex(index if index >= 0 else combo.findData(default))
        self.precipitation_intensity.setValue(0.6)
        self.wind_north.setValue(0.0)
        self.wind_east.setValue(0.0)
        self._environment_state = self._selected_environment()
        self._update_environment_input_state()

    def _save_settings(self) -> None:
        self.settings.setValue("takeoff_altitude", self.takeoff_altitude.value())
        self.settings.setValue("destination_x", self.destination_x.value())
        self.settings.setValue("destination_y", self.destination_y.value())
        self.settings.setValue("destination_altitude", self.destination_altitude.value())
        self.settings.setValue("speed", self.speed.value())
        self.settings.setValue(
            "patrol_duration_minutes",
            self.patrol_duration_minutes.value(),
        )
        self.settings.setValue("route_waypoints", json.dumps(self._route_waypoints))
        self.settings.setValue("collection_dataset_name", self.dataset_name_edit.text())
        self.settings.setValue("collection_city", self.city_edit.text())
        self.settings.setValue("collection_region", self.region_edit.text())
        self.settings.setValue("collection_terrain", self.terrain_combo.currentText())
        self.settings.setValue("collection_root", self.collection_root_edit.text())
        self.settings.setValue("collection_rate_hz", self.collection_rate.value())
        for key, checkbox in self.collection_sensor_checks.items():
            self.settings.setValue(f"collection_sensor_{key}", checkbox.isChecked())
        self.settings.setValue(
            "collection_auto_report",
            self.auto_report_checkbox.isChecked(),
        )
    def closeEvent(self, event: QCloseEvent) -> None:  # noqa: N802 - Qt API
        self._save_settings()
        self.recorder.stop(timeout_seconds=5.0)
        if self._report_worker is not None and self._report_worker.isRunning():
            self._report_worker.wait(15000)
        self.worker.stop()
        self.worker.wait(2500)
        event.accept()

    def _apply_style(self) -> None:
        self.setStyleSheet(
            """
            QMainWindow, QWidget { background:#151a22; color:#e7edf5; font-size:13px; }
            QFrame { border:none; }
            QGroupBox { border:1px solid #354052; border-radius:7px; margin-top:12px; padding:12px 8px 8px; font-weight:600; }
            QGroupBox::title { subcontrol-origin:margin; left:10px; padding:0 5px; color:#8ecbff; }
            QPushButton { background:#293445; border:1px solid #45546a; border-radius:5px; padding:8px; }
            QPushButton:hover { background:#34445a; }
            QPushButton:disabled { color:#66707e; background:#202630; }
            QPushButton#emergency { background:#8f2735; border-color:#d94c5d; font-weight:700; }
            QDoubleSpinBox { background:#0f141b; border:1px solid #3a4658; border-radius:4px; padding:5px; }
            QTabWidget::pane { border:1px solid #354052; }
            QScrollArea#controlScroll { background:#151a22; border:none; }
            QScrollBar:vertical { background:#151a22; width:12px; margin:0; }
            QScrollBar::handle:vertical { background:#45546a; min-height:36px; border-radius:5px; }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height:0; }
            QTabBar::tab { background:#202936; padding:9px 16px; }
            QTabBar::tab:selected { background:#31547a; }
            QLabel#title { font-size:18px; font-weight:800; color:#d8edff; }
            QLabel#connected { color:#55db8a; font-weight:700; }
            QLabel#disconnected { color:#f17a84; font-weight:700; }
            QLabel#message { background:#0f141b; border:1px solid #303a49; padding:8px; color:#b7c4d5; }
            QLabel#sensorTitle { color:#8ecbff; font-weight:700; }
            """
        )


def main() -> int:
    app = QApplication(sys.argv)
    app.setApplicationName("Autonomous Drone Mission Control")
    window = MissionControlWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())

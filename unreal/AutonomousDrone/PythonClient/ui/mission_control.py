"""Cosys-AirSim mission-control desktop application."""

from __future__ import annotations

import itertools
import math
import queue
import sys
import time
from pathlib import Path

PYTHON_CLIENT_ROOT = Path(__file__).resolve().parents[1]
if str(PYTHON_CLIENT_ROOT) not in sys.path:
    sys.path.insert(0, str(PYTHON_CLIENT_ROOT))

import numpy as np
from PySide6.QtCore import QSettings, QThread, Signal
from PySide6.QtGui import QCloseEvent
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QDoubleSpinBox,
    QFormLayout,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSplitter,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from common.airsim_client import AirSimController
from missions.heart_mission import build_heart_path
from missions.square_mission import build_square_path
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
    telemetry_updated = Signal(dict)
    images_updated = Signal(dict)
    lidar_updated = Signal(object)
    radar_updated = Signal(object)
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
        self._last_sensor_error = 0.0

    def submit(self, name: str, *args, priority: int = 10) -> None:
        self._commands.put((priority, next(self._sequence), name, args))

    def stop(self) -> None:
        self._running = False

    def set_sensor_streaming(self, enabled: bool) -> None:
        self._stream_sensors = enabled

    def set_lidar_enabled(self, enabled: bool) -> None:
        self._lidar_enabled = bool(enabled)
        if self.controller.connected:
            self.submit("sensor_debug", "lidar", self._lidar_enabled, priority=2)

    def set_radar_enabled(self, enabled: bool) -> None:
        self._radar_enabled = bool(enabled)
        if self.controller.connected:
            self.submit("sensor_debug", "radar", self._radar_enabled, priority=2)

    def run(self) -> None:
        next_telemetry = 0.0
        next_images = 0.0
        next_lidar = 0.0
        next_radar = 0.0
        while self._running:
            self._process_pending_commands()
            now = time.monotonic()
            if self.controller.connected and now >= next_telemetry:
                self._poll_telemetry()
                next_telemetry = now + 0.1
            # LiDAR is a flight-safety input and must keep running even when
            # the optional camera preview stream is disabled.
            # LiDAR는 비행 안전에 필요한 입력이므로 선택형 카메라 미리보기
            # 스트리밍이 꺼져 있어도 계속 작동해야 합니다.
            if self.controller.connected and self._lidar_enabled and now >= next_lidar:
                self._poll_lidar()
                next_lidar = now + 0.15
            # Radar is polled independently so either ranging sensor can be
            # enabled without forcing the other sensor to run.
            # 두 거리 센서를 독립적으로 켜고 끌 수 있도록 Radar는 별도
            # 주기로 수신합니다.
            if self.controller.connected and self._radar_enabled and now >= next_radar:
                self._poll_radar()
                next_radar = now + 0.1
            if self.controller.connected and self._stream_sensors and now >= next_images:
                self._poll_images()
                next_images = now + 0.5
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
                    self.controller.connect()
                    # Reapply both default-on UI states after each connection.
                    # 연결할 때마다 기본 ON인 두 UI 상태를 언리얼 표시에 동기화합니다.
                    self.controller.set_sensor_debug_visualization(
                        "lidar", self._lidar_enabled
                    )
                    self.controller.set_sensor_debug_visualization(
                        "radar", self._radar_enabled
                    )
                    self.connection_changed.emit(True, "연결됨")
                elif name == "disconnect":
                    self.controller.disconnect()
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
                    self.controller.land()
                elif name == "emergency":
                    self.controller.emergency_stop()
                elif name == "sensor_debug":
                    self.controller.set_sensor_debug_visualization(
                        str(args[0]), bool(args[1])
                    )
                else:
                    raise ValueError(f"알 수 없는 명령: {name}")
                if name not in {"connect", "disconnect"}:
                    self.command_completed.emit(name)
            except Exception as exc:
                if name == "connect":
                    self.connection_changed.emit(False, "연결 실패")
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


class MissionControlWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Autonomous Drone Mission Control · Avoidance v8")
        self.resize(1500, 920)
        self.settings = QSettings("AutonomousDrone", "MissionControl")
        self._connected = False
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
        self._pending_descent_altitude: float | None = None
        self._pending_descent_safe_altitude: float | None = None
        self._pending_descent_commanded = False
        self._last_replan = 0.0
        self._last_emergency_stop = 0.0
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

        self.worker = AirSimWorker()
        self.worker.connection_changed.connect(self._on_connection_changed)
        self.worker.telemetry_updated.connect(self._on_telemetry)
        self.worker.images_updated.connect(self._on_images)
        self.worker.lidar_updated.connect(self._on_lidar)
        self.worker.radar_updated.connect(self._on_radar)
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
        header.addWidget(title)
        header.addStretch()
        header.addWidget(self.sensor_checkbox)
        header.addWidget(self.lidar_checkbox)
        header.addWidget(self.radar_checkbox)
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
        panel.setMaximumWidth(430)
        layout = QVBoxLayout(panel)

        flight_group = QGroupBox("비행 제어")
        flight_layout = QGridLayout(flight_group)
        self.arm_button = QPushButton("ARM")
        self.disarm_button = QPushButton("DISARM")
        self.takeoff_button = QPushButton("이륙")
        self.hover_button = QPushButton("호버링")
        self.land_button = QPushButton("착륙")
        self.emergency_button = QPushButton("긴급 정지")
        self.emergency_button.setObjectName("emergency")
        self.arm_button.clicked.connect(lambda: self.worker.submit("arm", True))
        self.disarm_button.clicked.connect(lambda: self.worker.submit("arm", False))
        self.takeoff_button.clicked.connect(self._takeoff)
        self.hover_button.clicked.connect(lambda: self.worker.submit("hover"))
        self.land_button.clicked.connect(lambda: self.worker.submit("land"))
        self.emergency_button.clicked.connect(lambda: self.worker.submit("emergency", priority=0))
        flight_layout.addWidget(self.arm_button, 0, 0)
        flight_layout.addWidget(self.disarm_button, 0, 1)
        flight_layout.addWidget(self.takeoff_button, 1, 0)
        flight_layout.addWidget(self.hover_button, 1, 1)
        flight_layout.addWidget(self.land_button, 2, 0)
        flight_layout.addWidget(self.emergency_button, 2, 1)
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
        self.target_select_button = QPushButton("목표 B 선택")
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
        self.auto_replan_checkbox = QCheckBox("LiDAR 장애물 감지 시 자동 재탐색")
        self.auto_replan_checkbox.setChecked(True)
        destination_form.addRow(self.auto_replan_checkbox)
        self.enemy_avoidance_checkbox = QCheckBox("적 드론·사람 인식 및 동적 회피")
        self.enemy_avoidance_checkbox.setChecked(True)
        destination_form.addRow(self.enemy_avoidance_checkbox)
        self.move_button = QPushButton("A* 경로로 목표 B 이동")
        self.move_button.clicked.connect(self._move)
        destination_form.addRow(self.move_button)
        layout.addWidget(destination_group)

        mission_group = QGroupBox("패턴 미션")
        mission_layout = QHBoxLayout(mission_group)
        self.square_button = QPushButton("사각형 비행")
        self.heart_button = QPushButton("하트 비행")
        self.square_button.clicked.connect(self._square_mission)
        self.heart_button.clicked.connect(self._heart_mission)
        mission_layout.addWidget(self.square_button)
        mission_layout.addWidget(self.heart_button)
        layout.addWidget(mission_group)

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
        ]
        for label, key in fields:
            value = QLabel("—")
            value.setTextInteractionFlags(value.textInteractionFlags())
            self.telemetry_labels[key] = value
            telemetry_layout.addRow(label, value)
        layout.addWidget(telemetry_group)
        layout.addStretch()
        return panel

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
        tabs.addTab(self.camera_viewer, "RGB · Depth · Segmentation")
        tabs.addTab(self.lidar_viewer, "LiDAR 3D 점군")
        tabs.addTab(self.radar_viewer, "Radar 3D 점군")
        return tabs

    @staticmethod
    def _spinbox(minimum: float, maximum: float, value: float, suffix: str) -> QDoubleSpinBox:
        widget = QDoubleSpinBox()
        widget.setRange(minimum, maximum)
        widget.setDecimals(1)
        widget.setSingleStep(1.0)
        widget.setValue(value)
        widget.setSuffix(suffix)
        return widget

    def _toggle_connection(self) -> None:
        if self._connected:
            self.worker.submit("disconnect", priority=1)
        else:
            self.message_label.setText("AirSim 연결 중…")
            self.worker.submit("connect", priority=1)

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
        self.worker.submit("takeoff", self.takeoff_altitude.value())

    def _move(self) -> None:
        try:
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
            else "미니맵에서 목표 B를 클릭하세요."
        )

    def _on_spawn_selected(self, x_m: float, y_m: float) -> None:
        self._spawn_xy = (x_m, y_m)
        self.message_label.setText(
            f"스폰 A 선택: X={x_m:.1f}, Y={y_m:.1f}m · 적용 버튼을 누르세요."
        )

    def _on_target_selected(self, x_m: float, y_m: float) -> None:
        self.destination_x.setValue(x_m)
        self.destination_y.setValue(y_m)
        self._active_target = None
        self._pending_descent_altitude = None
        self._pending_descent_safe_altitude = None
        self._pending_descent_commanded = False
        self.message_label.setText(f"목표 B 선택: X={x_m:.1f}, Y={y_m:.1f}m")

    def _apply_spawn(self) -> None:
        self.worker.submit("spawn", self._spawn_xy[0], self._spawn_xy[1], priority=2)

    def _plan_and_fly(
        self,
        replan: bool,
        start_override: tuple[float, float] | None = None,
        altitude_override: float | None = None,
        collision_escape: tuple[float, float, float, float, float] | None = None,
    ) -> None:
        if self._telemetry is None:
            raise RuntimeError("드론 위치를 아직 받지 못했습니다.")
        if not replan:
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
        target = (
            self.destination_x.value(),
            self.destination_y.value(),
            self.destination_altitude.value(),
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
        path = self.planner.plan(
            start,
            (target[0], target[1]),
            safe_target_altitude,
            (
                float(altitude_override)
                if altitude_override is not None
                else float(self._telemetry["altitude"])
            ),
            max_altitude_m=self._avoidance_altitude_ceiling_m,
        )
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
            f"{'재탐색' if replan else '경로 생성'} 완료: "
            f"웨이포인트 {len(path)}개, 최고 {cruise:.1f}m"
        )

    def _square_mission(self) -> None:
        points = build_square_path(
            self.destination_x.value(),
            self.destination_y.value(),
            self.destination_altitude.value(),
        )
        self.worker.submit("path", points, self.speed.value())

    def _heart_mission(self) -> None:
        points = build_heart_path(
            self.destination_x.value(),
            self.destination_y.value(),
            self.destination_altitude.value(),
        )
        self.worker.submit("path", points, self.speed.value())

    def _on_connection_changed(self, connected: bool, message: str) -> None:
        self._connected = connected
        self.status_indicator.setText(f"● {message}")
        self.status_indicator.setObjectName("connected" if connected else "disconnected")
        self.status_indicator.style().unpolish(self.status_indicator)
        self.status_indicator.style().polish(self.status_indicator)
        self.connect_button.setText("연결 해제" if connected else "AirSim 연결")
        self._set_controls_enabled(connected)
        self.message_label.setText(message)

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
            self.square_button,
            self.heart_button,
        ):
            widget.setEnabled(enabled)

    def _on_telemetry(self, data: dict) -> None:
        self._telemetry = data
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
        self._advance_terminal_descent(data)

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
        self.planner.set_obstacle_points(self._combined_obstacle_points())

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

    def _refresh_obstacle_map(self) -> None:
        """Push the currently enabled sensor clouds to planner and minimap.

        현재 활성화된 센서 점군을 경로계획기와 미니맵에 반영합니다.
        """
        combined = self._combined_obstacle_points()
        self.planner.set_obstacle_points(combined)
        self.minimap.set_obstacles(combined)

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
            drone_count = len(detections) - human_count
            self.telemetry_labels["enemy"].setText(
                f"사람 {human_count}명 · 드론 {drone_count}대 · "
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
            self.planner.set_obstacle_points(self._combined_obstacle_points())
            return

        combined = self._combined_obstacle_points()
        self.planner.set_obstacle_points(combined)
        self.minimap.set_obstacles(combined)

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
        if not self.radar_checkbox.isChecked():
            return
        data = dict(snapshot)
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
            drone_count = len(tracks) - human_count
            self.telemetry_labels["radar"].setText(
                f'{len(local):,} returns · 사람 {human_count}명 · '
                f'드론 {drone_count}대 · '
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
        lateral = np.abs(-delta_x * direction_y + delta_y * direction_x)
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
        nearest_index = int(candidates[np.argmin(forward[candidates])])
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
        barrier_half_span = min(35.0, max(12.0, visible_half_span + 8.0))
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
    ) -> None:
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
        if self._collision_obstacle_points.size:
            self._collision_obstacle_points = np.vstack(
                (self._collision_obstacle_points, barrier)
            )[-20000:]
        else:
            self._collision_obstacle_points = barrier
        self.planner.set_obstacle_points(self._combined_obstacle_points())

    def _on_lidar(self, snapshot: object) -> None:
        if not self.lidar_checkbox.isChecked():
            return
        points, pose = snapshot
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

            # A single sparse or noisy LiDAR return must not cancel a flight.
            # Two consecutive frames reject isolated noise while reacting
            # sooner than the previous three-frame requirement.
            # 두 프레임 연속 감지로 단일 노이즈는 제거하면서 기존 세 프레임보다
            # 더 빠르게 벽에 반응합니다.
            if self._obstacle_detection_count < 2:
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
                try:
                    self._remember_detected_wall(
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
                    self.worker.submit("emergency", priority=0)
                    self._on_error(f"회피 경로 생성 실패 · 호버링: {exc}")
                return
        else:
            self._obstacle_detection_count = 0

    def _on_command_completed(self, command: str) -> None:
        names = {
            "arm": "ARM/DISARM 명령 전송",
            "spawn": "선택한 스폰 A 위치를 적용했습니다.",
            "takeoff": "이륙 명령 전송",
            "hover": "호버링 명령 전송",
            "move": "목적지 이동 시작",
            "path": "패턴 미션 시작",
            "land": "착륙 명령 전송",
            "emergency": "긴급 정지 명령 전송",
        }
        self.message_label.setText(names.get(command, command))

    def _on_error(self, message: str) -> None:
        self.message_label.setText(message)
        if not message.startswith(("센서:", "카메라:", "LiDAR:", "Radar:")):
            QMessageBox.warning(self, "Mission Control", message)

    def _restore_settings(self) -> None:
        self.takeoff_altitude.setValue(float(self.settings.value("takeoff_altitude", 5.0)))
        self.destination_x.setValue(float(self.settings.value("destination_x", 10.0)))
        self.destination_y.setValue(float(self.settings.value("destination_y", 0.0)))
        self.destination_altitude.setValue(float(self.settings.value("destination_altitude", 5.0)))
        self.speed.setValue(float(self.settings.value("speed", 3.0)))

    def _save_settings(self) -> None:
        self.settings.setValue("takeoff_altitude", self.takeoff_altitude.value())
        self.settings.setValue("destination_x", self.destination_x.value())
        self.settings.setValue("destination_y", self.destination_y.value())
        self.settings.setValue("destination_altitude", self.destination_altitude.value())
        self.settings.setValue("speed", self.speed.value())

    def closeEvent(self, event: QCloseEvent) -> None:  # noqa: N802 - Qt API
        self._save_settings()
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

"""Tests for reliable one-click take-off sequencing."""

from __future__ import annotations

import threading
import time
import math
from types import SimpleNamespace

import cosysairsim as airsim
import numpy as np

from common.airsim_client import (
    AirSimController,
    navigation_tracking_limits,
)
from ui.mission_control import AirSimWorker, MissionControlWindow


class _Future:
    def __init__(self, on_join=None) -> None:
        self._on_join = on_join
        self.joined = False

    def join(self) -> None:
        self.joined = True
        if self._on_join is not None:
            self._on_join()


class _FakeFlightClient:
    def __init__(self) -> None:
        self.calls: list[str] = []
        self.landed_state = airsim.LandedState.Landed
        self.z = 0.0
        self.takeoff_future: _Future | None = None
        self.move_future: _Future | None = None

    def cancelLastTask(self, vehicle_name: str) -> None:
        self.calls.append("cancel")

    def enableApiControl(self, enabled: bool, vehicle_name: str) -> None:
        assert enabled
        self.calls.append("api")

    def isApiControlEnabled(self, vehicle_name: str) -> bool:
        self.calls.append("api_check")
        return True

    def armDisarm(self, armed: bool, vehicle_name: str) -> bool:
        assert armed
        self.calls.append("arm")
        return True

    def getMultirotorState(self, vehicle_name: str):
        return SimpleNamespace(
            landed_state=self.landed_state,
            kinematics_estimated=SimpleNamespace(
                position=SimpleNamespace(z_val=self.z)
            ),
        )

    def takeoffAsync(self, timeout_sec: float, vehicle_name: str) -> _Future:
        self.calls.append("takeoff")

        def complete() -> None:
            self.landed_state = airsim.LandedState.Flying
            self.z = -3.0

        self.takeoff_future = _Future(complete)
        return self.takeoff_future

    def moveToZAsync(self, z: float, **kwargs) -> _Future:
        self.calls.append("move_z")
        self.move_future = _Future(lambda: setattr(self, "z", float(z)))
        return self.move_future

    def hoverAsync(self, vehicle_name: str) -> _Future:
        self.calls.append("hover")
        return _Future()


class _FakeEnvironmentClient:
    def __init__(self) -> None:
        self.time_of_day: tuple | None = None
        self.weather_enabled = False
        self.weather: dict[int, float] = {}
        self.console_commands: list[str] = []
        self.wind = None

    def simSetTimeOfDay(self, *args) -> None:
        self.time_of_day = args

    def simEnableWeather(self, enabled: bool) -> None:
        self.weather_enabled = enabled

    def simSetWeatherParameter(self, parameter: int, value: float) -> None:
        self.weather[int(parameter)] = float(value)

    def simRunConsoleCommand(self, command: str) -> bool:
        self.console_commands.append(command)
        return True

    def simSetWind(self, wind) -> None:
        self.wind = wind


def test_takeoff_cancels_old_task_and_waits_for_target_altitude() -> None:
    controller = AirSimController()
    client = _FakeFlightClient()
    controller.client = client

    controller.takeoff(5.0)

    assert client.calls[:4] == ["cancel", "api", "api_check", "arm"]
    assert client.takeoff_future is not None and client.takeoff_future.joined
    assert client.move_future is not None and client.move_future.joined
    assert client.z == -5.0


def test_safe_midflight_descent_keeps_clearance_over_rooftop() -> None:
    window = SimpleNamespace(
        planner=SimpleNamespace(
            config=SimpleNamespace(
                drone_radius_m=6.0,
                vertical_clearance_m=2.0,
            )
        ),
        _combined_obstacle_points=lambda: np.asarray(
            [[0.0, 0.0, -4.0]],
            dtype=np.float32,
        ),
    )

    altitude, surface = MissionControlWindow._safe_altitude_at_xy(
        window,
        0.0,
        0.0,
        3.0,
        7.0,
    )

    assert surface == 4.0
    assert altitude == 6.75


def test_flat_rooftop_detection_is_not_extruded_into_a_tall_wall() -> None:
    offsets = np.arange(-3.0, 3.1, 1.0, dtype=np.float32)
    local_x, local_y = np.meshgrid(offsets, offsets)
    rooftop = np.column_stack(
        (
            20.0 + local_x.ravel(),
            local_y.ravel(),
            np.full(local_x.size, -4.0, dtype=np.float32),
        )
    )
    window = SimpleNamespace(
        _active_target=(100.0, 0.0, 5.0),
        _telemetry={"altitude": 5.0},
        _latest_lidar_world=rooftop,
        _avoidance_altitude_floor_m=1.0,
        _avoidance_altitude_ceiling_m=None,
        _collision_obstacle_points=np.empty((0, 3), dtype=np.float32),
        planner=SimpleNamespace(
            config=SimpleNamespace(max_extra_altitude_m=12.0)
        ),
        destination_altitude=SimpleNamespace(value=lambda: 5.0),
        _refresh_obstacle_map=lambda: None,
    )

    MissionControlWindow._remember_detected_wall(
        window,
        (20.0, 0.0, -4.0),
        (1.0, 0.0),
        6.0,
    )

    assert len(window._collision_obstacle_points) > 0
    assert np.ptp(window._collision_obstacle_points[:, 2]) == 0.0


def test_fallback_bypass_moves_sideways_and_forward_without_reverse() -> None:
    submitted: list[tuple[tuple, dict]] = []
    shown_paths: list[list[tuple[float, float, float]]] = []
    window = SimpleNamespace(
        _telemetry={"x": 0.0, "y": 0.0, "altitude": 5.0},
        _active_target=(100.0, 0.0, 5.0),
        _latest_lidar_world=np.empty((0, 3), dtype=np.float32),
        _pending_descent_altitude=None,
        _pending_descent_safe_altitude=None,
        _pending_descent_commanded=False,
        _planned_path=[],
        _mission_stall_started=1.0,
        planner=SimpleNamespace(
            config=SimpleNamespace(
                drone_radius_m=4.0,
                vertical_clearance_m=2.0,
            )
        ),
        minimap=SimpleNamespace(set_path=lambda path: shown_paths.append(path)),
        speed=SimpleNamespace(value=lambda: 8.0),
        worker=SimpleNamespace(
            submit=lambda *args, **kwargs: submitted.append((args, kwargs))
        ),
    )

    MissionControlWindow._command_forward_bypass(
        window,
        12.0,
        (1.0, 0.0),
        1.0,
    )

    assert shown_paths
    path = shown_paths[-1]
    assert path[0][0] > 0.0 and path[1][0] > path[0][0]
    assert all(x_m >= 0.0 for x_m, _y_m, _altitude_m in path)
    assert abs(path[0][1]) >= 8.0
    assert submitted[-1][0][0] == "path"


def test_environment_applies_reproducible_rain_midnight_and_wind() -> None:
    controller = AirSimController()
    client = _FakeEnvironmentClient()
    controller.client = client

    result = controller.set_environment(
        "winter", "midnight", "cloudy", "rain", 0.75, 8.0, -3.0
    )

    assert client.time_of_day is None
    assert client.weather_enabled
    assert client.weather[int(airsim.WeatherParameter.Rain)] == 0.75
    assert math.isclose(
        client.weather[int(airsim.WeatherParameter.Roadwetness)], 0.6
    )
    assert client.weather[int(airsim.WeatherParameter.Snow)] == 0.0
    assert client.weather[int(airsim.WeatherParameter.Fog)] == 0.05
    assert client.console_commands == [
        "r.VolumetricCloud 1",
        "r.EyeAdaptationQuality 0",
        "DroneEnv.ApplyGoodSky 0.0 cloudy rain 0.750 winter",
    ]
    assert result["volumetric_clouds"] is True
    assert client.wind.x_val == 8.0
    assert client.wind.y_val == -3.0
    assert result["precipitation"] == "rain"
    assert result["season"] == "winter"


def test_environment_apply_does_not_block_worker_command_loop() -> None:
    worker = AirSimWorker()
    worker.controller.client = object()
    started = threading.Event()
    release = threading.Event()
    original_apply = AirSimController.apply_environment_once

    def slow_apply(*_args) -> dict[str, object]:
        started.set()
        release.wait(timeout=1.0)
        return {}

    AirSimController.apply_environment_once = staticmethod(slow_apply)
    try:
        started_at = time.monotonic()
        worker._start_environment_apply(
            ("summer", "noon", "clear", "none", 0.0, 0.0, 0.0)
        )
        elapsed = time.monotonic() - started_at
        assert elapsed < 0.2
        assert started.wait(timeout=0.5)
        assert worker._environment_apply_running
    finally:
        release.set()
        AirSimController.apply_environment_once = original_apply


def test_connect_does_not_wait_for_semantic_setup() -> None:
    worker = AirSimWorker()
    controller = worker.controller
    semantic_started = threading.Event()
    allow_semantic_finish = threading.Event()
    original_builder = AirSimController.build_segmentation_report

    def slow_semantic_setup(host: str, port: int) -> dict[str, object]:
        semantic_started.set()
        allow_semantic_finish.wait(timeout=1.0)
        return {"class_count": 9, "object_count": 10, "assigned_count": 10}

    controller.connect = lambda: setattr(controller, "client", object())
    controller.set_sensor_debug_visualization = lambda *_args: None
    AirSimController.build_segmentation_report = staticmethod(slow_semantic_setup)
    try:
        worker.submit("connect")
        started = time.monotonic()
        worker._process_pending_commands()
        elapsed = time.monotonic() - started

        assert elapsed < 0.2
        assert controller.connected
        assert semantic_started.wait(timeout=0.5)
    finally:
        allow_semantic_finish.set()
        AirSimController.build_segmentation_report = staticmethod(original_builder)
        controller.client = None


def test_stalled_active_mission_replans_after_confirmation_window() -> None:
    calls: list[tuple[str, object]] = []
    label = SimpleNamespace(
        setText=lambda value: calls.append(("message", value))
    )
    worker = SimpleNamespace(
        discard_pending_navigation=lambda: calls.append(("discard", None))
    )
    window = SimpleNamespace(
        _connected=True,
        _active_target=(20.0, 0.0, 5.0),
        _mission_stall_started=time.monotonic() - 3.0,
        _last_stall_recovery=0.0,
        _avoidance_grace_until=0.0,
        message_label=label,
        worker=worker,
        _plan_and_fly=lambda **kwargs: calls.append(("replan", kwargs)),
    )

    MissionControlWindow._recover_stalled_mission(
        window,
        {"x": 0.0, "y": 0.0, "speed": 0.0},
    )

    assert calls[0] == ("discard", None)
    assert calls[1] == (
        "replan",
        {"replan": True, "reset_avoidance": False},
    )
    assert "자동 재전송" in str(calls[2][1])


def test_brief_slowdown_does_not_trigger_stall_replan() -> None:
    calls: list[tuple[str, object]] = []
    window = SimpleNamespace(
        _connected=True,
        _active_target=(20.0, 0.0, 5.0),
        _mission_stall_started=time.monotonic() - 0.4,
        _last_stall_recovery=0.0,
        _avoidance_grace_until=0.0,
        message_label=SimpleNamespace(setText=lambda value: calls.append(("message", value))),
        worker=SimpleNamespace(
            discard_pending_navigation=lambda: calls.append(("discard", None))
        ),
        _plan_and_fly=lambda **kwargs: calls.append(("replan", kwargs)),
    )

    MissionControlWindow._recover_stalled_mission(
        window,
        {"x": 0.0, "y": 0.0, "speed": 0.0},
    )

    assert calls == []


def test_close_single_lidar_return_detects_thin_pole() -> None:
    """A close pole can be only one LiDAR point and must still be guarded."""
    window = SimpleNamespace(
        _telemetry={"vx": 8.0, "vy": 0.0},
        _planned_path=[(20.0, 0.0, 5.0)],
        _active_target=(20.0, 0.0, 5.0),
        planner=SimpleNamespace(
            config=SimpleNamespace(
                drone_radius_m=3.6,
                vertical_clearance_m=1.5,
            )
        ),
    )
    world = np.asarray([[8.0, 0.1, -5.0]], dtype=np.float32)

    obstacle = MissionControlWindow._nearest_obstacle_ahead(
        window,
        world,
        {"x": 0.0, "y": 0.0, "z": -5.0},
    )

    assert obstacle is not None
    assert math.isclose(obstacle[0], 8.0)
    assert math.isclose(obstacle[3], 0.75)


def test_distant_single_lidar_return_remains_unconfirmed() -> None:
    """A lone distant particle must not recreate empty-space stops."""
    window = SimpleNamespace(
        _telemetry={"vx": 3.0, "vy": 0.0},
        _planned_path=[(40.0, 0.0, 5.0)],
        _active_target=(40.0, 0.0, 5.0),
        planner=SimpleNamespace(
            config=SimpleNamespace(
                drone_radius_m=3.6,
                vertical_clearance_m=1.5,
            )
        ),
    )
    world = np.asarray([[18.0, 0.0, -5.0]], dtype=np.float32)

    obstacle = MissionControlWindow._nearest_obstacle_ahead(
        window,
        world,
        {"x": 0.0, "y": 0.0, "z": -5.0},
    )

    assert obstacle is None


def test_wide_lidar_returns_remain_a_wall_not_a_pole() -> None:
    window = SimpleNamespace(
        _telemetry={"vx": 8.0, "vy": 0.0},
        _planned_path=[(20.0, 0.0, 5.0)],
        _active_target=(20.0, 0.0, 5.0),
        planner=SimpleNamespace(
            config=SimpleNamespace(
                drone_radius_m=3.6,
                vertical_clearance_m=1.5,
            )
        ),
    )
    world = np.asarray(
        [[10.0, float(y), -5.0] for y in range(-4, 5)],
        dtype=np.float32,
    )

    obstacle = MissionControlWindow._nearest_obstacle_ahead(
        window,
        world,
        {"x": 0.0, "y": 0.0, "z": -5.0},
    )

    assert obstacle is not None
    assert obstacle[3] >= 3.0


def test_detour_tracking_does_not_cut_across_obstacle_clearance() -> None:
    speed, lookahead = navigation_tracking_limits(12.0, 3)

    assert speed == 10.0
    assert math.isclose(lookahead, 4.5)


def test_direct_flight_keeps_requested_speed_with_bounded_lookahead() -> None:
    speed, lookahead = navigation_tracking_limits(12.0, 1)

    assert speed == 12.0
    assert lookahead == 5.0

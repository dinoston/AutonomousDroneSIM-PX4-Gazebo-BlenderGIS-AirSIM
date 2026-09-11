"""Tests for reliable one-click take-off sequencing."""

from __future__ import annotations

import threading
import time
from types import SimpleNamespace

import cosysairsim as airsim

from common.airsim_client import AirSimController
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


def test_takeoff_cancels_old_task_and_waits_for_target_altitude() -> None:
    controller = AirSimController()
    client = _FakeFlightClient()
    controller.client = client

    controller.takeoff(5.0)

    assert client.calls[:4] == ["cancel", "api", "api_check", "arm"]
    assert client.takeoff_future is not None and client.takeoff_future.joined
    assert client.move_future is not None and client.move_future.joined
    assert client.z == -5.0


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

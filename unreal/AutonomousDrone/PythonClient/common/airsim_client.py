"""Pure-Python wrapper around the Cosys-AirSim RPC client."""

from __future__ import annotations

import socket
import time
from typing import Iterable

import cosysairsim as airsim
import numpy as np

from common.coordinates import altitude_to_ned_z, radians_to_degrees
from common.safety import validate_destination
from perception.radar_processor import RadarProcessor
from perception.segmentation_labels import configure_air_sim_segmentation
from perception.target_detector import TargetDetector


class AirSimController:
    def __init__(
        self,
        vehicle_name: str = "SimpleFlight",
        lidar_name: str = "Lidar1",
        radar_name: str = "FrontRadar",
        host: str = "127.0.0.1",
        port: int = 41451,
    ) -> None:
        self.vehicle_name = vehicle_name
        self.lidar_name = lidar_name
        self.radar_name = radar_name
        self.host = host
        self.port = port
        self.client: airsim.MultirotorClient | None = None
        self._enemy_detection_ready = False
        self._enemy_detection_error = ""
        self._segmentation_report: dict[str, object] = {}
        self._segmentation_error = ""
        self._last_segmentation_refresh = 0.0

    @property
    def connected(self) -> bool:
        return self.client is not None

    def connect(self) -> None:
        try:
            with socket.create_connection((self.host, self.port), timeout=1.5):
                pass
        except OSError as exc:
            raise ConnectionError(
                "AirSim RPC 서버가 열려 있지 않습니다. Unreal에서 Play를 먼저 실행하세요."
            ) from exc
        client = airsim.MultirotorClient(ip=self.host, port=self.port)
        if not client.ping():
            raise ConnectionError("AirSim RPC ping에 실패했습니다.")
        self.client = client
        self._configure_enemy_detection()

    @property
    def segmentation_status(self) -> str:
        if self._segmentation_error:
            return "Segmentation 규칙 미적용"
        class_count = int(self._segmentation_report.get("class_count", 0))
        object_count = int(self._segmentation_report.get("object_count", 0))
        assigned_count = int(self._segmentation_report.get("assigned_count", 0))
        if class_count:
            return (
                f"Segmentation {class_count}개 클래스 · "
                f"객체 {assigned_count}/{object_count} 적용"
            )
        return "Segmentation 대기"

    def _configure_segmentation_labels(self) -> None:
        """Apply the shared semantic class map to the current Unreal level.

        현재 Unreal 레벨의 메시를 공통 Semantic 클래스 ID로 통일합니다.
        """
        self._segmentation_report = {}
        self._segmentation_error = ""
        try:
            self._segmentation_report = configure_air_sim_segmentation(
                self._require_client()
            )
            self._last_segmentation_refresh = time.monotonic()
        except Exception as exc:
            # Flight and the existing preview remain usable if an older server
            # does not expose the segmentation assignment RPC.
            self._segmentation_error = str(exc)

    def configure_segmentation_labels(self) -> str:
        """Reapply semantic IDs after a level or spawned-object change."""
        self._configure_segmentation_labels()
        return self.segmentation_status

    @staticmethod
    def build_segmentation_report(
        host: str = "127.0.0.1",
        port: int = 41451,
    ) -> dict[str, object]:
        """Build semantic assignments with an independent RPC connection.

        The main flight client must remain responsive while a large Unreal
        level updates thousands of segmentation components.
        대형 레벨의 수천 Segmentation 컴포넌트를 갱신하는 동안에도 주 비행
        클라이언트가 응답하도록 별도의 RPC 연결을 사용합니다.
        """
        with socket.create_connection((host, port), timeout=1.5):
            pass
        client = airsim.MultirotorClient(
            ip=host,
            port=port,
            timeout_value=45,
        )
        if not client.ping():
            raise ConnectionError("Segmentation 설정용 AirSim RPC ping에 실패했습니다.")
        return configure_air_sim_segmentation(client)

    def apply_segmentation_report(self, report: dict[str, object]) -> None:
        """Publish a report produced by the background segmentation client."""
        self._segmentation_report = dict(report)
        self._segmentation_error = ""
        self._last_segmentation_refresh = time.monotonic()

    def apply_segmentation_error(self, error: str) -> None:
        """Publish a non-fatal background segmentation failure."""
        self._segmentation_error = str(error)

    def _refresh_segmentation_labels(self) -> None:
        """Assign semantic IDs to people/birds spawned after connection."""
        now = time.monotonic()
        if now - self._last_segmentation_refresh < 2.0:
            return
        self._last_segmentation_refresh = now
        client = self._require_client()
        try:
            runtime_names = {
                str(name) for name in client.simListInstanceSegmentationObjects()
            }
            known_assignments = dict(
                self._segmentation_report.get("assignments", {})
            )
            new_names = sorted(runtime_names - set(known_assignments))
            if not new_names:
                return
            update = configure_air_sim_segmentation(
                client,
                object_names=new_names,
            )
            known_assignments.update(dict(update.get("assignments", {})))
            class_counts = dict(self._segmentation_report.get("class_counts", {}))
            for name, count in dict(update.get("class_counts", {})).items():
                class_counts[str(name)] = int(class_counts.get(str(name), 0)) + int(count)
            self._segmentation_report.update(
                {
                    "object_count": len(runtime_names),
                    "assigned_count": len(known_assignments),
                    "assignments": known_assignments,
                    "class_counts": class_counts,
                }
            )
            self._segmentation_error = ""
        except Exception as exc:
            self._segmentation_error = str(exc)

    def _configure_enemy_detection(self) -> None:
        """Register simulator filters for drones, people, and flocking birds.

        적 드론, 사람, 군집 새를 찾도록 시뮬레이터 이름 필터를 등록합니다.
        """
        client = self._require_client()
        self._enemy_detection_ready = False
        self._enemy_detection_error = ""
        try:
            client.simClearDetectionMeshNames(
                "0",
                airsim.ImageType.Scene,
                vehicle_name=self.vehicle_name,
            )
            client.simSetDetectionFilterRadius(
                "0",
                airsim.ImageType.Scene,
                5000,
                vehicle_name=self.vehicle_name,
            )
            # Blueprint instances normally include BP_EnemyDrone in their
            # generated name. The imported mesh pattern is a fallback.
            # Blueprint 인스턴스 이름에는 보통 BP_EnemyDrone이 포함되며,
            # 가져온 메시 이름 패턴은 이를 찾지 못할 때의 대체 필터입니다.
            for pattern in (
                "BP_EnemyDrone*",
                "*EnemyDrone*",
                "*Drone_FuturisticSleek*",
                "BP_AINormalPeople_Drone*",
                "*AINormalPeople*",
                "*HumanTarget*",
                "*Ch01*",
                "*Ch02*",
                "*FlockCharacter*",
                "*BoidCharacter*",
                "*BirdTarget*",
                "*Crow*",
            ):
                client.simAddDetectionFilterMeshName(
                    "0",
                    airsim.ImageType.Scene,
                    pattern,
                    vehicle_name=self.vehicle_name,
                )
            self._enemy_detection_ready = True
        except Exception as exc:
            # Camera and flight control remain usable when an older AirSim
            # build does not expose the detection RPC endpoints.
            # 이전 AirSim 빌드에 탐지 RPC가 없어도 카메라와 비행 제어는
            # 계속 사용할 수 있도록 연결 자체는 유지합니다.
            self._enemy_detection_error = str(exc)

    def disconnect(self) -> None:
        if self.client is not None:
            # Closing the desktop UI does not automatically cancel an async
            # path command that is already running inside the AirSim server.
            # Cancel it before releasing API control so PIE can tear down the
            # vehicle and RPC server without waiting on residual flight work.
            # 데스크톱 UI를 닫아도 AirSim 서버 내부에서 실행 중인 비동기 경로
            # 명령은 자동 취소되지 않습니다. API 제어권을 해제하기 전에 명령을
            # 취소하여 PIE 종료 시 남은 비행 작업을 기다리지 않도록 합니다.
            try:
                self.client.cancelLastTask(vehicle_name=self.vehicle_name)
            except Exception:
                pass
            try:
                self.client.enableApiControl(False, vehicle_name=self.vehicle_name)
            except Exception:
                pass
        self.client = None
        self._enemy_detection_ready = False
        self._segmentation_report = {}
        self._segmentation_error = ""
        self._last_segmentation_refresh = 0.0

    def _require_client(self) -> airsim.MultirotorClient:
        if self.client is None:
            raise ConnectionError("먼저 AirSim에 연결하세요.")
        return self.client

    def arm(self, armed: bool) -> None:
        client = self._require_client()
        client.enableApiControl(True, vehicle_name=self.vehicle_name)
        client.armDisarm(armed, vehicle_name=self.vehicle_name)

    def set_spawn(self, x_m: float, y_m: float) -> None:
        """Teleport the simulated vehicle before flight; this is not a real-aircraft API."""
        client = self._require_client()
        state = client.getMultirotorState(vehicle_name=self.vehicle_name)
        if state.landed_state != airsim.LandedState.Landed:
            raise RuntimeError("스폰 위치는 착륙 상태에서만 변경할 수 있습니다.")
        client.enableApiControl(False, vehicle_name=self.vehicle_name)
        pose = airsim.Pose(
            airsim.Vector3r(float(x_m), float(y_m), 0.0),
            state.kinematics_estimated.orientation,
        )
        client.simSetVehiclePose(pose, True, vehicle_name=self.vehicle_name)

    def lidar_snapshot(self) -> tuple[np.ndarray, dict[str, float]]:
        """Return local LiDAR points plus the vehicle pose needed for map projection."""
        client = self._require_client()
        data = client.getLidarData(lidar_name=self.lidar_name, vehicle_name=self.vehicle_name)
        values = np.asarray(data.point_cloud, dtype=np.float32)
        if values.size < 3:
            points = np.empty((0, 3), dtype=np.float32)
        else:
            points = values[: values.size - (values.size % 3)].reshape((-1, 3))
        state = client.getMultirotorState(vehicle_name=self.vehicle_name)
        kin = state.kinematics_estimated
        roll, pitch, yaw = airsim.quaternion_to_euler_angles(kin.orientation)
        return points, {
            "timestamp": int(getattr(data, "time_stamp", 0)),
            "x": float(kin.position.x_val),
            "y": float(kin.position.y_val),
            "z": float(kin.position.z_val),
            "roll": float(roll),
            "pitch": float(pitch),
            "yaw": float(yaw),
            "qx": float(kin.orientation.x_val),
            "qy": float(kin.orientation.y_val),
            "qz": float(kin.orientation.z_val),
            "qw": float(kin.orientation.w_val),
        }

    def radar_snapshot(self) -> dict[str, object]:
        """Return active Echo/Radar samples and the vehicle pose.

        Echo/Radar 활성 반사점과 월드 변환에 필요한 기체 자세를 반환합니다.
        """
        client = self._require_client()
        data = client.getEchoData(
            echo_name=self.radar_name,
            vehicle_name=self.vehicle_name,
        )
        parsed = RadarProcessor.parse_active_echo(data.point_cloud)
        point_count = len(parsed["points"])
        state = client.getMultirotorState(vehicle_name=self.vehicle_name)
        kin = state.kinematics_estimated
        roll, pitch, yaw = airsim.quaternion_to_euler_angles(kin.orientation)
        return {
            **parsed,
            "labels": RadarProcessor.normalize_labels(
                getattr(data, "groundtruth", None),
                point_count,
            ),
            "timestamp": int(getattr(data, "time_stamp", 0)),
            "pose": {
                "x": float(kin.position.x_val),
                "y": float(kin.position.y_val),
                "z": float(kin.position.z_val),
                "roll": float(roll),
                "pitch": float(pitch),
                "yaw": float(yaw),
                "qx": float(kin.orientation.x_val),
                "qy": float(kin.orientation.y_val),
                "qz": float(kin.orientation.z_val),
                "qw": float(kin.orientation.w_val),
            },
        }

    def set_sensor_debug_visualization(self, sensor_name: str, enabled: bool) -> None:
        """Toggle the matching Unreal sensor debug box through AirSim.

        AirSim을 통해 해당 언리얼 센서 디버그 박스를 켜거나 끕니다.
        """
        console_variables = {
            "lidar": "autodrone.LidarDebug",
            "radar": "autodrone.RadarDebug",
            "sensor_ray": "autodrone.SensorRayDebug",
        }
        normalized = str(sensor_name).strip().lower()
        if normalized not in console_variables:
            raise ValueError(f"지원하지 않는 센서 디버그 표시: {sensor_name}")
        command = f"{console_variables[normalized]} {1 if enabled else 0}"
        if not self._require_client().simRunConsoleCommand(command):
            raise RuntimeError(f"언리얼 콘솔 명령을 실행하지 못했습니다: {command}")

    def takeoff(self, altitude_m: float) -> None:
        client = self._require_client()
        validate_destination(0.0, 0.0, altitude_m, 2.0)
        # A path/hover command left active on the server can immediately
        # replace a new take-off request. Start from a known command state.
        # 서버에 남은 경로/호버 명령이 새 이륙 요청을 덮어쓸 수 있으므로
        # 기존 비동기 작업을 먼저 취소합니다.
        client.cancelLastTask(vehicle_name=self.vehicle_name)
        client.enableApiControl(True, vehicle_name=self.vehicle_name)
        api_enabled = bool(
            client.isApiControlEnabled(vehicle_name=self.vehicle_name)
        )
        if not api_enabled:
            # SimpleFlight can need one short retry immediately after PIE or
            # after control was released by a previous disconnect.
            # PIE 직후 또는 연결 해제 뒤에는 제어권 반영이 늦을 수 있어
            # 한 번 짧게 재시도합니다.
            time.sleep(0.2)
            client.enableApiControl(True, vehicle_name=self.vehicle_name)
            api_enabled = bool(
                client.isApiControlEnabled(vehicle_name=self.vehicle_name)
            )
        if not api_enabled:
            raise RuntimeError("AirSim API 제어권을 얻지 못했습니다.")

        armed = bool(client.armDisarm(True, vehicle_name=self.vehicle_name))
        if not armed:
            time.sleep(0.25)
            armed = bool(client.armDisarm(True, vehicle_name=self.vehicle_name))
        if not armed:
            raise RuntimeError("드론 ARM에 실패했습니다. 충돌 또는 스폰 상태를 확인하세요.")

        state = client.getMultirotorState(vehicle_name=self.vehicle_name)
        if state.landed_state == airsim.LandedState.Landed:
            client.takeoffAsync(
                timeout_sec=20,
                vehicle_name=self.vehicle_name,
            ).join()

        target_z = altitude_to_ned_z(altitude_m)
        state = client.getMultirotorState(vehicle_name=self.vehicle_name)
        current_altitude = -float(state.kinematics_estimated.position.z_val)
        timeout_seconds = max(
            8.0,
            abs(float(altitude_m) - current_altitude) + 3.0,
        )
        client.moveToZAsync(
            target_z,
            velocity=2.0,
            timeout_sec=timeout_seconds,
            yaw_mode=airsim.YawMode(False, 0),
            vehicle_name=self.vehicle_name,
        ).join()

        reached_state = client.getMultirotorState(vehicle_name=self.vehicle_name)
        reached_altitude = -float(
            reached_state.kinematics_estimated.position.z_val
        )
        if abs(reached_altitude - float(altitude_m)) > 0.8:
            client.hoverAsync(vehicle_name=self.vehicle_name)
            raise RuntimeError(
                f"이륙은 시작됐지만 목표 고도에 도달하지 못했습니다. "
                f"현재 {reached_altitude:.1f}m / 목표 {float(altitude_m):.1f}m"
            )

    def hover(self) -> None:
        self._require_client().hoverAsync(vehicle_name=self.vehicle_name)

    def move_to(self, x_m: float, y_m: float, altitude_m: float, speed_mps: float) -> None:
        validate_destination(x_m, y_m, altitude_m, speed_mps)
        lookahead_m = max(5.0, float(speed_mps) * 2.0)
        self._require_client().moveToPositionAsync(
            float(x_m),
            float(y_m),
            altitude_to_ned_z(altitude_m),
            float(speed_mps),
            drivetrain=airsim.DrivetrainType.MaxDegreeOfFreedom,
            yaw_mode=airsim.YawMode(False, 0),
            lookahead=lookahead_m,
            adaptive_lookahead=0,
            vehicle_name=self.vehicle_name,
        )

    def move_path(self, points: Iterable[tuple[float, float, float]], speed_mps: float) -> None:
        point_list = list(points)
        if not point_list:
            raise ValueError("비행 경로가 비어 있습니다.")
        for point in point_list:
            validate_destination(point[0], point[1], point[2], speed_mps)

        # A single destination does not need path-following control. Using
        # moveOnPathAsync for one point can repeatedly adjust path heading and
        # make the multirotor appear to oscillate near the target direction.
        # 목적지가 하나뿐이면 경로 추종 제어가 필요하지 않습니다. 단일 지점에
        # moveOnPathAsync를 사용하면 방향을 반복 보정해 기체가 떨릴 수 있습니다.
        if len(point_list) == 1:
            x_m, y_m, altitude_m = point_list[0]
            self.move_to(x_m, y_m, altitude_m, speed_mps)
            return

        client = self._require_client()
        state = client.getMultirotorState(vehicle_name=self.vehicle_name)
        position = state.kinematics_estimated.position
        first_x, first_y, first_altitude = point_list[0]
        horizontal_to_first = float(
            np.hypot(first_x - position.x_val, first_y - position.y_val)
        )
        current_altitude = -float(position.z_val)
        if (
            horizontal_to_first <= 4.0
            and abs(first_altitude - current_altitude) > 0.5
        ):
            # Complete the vertical stage before starting horizontal path
            # following. Otherwise AirSim lookahead cuts the corner diagonally.
            # 수평 경로 추종을 시작하기 전에 수직 이동을 완료합니다. 그렇지
            # 않으면 AirSim 선행거리 때문에 모서리를 대각선으로 잘라 이동합니다.
            vertical_speed = min(max(float(speed_mps), 1.0), 2.0)
            client.moveToZAsync(
                altitude_to_ned_z(first_altitude),
                velocity=vertical_speed,
                timeout_sec=15.0,
                yaw_mode=airsim.YawMode(False, 0),
                vehicle_name=self.vehicle_name,
            ).join()
            reached_state = client.getMultirotorState(vehicle_name=self.vehicle_name)
            reached_altitude = -float(
                reached_state.kinematics_estimated.position.z_val
            )
            if abs(reached_altitude - first_altitude) > 0.8:
                client.hoverAsync(vehicle_name=self.vehicle_name)
                raise RuntimeError(
                    "수직 회피 고도에 도달하지 못해 수평 이동을 중단했습니다."
                )
            point_list.pop(0)
            if not point_list:
                client.hoverAsync(vehicle_name=self.vehicle_name)
                return

        path = [
            airsim.Vector3r(x, y, altitude_to_ned_z(altitude))
            for x, y, altitude in point_list
        ]
        # The planner grid is 2.5 m. AirSim's automatic lookahead was only
        # about 1.8 m at the normal mission speed, so the controller reacted
        # to nearly every grid corner. Looking several metres ahead produces
        # one continuous trajectory through the short A* segments.
        # 경로계획 격자는 2.5m이지만 기본 선행거리는 약 1.8m여서 각 격자 모서리마다
        # 제어가 반응했습니다. 선행거리를 늘려 짧은 A* 구간을 연속 경로로 추종합니다.
        lookahead_m = max(5.0, float(speed_mps) * 2.0)
        client.moveOnPathAsync(
            path,
            float(speed_mps),
            # A multirotor can translate without continuously turning toward
            # every short A* segment. This prevents heading corrections from
            # producing visible left/right jitter.
            # 멀티로터는 각 A* 구간 방향으로 계속 회전하지 않고도 이동할 수 있습니다.
            # 불필요한 방향 보정 때문에 좌우로 흔들리는 현상을 방지합니다.
            drivetrain=airsim.DrivetrainType.MaxDegreeOfFreedom,
            yaw_mode=airsim.YawMode(False, 0),
            lookahead=lookahead_m,
            adaptive_lookahead=0,
            vehicle_name=self.vehicle_name,
        )

    def recover_and_move_path(
        self,
        points: Iterable[tuple[float, float, float]],
        speed_mps: float,
        normal_x: float,
        normal_y: float,
        normal_z: float,
        altitude_m: float,
        escape_altitude_m: float,
        retreat_distance_m: float = 4.5,
    ) -> None:
        """Back away from a collision, then execute the replanned path.

        충돌면에서 먼저 후퇴한 다음 새로 계산한 회피 경로를 실행합니다.
        """
        client = self._require_client()
        client.enableApiControl(True, vehicle_name=self.vehicle_name)
        client.cancelLastTask(vehicle_name=self.vehicle_name)
        # Do not wait indefinitely for hover while the collision solver is
        # holding the vehicle against a surface. The finite escape command
        # below becomes the new active task immediately.
        # 충돌 솔버가 기체를 벽에 붙잡은 상태에서 hover 완료를 무기한 기다리지
        # 않습니다. 아래의 유한 시간 탈출 명령을 즉시 새 작업으로 실행합니다.
        if abs(float(normal_z)) >= 0.55:
            # Ceiling normals point down in NED, so moving along the normal
            # lowers altitude. Floor normals perform the opposite escape.
            # NED에서 천장 법선은 아래쪽을 향하므로 법선 방향 이동은 고도를
            # 낮춥니다. 바닥 법선은 반대로 상승 회피를 수행합니다.
            client.moveToZAsync(
                altitude_to_ned_z(escape_altitude_m),
                velocity=1.5,
                timeout_sec=10.0,
                yaw_mode=airsim.YawMode(False, 0),
                vehicle_name=self.vehicle_name,
            ).join()
            reached_state = client.getMultirotorState(
                vehicle_name=self.vehicle_name
            )
            reached_altitude = -float(
                reached_state.kinematics_estimated.position.z_val
            )
            if abs(reached_altitude - escape_altitude_m) > 0.8:
                client.hoverAsync(vehicle_name=self.vehicle_name)
                raise RuntimeError(
                    "천장/바닥 충돌면에서 안전 고도로 벗어나지 못했습니다."
                )
        else:
            retreat_speed = 1.5
            retreat_duration = max(
                0.5,
                float(retreat_distance_m) / retreat_speed,
            )
            # A fixed-duration velocity command can escape contact even when a
            # position command cannot converge because the body touches the wall.
            # 고정 시간 속도 명령은 기체가 벽에 닿아 위치 명령이 수렴하지 못하는
            # 상황에서도 충돌면에서 빠져나올 수 있습니다.
            client.moveByVelocityZAsync(
                float(normal_x) * retreat_speed,
                float(normal_y) * retreat_speed,
                altitude_to_ned_z(altitude_m),
                retreat_duration,
                drivetrain=airsim.DrivetrainType.MaxDegreeOfFreedom,
                yaw_mode=airsim.YawMode(False, 0),
                vehicle_name=self.vehicle_name,
            ).join()
        self.move_path(points, speed_mps)

    def land(
        self,
        approach_altitude_m: float | None = None,
        descent_speed_mps: float = 1.5,
    ) -> None:
        """Descend quickly to a safe approach height, then use AirSim landing.

        안전 접근 고도까지 빠르게 하강한 뒤 AirSim 기본 착륙으로 접지합니다.
        """
        client = self._require_client()
        client.cancelLastTask(vehicle_name=self.vehicle_name)
        if approach_altitude_m is not None:
            state = client.getMultirotorState(vehicle_name=self.vehicle_name)
            current_altitude = -float(state.kinematics_estimated.position.z_val)
            approach_altitude = max(0.3, float(approach_altitude_m))
            descent_speed = min(3.0, max(0.5, float(descent_speed_mps)))
            descent_distance = current_altitude - approach_altitude
            if descent_distance > 0.5:
                timeout_seconds = max(8.0, descent_distance / descent_speed * 2.0 + 3.0)
                client.moveToZAsync(
                    altitude_to_ned_z(approach_altitude),
                    velocity=descent_speed,
                    timeout_sec=timeout_seconds,
                    yaw_mode=airsim.YawMode(False, 0),
                    vehicle_name=self.vehicle_name,
                ).join()
        client.landAsync(timeout_sec=30, vehicle_name=self.vehicle_name)

    def emergency_stop(self) -> None:
        client = self._require_client()
        client.cancelLastTask(vehicle_name=self.vehicle_name)
        client.hoverAsync(vehicle_name=self.vehicle_name)

    def telemetry(self) -> dict[str, float | str]:
        client = self._require_client()
        state = client.getMultirotorState(vehicle_name=self.vehicle_name)
        collision = client.simGetCollisionInfo(vehicle_name=self.vehicle_name)
        kin = state.kinematics_estimated
        roll, pitch, yaw = airsim.quaternion_to_euler_angles(kin.orientation)
        velocity = kin.linear_velocity
        return {
            "timestamp": int(getattr(state, "timestamp", 0)),
            "landed": str(state.landed_state).split(".")[-1],
            "x": float(kin.position.x_val),
            "y": float(kin.position.y_val),
            "z": float(kin.position.z_val),
            "altitude": -float(kin.position.z_val),
            "vx": float(velocity.x_val),
            "vy": float(velocity.y_val),
            "vz": float(velocity.z_val),
            "speed": float(np.linalg.norm([velocity.x_val, velocity.y_val, velocity.z_val])),
            "roll": radians_to_degrees(roll),
            "pitch": radians_to_degrees(pitch),
            "yaw": radians_to_degrees(yaw),
            "has_collided": bool(collision.has_collided),
            "collision_timestamp": float(collision.time_stamp),
            "collision_object": str(collision.object_name),
            "collision_x": float(collision.impact_point.x_val),
            "collision_y": float(collision.impact_point.y_val),
            "collision_z": float(collision.impact_point.z_val),
            "collision_normal_x": float(collision.normal.x_val),
            "collision_normal_y": float(collision.normal.y_val),
            "collision_normal_z": float(collision.normal.z_val),
        }

    def camera_images(self) -> dict[str, object]:
        requests = [
            airsim.ImageRequest("0", airsim.ImageType.Scene, False, True),
            airsim.ImageRequest("0", airsim.ImageType.DepthVis, False, True),
            airsim.ImageRequest("0", airsim.ImageType.Segmentation, False, True),
        ]
        responses = self._require_client().simGetImages(requests, vehicle_name=self.vehicle_name)
        names = ("RGB", "Depth", "Segmentation")
        images: dict[str, object] = {
            name: bytes(response.image_data_uint8)
            for name, response in zip(names, responses)
            if response.image_data_uint8
        }
        images["_camera_timestamps"] = {
            name: int(getattr(response, "time_stamp", 0))
            for name, response in zip(names, responses)
        }
        images["_camera_sizes"] = {
            name: {
                "width": int(getattr(response, "width", 0)),
                "height": int(getattr(response, "height", 0)),
            }
            for name, response in zip(names, responses)
        }
        if self._enemy_detection_ready and responses:
            try:
                raw_detections = self._require_client().simGetDetections(
                    "0",
                    airsim.ImageType.Scene,
                    vehicle_name=self.vehicle_name,
                )
                scene_response = responses[0]
                images["_detections"] = TargetDetector.normalize(
                    raw_detections or [],
                    int(scene_response.width),
                    int(scene_response.height),
                )
            except Exception as exc:
                self._enemy_detection_ready = False
                self._enemy_detection_error = str(exc)
        if self._enemy_detection_error:
            images["_detection_error"] = self._enemy_detection_error
        if self._segmentation_report:
            images["_segmentation_class_map"] = self._segmentation_report.get(
                "classes",
                [],
            )
        if self._segmentation_error:
            images["_segmentation_error"] = self._segmentation_error
        return images

    def lidar_points(self) -> np.ndarray:
        points, _pose = self.lidar_snapshot()
        return points

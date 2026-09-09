"""Altitude-aware grid A* planner for an AirSim local NED world.

AirSim 로컬 NED 좌표계를 위한 고도 인식 격자 A* 경로 계획기입니다.
"""

from __future__ import annotations

import heapq
from dataclasses import dataclass
from math import ceil, hypot
from typing import Iterable

import numpy as np


def split_terminal_vertical_leg(
    points: Iterable[tuple[float, float, float]],
    xy_tolerance_m: float = 0.25,
    altitude_tolerance_m: float = 0.25,
) -> tuple[list[tuple[float, float, float]], float | None]:
    """Separate a final descent from the horizontal AirSim path.

    AirSim 수평 경로에서 마지막 수직 하강 구간을 분리합니다.

    AirSim lookahead begins turning toward the final low point before reaching
    its XY coordinate. Separating the leg keeps the vehicle level until it has
    completely cleared the obstacle below.

    AirSim 선행 제어는 목표 X/Y에 도달하기 전에 마지막 낮은 점을 향해 하강을
    시작합니다. 하강 구간을 분리하면 아래 장애물을 완전히 통과할 때까지 현재
    고도를 유지할 수 있습니다.
    """
    path = list(points)
    if len(path) < 2:
        return path, None
    previous = path[-2]
    final = path[-1]
    same_xy = hypot(final[0] - previous[0], final[1] - previous[1]) <= xy_tolerance_m
    descending = final[2] < previous[2] - altitude_tolerance_m
    if same_xy and descending:
        return path[:-1], float(final[2])
    return path, None


def build_vertical_barrier(
    hit_xyz: tuple[float, float, float],
    travel_direction_xy: tuple[float, float],
    half_span_m: float,
    minimum_altitude_m: float,
    maximum_altitude_m: float,
    spacing_m: float = 1.0,
) -> np.ndarray:
    """Extrude a frontal LiDAR hit into a conservative vertical wall.

    전방 LiDAR 반사점을 보수적인 수직 벽으로 확장합니다.

    A single scan only contains the currently visible part of a facade. The
    extrusion prevents A* from treating the same tall wall as empty a few
    metres above or beside the measured points.

    한 번의 스캔에는 외벽의 현재 보이는 부분만 포함됩니다. 수직·수평 확장으로
    A*가 측정점의 몇 m 위나 옆을 빈 공간으로 오판하지 않게 합니다.
    """
    direction_x, direction_y = map(float, travel_direction_xy)
    direction_length = hypot(direction_x, direction_y)
    if direction_length < 1e-6:
        raise ValueError("이동 방향 벡터의 길이는 0보다 커야 합니다.")
    direction_x /= direction_length
    direction_y /= direction_length
    tangent_x, tangent_y = -direction_y, direction_x
    spacing = max(0.25, float(spacing_m))
    span = max(spacing, float(half_span_m))
    low = max(0.0, min(float(minimum_altitude_m), float(maximum_altitude_m)))
    high = max(low, max(float(minimum_altitude_m), float(maximum_altitude_m)))
    tangent_offsets = np.arange(-span, span + spacing * 0.5, spacing, dtype=np.float32)
    altitudes = np.arange(low, high + spacing * 0.5, spacing, dtype=np.float32)
    tangent_grid, altitude_grid = np.meshgrid(tangent_offsets, altitudes)
    hit_x, hit_y, _hit_z = map(float, hit_xyz)
    return np.column_stack(
        (
            hit_x + tangent_x * tangent_grid.ravel(),
            hit_y + tangent_y * tangent_grid.ravel(),
            -altitude_grid.ravel(),
        )
    ).astype(np.float32, copy=False)


@dataclass(frozen=True)
class PlannerConfig:
    half_extent_m: float = 100.0
    resolution_m: float = 1.0
    drone_radius_m: float = 1.5
    vertical_clearance_m: float = 0.8
    altitude_step_m: float = 2.0
    max_extra_altitude_m: float = 12.0


class AltitudeGridPlanner:
    """Plan in XY and optionally try higher layers when a route is blocked.

    XY 평면에서 경로를 계획하고, 설정된 경우 막힌 경로의 상위 고도도 탐색합니다.
    """

    def __init__(self, config: PlannerConfig | None = None) -> None:
        self.config = config or PlannerConfig()
        self._points = np.empty((0, 3), dtype=np.float32)

    @property
    def obstacle_points(self) -> np.ndarray:
        return self._points.copy()

    def set_obstacle_points(self, points_ned: np.ndarray) -> None:
        points = np.asarray(points_ned, dtype=np.float32)
        if points.ndim != 2 or points.shape[1] != 3:
            self._points = np.empty((0, 3), dtype=np.float32)
            return
        finite = np.all(np.isfinite(points), axis=1)
        in_map = (
            (np.abs(points[:, 0]) <= self.config.half_extent_m)
            & (np.abs(points[:, 1]) <= self.config.half_extent_m)
        )
        self._points = points[finite & in_map]

    def plan(
        self,
        start_xy: tuple[float, float],
        goal_xy: tuple[float, float],
        requested_altitude_m: float,
        current_altitude_m: float,
        max_altitude_m: float | None = None,
    ) -> list[tuple[float, float, float]]:
        self._validate_xy(start_xy)
        self._validate_xy(goal_xy)
        # Once avoidance has climbed above the requested altitude, do not plan
        # a descent until the goal. This prevents repeated down/up oscillation.
        # 회피 중 요청 고도보다 높아졌다면 목적지 전에는 하강 경로를 만들지
        # 않습니다. 장애물 앞에서 상하로 반복 진동하는 현상을 방지합니다.
        base = max(
            1.0,
            float(requested_altitude_m),
            float(current_altitude_m),
        )
        # The climb limit is relative to the requested mission altitude, not
        # the latest replan altitude, so repeated scans cannot climb forever.
        # 상승 제한은 최근 재탐색 고도가 아니라 사용자가 지정한 임무 고도를
        # 기준으로 계산하여 반복 감지 때 무한히 상승하지 않도록 합니다.
        configured_maximum = (
            float(requested_altitude_m) + self.config.max_extra_altitude_m
        )
        if max_altitude_m is not None:
            # A ceiling collision establishes a temporary mission altitude cap.
            # 천장 충돌이 발생하면 현재 임무에 임시 최대 고도를 설정합니다.
            configured_maximum = min(
                configured_maximum,
                max(1.0, float(max_altitude_m)),
            )
            base = min(base, configured_maximum)
        maximum_altitude = max(base, configured_maximum)
        steps = max(
            0,
            int((maximum_altitude - base) / self.config.altitude_step_m),
        )

        best: tuple[float, list[tuple[int, int]], float] | None = None
        for index in range(steps + 1):
            altitude = base + index * self.config.altitude_step_m
            blocked = self._blocked_cells(altitude)
            cells = self._astar(self._to_cell(start_xy), self._to_cell(goal_xy), blocked)
            if not cells:
                continue
            cells = self._simplify(cells, blocked)
            horizontal = sum(
                hypot(b[0] - a[0], b[1] - a[1]) * self.config.resolution_m
                for a, b in zip(cells, cells[1:])
            )
            cost = horizontal + abs(altitude - current_altitude_m) * 1.7
            if best is None or cost < best[0]:
                best = (cost, cells, altitude)

        if best is None:
            raise RuntimeError("현재 장애물 지도에서 목적지까지 안전한 경로를 찾지 못했습니다.")

        _, cells, cruise_altitude = best
        path: list[tuple[float, float, float]] = []
        # When a higher layer is selected, climb in place before moving toward
        # the obstacle. A diagonal climb can still touch a tall facade.
        # 더 높은 고도층을 선택하면 장애물 쪽으로 이동하기 전에 제자리에서
        # 먼저 상승합니다. 대각선 상승은 높은 외벽에 계속 닿을 수 있습니다.
        if abs(cruise_altitude - current_altitude_m) > 0.25:
            path.append(
                (float(start_xy[0]), float(start_xy[1]), cruise_altitude)
            )
        path.extend(
            (*self._from_cell(cell), cruise_altitude) for cell in cells[1:]
        )
        # Return to the requested mission altitude only after reaching the goal.
        # 목적지에 도착한 뒤에만 사용자가 지정한 임무 고도로 복귀합니다.
        if cruise_altitude != requested_altitude_m:
            path.append((float(goal_xy[0]), float(goal_xy[1]), float(requested_altitude_m)))
        if not path:
            path.append((float(goal_xy[0]), float(goal_xy[1]), float(requested_altitude_m)))
        return path

    def route_blocked(self, path: Iterable[tuple[float, float, float]]) -> bool:
        points = list(path)
        if not points:
            return False
        previous = points[0]
        for point in points:
            blocked = self._blocked_cells(point[2])
            if not self._line_clear(
                self._to_cell((previous[0], previous[1])),
                self._to_cell((point[0], point[1])),
                blocked,
            ):
                return True
            previous = point
        return False

    def _blocked_cells(self, altitude_m: float) -> set[tuple[int, int]]:
        if not self._points.size:
            return set()
        obstacle_altitudes = -self._points[:, 2]
        layer = np.abs(obstacle_altitudes - altitude_m) <= self.config.vertical_clearance_m
        layer_points = self._points[layer, :2]
        blocked: set[tuple[int, int]] = set()
        inflation = int(ceil(self.config.drone_radius_m / self.config.resolution_m))
        for xy in layer_points:
            center = self._to_cell((float(xy[0]), float(xy[1])))
            for dx in range(-inflation, inflation + 1):
                for dy in range(-inflation, inflation + 1):
                    if hypot(dx, dy) * self.config.resolution_m <= self.config.drone_radius_m:
                        candidate = (center[0] + dx, center[1] + dy)
                        if self._inside(candidate):
                            blocked.add(candidate)
        return blocked

    def _astar(
        self,
        start: tuple[int, int],
        goal: tuple[int, int],
        blocked: set[tuple[int, int]],
    ) -> list[tuple[int, int]]:
        # The vehicle already occupies its current footprint. LiDAR returns
        # from landing gear/attachments or a nearby surface can otherwise
        # inflate around the start cell and make an open scene look sealed.
        # Clear only the already occupied start footprint; the goal keeps its
        # surrounding clearance so a target inside a building is not accepted.
        # 기체가 현재 차지한 영역은 실제로 비어 있습니다. 랜딩기어/부착물의
        # LiDAR 반사나 가까운 표면이 시작 셀 주위를 팽창시켜 열린 공간을
        # 완전히 막힌 것으로 만들지 않도록 현재 기체 영역만 비웁니다.
        start_clearance = max(
            1,
            int(ceil(self.config.drone_radius_m / self.config.resolution_m)),
        )
        blocked = {
            cell
            for cell in blocked
            if hypot(cell[0] - start[0], cell[1] - start[1]) > start_clearance
        }
        blocked.discard(goal)
        frontier: list[tuple[float, tuple[int, int]]] = [(0.0, start)]
        came_from: dict[tuple[int, int], tuple[int, int]] = {}
        cost = {start: 0.0}
        moves = ((1, 0, 1.0), (-1, 0, 1.0), (0, 1, 1.0), (0, -1, 1.0),
                 (1, 1, 1.414), (1, -1, 1.414), (-1, 1, 1.414), (-1, -1, 1.414))
        while frontier:
            _, current = heapq.heappop(frontier)
            if current == goal:
                result = [goal]
                while result[-1] != start:
                    result.append(came_from[result[-1]])
                result.reverse()
                return result
            for dx, dy, step in moves:
                nxt = (current[0] + dx, current[1] + dy)
                if not self._inside(nxt) or nxt in blocked:
                    continue
                # A diagonal move must not squeeze between two blocked corner
                # cells. Without this guard a simplified path can clip a wall.
                # 대각선 이동 시 막힌 두 모서리 사이를 비집고 지나가지 않게 하여
                # 단순화된 경로가 벽을 스치는 현상을 방지합니다.
                if dx and dy and (
                    (current[0] + dx, current[1]) in blocked
                    or (current[0], current[1] + dy) in blocked
                ):
                    continue
                new_cost = cost[current] + step
                if new_cost >= cost.get(nxt, float("inf")):
                    continue
                cost[nxt] = new_cost
                came_from[nxt] = current
                heuristic = hypot(goal[0] - nxt[0], goal[1] - nxt[1])
                heapq.heappush(frontier, (new_cost + heuristic, nxt))
        return []

    def _simplify(
        self, cells: list[tuple[int, int]], blocked: set[tuple[int, int]]
    ) -> list[tuple[int, int]]:
        if len(cells) < 3:
            return cells
        result = [cells[0]]
        anchor = 0
        while anchor < len(cells) - 1:
            furthest = anchor + 1
            for candidate in range(anchor + 2, len(cells)):
                if self._line_clear(cells[anchor], cells[candidate], blocked):
                    furthest = candidate
                else:
                    break
            result.append(cells[furthest])
            anchor = furthest
        return result

    @staticmethod
    def _line_clear(a: tuple[int, int], b: tuple[int, int], blocked: set[tuple[int, int]]) -> bool:
        """Check every grid cell touched by a segment (supercover line).

        선분이 닿는 모든 격자 셀을 검사합니다(슈퍼커버 선 검사).
        """
        x, y = a
        delta_x = b[0] - a[0]
        delta_y = b[1] - a[1]
        steps_x = abs(delta_x)
        steps_y = abs(delta_y)
        sign_x = 1 if delta_x > 0 else -1 if delta_x < 0 else 0
        sign_y = 1 if delta_y > 0 else -1 if delta_y < 0 else 0
        moved_x = 0
        moved_y = 0
        if (x, y) in blocked:
            return False
        while moved_x < steps_x or moved_y < steps_y:
            decision = (1 + 2 * moved_x) * steps_y - (1 + 2 * moved_y) * steps_x
            if decision == 0:
                # The segment crosses an exact cell corner. Both side cells
                # must be free or the vehicle would clip the obstacle radius.
                # 선분이 셀 모서리를 정확히 지날 때 양옆 셀을 모두 확인하여
                # 기체 안전 반경이 장애물을 스치지 않게 합니다.
                if (x + sign_x, y) in blocked or (x, y + sign_y) in blocked:
                    return False
                x += sign_x
                y += sign_y
                moved_x += 1
                moved_y += 1
            elif decision < 0:
                x += sign_x
                moved_x += 1
            else:
                y += sign_y
                moved_y += 1
            if (x, y) in blocked:
                return False
        return True

    def _to_cell(self, xy: tuple[float, float]) -> tuple[int, int]:
        half = self.config.half_extent_m
        resolution = self.config.resolution_m
        return (round((xy[0] + half) / resolution), round((xy[1] + half) / resolution))

    def _from_cell(self, cell: tuple[int, int]) -> tuple[float, float]:
        half = self.config.half_extent_m
        resolution = self.config.resolution_m
        return (cell[0] * resolution - half, cell[1] * resolution - half)

    def _inside(self, cell: tuple[int, int]) -> bool:
        maximum = round(2 * self.config.half_extent_m / self.config.resolution_m)
        return 0 <= cell[0] <= maximum and 0 <= cell[1] <= maximum

    def _validate_xy(self, xy: tuple[float, float]) -> None:
        if abs(xy[0]) > self.config.half_extent_m or abs(xy[1]) > self.config.half_extent_m:
            raise ValueError(f"미니맵 범위는 ±{self.config.half_extent_m:.0f}m입니다.")

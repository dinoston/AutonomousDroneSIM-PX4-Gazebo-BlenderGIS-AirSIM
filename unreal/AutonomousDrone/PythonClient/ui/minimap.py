"""Clickable local NED minimap for the mission-control UI."""

from __future__ import annotations

import math
import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QMouseEvent, QPainter, QPainterPath, QPen, QPixmap, QWheelEvent
from PySide6.QtWidgets import QWidget


class MiniMapWidget(QWidget):
    spawn_selected = Signal(float, float)
    target_selected = Signal(float, float)

    def __init__(
        self,
        half_extent_m: float = 700.0,
        center_xy_m: tuple[float, float] = (53.31, 159.39),
        background_path: str | None = None,
    ) -> None:
        super().__init__()
        self.half_extent_m = float(half_extent_m)
        self.center_xy_m = (float(center_xy_m[0]), float(center_xy_m[1]))
        self._base_half_extent_m = float(half_extent_m)
        self._base_center_xy_m = self.center_xy_m
        self._minimum_half_extent_m = max(20.0, self._base_half_extent_m / 32.0)
        self._drag_start: QPointF | None = None
        self._drag_center_xy: tuple[float, float] | None = None
        self._dragging = False
        self.background = QPixmap(background_path or "")
        self.drone_xy = (0.0, 0.0)
        self.spawn_xy = (0.0, 0.0)
        self.target_xy: tuple[float, float] | None = None
        self.route_waypoints: list[tuple[float, float, float]] = []
        self.patrol_waypoints: list[tuple[float, float, float]] = []
        self.selection_mode = "target"
        self.obstacles = np.empty((0, 3), dtype=np.float32)
        self.path: list[tuple[float, float, float]] = []
        self.setMinimumSize(560, 560)
        self.setCursor(Qt.CursorShape.CrossCursor)
        self.setToolTip(
            "클릭: A/B 선택 · 왼쪽 드래그: 지도 이동 · 휠: 확대/축소 · 더블클릭: 전체 보기"
        )

    def set_selection_mode(self, mode: str) -> None:
        if mode not in {"spawn", "target"}:
            raise ValueError(f"지원하지 않는 지도 선택 모드: {mode}")
        self.selection_mode = mode
        self.update()

    def set_drone(self, x_m: float, y_m: float) -> None:
        self.drone_xy = (float(x_m), float(y_m))
        self.update()

    def set_target(self, x_m: float, y_m: float) -> None:
        self.target_xy = (float(x_m), float(y_m))
        self.update()

    def set_spawn(self, x_m: float, y_m: float) -> None:
        self.spawn_xy = (float(x_m), float(y_m))
        self.update()

    def set_obstacles(self, points: np.ndarray) -> None:
        self.obstacles = np.asarray(points, dtype=np.float32)
        self.update()

    def set_path(self, path: list[tuple[float, float, float]]) -> None:
        self.path = list(path)
        self.update()

    def set_route_waypoints(
        self,
        waypoints: list[tuple[float, float, float]],
    ) -> None:
        """Show the user-defined B, C, D... reservation points."""
        self.route_waypoints = list(waypoints)
        self.update()

    def set_patrol_waypoints(
        self,
        waypoints: list[tuple[float, float, float]],
    ) -> None:
        """Show the current autonomous central-coverage cycle."""
        self.patrol_waypoints = list(waypoints)
        self.update()

    def mousePressEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        if event.button() != Qt.MouseButton.LeftButton:
            return
        self._drag_start = event.position()
        self._drag_center_xy = self.center_xy_m
        self._dragging = False

    def mouseMoveEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        if not (event.buttons() & Qt.MouseButton.LeftButton):
            return
        if self._drag_start is None or self._drag_center_xy is None:
            return
        delta = event.position() - self._drag_start
        if not self._dragging and delta.manhattanLength() < 6.0:
            return
        self._dragging = True
        rect = self._map_rect()
        meters_per_pixel = (2.0 * self.half_extent_m) / max(1.0, rect.width())
        start_x, start_y = self._drag_center_xy
        self.center_xy_m = self._clamped_center(
            start_x + delta.y() * meters_per_pixel,
            start_y - delta.x() * meters_per_pixel,
        )
        self.setCursor(Qt.CursorShape.ClosedHandCursor)
        self.update()

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        if event.button() != Qt.MouseButton.LeftButton:
            return
        was_dragging = self._dragging
        self._drag_start = None
        self._drag_center_xy = None
        self._dragging = False
        self.setCursor(Qt.CursorShape.CrossCursor)
        if was_dragging:
            return
        x_m, y_m = self._pixel_to_world(event.position())
        if self.selection_mode == "spawn":
            self.set_spawn(x_m, y_m)
            self.spawn_selected.emit(x_m, y_m)
        else:
            self.set_target(x_m, y_m)
            self.target_selected.emit(x_m, y_m)

    def wheelEvent(self, event: QWheelEvent) -> None:  # noqa: N802
        if event.angleDelta().y() == 0:
            return
        cursor_world_before = self._pixel_to_world(event.position())
        zoom_factor = 0.82 if event.angleDelta().y() > 0 else 1.0 / 0.82
        new_half_extent = max(
            self._minimum_half_extent_m,
            min(self._base_half_extent_m, self.half_extent_m * zoom_factor),
        )
        if math.isclose(new_half_extent, self.half_extent_m):
            event.accept()
            return
        self.half_extent_m = new_half_extent
        cursor_world_after = self._pixel_to_world(event.position())
        cx, cy = self.center_xy_m
        self.center_xy_m = self._clamped_center(
            cx + cursor_world_before[0] - cursor_world_after[0],
            cy + cursor_world_before[1] - cursor_world_after[1],
        )
        self.update()
        event.accept()

    def mouseDoubleClickEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        if event.button() == Qt.MouseButton.LeftButton:
            self.half_extent_m = self._base_half_extent_m
            self.center_xy_m = self._base_center_xy_m
            self._dragging = True
            self.update()
            event.accept()

    def paintEvent(self, _event) -> None:  # noqa: N802
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.fillRect(self.rect(), QColor("#0b1118"))
        map_rect = self._map_rect()
        if not self.background.isNull():
            base_cx, base_cy = self._base_center_xy_m
            base_half = self._base_half_extent_m
            top_left = self._world_to_pixel((base_cx + base_half, base_cy - base_half))
            bottom_right = self._world_to_pixel((base_cx - base_half, base_cy + base_half))
            background_rect = QRectF(top_left, bottom_right).normalized()
            painter.save()
            painter.setClipRect(map_rect)
            painter.drawPixmap(background_rect, self.background, QRectF(self.background.rect()))
            painter.restore()
        self._draw_grid(painter)
        self._draw_obstacles(painter)
        self._draw_route_waypoints(painter)
        self._draw_patrol_waypoints(painter)
        self._draw_path(painter)
        self._draw_marker(painter, self.spawn_xy, QColor("#66a8ff"), "SPAWN A")
        self._draw_marker(painter, self.drone_xy, QColor("#51d88a"), "DRONE")
        if self.target_xy is not None:
            self._draw_marker(painter, self.target_xy, QColor("#ffcc55"), "B")
        painter.setPen(QColor("#9fb1c5"))
        mode = "스폰 A" if self.selection_mode == "spawn" else "경유지 후보"
        zoom = self._base_half_extent_m / self.half_extent_m
        painter.drawText(12, 22, f"N(+X) ↑    E(+Y) →    현재 선택: {mode}    줌: {zoom:.1f}x")

    def _map_rect(self):
        margin = 28.0
        side = max(1.0, min(self.width(), self.height()) - 2 * margin)
        left = (self.width() - side) / 2.0
        top = (self.height() - side) / 2.0
        return QRectF(left, top, side, side)

    def _draw_grid(self, painter: QPainter) -> None:
        painter.setPen(QPen(QColor("#263342"), 1))
        spacing = 100
        cx, cy = self.center_xy_m
        min_x, max_x = cx - self.half_extent_m, cx + self.half_extent_m
        min_y, max_y = cy - self.half_extent_m, cy + self.half_extent_m
        first_x = math.ceil(min_x / spacing) * spacing
        first_y = math.ceil(min_y / spacing) * spacing
        for value in range(first_x, int(max_x) + 1, spacing):
            p1 = self._world_to_pixel((value, min_y))
            p2 = self._world_to_pixel((value, max_y))
            painter.drawLine(p1, p2)
        for value in range(first_y, int(max_y) + 1, spacing):
            p3 = self._world_to_pixel((min_x, value))
            p4 = self._world_to_pixel((max_x, value))
            painter.drawLine(p3, p4)
        painter.setPen(QPen(QColor("#50657b"), 2))
        painter.drawLine(self._world_to_pixel((0, min_y)), self._world_to_pixel((0, max_y)))
        painter.drawLine(self._world_to_pixel((min_x, 0)), self._world_to_pixel((max_x, 0)))

    def _draw_obstacles(self, painter: QPainter) -> None:
        if self.obstacles.ndim != 2 or not self.obstacles.size:
            return
        painter.setPen(QPen(QColor(235, 87, 87, 150), 3))
        stride = max(1, len(self.obstacles) // 4000)
        for point in self.obstacles[::stride]:
            painter.drawPoint(self._world_to_pixel((float(point[0]), float(point[1]))))

    def _draw_path(self, painter: QPainter) -> None:
        if not self.path:
            return
        painter.setPen(QPen(QColor("#4db7ff"), 3))
        draw_path = QPainterPath(self._world_to_pixel(self.drone_xy))
        for x_m, y_m, _altitude in self.path:
            draw_path.lineTo(self._world_to_pixel((x_m, y_m)))
        painter.drawPath(draw_path)

    def _draw_route_waypoints(self, painter: QPainter) -> None:
        if not self.route_waypoints:
            return
        route_pen = QPen(QColor(255, 204, 85, 190), 2)
        route_pen.setStyle(Qt.PenStyle.DashLine)
        painter.setPen(route_pen)
        route_path = QPainterPath(self._world_to_pixel(self.spawn_xy))
        for x_m, y_m, _altitude in self.route_waypoints:
            route_path.lineTo(self._world_to_pixel((x_m, y_m)))
        painter.drawPath(route_path)
        for index, (x_m, y_m, altitude) in enumerate(self.route_waypoints):
            label = self._route_label(index)
            self._draw_marker(
                painter,
                (x_m, y_m),
                QColor("#ffcc55"),
                f"{label} · {altitude:.1f}m",
            )

    def _draw_patrol_waypoints(self, painter: QPainter) -> None:
        if not self.patrol_waypoints:
            return
        patrol_color = QColor(65, 224, 210, 205)
        patrol_pen = QPen(patrol_color, 2)
        patrol_pen.setStyle(Qt.PenStyle.DotLine)
        painter.setPen(patrol_pen)
        patrol_path = QPainterPath(self._world_to_pixel(self.drone_xy))
        for x_m, y_m, _altitude in self.patrol_waypoints:
            patrol_path.lineTo(self._world_to_pixel((x_m, y_m)))
        painter.drawPath(patrol_path)
        for index, (x_m, y_m, _altitude) in enumerate(self.patrol_waypoints):
            self._draw_marker(
                painter,
                (x_m, y_m),
                patrol_color,
                f"정찰 {index + 1}",
            )

    @staticmethod
    def _route_label(index: int) -> str:
        """Return B, C ... Z, AA style labels (A is the spawn point)."""
        value = max(0, int(index)) + 2
        label = ""
        while value:
            value, remainder = divmod(value - 1, 26)
            label = chr(ord("A") + remainder) + label
        return label

    def _draw_marker(self, painter: QPainter, xy: tuple[float, float], color: QColor, label: str) -> None:
        point = self._world_to_pixel(xy)
        painter.setBrush(color)
        painter.setPen(QPen(QColor("#101820"), 2))
        painter.drawEllipse(point, 8, 8)
        painter.setPen(color)
        painter.drawText(point + QPointF(10, -8), label)

    def _world_to_pixel(self, xy: tuple[float, float]) -> QPointF:
        rect = self._map_rect()
        width, height = rect.width(), rect.height()
        cx, cy = self.center_xy_m
        # North/X is screen-up; East/Y is screen-right.
        px = rect.left() + (xy[1] - cy + self.half_extent_m) / (2 * self.half_extent_m) * width
        py = rect.top() + (self.half_extent_m - (xy[0] - cx)) / (2 * self.half_extent_m) * height
        return QPointF(px, py)

    def _pixel_to_world(self, point: QPointF) -> tuple[float, float]:
        rect = self._map_rect()
        width, height = rect.width(), rect.height()
        cx, cy = self.center_xy_m
        y_m = cy + ((point.x() - rect.left()) / width) * 2 * self.half_extent_m - self.half_extent_m
        x_m = cx + self.half_extent_m - ((point.y() - rect.top()) / height) * 2 * self.half_extent_m
        return (
            max(cx - self.half_extent_m, min(cx + self.half_extent_m, x_m)),
            max(cy - self.half_extent_m, min(cy + self.half_extent_m, y_m)),
        )

    def _clamped_center(self, x_m: float, y_m: float) -> tuple[float, float]:
        base_x, base_y = self._base_center_xy_m
        available_pan = max(0.0, self._base_half_extent_m - self.half_extent_m)
        return (
            max(base_x - available_pan, min(base_x + available_pan, float(x_m))),
            max(base_y - available_pan, min(base_y + available_pan, float(y_m))),
        )

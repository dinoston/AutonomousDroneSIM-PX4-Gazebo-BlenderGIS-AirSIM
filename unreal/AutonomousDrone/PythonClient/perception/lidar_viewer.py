"""Embedded OpenGL LiDAR point-cloud viewer."""

from __future__ import annotations

import numpy as np
import pyqtgraph.opengl as gl
from PySide6.QtWidgets import QVBoxLayout, QWidget

from common.coordinates import ned_points_to_display


class LidarViewer(QWidget):
    def __init__(
        self,
        parent: QWidget | None = None,
        fixed_color: tuple[float, float, float, float] | None = None,
        point_size: float = 2.0,
    ) -> None:
        super().__init__(parent)
        self._fixed_color = fixed_color
        self._point_size = float(point_size)
        layout = QVBoxLayout(self)
        self.view = gl.GLViewWidget()
        self.view.setCameraPosition(distance=35, elevation=25, azimuth=45)
        self.view.setBackgroundColor((12, 17, 24))

        grid = gl.GLGridItem()
        grid.setSize(60, 60)
        grid.setSpacing(5, 5)
        self.view.addItem(grid)

        axes = gl.GLAxisItem()
        axes.setSize(5, 5, 5)
        self.view.addItem(axes)

        self.cloud = gl.GLScatterPlotItem(
            pos=np.empty((0, 3), dtype=np.float32),
            color=(0.1, 0.85, 1.0, 0.8),
            size=self._point_size,
            pxMode=True,
        )
        self.view.addItem(self.cloud)
        layout.addWidget(self.view)

    def update_points(self, points: np.ndarray) -> None:
        if points.size == 0:
            self.cloud.setData(pos=np.empty((0, 3), dtype=np.float32))
            return
        display_points = ned_points_to_display(points)
        if self._fixed_color is not None:
            colors = np.tile(
                np.asarray(self._fixed_color, dtype=np.float32),
                (len(display_points), 1),
            )
        else:
            distances = np.linalg.norm(display_points, axis=1)
            normalized = np.clip(
                distances / max(float(distances.max()), 1.0),
                0.0,
                1.0,
            )
            colors = np.column_stack(
                (
                    normalized,
                    1.0 - normalized * 0.55,
                    np.ones_like(normalized),
                    np.full_like(normalized, 0.85),
                )
            ).astype(np.float32)
        self.cloud.setData(pos=display_points, color=colors, size=self._point_size)

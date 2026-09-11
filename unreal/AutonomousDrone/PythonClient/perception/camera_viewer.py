"""Qt widget for RGB, depth and segmentation streams."""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QFont, QImage, QPainter, QPen, QPixmap
from PySide6.QtWidgets import QFrame, QGridLayout, QLabel, QWidget

from perception.segmentation_labels import load_segmentation_classes


class CameraPanel(QFrame):
    def __init__(self, title: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setFrameShape(QFrame.Shape.StyledPanel)
        layout = QGridLayout(self)
        title_label = QLabel(title)
        title_label.setObjectName("sensorTitle")
        self.image_label = QLabel("영상 대기 중")
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setMinimumSize(280, 180)
        self.image_label.setStyleSheet("background:#10151d; color:#7f8b9a;")
        layout.addWidget(title_label, 0, 0)
        layout.addWidget(self.image_label, 1, 0)
        self._image = QImage()

    def set_encoded_image(
        self,
        data: bytes,
        detections: list[dict] | None = None,
    ) -> None:
        image = QImage()
        if not image.loadFromData(data):
            self.image_label.setText("영상 디코딩 실패")
            return
        if detections:
            image = image.copy()
            painter = QPainter(image)
            painter.setRenderHint(QPainter.RenderHint.Antialiasing)
            font = QFont()
            font.setBold(True)
            font.setPointSize(10)
            painter.setFont(font)
            for detection in detections:
                target_kind = detection.get("target_kind")
                is_human = target_kind == "human"
                is_bird = target_kind == "bird"
                short_target_label = "HUMAN" if is_human else ("BIRD" if is_bird else "DRONE")
                draw_lidar_box = bool(detection.get("lidar_visible", True))
                draw_radar_box = bool(detection.get("radar_confirmed", False))
                if not draw_lidar_box and not draw_radar_box:
                    continue
                x_min = int(detection.get("x_min", 0))
                y_min = int(detection.get("y_min", 0))
                x_max = int(detection.get("x_max", x_min))
                y_max = int(detection.get("y_max", y_min))

                # Radar confirmation uses a larger blue outer box so it stays
                # visually distinct from the tighter camera detection box.
                # Radar 확인 표적은 더 큰 파란 외곽 박스를 사용하여 안쪽의
                # 카메라 탐지 박스와 바로 구분할 수 있게 합니다.
                if draw_radar_box:
                    box_width = max(1, x_max - x_min)
                    box_height = max(1, y_max - y_min)
                    padding_x = max(8, int(round(box_width * 0.1)))
                    padding_y = max(8, int(round(box_height * 0.1)))
                    radar_x_min = max(0, x_min - padding_x)
                    radar_y_min = max(0, y_min - padding_y)
                    radar_x_max = min(image.width() - 1, x_max + padding_x)
                    radar_y_max = min(image.height() - 1, y_max + padding_y)
                    painter.setPen(QPen(QColor("#2494ff"), 4))
                    painter.drawRect(
                        radar_x_min,
                        radar_y_min,
                        radar_x_max - radar_x_min,
                        radar_y_max - radar_y_min,
                    )
                    radar_label = (
                        f"RADAR {short_target_label} "
                        f'{float(detection.get("radar_distance_m", 0.0)):.1f}m'
                    )
                    radar_text_y = max(16, radar_y_min - 5)
                    painter.fillRect(
                        radar_x_min,
                        radar_text_y - 15,
                        max(105, len(radar_label) * 7),
                        18,
                        QColor(18, 92, 190, 220),
                    )
                    painter.setPen(QPen(QColor("white"), 1))
                    painter.drawText(radar_x_min + 3, radar_text_y, radar_label)

                if draw_lidar_box:
                    lidar_color = (
                        QColor("#35d66f")
                        if is_human or is_bird
                        else QColor("#ff3b4f")
                    )
                    painter.setPen(QPen(lidar_color, 3))
                    painter.drawRect(x_min, y_min, x_max - x_min, y_max - y_min)
                    label = (
                        f"LIDAR {short_target_label if is_human or is_bird else 'ENEMY DRONE'} "
                        f'{float(detection.get("distance_m", 0.0)):.1f}m'
                    )
                    text_y = max(16, y_min - 5)
                    painter.fillRect(
                        x_min,
                        text_y - 15,
                        max(100, len(label) * 7),
                        18,
                        QColor(21, 130, 67, 220)
                        if is_human or is_bird
                        else QColor(180, 0, 20, 210),
                    )
                    painter.setPen(QPen(QColor("white"), 1))
                    painter.drawText(x_min + 3, text_y, label)
                    painter.setPen(QPen(lidar_color, 3))
            painter.end()
        self._image = image
        self._refresh_pixmap()

    def resizeEvent(self, event) -> None:  # noqa: N802 - Qt API
        super().resizeEvent(event)
        self._refresh_pixmap()

    def _refresh_pixmap(self) -> None:
        if self._image.isNull():
            return
        pixmap = QPixmap.fromImage(self._image).scaled(
            self.image_label.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self.image_label.setPixmap(pixmap)


class CameraViewer(QWidget):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QGridLayout(self)
        self.panels = {
            "RGB": CameraPanel("RGB"),
            "Depth": CameraPanel("Depth"),
            "Segmentation": CameraPanel("Segmentation"),
        }
        layout.addWidget(self.panels["RGB"], 0, 0)
        layout.addWidget(self.panels["Depth"], 0, 1)
        layout.addWidget(self.panels["Segmentation"], 1, 0, 1, 2)
        classes = load_segmentation_classes()
        legend_items = []
        for definition in classes:
            if definition.name == "background":
                continue
            red, green, blue = definition.color_rgb
            legend_items.append(
                f'<span style="color:rgb({red},{green},{blue});">'
                f"■ {definition.name_ko}</span>"
            )
        self.segmentation_legend = QLabel(
            "Semantic 클래스 · " + " &nbsp; ".join(legend_items)
        )
        self.segmentation_legend.setWordWrap(True)
        self.segmentation_legend.setStyleSheet(
            "background:#10151d; color:#d4dde8; padding:6px;"
        )
        layout.addWidget(self.segmentation_legend, 2, 0, 1, 2)

    def update_images(
        self,
        images: dict[str, bytes],
        detections: list[dict] | None = None,
    ) -> None:
        for name, data in images.items():
            panel = self.panels.get(name)
            if panel is not None:
                panel.set_encoded_image(
                    data,
                    detections if name == "RGB" else None,
                )

"""Generate analysis CSV files and a visual PDF report for a dataset session."""

from __future__ import annotations

import csv
import json
import math
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from statistics import mean
from typing import Any, Iterable


@dataclass(frozen=True)
class ReportResult:
    pdf_path: Path
    frames_csv: Path
    objects_csv: Path
    summary_csv: Path
    summary: dict[str, Any]


def _number(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def _at(sequence: Any, index: int, default: float = 0.0) -> float:
    if isinstance(sequence, (list, tuple)) and len(sequence) > index:
        return _number(sequence[index], default)
    return default


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    value = json.loads(path.read_text(encoding="utf-8"))
    return value if isinstance(value, dict) else {}


def _load_frames(path: Path) -> tuple[list[dict[str, Any]], int]:
    frames: list[dict[str, Any]] = []
    invalid_lines = 0
    if not path.exists():
        return frames, invalid_lines
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            invalid_lines += 1
            continue
        if isinstance(value, dict):
            frames.append(value)
        else:
            invalid_lines += 1
    return frames, invalid_lines


def _timestamp_seconds(value: Any) -> float | None:
    if value in (None, ""):
        return None
    if isinstance(value, (int, float)):
        numeric = float(value)
        if numeric > 1e15:
            return numeric / 1e9
        if numeric > 1e12:
            return numeric / 1e3
        return numeric
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00")).timestamp()
    except ValueError:
        return None


def _camera_skew_ms(timestamps: Any) -> float:
    if not isinstance(timestamps, dict):
        return 0.0
    parsed = [
        value
        for value in (_timestamp_seconds(item) for item in timestamps.values())
        if value is not None
    ]
    if len(parsed) < 2:
        return 0.0
    return max(parsed) * 1000.0 - min(parsed) * 1000.0


def _write_csv(path: Path, fieldnames: list[str], rows: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _flatten_session(
    session_dir: Path,
    session: dict[str, Any],
    frames: list[dict[str, Any]],
    invalid_lines: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any], dict[str, list[float]]]:
    session_id = session_dir.name
    config = session.get("config", {}) if isinstance(session.get("config"), dict) else {}
    class_counts: Counter[str] = Counter()
    mission_counts: Counter[str] = Counter()
    sensor_missing: Counter[str] = Counter()
    frame_rows: list[dict[str, Any]] = []
    object_rows: list[dict[str, Any]] = []
    speeds: list[float] = []
    altitudes: list[float] = []
    lidar_counts: list[float] = []
    radar_counts: list[float] = []
    positions: list[tuple[float, float, float]] = []
    radar_confirmed = 0
    lidar_visible = 0
    sensor_keys = ("rgb", "depth", "segmentation", "lidar", "radar")
    selected_sensors = {
        str(sensor) for sensor in config.get("sensors", [])
    } if isinstance(config.get("sensors"), (list, tuple)) else set(sensor_keys)

    for sequence_index, frame in enumerate(frames):
        frame_id = int(frame.get("frame_id", sequence_index))
        files = frame.get("files", {}) if isinstance(frame.get("files"), dict) else {}
        telemetry = (
            frame.get("telemetry", {})
            if isinstance(frame.get("telemetry"), dict)
            else {}
        )
        mission = frame.get("mission", {}) if isinstance(frame.get("mission"), dict) else {}
        detections = frame.get("detections", [])
        if not isinstance(detections, list):
            detections = []
        lidar = frame.get("lidar", {}) if isinstance(frame.get("lidar"), dict) else {}
        radar = frame.get("radar", {}) if isinstance(frame.get("radar"), dict) else {}
        x = _number(telemetry.get("x"))
        y = _number(telemetry.get("y"))
        altitude = _number(telemetry.get("altitude"))
        speed = _number(telemetry.get("speed"))
        positions.append((x, y, altitude))
        speeds.append(speed)
        altitudes.append(altitude)
        lidar_point_count = int(_number(lidar.get("point_count")))
        radar_point_count = int(_number(radar.get("point_count")))
        lidar_counts.append(float(lidar_point_count))
        radar_counts.append(float(radar_point_count))
        target = mission.get("active_target")
        remaining = math.hypot(_at(target, 0, x) - x, _at(target, 1, y) - y)
        mission_type = str(mission.get("type", "unknown"))
        mission_counts[mission_type] += 1
        per_frame_counts: Counter[str] = Counter()

        for object_index, detection in enumerate(detections):
            if not isinstance(detection, dict):
                continue
            class_name = str(
                detection.get("target_kind")
                or detection.get("label")
                or "unknown"
            )
            class_counts[class_name] += 1
            per_frame_counts[class_name] += 1
            radar_match = bool(detection.get("radar_confirmed", False))
            lidar_match = bool(detection.get("lidar_visible", False))
            radar_confirmed += int(radar_match)
            lidar_visible += int(lidar_match)
            object_rows.append(
                {
                    "session_id": session_id,
                    "frame_id": frame_id,
                    "object_index": object_index,
                    "name": detection.get("name", ""),
                    "class_name": class_name,
                    "radar_confirmed": radar_match,
                    "lidar_enabled": lidar_match,
                    "distance_m": _number(detection.get("distance_m")),
                    "radar_distance_m": _number(detection.get("radar_distance_m")),
                    "radar_attenuation_db": _number(
                        detection.get("radar_attenuation_db")
                    ),
                    "x_min": detection.get("x_min", ""),
                    "y_min": detection.get("y_min", ""),
                    "x_max": detection.get("x_max", ""),
                    "y_max": detection.get("y_max", ""),
                    "relative_x": detection.get("relative_x", ""),
                    "relative_y": detection.get("relative_y", ""),
                    "relative_z": detection.get("relative_z", ""),
                    "world_x_min": detection.get("world_x_min", ""),
                    "world_y_min": detection.get("world_y_min", ""),
                    "world_z_min": detection.get("world_z_min", ""),
                    "world_x_max": detection.get("world_x_max", ""),
                    "world_y_max": detection.get("world_y_max", ""),
                    "world_z_max": detection.get("world_z_max", ""),
                }
            )

        for sensor in sensor_keys:
            if sensor in selected_sensors and sensor not in files:
                sensor_missing[sensor] += 1
        frame_rows.append(
            {
                "session_id": session_id,
                "frame_id": frame_id,
                "captured_at_utc": frame.get("captured_at_utc", ""),
                "x_m": x,
                "y_m": y,
                "altitude_m": altitude,
                "speed_mps": speed,
                "vx_mps": _number(telemetry.get("vx")),
                "vy_mps": _number(telemetry.get("vy")),
                "vz_mps": _number(telemetry.get("vz")),
                "roll_deg": _number(telemetry.get("roll")),
                "pitch_deg": _number(telemetry.get("pitch")),
                "yaw_deg": _number(telemetry.get("yaw")),
                "mission_type": mission_type,
                "route_index": mission.get("route_index", ""),
                "target_x_m": _at(target, 0, x),
                "target_y_m": _at(target, 1, y),
                "target_altitude_m": _at(target, 2, altitude),
                "remaining_distance_m": remaining,
                "detection_count": len(detections),
                "human_count": per_frame_counts.get("human", 0),
                "bird_count": per_frame_counts.get("bird", 0),
                "drone_count": per_frame_counts.get("enemy_drone", 0)
                + per_frame_counts.get("drone", 0),
                "radar_confirmed_count": sum(
                    bool(item.get("radar_confirmed", False))
                    for item in detections
                    if isinstance(item, dict)
                ),
                "lidar_point_count": lidar_point_count,
                "radar_point_count": radar_point_count,
                "camera_skew_ms": _camera_skew_ms(frame.get("camera_timestamps")),
                "rgb_path": files.get("rgb", ""),
                "depth_path": files.get("depth", ""),
                "segmentation_path": files.get("segmentation", ""),
                "lidar_path": files.get("lidar", ""),
                "radar_path": files.get("radar", ""),
            }
        )

    distance_m = 0.0
    for previous, current in zip(positions, positions[1:]):
        distance_m += math.sqrt(sum((b - a) ** 2 for a, b in zip(previous, current)))
    timestamps = [
        parsed
        for parsed in (_timestamp_seconds(frame.get("captured_at_utc")) for frame in frames)
        if parsed is not None
    ]
    duration_s = max(timestamps) - min(timestamps) if len(timestamps) >= 2 else 0.0
    written_frames = int(session.get("written_frames", len(frames)))
    dropped_frames = int(session.get("dropped_frames", 0))
    attempted = written_frames + dropped_frames
    summary = {
        "schema_version": 1,
        "session_id": session_id,
        "dataset_name": config.get("dataset_name", ""),
        "city": config.get("city", ""),
        "region": config.get("region", ""),
        "terrain_type": config.get("terrain_type", ""),
        "season": config.get("season", ""),
        "time_of_day": config.get("time_of_day", ""),
        "visibility": config.get("visibility", ""),
        "precipitation": config.get("precipitation", ""),
        "precipitation_intensity": _number(
            config.get("precipitation_intensity")
        ),
        "wind_north_mps": _number(config.get("wind_north_mps")),
        "wind_east_mps": _number(config.get("wind_east_mps")),
        "sample_rate_hz": _number(config.get("sample_rate_hz")),
        "status": session.get("status", "unknown"),
        "started_at_utc": session.get("started_at_utc", ""),
        "ended_at_utc": session.get("ended_at_utc", ""),
        "duration_seconds": round(duration_s, 3),
        "written_frames": written_frames,
        "dropped_frames": dropped_frames,
        "drop_rate_percent": round(dropped_frames / attempted * 100.0, 3)
        if attempted
        else 0.0,
        "invalid_manifest_lines": invalid_lines,
        "flight_distance_m": round(distance_m, 3),
        "average_speed_mps": round(mean(speeds), 3) if speeds else 0.0,
        "maximum_speed_mps": round(max(speeds), 3) if speeds else 0.0,
        "average_altitude_m": round(mean(altitudes), 3) if altitudes else 0.0,
        "maximum_altitude_m": round(max(altitudes), 3) if altitudes else 0.0,
        "detection_rows": len(object_rows),
        "radar_confirmed_detections": radar_confirmed,
        "lidar_enabled_detections": lidar_visible,
        "average_lidar_points": round(mean(lidar_counts), 1) if lidar_counts else 0.0,
        "average_radar_points": round(mean(radar_counts), 1) if radar_counts else 0.0,
        "class_counts": dict(sorted(class_counts.items())),
        "mission_frame_counts": dict(sorted(mission_counts.items())),
        "missing_sensor_frames": dict(sorted(sensor_missing.items())),
    }
    series = {
        "speed": speeds,
        "altitude": altitudes,
        "lidar": lidar_counts,
        "radar": radar_counts,
    }
    return frame_rows, object_rows, summary, series


def _font_names() -> tuple[str, str]:
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont

    candidates = [
        (
            Path("C:/Windows/Fonts/malgun.ttf"),
            Path("C:/Windows/Fonts/malgunbd.ttf"),
        ),
        (
            Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"),
            Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc"),
        ),
    ]
    for regular_path, bold_path in candidates:
        if regular_path.exists() and bold_path.exists():
            pdfmetrics.registerFont(TTFont("ReportKorean", str(regular_path)))
            pdfmetrics.registerFont(TTFont("ReportKoreanBold", str(bold_path)))
            return "ReportKorean", "ReportKoreanBold"
    return "Helvetica", "Helvetica-Bold"


def _line_chart(values: list[float], title: str, color: Any, font_name: str) -> Any:
    from reportlab.graphics.charts.lineplots import LinePlot
    from reportlab.graphics.shapes import Drawing, String
    from reportlab.lib.colors import HexColor

    drawing = Drawing(480, 180)
    drawing.add(String(10, 164, title, fontName=font_name, fontSize=10, fillColor=HexColor("#22364d")))
    if not values:
        drawing.add(String(20, 85, "기록된 데이터가 없습니다.", fontName=font_name, fontSize=9))
        return drawing
    chart = LinePlot()
    chart.x, chart.y, chart.width, chart.height = 45, 30, 415, 120
    chart.data = [[(index, value) for index, value in enumerate(values)]]
    chart.lines[0].strokeColor = color
    chart.lines[0].strokeWidth = 1.6
    chart.xValueAxis.valueMin = 0
    chart.xValueAxis.valueMax = max(1, len(values) - 1)
    if len(values) <= 8:
        chart.xValueAxis.valueSteps = list(range(len(values))) or [0]
    else:
        chart.xValueAxis.valueSteps = sorted(
            {round(index * (len(values) - 1) / 5) for index in range(6)}
        )
    chart.yValueAxis.valueMin = min(0.0, min(values))
    chart.yValueAxis.valueMax = max(1.0, max(values) * 1.1)
    chart.xValueAxis.labelTextFormat = lambda value: str(int(value))
    chart.xValueAxis.labels.fontName = font_name
    chart.yValueAxis.labels.fontName = font_name
    chart.xValueAxis.labels.fontSize = 7
    chart.yValueAxis.labels.fontSize = 7
    chart.joinedLines = 1
    drawing.add(chart)
    return drawing


def _bar_chart(counts: dict[str, int], font_name: str) -> Any:
    from reportlab.graphics.charts.barcharts import VerticalBarChart
    from reportlab.graphics.shapes import Drawing, String
    from reportlab.lib.colors import HexColor

    drawing = Drawing(480, 205)
    drawing.add(String(10, 188, "클래스별 탐지 건수", fontName=font_name, fontSize=10, fillColor=HexColor("#22364d")))
    items = sorted(counts.items(), key=lambda item: item[1], reverse=True)[:8]
    if not items:
        drawing.add(String(20, 95, "탐지된 객체가 없습니다.", fontName=font_name, fontSize=9))
        return drawing
    chart = VerticalBarChart()
    chart.x, chart.y, chart.width, chart.height = 45, 50, 415, 120
    chart.data = [[value for _, value in items]]
    chart.categoryAxis.categoryNames = [name for name, _ in items]
    chart.valueAxis.valueMin = 0
    chart.valueAxis.valueMax = max(1, max(value for _, value in items) * 1.15)
    chart.bars[0].fillColor = HexColor("#2f80c9")
    chart.bars[0].strokeColor = None
    chart.categoryAxis.labels.fontName = font_name
    chart.categoryAxis.labels.fontSize = 7
    chart.categoryAxis.labels.angle = 25
    chart.categoryAxis.labels.dy = -8
    chart.valueAxis.labels.fontName = font_name
    chart.valueAxis.labels.fontSize = 7
    drawing.add(chart)
    return drawing


def _build_pdf(
    path: Path,
    session: dict[str, Any],
    summary: dict[str, Any],
    series: dict[str, list[float]],
) -> None:
    try:
        from reportlab.lib import colors
        from reportlab.lib.enums import TA_CENTER
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
        from reportlab.lib.units import mm
        from reportlab.platypus import (
            Paragraph,
            SimpleDocTemplate,
            Spacer,
            Table,
            TableStyle,
        )
    except ImportError as exc:
        raise RuntimeError(
            "PDF 보고서 패키지가 없습니다. requirements.txt를 다시 설치하세요."
        ) from exc

    regular_font, bold_font = _font_names()
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "KoreanTitle",
        parent=styles["Title"],
        fontName=bold_font,
        fontSize=21,
        leading=28,
        alignment=TA_CENTER,
        textColor=colors.HexColor("#16324f"),
        spaceAfter=8 * mm,
    )
    heading_style = ParagraphStyle(
        "KoreanHeading",
        parent=styles["Heading2"],
        fontName=bold_font,
        fontSize=13,
        leading=17,
        textColor=colors.HexColor("#1f5e8c"),
        spaceBefore=4 * mm,
        spaceAfter=2 * mm,
    )
    body_style = ParagraphStyle(
        "KoreanBody",
        parent=styles["BodyText"],
        fontName=regular_font,
        fontSize=9,
        leading=14,
        textColor=colors.HexColor("#25313d"),
    )
    doc = SimpleDocTemplate(
        str(path),
        pagesize=A4,
        rightMargin=16 * mm,
        leftMargin=16 * mm,
        topMargin=17 * mm,
        bottomMargin=17 * mm,
        title="Autonomous Drone Dataset Report",
        author="Autonomous Drone Mission Control",
    )

    def page_footer(canvas: Any, document: Any) -> None:
        canvas.saveState()
        canvas.setFont(regular_font, 7)
        canvas.setFillColor(colors.HexColor("#6b7785"))
        canvas.drawString(16 * mm, 9 * mm, f"Session: {summary['session_id']}")
        canvas.drawRightString(A4[0] - 16 * mm, 9 * mm, f"Page {document.page}")
        canvas.restoreState()

    config = session.get("config", {}) if isinstance(session.get("config"), dict) else {}
    story: list[Any] = [
        Paragraph("자율 드론 데이터 수집 결과 보고서", title_style),
        Paragraph(
            "원본 센서 파일을 변경하지 않고 frames.jsonl을 분석해 생성한 자동 보고서입니다.",
            body_style,
        ),
        Spacer(1, 4 * mm),
        Paragraph("세션 정보", heading_style),
    ]
    session_rows = [
        ["세션", summary["session_id"], "상태", summary["status"]],
        ["도시/맵", summary["city"] or "-", "구역", summary["region"] or "-"],
        ["지형", summary["terrain_type"] or "-", "저장 주기", f"{summary['sample_rate_hz']:.1f} Hz"],
        [
            "환경",
            f"{summary['season']} · {summary['time_of_day']} · {summary['visibility']}",
            "강수",
            f"{summary['precipitation']} {summary['precipitation_intensity']:.1f}",
        ],
        [
            "바람 N/E",
            f"{summary['wind_north_mps']:.1f} / {summary['wind_east_mps']:.1f} m/s",
            "좌표계",
            "NED",
        ],
        ["시작", str(summary["started_at_utc"]), "종료", str(summary["ended_at_utc"])],
    ]
    info_table = Table(session_rows, colWidths=[24 * mm, 55 * mm, 24 * mm, 66 * mm])
    info_table.setStyle(
        TableStyle(
            [
                ("FONTNAME", (0, 0), (-1, -1), regular_font),
                ("FONTNAME", (0, 0), (0, -1), bold_font),
                ("FONTNAME", (2, 0), (2, -1), bold_font),
                ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#eaf2f8")),
                ("BACKGROUND", (2, 0), (2, -1), colors.HexColor("#eaf2f8")),
                ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#b6c7d6")),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("FONTSIZE", (0, 0), (-1, -1), 8),
                ("LEADING", (0, 0), (-1, -1), 11),
                ("TOPPADDING", (0, 0), (-1, -1), 5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ]
        )
    )
    story.extend([info_table, Paragraph("핵심 결과", heading_style)])
    metric_rows = [
        ["저장 프레임", f"{summary['written_frames']:,}", "누락률", f"{summary['drop_rate_percent']:.2f}%"],
        ["비행 시간", f"{summary['duration_seconds']:.1f}초", "이동 거리", f"{summary['flight_distance_m']:.1f}m"],
        ["평균/최대 속도", f"{summary['average_speed_mps']:.2f} / {summary['maximum_speed_mps']:.2f} m/s", "평균/최대 고도", f"{summary['average_altitude_m']:.2f} / {summary['maximum_altitude_m']:.2f}m"],
        ["객체 탐지 행", f"{summary['detection_rows']:,}", "Radar 확인", f"{summary['radar_confirmed_detections']:,}"],
        ["평균 LiDAR 점", f"{summary['average_lidar_points']:,.1f}", "평균 Radar 점", f"{summary['average_radar_points']:,.1f}"],
    ]
    metric_table = Table(metric_rows, colWidths=[32 * mm, 48 * mm, 32 * mm, 57 * mm])
    metric_table.setStyle(
        TableStyle(
            [
                ("FONTNAME", (0, 0), (-1, -1), regular_font),
                ("FONTNAME", (0, 0), (0, -1), bold_font),
                ("FONTNAME", (2, 0), (2, -1), bold_font),
                ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#edf6ff")),
                ("BACKGROUND", (2, 0), (2, -1), colors.HexColor("#edf6ff")),
                ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#91aec5")),
                ("INNERGRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#c7d6e2")),
                ("FONTSIZE", (0, 0), (-1, -1), 8),
                ("TOPPADDING", (0, 0), (-1, -1), 6),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ]
        )
    )
    story.extend(
        [
            metric_table,
            Paragraph("비행 상태 그래프", heading_style),
            _line_chart(series["speed"], "프레임별 속도 (m/s)", colors.HexColor("#2478b4"), regular_font),
            _line_chart(series["altitude"], "프레임별 고도 (m)", colors.HexColor("#23a36d"), regular_font),
            Paragraph("객체 및 센서 분석", heading_style),
            _bar_chart(summary["class_counts"], regular_font),
        ]
    )
    class_rows = [["클래스", "탐지 건수"]] + [
        [name, f"{count:,}"] for name, count in summary["class_counts"].items()
    ]
    if len(class_rows) == 1:
        class_rows.append(["탐지 없음", "0"])
    class_table = Table(class_rows, colWidths=[90 * mm, 45 * mm], repeatRows=1)
    class_table.setStyle(
        TableStyle(
            [
                ("FONTNAME", (0, 0), (-1, -1), regular_font),
                ("FONTNAME", (0, 0), (-1, 0), bold_font),
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#274c69")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#b6c7d6")),
                ("FONTSIZE", (0, 0), (-1, -1), 8),
                ("ALIGN", (1, 1), (1, -1), "RIGHT"),
            ]
        )
    )
    story.extend([class_table, Paragraph("데이터 품질 확인", heading_style)])
    issues: list[str] = []
    if summary["drop_rate_percent"] > 1.0:
        issues.append(f"저장 누락률이 {summary['drop_rate_percent']:.2f}%로 높습니다.")
    if summary["invalid_manifest_lines"]:
        issues.append(f"읽지 못한 manifest 행이 {summary['invalid_manifest_lines']}개 있습니다.")
    missing = {
        key: value
        for key, value in summary["missing_sensor_frames"].items()
        if value > 0
    }
    if missing:
        issues.append("센서 파일 누락: " + ", ".join(f"{key} {value}개" for key, value in missing.items()))
    if not summary["detection_rows"]:
        issues.append("객체 탐지 기록이 없습니다. 탐지 API와 annotation 선택 상태를 확인하세요.")
    if not issues:
        issues.append("manifest 손상이나 주요 저장 누락이 발견되지 않았습니다.")
    for issue in issues:
        story.append(Paragraph(f"- {issue}", body_style))
    story.extend(
        [
            Spacer(1, 4 * mm),
            Paragraph(
                "참고: 현재 탐지 행은 시뮬레이터 탐지 결과입니다. Precision, Recall, F1을 계산하려면 추후 Unreal Ground Truth 객체 ID와 예측 결과를 분리해 저장해야 합니다.",
                body_style,
            ),
        ]
    )
    doc.build(story, onFirstPage=page_footer, onLaterPages=page_footer)


def generate_session_report(session_dir: Path | str) -> ReportResult:
    """Create normalized CSV outputs and a Korean PDF summary for one session."""
    root = Path(session_dir).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"세션 폴더를 찾을 수 없습니다: {root}")
    session_path = root / "session.json"
    manifest_path = root / "frames.jsonl"
    if not session_path.exists() or not manifest_path.exists():
        raise FileNotFoundError("session.json과 frames.jsonl이 있는 세션 폴더를 선택하세요.")

    session = _load_json(session_path)
    frames, invalid_lines = _load_frames(manifest_path)
    frame_rows, object_rows, summary, series = _flatten_session(
        root,
        session,
        frames,
        invalid_lines,
    )
    analysis_dir = root / "analysis"
    analysis_dir.mkdir(exist_ok=True)
    frames_csv = analysis_dir / "frames.csv"
    objects_csv = analysis_dir / "objects.csv"
    summary_csv = analysis_dir / "session_summary.csv"
    pdf_path = analysis_dir / "session_report.pdf"

    frame_fields = list(frame_rows[0]) if frame_rows else [
        "session_id", "frame_id", "captured_at_utc", "x_m", "y_m",
        "altitude_m", "speed_mps", "detection_count",
    ]
    object_fields = list(object_rows[0]) if object_rows else [
        "session_id", "frame_id", "object_index", "name", "class_name",
        "radar_confirmed", "lidar_enabled", "distance_m",
    ]
    scalar_summary = {
        key: json.dumps(value, ensure_ascii=False, separators=(",", ":"))
        if isinstance(value, dict)
        else value
        for key, value in summary.items()
    }
    _write_csv(frames_csv, frame_fields, frame_rows)
    _write_csv(objects_csv, object_fields, object_rows)
    _write_csv(summary_csv, list(scalar_summary), [scalar_summary])
    _build_pdf(pdf_path, session, summary, series)
    return ReportResult(pdf_path, frames_csv, objects_csv, summary_csv, summary)

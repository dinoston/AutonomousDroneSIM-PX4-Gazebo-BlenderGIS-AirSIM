"""Stable semantic-segmentation labels for every simulated city map.

모든 시뮬레이션 도시 맵에서 동일하게 사용하는 Semantic Segmentation
클래스와 AirSim 메시 할당 규칙입니다.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DEFAULT_CLASS_MAP_PATH = (
    Path(__file__).resolve().parents[1] / "config" / "class_map.json"
)


@dataclass(frozen=True)
class SegmentationClass:
    class_id: int
    name: str
    name_ko: str
    color_rgb: tuple[int, int, int]
    priority: int
    patterns: tuple[str, ...]


def load_segmentation_classes(
    path: str | Path = DEFAULT_CLASS_MAP_PATH,
) -> list[SegmentationClass]:
    """Load and validate the shared class map."""
    source = Path(path)
    payload = json.loads(source.read_text(encoding="utf-8"))
    if int(payload.get("schema_version", 0)) != 1:
        raise ValueError(f"지원하지 않는 class_map 스키마입니다: {source}")

    classes: list[SegmentationClass] = []
    used_ids: set[int] = set()
    used_names: set[str] = set()
    for raw in payload.get("classes", []):
        class_id = int(raw["id"])
        name = str(raw["name"])
        color = tuple(int(channel) for channel in raw["color_rgb"])
        if not 0 <= class_id <= 255:
            raise ValueError(f"Segmentation ID는 0~255여야 합니다: {class_id}")
        if class_id in used_ids or name in used_names:
            raise ValueError(f"중복된 Segmentation 클래스입니다: {name}/{class_id}")
        if len(color) != 3 or any(channel < 0 or channel > 255 for channel in color):
            raise ValueError(f"잘못된 RGB 색상입니다: {name}={color}")
        classes.append(
            SegmentationClass(
                class_id=class_id,
                name=name,
                name_ko=str(raw.get("name_ko", name)),
                color_rgb=(color[0], color[1], color[2]),
                priority=int(raw.get("priority", 0)),
                patterns=tuple(str(pattern) for pattern in raw.get("patterns", [])),
            )
        )
        used_ids.add(class_id)
        used_names.add(name)

    required = {"background", "building", "human", "drone", "bird"}
    missing = required - used_names
    if missing:
        raise ValueError(f"필수 Segmentation 클래스가 없습니다: {sorted(missing)}")
    return classes


def configure_air_sim_segmentation(
    client: Any,
    classes: list[SegmentationClass] | None = None,
    object_names: list[str] | None = None,
) -> dict[str, object]:
    """Assign stable semantic IDs to Unreal meshes through the AirSim RPC.

    Lower-priority defaults are applied first, then specific object classes
    overwrite them. A false RPC return only means that a pattern had no match
    in the current level; it is not an error.
    """
    definitions = classes or load_segmentation_classes()
    runtime_names = (
        [str(name) for name in object_names]
        if object_names is not None
        else [str(name) for name in client.simListInstanceSegmentationObjects()]
    )
    assignments_by_class: dict[str, list[str]] = {
        definition.name: [] for definition in definitions
    }
    class_counts: dict[str, int] = {
        definition.name: 0 for definition in definitions
    }
    for object_name in runtime_names:
        definition = classify_segmentation_object(object_name, definitions)
        assignments_by_class[definition.name].append(object_name)

    assignments: dict[str, int] = {}
    failed_objects: list[str] = []
    if object_names is None:
        # A large Unreal level can expose thousands of segmentation objects.
        # One RPC per object made connection take tens of minutes. Apply one
        # combined regex per semantic class instead (normally 6-9 RPC calls).
        # 대형 레벨의 수천 개 객체를 하나씩 RPC 호출하지 않고 클래스마다
        # 결합 정규식 한 번으로 지정합니다.
        for definition in sorted(definitions, key=lambda item: item.priority):
            class_objects = assignments_by_class[definition.name]
            if not class_objects or not definition.patterns:
                continue
            combined_pattern = "|".join(
                f"(?:{pattern})" for pattern in definition.patterns
            )
            result = bool(
                client.simSetSegmentationObjectID(
                    combined_pattern,
                    definition.class_id,
                    True,
                )
            )
            if result:
                for object_name in class_objects:
                    assignments[object_name] = definition.class_id
                class_counts[definition.name] = len(class_objects)
            else:
                failed_objects.extend(class_objects)
    else:
        # Dynamic NPCs and birds normally arrive a few at a time, so exact
        # names avoid resetting the whole map on every refresh.
        for object_name in runtime_names:
            definition = classify_segmentation_object(object_name, definitions)
            result = bool(
                client.simSetSegmentationObjectID(
                    object_name,
                    definition.class_id,
                    False,
                )
            )
            if result:
                assignments[object_name] = definition.class_id
                class_counts[definition.name] += 1
            else:
                failed_objects.append(object_name)

    return {
        "class_count": len(definitions),
        "object_count": len(runtime_names),
        "assigned_count": len(assignments),
        "class_counts": class_counts,
        "assignments": assignments,
        "failed_objects": failed_objects,
        "classes": [
            {
                "id": definition.class_id,
                "name": definition.name,
                "name_ko": definition.name_ko,
                "color_rgb": list(definition.color_rgb),
            }
            for definition in sorted(definitions, key=lambda item: item.class_id)
        ],
    }


def classify_segmentation_object(
    object_name: str,
    classes: list[SegmentationClass] | None = None,
) -> SegmentationClass:
    """Return the highest-priority semantic rule matching an AirSim object."""
    definitions = classes or load_segmentation_classes()
    ordered = sorted(definitions, key=lambda item: item.priority, reverse=True)
    for definition in ordered:
        if any(
            re.fullmatch(pattern, str(object_name), flags=re.IGNORECASE)
            for pattern in definition.patterns
        ):
            return definition
    return next(
        definition for definition in definitions if definition.name == "background"
    )

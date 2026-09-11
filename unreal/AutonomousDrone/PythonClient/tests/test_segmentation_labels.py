"""Tests for stable semantic class assignment."""

from __future__ import annotations

from perception.segmentation_labels import (
    classify_segmentation_object,
    configure_air_sim_segmentation,
    load_segmentation_classes,
)


class _FakeClient:
    def __init__(self) -> None:
        self.calls: list[tuple[str, int, bool]] = []

    @staticmethod
    def simListInstanceSegmentationObjects() -> list[str]:
        return [
            "SM_Hangar2_4",
            "SM_Road_Asphalt_Corner_4",
            "BP_AINormalPeople_Drone_C_2",
            "FlockCharacter_12",
            "UnknownProp_7",
        ]

    def simSetSegmentationObjectID(
        self,
        pattern: str,
        class_id: int,
        is_regex: bool,
    ) -> bool:
        self.calls.append((pattern, class_id, is_regex))
        return True


def test_class_map_has_stable_required_ids() -> None:
    classes = load_segmentation_classes()
    by_name = {definition.name: definition for definition in classes}

    assert by_name["background"].class_id == 0
    assert by_name["building"].class_id == 219
    assert by_name["building"].color_rgb == (255, 95, 79)
    assert by_name["road"].class_id == 60
    assert by_name["road"].color_rgb == (95, 95, 255)
    assert by_name["human"].class_id == 216
    assert by_name["drone"].class_id == 252
    assert by_name["bird"].class_id == 48
    assert len({definition.class_id for definition in classes}) == len(classes)


def test_initial_level_is_assigned_in_class_batches() -> None:
    client = _FakeClient()
    report = configure_air_sim_segmentation(client)

    assert report["class_count"] == 9
    assert report["object_count"] == 5
    assert report["assigned_count"] == 5
    assert len(client.calls) == 5
    assert all(call[2] is True for call in client.calls)
    assert {class_id for _pattern, class_id, _is_regex in client.calls} == {
        7,
        48,
        60,
        216,
        219,
    }
    assert report["assignments"]["SM_Hangar2_4"] == 219
    assert report["assignments"]["SM_Road_Asphalt_Corner_4"] == 60
    assert report["assignments"]["BP_AINormalPeople_Drone_C_2"] == 216
    assert report["assignments"]["FlockCharacter_12"] == 48
    assert report["assignments"]["UnknownProp_7"] == 7


def test_new_dynamic_objects_use_exact_names() -> None:
    client = _FakeClient()
    report = configure_air_sim_segmentation(
        client,
        object_names=["FlockCharacter_12", "BP_AINormalPeople_Drone_C_2"],
    )

    assert report["assigned_count"] == 2
    assert client.calls == [
        ("FlockCharacter_12", 48, False),
        ("BP_AINormalPeople_Drone_C_2", 216, False),
    ]


def test_human_rule_overrides_drone_substring() -> None:
    result = classify_segmentation_object("BP_AINormalPeople_Drone_C_2")
    assert result.name == "human"

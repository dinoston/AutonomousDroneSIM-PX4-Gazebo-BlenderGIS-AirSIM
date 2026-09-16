from common.minimap_profiles import (
    MinimapProfile,
    build_scene_identity,
    match_minimap_profile,
    profile_id_from_name,
    unique_profile_id,
    with_scene_identity,
)


def sample_profile(profile_id: str = "seoul") -> MinimapProfile:
    return MinimapProfile(
        profile_id=profile_id,
        display_name="서울 테스트 맵",
        image="assets/minimaps/seoul.png",
        center_x_m=0.0,
        center_y_m=0.0,
        half_extent_m=1500.0,
    )


def test_scene_signature_is_order_independent_and_ignores_dynamic_objects():
    first = build_scene_identity(["Building_A_12", "Road_3", "BP_EnemyDrone_C_2"])
    second = build_scene_identity(["Road_99", "Building_A_7"])
    assert first["signature"] == second["signature"]


def test_exact_scene_signature_selects_registered_profile():
    names = [f"Seoul_Block_{chr(65 + index)}" for index in range(20)]
    profile = with_scene_identity(sample_profile(), build_scene_identity(names))
    match, reason, score = match_minimap_profile(reversed(names), [profile])
    assert match == profile
    assert reason == "장면 서명"
    assert score == 1.0


def test_mapid_marker_has_priority():
    profile = sample_profile("seoul_city")
    match, reason, score = match_minimap_profile(
        ["SomeActor", "MAPID_Seoul_City_0"],
        [profile],
    )
    assert match == profile
    assert reason == "MAPID 마커"
    assert score == 1.0


def test_korean_name_gets_stable_profile_id():
    assert profile_id_from_name("서울 강남 지도") == profile_id_from_name("서울 강남 지도")
    assert profile_id_from_name("서울 강남 지도").startswith("map_")


def test_new_profile_id_never_overwrites_existing_map():
    existing = sample_profile("pangyo")
    assert unique_profile_id([existing], "pangyo") == "pangyo_2"


def test_generic_engine_actors_do_not_identify_a_blendergis_level():
    generic = [
        "PlayerStart",
        "SceneCapture2D",
        "StaticMeshActor_12",
        "SkyLight",
        "WorldSettings",
        "VolumetricCloud",
        "DirectionalLight",
        "AirSimGameMode",
        "SimHUD",
        "Brush",
        "PlayerController",
        "WeatherActor",
    ]
    profile = with_scene_identity(sample_profile(), build_scene_identity(generic))
    match, reason, _score = match_minimap_profile(generic, [profile])
    assert match is None
    assert reason == "미등록 장면"

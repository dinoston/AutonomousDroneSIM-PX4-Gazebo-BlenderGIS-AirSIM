from common.safety import minimum_braking_distance_m


def test_braking_distance_grows_with_speed() -> None:
    assert minimum_braking_distance_m(8.0) > minimum_braking_distance_m(3.0)
    assert minimum_braking_distance_m(8.0) >= 15.0


def test_braking_distance_has_stationary_margin() -> None:
    assert minimum_braking_distance_m(0.0) == 3.5

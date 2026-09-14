from navigation.grid_planner import AltitudeGridPlanner
from navigation.grid_planner import PlannerConfig
from navigation.grid_planner import build_vertical_barrier
from navigation.grid_planner import split_terminal_vertical_leg

import numpy as np
import pytest


def test_direct_path_without_obstacles() -> None:
    planner = AltitudeGridPlanner()
    path = planner.plan((0.0, 0.0), (10.0, 0.0), 5.0, 5.0)
    assert path[-1] == (10.0, 0.0, 5.0)


def test_planner_avoids_wall_horizontally_or_vertically() -> None:
    planner = AltitudeGridPlanner()
    wall = np.array([(5.0, y, -5.0) for y in np.linspace(-4.0, 4.0, 30)], dtype=np.float32)
    planner.set_obstacle_points(wall)
    path = planner.plan((0.0, 0.0), (10.0, 0.0), 5.0, 5.0)
    assert path
    assert any(abs(y) > 4.0 or altitude > 5.0 for _x, y, altitude in path)


def test_planner_can_choose_higher_layer() -> None:
    planner = AltitudeGridPlanner()
    wall = np.array([(0.0, y, -5.0) for y in np.linspace(-100.0, 100.0, 401)], dtype=np.float32)
    planner.set_obstacle_points(wall)
    path = planner.plan((-10.0, 0.0), (10.0, 0.0), 5.0, 5.0)
    assert max(altitude for _x, _y, altitude in path) > 5.0
    assert path[0][0:2] == (-10.0, 0.0)
    assert path[-1] == (10.0, 0.0, 5.0)


def test_planner_keeps_climbing_when_lidar_sees_the_wall_again() -> None:
    planner = AltitudeGridPlanner()
    wall_layers = np.array(
        [
            (0.0, y, -altitude)
            for altitude in (5.0, 7.0, 9.0)
            for y in np.linspace(-100.0, 100.0, 401)
        ],
        dtype=np.float32,
    )
    planner.set_obstacle_points(wall_layers)
    path = planner.plan((-10.0, 0.0), (10.0, 0.0), 5.0, 5.0)
    assert path[0] == (-10.0, 0.0, 11.0)
    assert path[-1] == (10.0, 0.0, 5.0)


def test_replan_does_not_descend_before_clearing_an_obstacle() -> None:
    planner = AltitudeGridPlanner()
    wall = np.array(
        [(0.0, y, -7.0) for y in np.linspace(-100.0, 100.0, 401)],
        dtype=np.float32,
    )
    planner.set_obstacle_points(wall)
    path = planner.plan((-10.0, 0.0), (10.0, 0.0), 5.0, 7.0)
    assert path[0] == (-10.0, 0.0, 9.0)
    assert all(altitude >= 7.0 for _x, _y, altitude in path[:-1])
    assert path[-1] == (10.0, 0.0, 5.0)


def test_replan_respects_the_absolute_climb_limit() -> None:
    planner = AltitudeGridPlanner(PlannerConfig(max_extra_altitude_m=8.0))
    wall = np.array(
        [(0.0, y, -13.0) for y in np.linspace(-100.0, 100.0, 401)],
        dtype=np.float32,
    )
    planner.set_obstacle_points(wall)
    with pytest.raises(RuntimeError):
        planner.plan((-10.0, 0.0), (10.0, 0.0), 5.0, 13.0)


def test_ceiling_cap_forces_a_lower_cruise_layer() -> None:
    planner = AltitudeGridPlanner(PlannerConfig(max_extra_altitude_m=8.0))
    path = planner.plan(
        (-10.0, 0.0),
        (10.0, 0.0),
        5.0,
        13.0,
        max_altitude_m=10.0,
    )
    assert path[0] == (-10.0, 0.0, 10.0)
    assert max(altitude for _x, _y, altitude in path) <= 10.0
    assert path[-1] == (10.0, 0.0, 5.0)


def test_detected_facade_is_blocked_at_every_flight_layer() -> None:
    planner = AltitudeGridPlanner(PlannerConfig(max_extra_altitude_m=8.0))
    barrier = build_vertical_barrier(
        (5.0, 0.0, -5.0),
        (1.0, 0.0),
        half_span_m=12.0,
        minimum_altitude_m=1.0,
        maximum_altitude_m=13.0,
    )
    planner.set_obstacle_points(barrier)
    for altitude in (1.0, 5.0, 9.0, 13.0):
        blocked = planner._blocked_cells(altitude)
        assert planner._to_cell((5.0, 0.0)) in blocked


def test_detected_facade_produces_a_lateral_fly_by_route() -> None:
    config = PlannerConfig(
        half_extent_m=900.0,
        resolution_m=2.5,
        drone_radius_m=2.5,
        vertical_clearance_m=1.5,
        altitude_step_m=2.0,
        max_extra_altitude_m=8.0,
    )
    planner = AltitudeGridPlanner(config)
    planner.set_obstacle_points(
        build_vertical_barrier((35.0, 0.0, -5.0), (1.0, 0.0), 12.0, 1.0, 13.0)
    )
    path = planner.plan((0.0, 0.0), (100.0, 0.0), 5.0, 5.0)
    assert any(abs(y) >= 12.5 for _x, y, _altitude in path)
    assert path[-1] == (100.0, 0.0, 5.0)


def test_forward_advance_detour_does_not_begin_by_reversing() -> None:
    config = PlannerConfig(
        half_extent_m=900.0,
        resolution_m=4.0,
        drone_radius_m=4.0,
        vertical_clearance_m=2.0,
        altitude_step_m=4.0,
        max_extra_altitude_m=12.0,
    )
    planner = AltitudeGridPlanner(config)
    planner.set_obstacle_points(
        build_vertical_barrier(
            (20.0, 0.0, -5.0),
            (1.0, 0.0),
            8.0,
            1.0,
            17.0,
        )
    )

    path = planner.plan((0.0, 0.0), (100.0, 0.0), 5.0, 5.0)

    assert path
    assert all(x_m >= 0.0 for x_m, _y_m, _altitude_m in path)
    assert any(abs(y_m) >= 12.0 for _x_m, y_m, _altitude_m in path)


def test_horizontal_rooftop_keeps_a_forward_climb_route_open() -> None:
    config = PlannerConfig(
        half_extent_m=900.0,
        resolution_m=4.0,
        drone_radius_m=4.0,
        vertical_clearance_m=2.0,
        altitude_step_m=4.0,
        max_extra_altitude_m=12.0,
    )
    planner = AltitudeGridPlanner(config)
    offsets = np.arange(-12.0, 12.1, 2.0, dtype=np.float32)
    patch_x, patch_y = np.meshgrid(offsets, offsets)
    rooftop = np.column_stack(
        (
            20.0 + patch_x.ravel(),
            patch_y.ravel(),
            np.full(patch_x.size, -4.0, dtype=np.float32),
        )
    )
    planner.set_obstacle_points(rooftop)

    path = planner.plan((0.0, 0.0), (100.0, 0.0), 5.0, 5.0)

    assert path[0] == (0.0, 0.0, 9.0)
    assert all(x_m >= 0.0 for x_m, _y_m, _altitude_m in path)
    assert max(altitude_m for _x_m, _y_m, altitude_m in path) == 9.0


def test_terminal_descent_is_separated_from_horizontal_flight() -> None:
    horizontal, descent_altitude = split_terminal_vertical_leg(
        [(20.0, -10.0, 13.0), (100.0, 0.0, 13.0), (100.0, 0.0, 5.0)]
    )
    assert horizontal == [(20.0, -10.0, 13.0), (100.0, 0.0, 13.0)]
    assert descent_altitude == 5.0


def test_astar_does_not_cut_through_a_blocked_diagonal_corner() -> None:
    planner = AltitudeGridPlanner(
        PlannerConfig(half_extent_m=10.0, resolution_m=1.0, drone_radius_m=0.1)
    )
    start = planner._to_cell((0.0, 0.0))
    goal = planner._to_cell((1.0, 1.0))
    blocked = {
        planner._to_cell((1.0, 0.0)),
        planner._to_cell((0.0, 1.0)),
    }
    path = planner._astar(start, goal, blocked)
    assert len(path) > 2
    assert not planner._line_clear(start, goal, blocked)


def test_start_footprint_returns_do_not_trap_the_vehicle() -> None:
    planner = AltitudeGridPlanner(
        PlannerConfig(half_extent_m=20.0, resolution_m=1.0, drone_radius_m=2.0)
    )
    start = planner._to_cell((0.0, 0.0))
    goal = planner._to_cell((10.0, 0.0))
    blocked = {
        (start[0] + dx, start[1] + dy)
        for dx in range(-2, 3)
        for dy in range(-2, 3)
        if dx or dy
    }
    path = planner._astar(start, goal, blocked)
    assert path[0] == start
    assert path[-1] == goal

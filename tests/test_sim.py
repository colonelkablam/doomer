import math

import pytest

from ok_doomer.shared.collision import Obstacle, move_box
from ok_doomer.shared.config import ConfigError, MovementConfig, SimConfig, from_dict
from ok_doomer.shared.events import PlayerSpawned
from ok_doomer.shared.mapfile import parse_map
from ok_doomer.shared.player import PlayerInput
from ok_doomer.shared.sim import step
from ok_doomer.shared.things import THING_TYPES, ThingType, register_thing_type
from ok_doomer.shared.timestep import FixedTimestep
from ok_doomer.shared.world import ModeNotSupported, World, drain_events, new_world, spawn_player

ROOM = parse_map(
    """\
[walls]
#######
#.....#
#.....#
#..#..#
#.....#
#######

[things]
coop_start 1.5 1.5 0
exit 5.5 4.5 0
"""
)
R = 0.25


def world_with_player(x: float = 1.5, y: float = 1.5) -> World:
    world = new_world(ROOM, "coop")
    player = spawn_player(world, 1, "test")
    player.x, player.y = x, y
    return world


# --- collision ---------------------------------------------------------------


def test_free_move_is_unchanged() -> None:
    assert move_box(ROOM, 1.5, 1.5, 0.5, 0.25, R) == pytest.approx((2.0, 1.75))


def test_stops_flush_against_wall() -> None:
    # East wall of the room is column 6, so the box's right edge stops at x = 6.
    x, y = move_box(ROOM, 5.0, 1.5, 3.0, 0.0, R)
    assert (x, y) == pytest.approx((6 - R, 1.5))
    # West wall is column 0, so the left edge stops at x = 1.
    x, _ = move_box(ROOM, 2.0, 1.5, -3.0, 0.0, R)
    assert x == pytest.approx(1 + R)


def test_slides_along_wall_when_moving_diagonally() -> None:
    # Pushing north-east into the north wall: y blocked, x keeps moving.
    x, y = move_box(ROOM, 2.0, 1.5, 0.5, -0.5, R)
    assert x == pytest.approx(2.5)
    assert y == pytest.approx(1 + R)


def test_cannot_tunnel_through_pillar_with_large_move() -> None:
    # Pillar at cell (3, 3). A single huge move must still be stopped by it.
    x, _ = move_box(ROOM, 1.5, 3.5, 10.0, 0.0, R)
    assert x == pytest.approx(3 - R)


def test_resting_flush_against_wall_can_move_away_and_along() -> None:
    x, y = move_box(ROOM, 6 - R, 2.0, -0.1, 0.0, R)
    assert x == pytest.approx(6 - R - 0.1)
    x, y = move_box(ROOM, 6 - R, 2.0, 0.0, 0.5, R)
    assert (x, y) == pytest.approx((6 - R, 2.5))


# --- simulation step -----------------------------------------------------------


def test_forward_moves_along_facing_angle() -> None:
    world = world_with_player(2.5, 2.5)
    step(world, {1: PlayerInput(forward=1.0, angle=math.pi / 2)})  # 90 deg = south
    p = world.players[1]
    d = world.config.movement.walk_speed * world.config.dt
    assert (p.x, p.y) == pytest.approx((2.5, 2.5 + d))
    assert world.tick == 1


def test_strafe_right_is_clockwise_of_facing() -> None:
    world = world_with_player(2.5, 2.5)
    step(world, {1: PlayerInput(strafe=1.0, angle=0.0)})  # facing east, right = south
    p = world.players[1]
    assert p.y > 2.5
    assert p.x == pytest.approx(2.5)


def test_diagonal_is_not_faster() -> None:
    world = world_with_player(2.5, 2.5)
    step(world, {1: PlayerInput(forward=1.0, strafe=1.0, angle=0.0)})
    p = world.players[1]
    d = world.config.movement.walk_speed * world.config.dt
    assert math.hypot(p.x - 2.5, p.y - 2.5) == pytest.approx(d)


def test_angle_is_taken_from_input_and_normalised() -> None:
    world = world_with_player()
    step(world, {1: PlayerInput(angle=-math.pi / 2)})
    assert world.players[1].angle == pytest.approx(3 * math.pi / 2)


def test_deterministic() -> None:
    inputs = [PlayerInput(forward=1.0, strafe=(i % 3) - 1, angle=i * 0.1) for i in range(200)]
    results = []
    for _ in range(2):
        world = world_with_player()
        for inp in inputs:
            step(world, {1: inp})
        p = world.players[1]
        results.append((p.x, p.y, p.angle))
    assert results[0] == results[1]


def test_spawn_emits_event_and_uses_mode_spawn() -> None:
    world = new_world(ROOM, "coop")
    player = spawn_player(world, 7, "nick")
    assert (player.x, player.y) == (1.5, 1.5)
    assert drain_events(world) == [PlayerSpawned(0, 7, 1.5, 1.5)]
    assert drain_events(world) == []


def test_unsupported_mode_is_rejected() -> None:
    with pytest.raises(ModeNotSupported, match="needs 4 more DM spawns"):
        new_world(ROOM, "deathmatch")


# --- timestep ------------------------------------------------------------------


def test_fixed_timestep_accumulates_and_reports_alpha() -> None:
    ts = FixedTimestep(dt=0.01)
    assert ts.advance(0.025) == 2
    assert ts.alpha == pytest.approx(0.5)
    assert ts.advance(0.005) == 1
    assert ts.alpha == pytest.approx(0.0, abs=1e-9)


def test_fixed_timestep_caps_catch_up() -> None:
    ts = FixedTimestep(dt=0.01, max_steps=8)
    assert ts.advance(5.0) == 8
    assert ts.accumulator == 0.0


# --- config ----------------------------------------------------------------------


def test_config_from_dict_nested_with_int_to_float() -> None:
    cfg = from_dict(SimConfig, {"tick_rate": 30, "movement": {"walk_speed": 5}})
    assert cfg == SimConfig(tick_rate=30, movement=MovementConfig(walk_speed=5.0))
    assert isinstance(cfg.movement.walk_speed, float)


@pytest.mark.parametrize(
    "data",
    [{"tick_rte": 30}, {"tick_rate": "fast"}, {"tick_rate": True}, {"movement": 3}],
)
def test_config_rejects_bad_input(data: dict[str, object]) -> None:
    with pytest.raises(ConfigError):
        from_dict(SimConfig, data)


# --- blocking things -------------------------------------------------------------

OB = Obstacle.square(4.0, 2.5, 0.25)
OPEN = parse_map("[walls]\n########\n#......#\n#......#\n#......#\n########\n")


def test_obstacle_stops_movement_on_each_side() -> None:
    reach = R + OB.half_x
    assert move_box(OPEN, 2.0, 2.5, 3.0, 0.0, R, [OB])[0] == pytest.approx(OB.x - reach)
    assert move_box(OPEN, 5.5, 2.5, -3.0, 0.0, R, [OB])[0] == pytest.approx(OB.x + reach)
    assert move_box(OPEN, 4.0, 1.4, 0.0, 3.0, R, [OB])[1] == pytest.approx(OB.y - reach)


def test_slides_past_obstacle_and_misses_when_not_lined_up() -> None:
    x, y = move_box(OPEN, 3.0, 2.5, 0.5, 0.5, R, [OB])  # diagonal into its west face
    assert x == pytest.approx(OB.x - R - OB.half_x)
    assert y == pytest.approx(3.0)
    # Passing well to the north: no effect.
    assert move_box(OPEN, 2.0, 1.4, 3.0, 0.0, R, [OB]) == pytest.approx((5.0, 1.4))


def test_can_walk_out_if_overlapping() -> None:
    x, _ = move_box(OPEN, 4.1, 2.5, 0.5, 0.0, R, [OB])
    assert x == pytest.approx(4.6)


def test_world_builds_obstacles_from_blocking_things() -> None:
    m = parse_map(
        "[walls]\n######\n#....#\n######\n[things]\n"
        "coop_start 1.5 1.5 0\nexit 4.5 1.5 0\ndesk 3.5 1.5 0\nceiling_light 2.5 1.5 0\n"
    )
    world = new_world(m, "coop")
    assert world.obstacles == (Obstacle.square(3.5, 1.5, THING_TYPES["desk"].radius),)
    player = spawn_player(world, 1, "p")
    for _ in range(60):
        step(world, {1: PlayerInput(forward=1.0, angle=0.0)})
    assert player.x == pytest.approx(3.5 - THING_TYPES["desk"].radius - R)


def test_blocking_thing_type_needs_radius() -> None:
    with pytest.raises(ValueError, match="needs a radius"):
        register_thing_type(ThingType("broken", "no radius", blocking=True))

"""Spawning players (spread round the start, not stacked on it), idle input,
and moving the whole team on to the next floor."""

import math

from ok_doomer.shared.mapfile import load_map, parse_map
from ok_doomer.shared.player import PlayerInput, PlayerState
from ok_doomer.shared.sim import step
from ok_doomer.shared.spawn import arrive_on_floor, spawn_player
from ok_doomer.shared.world import World, new_world

ROOM = """\
[walls]
#######
#.....#
#.....#
#.....#
#######

[things]
coop_start 3.5 2.5 0
exit 5.5 3.5 0
"""

# A corridor one cell wide, running east: there's only room ahead or behind.
CORRIDOR = """\
[walls]
#######
#.....#
#######

[things]
coop_start 3.5 1.5 0
exit 5.5 1.5 0
"""


def _overlap(a: PlayerState, b: PlayerState, radius: float) -> bool:
    return abs(a.x - b.x) < 2 * radius - 1e-9 and abs(a.y - b.y) < 2 * radius - 1e-9


def _no_overlaps(world: World) -> bool:
    ps = list(world.players.values())
    r = world.config.movement.radius
    return not any(_overlap(a, b, r) for i, a in enumerate(ps) for b in ps[i + 1 :])


def test_the_first_player_spawns_on_the_start() -> None:
    world = new_world(parse_map(ROOM), "coop")
    p = spawn_player(world, 1, "p")
    assert (p.x, p.y, p.angle) == (3.5, 2.5, 0.0)


def test_more_players_spread_round_the_start() -> None:
    world = new_world(parse_map(ROOM), "coop")
    for i in range(1, 5):
        spawn_player(world, i, f"p{i}")
    assert _no_overlaps(world)
    r = world.config.movement.radius
    for p in world.players.values():  # all near the start, none in a wall
        assert math.dist((p.x, p.y), (3.5, 2.5)) <= 2 * r * math.sqrt(2) + 1e-9
        assert 1 + r - 1e-9 <= p.x <= 6 - r + 1e-9 and 1 + r - 1e-9 <= p.y <= 4 - r + 1e-9


def test_in_a_corridor_they_spawn_ahead_and_behind() -> None:
    world = new_world(parse_map(CORRIDOR), "coop")
    for i in range(1, 4):
        spawn_player(world, i, f"p{i}")
    assert _no_overlaps(world)
    assert all(p.y == 1.5 for p in world.players.values())  # walls stop the sideways tries


def test_no_input_keeps_the_view_and_the_crouch() -> None:
    world = new_world(parse_map(ROOM), "coop")
    p = spawn_player(world, 1, "p")
    step(world, {1: PlayerInput(angle=1.0, pitch=0.4, crouch=True)})
    step(world, {})  # this tick's input didn't arrive
    assert (p.angle, p.pitch, p.crouching) == (1.0, 0.4, True)
    assert (p.x, p.y) == (3.5, 2.5)


def test_the_team_arrives_on_the_next_floor_together() -> None:
    world = new_world(load_map("maps/office.map"), "coop")
    a, b = spawn_player(world, 1, "ann"), spawn_player(world, 2, "bob")
    a.health, b.coffee = 40, 25
    world.passes.add("blue")
    up = arrive_on_floor(world, "1")
    assert up.map.storey == "1" and not up.passes
    assert sorted(up.players) == [1, 2]
    assert (up.players[1].name, up.players[1].health) == ("ann", 40)
    assert (up.players[2].name, up.players[2].coffee) == ("bob", 25)
    # Both in the car they arrive in.
    (car,) = [d.car for d in up.doors if d.car_here and d.arrive_in > 0]
    assert all((math.floor(p.x), math.floor(p.y)) == car for p in up.players.values())


def test_arriving_is_deterministic() -> None:
    def positions() -> list[tuple[float, float]]:
        world = new_world(load_map("maps/office.map"), "coop", seed=7)
        for i in range(1, 4):
            spawn_player(world, i, f"p{i}")
        up = arrive_on_floor(world, "1")
        return [(p.x, p.y) for p in up.players.values()] + [(up.rng.random(), 0.0)]

    assert positions() == positions()

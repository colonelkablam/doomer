"""Doors (doomer/shared/doors.py): opening, closing, collision, drawing."""

import math

import pytest

from doomer.shared.config import MovementConfig
from doomer.shared.doors import DOOR_HALF, HINGED, OPEN_SECONDS, SLIDING, doors_from_map
from doomer.shared.events import DoorToggled
from doomer.shared.mapfile import parse_map
from doomer.shared.player import PlayerInput, PlayerState
from doomer.shared.sim import step
from doomer.shared.world import World, drain_events, new_world, spawn_player

MOVE = MovementConfig()
NORTH, SOUTH, EAST = -math.pi / 2, math.pi / 2, 0.0

# Two rooms, north (y 1..3) and south (y 5..7), divided by a wall along y = 4
# with doorways at x = 3 (hinged, opens north) and x = 8 (sliding).
ROOMS = """\
[walls]
############
#..........#
#..........#
#..........#
###.####.###
#..........#
#..........#
#..........#
############

[ceiling]
888888888888
888888888888
888888888888
888888888888
888688886888

[things]
coop_start 3.5 6.5 270
exit 10.5 1.5 0
door_hinged 3.5 4.5 270
door_sliding 8.5 4.5 90
"""


def world_at(x: float, y: float) -> tuple[World, PlayerState]:
    world = new_world(parse_map(ROOMS), "coop")
    player = spawn_player(world, 1, "p")
    player.x, player.y = x, y
    return world, player


def run(world: World, ticks: int, **held: float | bool) -> None:
    angle = held.pop("angle", world.players[1].angle)
    for _ in range(ticks):
        step(world, {1: PlayerInput(angle=float(angle), **held)})  # type: ignore[arg-type]


def ticks_for(kind: str) -> int:
    return math.ceil(OPEN_SECONDS[kind] * 60) + 2


def test_doors_come_from_the_map_shut() -> None:
    hinged, sliding = doors_from_map(parse_map(ROOMS))
    assert (hinged.kind, sliding.kind) == (HINGED, SLIDING)
    assert (hinged.x, hinged.y) == (3.5, 4.5) and hinged.facing == pytest.approx(3 * math.pi / 2)
    assert not hinged.across_x  # it stands in the plane y = 4.5
    assert hinged.open == sliding.open == 0.0


def test_a_shut_hinged_door_stops_you() -> None:
    world, p = world_at(3.5, 6.0)
    run(world, 90, forward=1.0, angle=NORTH)
    assert p.y == pytest.approx(4.5 + DOOR_HALF + MOVE.radius)


def test_use_opens_a_hinged_door_and_you_can_walk_through() -> None:
    world, p = world_at(3.5, 5.4)
    run(world, 1, use=True, angle=NORTH)
    events = drain_events(world)
    assert any(isinstance(e, DoorToggled) and e.opening for e in events)
    run(world, ticks_for(HINGED), angle=NORTH)  # let it swing open
    assert world.doors[0].open == 1.0
    run(world, 90, forward=1.0, angle=NORTH)
    assert p.y < 3.0  # through, into the north room


def test_holding_use_toggles_once() -> None:
    world, _ = world_at(3.5, 5.4)
    run(world, 30, use=True, angle=NORTH)  # held for half a second
    assert world.doors[0].opening


def test_use_needs_you_to_face_the_door_and_be_near_it() -> None:
    world, _ = world_at(3.5, 5.4)
    run(world, 1, use=True, angle=SOUTH)  # back to it
    assert not world.doors[0].opening
    world, _ = world_at(3.5, 6.9)  # 2.4 away
    run(world, 1, use=True, angle=NORTH)
    assert not world.doors[0].opening


def test_a_door_wont_shut_on_you() -> None:
    world, p = world_at(3.5, 5.4)
    run(world, 1, use=True, angle=NORTH)
    run(world, ticks_for(HINGED), angle=NORTH)
    p.y = 4.5  # standing in the doorway
    run(world, 1, angle=SOUTH)  # (release use)
    run(world, 1, use=True, angle=NORTH)  # try to shut it on yourself
    run(world, ticks_for(HINGED), angle=NORTH)
    assert world.doors[0].open == 1.0
    p.y = 5.4  # step back, clear of the doorway (> 0.75) but within reach (1.3)
    run(world, 1, angle=NORTH)
    run(world, 1, use=True, angle=NORTH)
    run(world, ticks_for(HINGED), angle=NORTH)
    assert world.doors[0].open == 0.0


def test_sliding_doors_open_as_you_come_and_close_behind_you() -> None:
    world, p = world_at(8.5, 7.0)
    run(world, 150, forward=1.0, angle=NORTH)  # walk up to them and through
    assert p.y < 2.0
    run(world, 120, forward=1.0, angle=EAST)  # move away (blocked by the wall: fine)
    p.y = 1.5
    run(world, ticks_for(SLIDING) + 30, angle=EAST)
    assert world.doors[1].open == 0.0


def test_a_shut_door_hides_what_is_behind_it_and_an_open_one_doesnt() -> None:
    import dataclasses

    from doomer.client.boxes import door_boxes
    from doomer.client.config import RenderConfig
    from doomer.client.render import Camera, Renderer

    game_map = parse_map(ROOMS)
    r = Renderer(RenderConfig())
    camera = Camera(8.5, 6.5, NORTH)  # looking at the sliding doors, 2 cells away
    centre_row, centre_col = int(r.config.height / 2) + 5, r.config.width // 2
    for fraction, expect_door in ((0.0, True), (1.0, False)):
        doors = [dataclasses.replace(d, open=fraction) for d in doors_from_map(game_map)]
        r.render_indices(game_map, camera, boxes=door_boxes(doors))
        depth = r.zbuffer[centre_col, centre_row]
        # Glass is a dither: check a few pixels across, so one hits a solid texel.
        near = (r.zbuffer[centre_col - 3 : centre_col + 3, centre_row] < 2.1).any()
        assert near == expect_door, (fraction, depth)

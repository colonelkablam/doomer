"""Cubicle screens on screened desk pairs ('j'): where they are, and that
they block you on the desk tops (shared simulation, so the server agrees)."""

import math

import pytest

from ok_doomer.shared.config import MovementConfig
from ok_doomer.shared.mapfile import parse_map
from ok_doomer.shared.player import PlayerInput, PlayerState
from ok_doomer.shared.screens import SCREEN_HALF, SCREEN_HEIGHT, Screen, desk_screens
from ok_doomer.shared.sim import step
from ok_doomer.shared.spawn import spawn_player
from ok_doomer.shared.world import World, new_world

MOVE = MovementConfig()
DESK_TOP = 0.25
EAST, SOUTH = 0.0, math.pi / 2

# Three screened pairs facing north at x 2..4, y 2 (sitters north and south):
# screens along y = 2.5 and across x = 2, 3, 4, 5. Aisles at y = 1 and 3.
POD = """\
[walls]
#######
#.....#
#.jjj.#
#.....#
#######

[floor]
0000000
0000000
0022200

[facing]
.......
.......
..nnn..

[things]
coop_start 1.5 1.5 0
exit 5.5 3.5 0
"""


def on_desk(x: float, y: float, text: str = POD) -> tuple[World, PlayerState]:
    world = new_world(parse_map(text), "coop")
    player = spawn_player(world, 1, "p")
    player.x, player.y, player.z, player.on_ground = x, y, DESK_TOP, True
    return world, player


def walk(world: World, angle: float, ticks: int, jump: bool = False) -> None:
    for _ in range(ticks):
        step(world, {1: PlayerInput(forward=1.0, angle=angle, jump=jump)})


def test_screens_are_where_the_pairs_meet_and_at_each_cell_edge() -> None:
    screens = desk_screens(parse_map(POD))
    assert sorted(screens) == sorted(
        [Screen(1, 2.5, x, x + 1, DESK_TOP) for x in (2, 3, 4)]
        + [Screen(0, x, 2, 3, DESK_TOP) for x in (2, 3, 4, 5)]
    )


def test_the_middle_screen_stops_you_crossing_the_pair() -> None:
    world, p = on_desk(3.5, 2.2)  # on the north desk of the middle pair
    walk(world, SOUTH, 40)
    assert p.y == pytest.approx(2.5 - SCREEN_HALF - MOVE.radius)
    assert p.z == pytest.approx(DESK_TOP)


def test_side_screens_stop_you_walking_along_the_row() -> None:
    world, p = on_desk(3.5, 2.2)
    walk(world, EAST, 40)
    assert p.x == pytest.approx(4 - SCREEN_HALF - MOVE.radius)


def test_screens_dont_matter_on_the_floor() -> None:
    world, p = on_desk(1.5, 1.5)
    p.z = 0.0  # in the aisle, alongside the pod: walk the length of it
    walk(world, EAST, 90)
    assert p.x > 5.0


def test_under_a_normal_ceiling_you_cant_jump_over_a_screen() -> None:
    # On a desk, a storey-high ceiling leaves 1 - 0.25 - 0.67 = 0.08 to jump.
    world, p = on_desk(3.5, 2.2)
    walk(world, SOUTH, 40, jump=True)
    assert p.y == pytest.approx(2.5 - SCREEN_HALF - MOVE.radius)


def test_under_a_high_ceiling_you_can_jump_over_a_screen() -> None:
    assert MOVE.jump_apex > SCREEN_HEIGHT
    tall = POD.replace("[things]", "[ceiling]\n" + "ggggggg\n" * 5 + "\n[things]")
    world, p = on_desk(3.5, 2.2, tall)
    walk(world, SOUTH, 40, jump=True)
    assert p.y > 2.5  # over the middle screen, onto the south desk (or beyond)

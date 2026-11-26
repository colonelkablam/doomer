"""Vertical movement: gravity, jumping, crouching, standing on desks, stairs,
ledges and ceilings. All in the shared simulation, so the server agrees."""

import math

import pytest

from ok_doomer.shared.config import MovementConfig
from ok_doomer.shared.mapfile import parse_map
from ok_doomer.shared.player import PlayerInput, PlayerState
from ok_doomer.shared.sim import step
from ok_doomer.shared.world import World, new_world, spawn_player

MOVE = MovementConfig()

# A corridor along x. Desk pair (d, height 2/8) at x = 4; counter (2, height
# 3/8) at x = 8; open floor between. Start at x = 1.5.
FURNITURE = """\
[walls]
#############
#...d...2...#
#############

[floor]
0000000000000
0000200030000

[things]
coop_start 1.5 1.5 0
exit 11.5 1.5 0
"""
# Stairs: steps of 1/8 at x = 3, 4, 5 up to a raised floor (4/8) at x = 6..9,
# under a double-height ceiling (as in an atrium) so there's headroom on top.
STAIRS = """\
[walls]
############
#..eeeeeee.#
############

[floor]
000000000000
000123444400

[ceiling]
gggggggggggg
gggggggggggg

[things]
coop_start 1.5 1.5 0
exit 10.5 1.5 0
"""


def world_for(text: str) -> tuple[World, PlayerState]:
    world = new_world(parse_map(text), "coop")
    return world, spawn_player(world, 1, "p")


def run(world: World, ticks: int, **held: float | bool) -> None:
    for _ in range(ticks):
        step(world, {1: PlayerInput(angle=0.0, **held)})  # type: ignore[arg-type]


def test_jump_speed_reaches_the_configured_apex() -> None:
    world, p = world_for(FURNITURE)
    run(world, 1, jump=True)
    peak = p.z
    for _ in range(120):
        run(world, 1)
        peak = max(peak, p.z)
    assert peak == pytest.approx(MOVE.jump_apex, abs=0.02)
    assert p.on_ground and p.z == 0.0  # back on the floor


def test_desk_blocks_you_on_the_floor_but_you_can_jump_onto_it() -> None:
    world, p = world_for(FURNITURE)
    run(world, 60, forward=1.0)
    assert p.x == pytest.approx(4 - MOVE.radius)  # stopped at the desk
    # Jump once and keep walking: blocked until your feet clear the desk top
    # (~8 ticks), then you move over it. Stop there and land on it.
    run(world, 1, forward=1.0, jump=True)
    run(world, 19, forward=1.0)
    run(world, 40)
    assert p.z == pytest.approx(0.25) and p.on_ground  # standing on the desk top
    assert 4.0 < p.x < 5.0
    run(world, 30, forward=1.0)  # walk off the far side and drop
    assert p.z == 0.0 and p.on_ground


def test_counter_is_too_high_to_jump_onto() -> None:
    world, p = world_for(FURNITURE)
    p.x = 6.5
    run(world, 90, forward=1.0, jump=True)
    assert p.x == pytest.approx(8 - MOVE.radius)
    assert p.z < 0.375


def test_walking_up_stairs_climbs_each_step() -> None:
    world, p = world_for(STAIRS)
    run(world, 120, forward=1.0)
    assert p.x > 6.5
    assert p.z == pytest.approx(0.5)  # on the raised floor


def test_walking_off_a_ledge_falls() -> None:
    world, p = world_for(STAIRS)
    p.x, p.z = 8.5, 0.5
    run(world, 70, forward=1.0)  # ~30 ticks to clear the edge, ~20 to fall
    assert p.x > 10.0
    assert p.z == 0.0 and p.on_ground


def test_crouch_lowers_speed_and_you_cant_stand_under_a_low_ceiling() -> None:
    low = parse_map(
        "[walls]\n##########\n#........#\n##########\n\n"
        "[ceiling]\n8888888888\n8884448888\n\n"  # 4/8 = 1.35 m ceiling at x = 3..5
        "[things]\ncoop_start 1.5 1.5 0\nexit 8.5 1.5 0\n"
    )
    world = new_world(low, "coop")
    p = spawn_player(world, 1, "p")
    run(world, 60, forward=1.0)
    assert p.x == pytest.approx(3 - MOVE.radius)  # too low to walk under standing
    run(world, 30, forward=1.0, crouch=True)
    assert p.crouching and p.x > 3.5
    run(world, 1)  # let go of crouch under the low ceiling
    assert p.crouching  # no room to stand

    world2 = new_world(low, "coop")
    q = spawn_player(world2, 1, "q")
    step(world2, {1: PlayerInput(forward=1.0, crouch=True)})
    crouch_step = q.x - 1.5
    q.x = 1.5
    q.crouching = False
    step(world2, {1: PlayerInput(forward=1.0)})
    assert crouch_step == pytest.approx((q.x - 1.5) * MOVE.crouch_speed_factor)


def test_head_bumps_the_ceiling_when_jumping_under_a_low_one() -> None:
    low = parse_map(
        "[walls]\n#####\n#...#\n#####\n\n[ceiling]\n88888\n85558\n\n"  # 5/8 = 0.625
        "[things]\ncoop_start 1.5 1.5 0\nexit 3.5 1.5 0\n"
    )
    world = new_world(low, "coop")
    p = spawn_player(world, 1, "p")
    run(world, 1, crouch=True, jump=True)
    peak = 0.0
    for _ in range(60):
        run(world, 1, crouch=True)
        peak = max(peak, p.z)
    assert peak <= 0.625 - MOVE.crouch_height + 1e-9


def test_spawning_on_a_raised_floor_stands_on_it() -> None:
    world = new_world(
        parse_map(
            "[walls]\n#####\n#.e.#\n#####\n\n[floor]\n00000\n00300\n\n"
            "[things]\ncoop_start 2.5 1.5 0\nexit 3.5 1.5 0\n"
        ),
        "coop",
    )
    p = spawn_player(world, 1, "p")
    assert p.z == pytest.approx(3 / 8)
    run(world, 5)
    assert p.z == pytest.approx(3 / 8) and p.on_ground


def test_vertical_movement_is_deterministic() -> None:
    results = []
    for _ in range(2):
        world, p = world_for(FURNITURE)
        for i in range(300):
            step(world, {1: PlayerInput(forward=1.0, angle=math.sin(i / 30), jump=i % 37 == 0)})
        results.append((p.x, p.y, p.z, p.z_velocity))
    assert results[0] == results[1]


# Two stair cells (h) facing north: 0 -> 4/8 at y = 3, 4/8 -> 8/8 at y = 2, then
# a full-storey landing (e, 8/8) at y = 1, under a double-height ceiling.
FLIGHT = """\
[walls]
#####
#.e.#
#.h.#
#.h.#
#...#
#####

[floor]
00000
00800
00800
00400
00000

[ceiling]
ggggg
ggggg
ggggg
ggggg
ggggg

[facing]
.....
.....
..n..
..n..

[things]
coop_start 2.5 4.5 0
exit 3.5 4.5 0
"""


def test_walk_up_a_flight_of_stairs_to_the_landing_and_back_down() -> None:
    world, p = world_for(FLIGHT)
    north = -math.pi / 2
    for _ in range(150):
        step(world, {1: PlayerInput(forward=1.0, angle=north)})
        assert p.on_ground  # follows the slope; never airborne
    assert p.y < 2.0
    assert p.z == pytest.approx(1.0)  # a full storey up
    for _ in range(150):
        step(world, {1: PlayerInput(forward=-1.0, angle=north)})
        assert p.on_ground  # walks down rather than falling and bouncing
    assert p.z == pytest.approx(0.0)


def test_cant_climb_onto_stairs_from_the_side_halfway_up() -> None:
    world, p = world_for(FLIGHT)
    p.x, p.y = 1.5, 2.5  # beside the upper stair cell (0.5..1.0 high there)
    for _ in range(30):
        step(world, {1: PlayerInput(forward=1.0, angle=0.0)})
    assert p.x == pytest.approx(2 - MOVE.radius)


# A balcony (i, a full storey up) at y = 1..2, stairs up to it at x = 2,
# open floor below at y = 3..4.
BALCONY = """\
[walls]
######
#iiii#
#iiii#
#.h..#
#.h..#
#....#
######

[floor]
000000
088880
088880
008000
004000
000000

[ceiling]
gggggg
gggggg
gggggg
gggggg
gggggg
gggggg
gggggg

[facing]
......
......
......
..n...
..n...

[things]
coop_start 2.5 5.5 0
exit 4.5 5.5 0
"""


def test_railings_stop_you_walking_off_a_balcony_but_not_down_the_stairs() -> None:
    world, p = world_for(BALCONY)
    p.x, p.y, p.z = 3.5, 1.5, 1.0  # on the balcony, beside the stairs
    south = math.pi / 2
    for _ in range(60):
        step(world, {1: PlayerInput(forward=1.0, angle=south)})
    assert p.y == pytest.approx(3 - MOVE.radius)  # held by the railing at the edge
    assert p.z == pytest.approx(1.0)
    p.x = 2.5  # now in line with the stairs: walk down them
    for _ in range(120):
        step(world, {1: PlayerInput(forward=1.0, angle=south)})
    assert p.y > 5.0 and p.z == pytest.approx(0.0)


def test_plain_raised_floor_has_no_railing() -> None:
    world, p = world_for(BALCONY.replace("#iiii#", "#eeee#"))
    p.x, p.y, p.z = 3.5, 1.5, 1.0
    for _ in range(90):
        step(world, {1: PlayerInput(forward=1.0, angle=math.pi / 2)})
    assert p.y > 3.0 and p.z == pytest.approx(0.0)  # walked off and fell

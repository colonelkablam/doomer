"""Lifts: calling one (only with power), getting in, and leaving the floor."""

import math

from doomer.shared.doors import LIFT, OPEN_SECONDS
from doomer.shared.events import LevelComplete, LiftNoPower
from doomer.shared.mapfile import parse_map
from doomer.shared.player import PlayerInput, PlayerState
from doomer.shared.sim import step
from doomer.shared.switches import POWER
from doomer.shared.world import World, drain_events, new_world, spawn_player

SOUTH, NORTH = math.pi / 2, -math.pi / 2

# A lobby (y 1..3) with a lift in its south wall: the car is the cell (4, 4),
# its doors its outer wall on the lobby line.
LOBBY = """\
[walls]
##########
#........#
#........#
#........#
####.#####
##########

[ceiling]
8888888888
8888888888
8888888888
8888888888
8888688888

[things]
coop_start 2.5 2.5 0
door_lift 4.5 4.5 270
"""


def lobby(players: int = 1, powered: bool = True) -> tuple[World, list[PlayerState]]:
    world = new_world(parse_map(LOBBY), "coop")
    ps = [spawn_player(world, i + 1, f"p{i}") for i in range(players)]
    if not powered:
        world.flags.discard(POWER)
    return world, ps


def run(world: World, ticks: int, inputs: dict[int, PlayerInput]) -> list[object]:
    events: list[object] = []
    for _ in range(ticks):
        step(world, inputs)
        events += drain_events(world)
    return events


def settle() -> int:
    return math.ceil(OPEN_SECONDS[LIFT] * 60) + 2


def test_the_lift_door_and_its_car() -> None:
    world, _ = lobby()
    (door,) = world.doors
    assert door.kind == LIFT and door.car == (4, 4)


def test_without_power_the_lift_wont_come() -> None:
    world, (p,) = lobby(powered=False)
    p.x, p.y = 4.5, 3.4
    events = run(world, 1, {1: PlayerInput(angle=SOUTH, use=True)})
    run(world, settle(), {1: PlayerInput(angle=SOUTH)})
    assert world.doors[0].open == 0.0
    assert any(isinstance(e, LiftNoPower) for e in events)


def test_call_it_step_in_and_the_floor_is_done() -> None:
    world, (p,) = lobby()
    p.x, p.y = 4.5, 3.4
    run(world, 1, {1: PlayerInput(angle=SOUTH, use=True)})
    run(world, settle(), {1: PlayerInput(angle=SOUTH)})
    assert world.doors[0].open == 1.0
    events = run(world, 90, {1: PlayerInput(forward=1.0, angle=SOUTH)})  # into the car
    assert (math.floor(p.x), math.floor(p.y)) == (4, 4)
    events += run(world, settle(), {1: PlayerInput(angle=SOUTH)})
    assert world.doors[0].open == 0.0  # the doors shut behind you
    assert world.complete
    assert sum(isinstance(e, LevelComplete) for e in events) == 1


def test_it_waits_for_everyone() -> None:
    world, (p1, p2) = lobby(players=2)
    p1.x, p1.y, p2.x, p2.y = 4.5, 3.4, 2.5, 1.5
    idle = PlayerInput(angle=SOUTH)
    run(world, 1, {1: PlayerInput(angle=SOUTH, use=True), 2: idle})
    run(world, settle(), {1: idle, 2: idle})
    run(world, 90, {1: PlayerInput(forward=1.0, angle=SOUTH), 2: idle})
    run(world, settle() + 30, {1: idle, 2: idle})
    assert not world.complete and world.doors[0].open == 1.0  # held for the other player
    p2.x, p2.y = 4.5, 4.6  # now both aboard (one cell: they overlap, which is fine)
    run(world, settle() + 2, {1: idle, 2: idle})
    assert world.complete


def test_hud_messages_for_lifts() -> None:
    from doomer.client.hud import message_for

    assert message_for(LiftNoPower(0, 0, 1)) == "The lift has no power"
    assert message_for(LevelComplete(0)) is None  # a banner instead (Hud.complete)


def test_a_lift_has_its_fittings_round_it() -> None:
    from doomer.client.boxes import door_boxes
    from doomer.shared.doors import LIFT, Door

    lift = Door(0, LIFT, 2.5, 2.5, -math.pi / 2)
    dark = door_boxes([lift], power=False)
    lit = door_boxes([lift], power=True)
    # Two leaves plus lining, architrave, sill, indicator, call plate and
    # button, and the car's panel: more than the leaves alone.
    assert len(dark) == len(lit) > 8
    assert sum(p.box.glow for p in lit) > sum(p.box.glow for p in dark)  # lit with power

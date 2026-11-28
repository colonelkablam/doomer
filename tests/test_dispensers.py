"""Dispensers: the water cooler gives health, a cupful a use, up to an
allowance for each player."""

import math

from ok_doomer.client.hud import message_for
from ok_doomer.shared.dispensers import DISPENSERS
from ok_doomer.shared.events import Dispensed
from ok_doomer.shared.mapfile import parse_map
from ok_doomer.shared.player import PlayerInput, PlayerState
from ok_doomer.shared.sim import step
from ok_doomer.shared.spawn import spawn_player
from ok_doomer.shared.world import World, drain_events, new_world

# A corridor; the water cooler at its east end, players facing it.
ROOM = """\
[walls]
#######
#.....#
#######

[things]
coop_start 4.5 1.5 0
exit 1.5 1.5 0
water_cooler 5.6 1.5 180
"""
COOLER = DISPENSERS["water_cooler"]


def _use(world: World, player: PlayerState) -> list[Dispensed]:
    """One press of E (and its release), facing east, at the cooler."""
    step(world, {player.id: PlayerInput(angle=0.0, use=True)})
    step(world, {player.id: PlayerInput(angle=0.0)})
    return [e for e in drain_events(world) if isinstance(e, Dispensed)]


def _world(*healths: int) -> tuple[World, list[PlayerState]]:
    world = new_world(parse_map(ROOM), "coop")
    players = []
    for i, health in enumerate(healths, start=1):
        p = spawn_player(world, i, f"p{i}")
        p.health, p.x = health, 5.1  # within reach (players don't block each other)
        players.append(p)
    drain_events(world)
    return world, players


def test_a_cupful_a_use_until_the_allowance_is_gone() -> None:
    world, (p,) = _world(50)
    gained = [
        e.amount for _ in range(COOLER.allowance // COOLER.per_use + 1) for e in _use(world, p)
    ]
    assert gained == [COOLER.per_use] * (COOLER.allowance // COOLER.per_use) + [0]
    assert p.health == 50 + COOLER.allowance
    (last,) = _use(world, p)
    assert last.left == 0 and message_for(last) == "The water cooler has nothing left for you"


def test_each_player_has_their_own_allowance() -> None:
    world, (a, b) = _world(10, 10)
    for _ in range(COOLER.allowance // COOLER.per_use):
        _use(world, a)
    (empty,) = _use(world, a)
    assert empty.amount == 0
    (first,) = _use(world, b)  # b's allowance is untouched
    assert first.amount == COOLER.per_use and first.left == COOLER.allowance - COOLER.per_use


def test_a_full_meter_takes_nothing_and_keeps_the_allowance() -> None:
    world, (p,) = _world(100)
    (event,) = _use(world, p)
    assert event.amount == 0 and event.left == COOLER.allowance
    assert message_for(event) == "Health's full"
    p.health = 98
    (topped,) = _use(world, p)
    assert topped.amount == 2 and p.health == 100  # only what fits
    assert message_for(topped) == "Health +2"


def test_you_must_face_it() -> None:
    world, (p,) = _world(50)
    step(world, {p.id: PlayerInput(angle=math.pi, use=True)})  # facing away
    assert not [e for e in drain_events(world) if isinstance(e, Dispensed)]


VENDING = """\
[walls]
#######
#.....#
#######

[things]
coop_start 4.5 1.5 0
exit 1.5 1.5 0
vending_machine 5.99 1.5 180
breaker 2.5 1.01 90
"""


def test_the_vending_machine_needs_the_power_on() -> None:
    from ok_doomer.shared.switches import POWER

    world = new_world(parse_map(VENDING), "coop")
    p = spawn_player(world, 1, "p")
    p.health, p.x = 50, 5.2
    drain_events(world)
    (dead,) = _use(world, p)
    assert dead.amount == 0 and dead.reason == "no power" and p.health == 50
    assert message_for(dead) == "The vending machine has no power"
    world.flags.add(POWER)
    (snack,) = _use(world, p)
    vending = DISPENSERS["vending_machine"]
    assert snack.amount == vending.per_use == 10 and p.health == 60
    assert snack.left == vending.allowance - 10 and message_for(snack) == "Health +10"


def test_the_office_has_a_vending_machine_on_a_solid_wall() -> None:
    from ok_doomer.shared.mapfile import load_map

    office = load_map("maps/office.map")
    (machine,) = office.things_of_type("vending_machine")
    bx = machine.x - 0.05 * math.cos(machine.angle)
    by = machine.y - 0.05 * math.sin(machine.angle)
    cell = (math.floor(bx), math.floor(by))
    assert office.is_full_wall(*cell) and office.walls[cell[1]][cell[0]] != 4

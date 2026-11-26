"""Coffee cups: a pickup that tops up the coffee meter (the armour)."""

from doomer.client.hud import message_for
from doomer.shared.events import Event, Refreshed
from doomer.shared.mapfile import parse_map
from doomer.shared.pickups import MAX_COFFEE
from doomer.shared.player import PlayerInput
from doomer.shared.sim import step
from doomer.shared.world import drain_events, new_world, spawn_player

ROOM = """\
[walls]
######
#....#
######

[things]
coop_start 1.5 1.5 0
exit 4.5 1.5 0
coffee_cup 3.0 1.5 0
"""


def _walk_east(ticks: int, coffee: int = 0) -> tuple[int, list[Event], bool]:
    world = new_world(parse_map(ROOM), "coop")
    player = spawn_player(world, 1, "p")
    player.coffee = coffee
    drain_events(world)
    for _ in range(ticks):
        step(world, {1: PlayerInput(forward=1.0, angle=0.0)})
    (cup,) = world.pickups
    return player.coffee, drain_events(world), cup.taken


def test_a_coffee_cup_gives_five_coffee() -> None:
    coffee, events, taken = _walk_east(40)
    assert coffee == 5 and taken
    (refreshed,) = [e for e in events if isinstance(e, Refreshed)]
    assert (refreshed.meter, refreshed.amount, refreshed.pickup) == ("coffee", 5, "coffee_cup")
    assert message_for(refreshed) == "Coffee +5"


def test_coffee_tops_out_and_a_full_meter_leaves_the_cup() -> None:
    coffee, events, _ = _walk_east(40, MAX_COFFEE - 2)
    assert coffee == MAX_COFFEE  # only 2 added
    assert [e.amount for e in events if isinstance(e, Refreshed)] == [2]
    coffee, events, taken = _walk_east(40, MAX_COFFEE)
    assert coffee == MAX_COFFEE and not taken  # left for someone who needs it
    assert not [e for e in events if isinstance(e, Refreshed)]


def test_the_office_has_cups_on_its_desks() -> None:
    from doomer.shared.furniture import resting_height
    from doomer.shared.mapfile import load_map

    office = load_map("maps/office.map")
    cups = office.things_of_type("coffee_cup")
    assert len(cups) >= 4
    assert all(resting_height(office, c) > 0.2 for c in cups)  # on desk tops and reception's

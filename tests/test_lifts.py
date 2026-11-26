"""Lifts: calling one (only with power) from another floor, its doors
(they stay open until shut from inside, and won't shut on anyone), riding it
up, and arriving."""

import math

from ok_doomer.shared.building import floor_index
from ok_doomer.shared.doors import ARRIVE_SECONDS, FLOOR_SECONDS, LIFT, OPEN_SECONDS
from ok_doomer.shared.events import (
    DoorLocked,
    LevelComplete,
    LiftCalled,
    LiftDeparting,
    LiftNoPower,
)
from ok_doomer.shared.mapfile import parse_map
from ok_doomer.shared.player import PlayerInput, PlayerState
from ok_doomer.shared.sim import RIDE_SECONDS, step
from ok_doomer.shared.spawn import spawn_player
from ok_doomer.shared.switches import POWER
from ok_doomer.shared.world import World, drain_events, new_world

SOUTH, NORTH = math.pi / 2, -math.pi / 2
G = floor_index("G")

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
door_lift 4.5 4.5 270{props}
"""
OUTSIDE = (4.5, 3.4)  # at the call button, facing the doors
INSIDE = (4.5, 4.5)


def lobby(
    players: int = 1, powered: bool = True, props: str = ""
) -> tuple[World, list[PlayerState]]:
    world = new_world(parse_map(LOBBY.format(props=props)), "coop")
    ps = [spawn_player(world, i + 1, f"p{i}") for i in range(players)]
    if not powered:
        world.flags.discard(POWER)
    return world, ps


def run(world: World, seconds: float, inputs: dict[int, PlayerInput]) -> list[object]:
    events: list[object] = []
    for _ in range(math.ceil(seconds * 60)):
        step(world, inputs)
        events += drain_events(world)
    return events


def press(world: World, player: int = 1, angle: float = SOUTH) -> list[object]:
    """One press of use (and a tick to let go)."""
    events = run(world, 1 / 60, {player: PlayerInput(angle=angle, use=True)})
    return events + run(world, 1 / 60, {player: PlayerInput(angle=angle)})


SETTLE = OPEN_SECONDS[LIFT] + 0.1
IDLE = PlayerInput(angle=SOUTH)


def test_the_lift_door_and_its_car() -> None:
    world, _ = lobby()
    (door,) = world.doors
    assert door.kind == LIFT and door.car == (4, 4)
    assert door.here == G and door.car_here  # no car= : waiting here


def test_without_power_the_lift_wont_come() -> None:
    world, (p,) = lobby(powered=False)
    p.x, p.y = OUTSIDE
    events = press(world)
    run(world, SETTLE, {1: IDLE})
    assert world.doors[0].open == 0.0
    assert any(isinstance(e, LiftNoPower) for e in events)


def test_called_from_b3_it_counts_up_to_g_then_opens() -> None:
    world, (p,) = lobby(props=" car=B3")
    (door,) = world.doors
    assert door.car_floor == floor_index("B3")
    p.x, p.y = OUTSIDE
    events = press(world)
    assert any(isinstance(e, LiftCalled) for e in events)
    run(world, FLOOR_SECONDS * 1.5, {1: IDLE})
    assert floor_index("B3") < door.car_floor < G and door.open == 0.0  # on its way, shut
    run(world, FLOOR_SECONDS * 1.6, {1: IDLE})
    assert door.car_here and door.opening  # here: opening
    run(world, SETTLE, {1: IDLE})
    assert door.open == 1.0


def test_open_it_stays_open_until_shut_from_inside() -> None:
    world, (p,) = lobby()
    (door,) = world.doors
    p.x, p.y = OUTSIDE
    press(world)
    p.x, p.y = 1.5, 1.5  # walk away: no automatic doors
    run(world, 3.0, {1: IDLE})
    assert door.open == 1.0
    p.x, p.y = INSIDE
    press(world, angle=NORTH)  # the car's panel, whichever way you face
    run(world, SETTLE, {1: IDLE})
    assert door.open == 0.0


def test_the_doors_wont_shut_on_someone_in_them() -> None:
    world, (p1, p2) = lobby(players=2)
    (door,) = world.doors
    p1.x, p1.y = OUTSIDE
    press(world)
    run(world, SETTLE, {1: IDLE, 2: IDLE})
    p1.x, p1.y = INSIDE
    p2.x, p2.y = door.centre  # standing in the doors
    press(world, angle=NORTH)
    run(world, SETTLE, {1: IDLE, 2: IDLE})
    assert door.open == 1.0  # opened again


def test_shut_in_with_everyone_aboard_it_rides_up_and_the_floor_is_done() -> None:
    world, (p,) = lobby()
    p.x, p.y = OUTSIDE
    press(world)
    run(world, SETTLE, {1: IDLE})
    p.x, p.y = INSIDE
    events = press(world, angle=NORTH)
    events += run(world, SETTLE, {1: IDLE})
    assert any(isinstance(e, LiftDeparting) for e in events)
    assert not world.complete
    run(world, RIDE_SECONDS / 2, {1: IDLE})
    assert G < world.doors[0].car_floor < G + 1  # climbing: the panel counts
    events = run(world, RIDE_SECONDS / 2 + 0.1, {1: IDLE})
    assert world.complete
    (done,) = [e for e in events if isinstance(e, LevelComplete)]
    assert done.next_floor == "1"


def test_it_waits_for_everyone() -> None:
    world, (p1, p2) = lobby(players=2)
    p1.x, p1.y, p2.x, p2.y = *OUTSIDE, 2.5, 1.5
    press(world)
    run(world, SETTLE, {1: IDLE, 2: IDLE})
    p1.x, p1.y = INSIDE
    press(world, angle=NORTH)
    run(world, SETTLE + 1.0, {1: IDLE, 2: IDLE})
    assert world.doors[0].open == 0.0 and world.ride < 0.0  # shut, but not going
    press(world, angle=NORTH)  # open up again for the other player
    run(world, SETTLE, {1: IDLE, 2: IDLE})
    assert world.doors[0].open == 1.0
    p2.x, p2.y = 4.5, 4.6  # both aboard (one cell: they overlap, which is fine)
    press(world, angle=NORTH)
    run(world, SETTLE + RIDE_SECONDS + 0.2, {1: IDLE, 2: IDLE})
    assert world.complete


def test_a_locked_lift_needs_its_pass_to_call_or_send() -> None:
    world, (p,) = lobby(props=" lock=gold")
    p.x, p.y = OUTSIDE
    events = press(world)
    assert any(isinstance(e, DoorLocked) for e in events)
    assert world.doors[0].open == 0.0
    world.passes.add("gold")
    press(world)
    run(world, SETTLE, {1: IDLE})
    assert world.doors[0].open == 1.0


def test_arriving_the_team_starts_in_the_car_and_it_opens() -> None:
    arrival = LOBBY.replace("coop_start 2.5 2.5 0", "coop_start 4.5 4.5 270")
    world = new_world(parse_map(arrival.format(props=" lock=gold")), "coop")
    spawn_player(world, 1, "p")
    (door,) = world.doors
    assert door.car_here and door.open == 0.0
    run(world, ARRIVE_SECONDS / 2, {1: IDLE})
    assert door.open == 0.0 and world.ride < 0.0  # not straight back up
    run(world, ARRIVE_SECONDS / 2 + SETTLE, {1: IDLE})
    assert door.open == 1.0
    # Locked (an executive lift): no going on up without the gold pass.
    events = press(world, angle=NORTH)
    assert any(isinstance(e, DoorLocked) for e in events) and door.open == 1.0


def test_hud_messages_for_lifts() -> None:
    from ok_doomer.client.hud import message_for

    assert message_for(LiftNoPower(0, 0, 1)) == "The lift has no power"
    assert message_for(LiftCalled(0, 0, 1)) == "The lift's on its way"
    assert message_for(LevelComplete(0, "1")) is None  # the next floor loads instead


def test_a_lift_has_its_fittings_round_it() -> None:
    from ok_doomer.client.boxes import door_boxes
    from ok_doomer.client.render import _powered
    from ok_doomer.shared.doors import LIFT, Door

    lift = Door(0, LIFT, 2.5, 2.5, -math.pi / 2, called=True)
    boxes = door_boxes([lift])
    # Two leaves plus lining, architrave, sill, indicator, call plate and
    # button, and the car's panel: more than the leaves alone.
    assert len(boxes) > 8
    # The indicator, call button and car panel are on the floor's power.
    powered = [p for p in boxes if p.box.mains and p.box.off_faces is not None]
    assert len(powered) == 3
    lit = [q for p in powered if (q := _powered(p, True, 0.0)) is not None]
    dark = [q for p in powered if (q := _powered(p, False, 0.0)) is not None]
    assert sum(p.box.glow for p in lit) == 2  # the G and the called button glow
    assert not any(p.box.glow for p in dark)  # dark without power
    assert all(p.box.faces == p0.box.off_faces for p, p0 in zip(dark, powered, strict=True))


def test_the_cars_handrail_stops_short_of_the_doors() -> None:
    from ok_doomer.client.palette import GREY
    from ok_doomer.client.textures import RAIL_END, lift_car_side

    tex = lift_car_side()
    rail = (tex[39] >> 4 == GREY) & (tex[39] & 15 == 15)  # its lit top edge
    assert not rail[:RAIL_END].any() and not rail[64 - RAIL_END :].any()
    assert rail[RAIL_END : 64 - RAIL_END].all()


def test_ride_from_the_ground_floor_to_floor_1() -> None:
    from ok_doomer.client.session import LocalSession
    from ok_doomer.shared.building import floor_map_path
    from ok_doomer.shared.mapfile import load_map

    session = LocalSession(new_world(load_map(floor_map_path("G")), "coop"), "p")
    world, p = session.world, session.player
    world.flags.add(POWER)  # the breaker's thrown
    p.health = 63
    door = next(d for d in world.doors if d.kind == LIFT)
    out = SOUTH  # into the car
    p.x, p.y = door.x, door.y - 1.1

    def tick(seconds: float, use: bool = False, angle: float = out) -> list[object]:
        events: list[object] = []
        for k in range(math.ceil(seconds * 60)):
            events += session.update(1 / 60, PlayerInput(angle=angle, use=use and k == 0))
        return events

    tick(0.1, use=True)  # call it: up from B3
    tick(3 * FLOOR_SECONDS + SETTLE)
    assert door.open == 1.0
    p.x, p.y = door.x, door.y  # step in
    events = tick(SETTLE + RIDE_SECONDS + 0.2, use=True)  # shut the doors; ride up
    (done,) = [e for e in events if isinstance(e, LevelComplete)]
    assert done.next_floor == "1"
    # The session moves on by itself, the same update the ride ended in.
    assert session.game_map.storey == "1" and session.player.health == 63
    assert not session.world.passes  # a floor's passes are its own
    arrival = next(d for d in session.world.doors if d.car_here and d.arrive_in > 0)
    assert (math.floor(session.player.x), math.floor(session.player.y)) == arrival.car
    tick(ARRIVE_SECONDS + SETTLE)
    assert arrival.open == 1.0  # out you go

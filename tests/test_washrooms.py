"""Washrooms: the washroom area, its fittings, the cubicle doors (and
hiding in a cubicle), and the ground floor's Gents and Ladies."""

import math

import pytest

from ok_doomer.client.areas import AREA_LOOKS
from ok_doomer.client.boxes import BOX_MODELS, Box, cubicle_door_boxes
from ok_doomer.client.hud import message_for
from ok_doomer.shared.cubicles import DOOR_SECONDS, CubicleDoor, cubicle_doors_from_map
from ok_doomer.shared.events import CubicleDoorUsed, Event
from ok_doomer.shared.mapfile import WASHROOM, format_map, load_map, parse_map
from ok_doomer.shared.player import PlayerInput, PlayerState
from ok_doomer.shared.sim import step
from ok_doomer.shared.spawn import spawn_player
from ok_doomer.shared.things import (
    CUBICLE_DEPTH,
    CUBICLE_WIDTH,
    PAN_TOP,
    PARTITION_HALF,
    RETURN_WIDTH,
    THING_TYPES,
)
from ok_doomer.shared.world import World, drain_events, new_world

FITTINGS = [
    f"{fitting}_{n}"
    for fitting in ("cubicles", "urinals", "basins")
    for n in ("two", "three", "four")
] + ["cubicles_five"]

# A washroom 4 cells wide: a run of four cubicles against the north wall.
ROOM = """\
[walls]
######
#....#
#....#
#....#
######

[areas]
......
.wwww.
.wwww.
.wwww.
......

[things]
coop_start 3.0 3.5 270
exit 4.5 3.5 0
dm_spawn 1.5 3.5 0
cubicles_four 3.0 1.5 0 kind=gents at=n
"""


def test_the_washroom_area_has_a_look_and_round_trips() -> None:
    assert WASHROOM in AREA_LOOKS
    m = parse_map("[walls]\n###\n#.#\n###\n\n[areas]\n...\n.w.\n...\n")
    assert m.area_at(1, 1) == WASHROOM
    assert parse_map(format_map(m)) == m


@pytest.mark.parametrize("name", FITTINGS)
def test_a_fittings_footprint_holds_its_model(name: str) -> None:
    fps = THING_TYPES[name].footprints
    boxes = BOX_MODELS[name]
    assert min(b.x[0] for b in boxes) >= min(f.x[0] for f in fps) - 1e-9
    assert max(b.x[1] for b in boxes) <= max(f.x[1] for f in fps) + 0.011  # the headrail
    assert min(b.y[0] for b in boxes) >= min(f.y[0] for f in fps) - 1e-9
    assert max(b.y[1] for b in boxes) <= max(f.y[1] for f in fps) + 1e-9


def test_a_cubicle_fits_a_player() -> None:
    # Between partitions, wider than a player's 0.5; deep enough in front of
    # the toilet's pan to stand with the door shut.
    assert CUBICLE_WIDTH - 2 * PARTITION_HALF > 0.5
    assert CUBICLE_WIDTH - 2 * (PARTITION_HALF + RETURN_WIDTH) > 0.5  # the doorway too
    pan = [f for f in THING_TYPES["cubicles_two"].footprints if f.top == PAN_TOP][0]
    assert CUBICLE_DEPTH - pan.x[1] > 0.5


def test_the_engaged_cubicle_shows_whose_feet() -> None:
    # Under its door: ladies' court shoes have heels, and there's a handbag;
    # the gents' shoes are flat, his trousers round his ankles.
    def low(name: str) -> set[tuple[float, float]]:
        return {b.z for b in BOX_MODELS[name] if b.z[1] <= 0.08}

    assert (0.0, 0.06) in low("cubicles_four:ladies")  # heels
    assert (0.0, 0.08) in low("cubicles_four:ladies")  # the handbag
    assert low("cubicles_four:gents") == {(0.0, 0.04)}  # shoes and trousers
    assert BOX_MODELS["cubicles_four"] == BOX_MODELS["cubicles_four:gents"]  # unless told


def _corners(box: Box) -> list[tuple[float, float]]:
    """A box's footprint corners in its model's frame, `turn` applied."""
    c, s = math.cos(box.turn), math.sin(box.turn)
    return [(x * c - y * s, x * s + y * c) for x in box.x for y in box.y]


@pytest.mark.parametrize("open_", [0.0, 0.5, 1.0])
def test_a_cubicle_door_hangs_from_its_hinge_and_swings_out(open_: float) -> None:
    door = cubicle_doors_from_map(parse_map(ROOM))[0]
    door.open = open_
    (placed,) = cubicle_door_boxes([door])
    corners = _corners(placed.box)
    half = 4 * CUBICLE_WIDTH / 2
    # The first cubicle's +y side, at the edge of its return.
    hinge = (CUBICLE_DEPTH, half - PARTITION_HALF - RETURN_WIDTH)
    assert min(math.dist(c, hinge) for c in corners) < 0.012
    if open_ == 0.0:
        assert max(c[0] for c in corners) <= CUBICLE_DEPTH + 1e-6  # across the front
    if open_ == 1.0:
        assert max(c[0] for c in corners) > CUBICLE_DEPTH + 0.5  # out into the room


def test_the_first_door_starts_open_the_second_engaged() -> None:
    doors = cubicle_doors_from_map(parse_map(ROOM))
    assert [d.open for d in doors] == [1.0, 0.0, 0.0, 0.0]
    assert [d.locked for d in doors] == [False, True, False, False]


def _ticks(world: World, inputs: PlayerInput, seconds: float) -> None:
    for _ in range(round(seconds / world.config.dt) + 2):
        step(world, {1: inputs})


def _in_front(world: World, door: CubicleDoor, out: float) -> PlayerState:
    """Player 1, `out` in front of a door's middle (negative: inside the
    cubicle), facing the door."""
    player = world.players.get(1) or spawn_player(world, 1, "p")
    x, y = door.to_world(CUBICLE_DEPTH + out, door.hinge_y - door.width / 2)
    player.x, player.y, player.z = x, y, 0.0
    player.angle = door.run_angle + (math.pi if out > 0 else 0.0)
    return player


def _use(world: World, player: PlayerState) -> list[Event]:
    step(world, {1: PlayerInput(angle=player.angle, use=True)})
    events = drain_events(world)
    _ticks(world, PlayerInput(angle=player.angle), DOOR_SECONDS)
    return events


def test_use_opens_a_vacant_cubicle_door_and_shuts_it_again() -> None:
    world = new_world(parse_map(ROOM), "coop")
    door = world.cubicle_doors[2]
    player = _in_front(world, door, 0.75)
    events = _use(world, player)
    assert door.open == 1.0
    used = [e for e in events if isinstance(e, CubicleDoorUsed)]
    assert used and used[0].opening and not used[0].engaged
    _use(world, player)  # standing where it would swing: it waits
    assert door.open == 1.0 and not door.opening
    _in_front(world, door, 0.88)  # step back (still in reach): it shuts
    _ticks(world, PlayerInput(angle=player.angle), DOOR_SECONDS)
    assert door.open == 0.0


def test_an_engaged_door_stays_shut_and_says_so() -> None:
    world = new_world(parse_map(ROOM), "coop")
    door = world.cubicle_doors[1]
    events = _use(world, _in_front(world, door, 0.75))
    assert door.open == 0.0
    (used,) = [e for e in events if isinstance(e, CubicleDoorUsed)]
    assert used.engaged
    assert message_for(used) == "Engaged: someone's in there"


def test_a_door_wont_shut_on_someone_in_the_doorway() -> None:
    world = new_world(parse_map(ROOM), "coop")
    door = world.cubicle_doors[0]  # open
    player = _in_front(world, door, 0.1)  # straddling the front
    _use(world, player)
    assert door.open == 1.0 and not door.opening  # told to shut; waiting


def test_hide_in_a_cubicle_shut_the_door_and_stand_on_the_loo() -> None:
    world = new_world(parse_map(ROOM), "coop")
    door = world.cubicle_doors[0]  # open
    player = _in_front(world, door, -0.27)  # inside, in front of the pan
    player.angle = door.run_angle  # facing out, at the door
    _use(world, player)
    assert door.open == 0.0  # shut, with them inside
    # Turn round and jump onto the toilet (its lid's above a step).
    into = door.run_angle + math.pi
    step(world, {1: PlayerInput(forward=1.0, angle=into, jump=True)})
    _ticks(world, PlayerInput(forward=1.0, angle=into), 0.6)
    assert player.z == pytest.approx(PAN_TOP)
    # And they can't walk out through the shut door.
    _ticks(world, PlayerInput(forward=1.0, angle=door.run_angle), 1.0)
    x, y = door.centre()
    assert math.hypot(player.x - x, player.y - y) < 0.6


def test_a_toilet_is_too_high_to_step_onto() -> None:
    world = new_world(parse_map(ROOM), "coop")
    door = world.cubicle_doors[0]
    player = _in_front(world, door, -0.27)
    _ticks(world, PlayerInput(forward=1.0, angle=door.run_angle + math.pi), 0.6)
    assert player.z == 0.0  # walked up to it, stopped


def test_the_ground_floor_has_gents_and_ladies_off_the_corridor() -> None:
    office = load_map("maps/office.map")
    plates = {p.prop("kind"): p for p in office.things_of_type("poster")}
    assert {"gents", "ladies", "wash_hands"} <= set(plates)
    rooms = {}
    for kind in ("gents", "ladies"):
        plate = plates[kind]
        # On the corridor's north wall, beside a door into a washroom.
        door = min(
            office.things_of_type("door_hinged"),
            key=lambda d: math.hypot(d.x - plate.x, d.y - plate.y),
        )
        assert math.hypot(door.x - plate.x, door.y - plate.y) < 1.5
        assert office.area_at(math.floor(door.x), math.floor(door.y) - 1) == WASHROOM
        rooms[kind] = math.floor(door.x) - 1, math.floor(door.x) + 1  # its x range

    def count(prefix: str, kind: str, per: dict[str, int]) -> int:
        x0, x1 = rooms[kind]
        return sum(
            per[t.type.rsplit("_", 1)[1]]
            for t in office.things
            if t.type.startswith(prefix) and x0 <= t.x < x1 + 1
        )

    sizes = {"two": 2, "three": 3, "four": 4, "five": 5}
    assert count("cubicles_", "gents", sizes) == 5
    assert count("cubicles_", "ladies", sizes) == 5
    assert count("urinals_", "gents", sizes) == 6
    assert count("basins_", "gents", sizes) == count("basins_", "ladies", sizes) == 4
    dryers = office.things_of_type("hand_dryer")
    for kind in ("gents", "ladies"):
        x0, x1 = rooms[kind]
        assert sum(x0 <= d.x < x1 + 1 for d in dryers) == 4


def test_loo_rolls_are_drawn_but_never_in_the_way() -> None:
    rolls = [b for b in BOX_MODELS["cubicles_five"] if b.z == (0.25, 0.32)]
    assert len(rolls) == 5  # one a cubicle
    for roll in rolls:
        for f in THING_TYPES["cubicles_five"].footprints:
            inside_x = f.x[0] < roll.x[1] and roll.x[0] < f.x[1]
            inside_y = f.y[0] < roll.y[1] - 1e-9 and roll.y[0] + 1e-9 < f.y[1]
            assert not (inside_x and inside_y)

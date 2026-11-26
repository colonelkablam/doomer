"""Washrooms: the washroom area, its fittings, the cubicle doors, and the
ground floor's Gents and Ladies."""

import math

import pytest

from doomer.client.areas import AREA_LOOKS
from doomer.client.boxes import BOX_MODELS, Box, cubicle_door_boxes
from doomer.client.hud import message_for
from doomer.shared.cubicles import DOOR_SECONDS, cubicle_doors_from_map
from doomer.shared.events import CubicleDoorUsed, Event
from doomer.shared.mapfile import WASHROOM, format_map, load_map, parse_map
from doomer.shared.player import PlayerInput
from doomer.shared.sim import step
from doomer.shared.things import CUBICLE_DEPTH, CUBICLE_WIDTH, THING_TYPES
from doomer.shared.world import drain_events, new_world, spawn_player

FITTINGS = [
    f"{fitting}_{n}"
    for fitting in ("cubicles", "urinals", "basins")
    for n in ("two", "three", "four")
]

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
coop_start 2.5 3.5 270
exit 4.5 3.5 0
dm_spawn 1.5 3.5 0
cubicles_four 2.5 1.5 0 kind=gents at=n
"""


def test_the_washroom_area_has_a_look_and_round_trips() -> None:
    assert WASHROOM in AREA_LOOKS
    m = parse_map("[walls]\n###\n#.#\n###\n\n[areas]\n...\n.w.\n...\n")
    assert m.area_at(1, 1) == WASHROOM
    assert parse_map(format_map(m)) == m


@pytest.mark.parametrize("name", FITTINGS)
def test_a_fittings_footprint_holds_its_model(name: str) -> None:
    (fp,) = THING_TYPES[name].footprints
    boxes = BOX_MODELS[name]
    assert min(b.x[0] for b in boxes) >= fp.x[0] - 1e-9
    assert max(b.x[1] for b in boxes) <= fp.x[1] + 0.011  # the headrail's lip
    assert min(b.y[0] for b in boxes) >= fp.y[0] - 1e-9
    assert max(b.y[1] for b in boxes) <= fp.y[1] + 1e-9


def test_the_engaged_cubicle_shows_whose_feet() -> None:
    # Under its door: ladies' court shoes have heels, and there's a handbag;
    # the gents' shoes are flat, his trousers round his ankles.
    def low(name: str) -> set[tuple[float, float]]:
        return {b.z for b in BOX_MODELS[name] if b.z[1] <= 0.08}

    assert (0.0, 0.05) in low("cubicles_four:ladies")  # heels
    assert (0.0, 0.07) in low("cubicles_four:ladies")  # the handbag
    assert low("cubicles_four:gents") == {(0.0, 0.035)}  # shoes and trousers
    assert BOX_MODELS["cubicles_four"] == BOX_MODELS["cubicles_four:gents"]  # unless told


def _corners(box: Box) -> list[tuple[float, float]]:
    """A box's footprint corners in its model's frame, `turn` applied."""
    c, s = math.cos(box.turn), math.sin(box.turn)
    return [(x * c - y * s, x * s + y * c) for x in box.x for y in box.y]


@pytest.mark.parametrize("open_", [0.0, 0.5, 1.0])
def test_a_cubicle_door_hangs_from_its_hinge_and_swings_in(open_: float) -> None:
    (door, *_) = cubicle_doors_from_map(parse_map(ROOM))
    door.open = open_
    (placed,) = cubicle_door_boxes([door])
    corners = _corners(placed.box)
    half = 4 * CUBICLE_WIDTH / 2
    hinge = (CUBICLE_DEPTH, half - CUBICLE_WIDTH)
    assert min(math.dist(c, hinge) for c in corners) < 0.02
    assert max(c[0] for c in corners) <= CUBICLE_DEPTH + 1e-6  # never out into the room
    if open_ == 1.0:
        assert min(c[0] for c in corners) < CUBICLE_DEPTH - 0.15  # well in


def test_the_first_door_starts_open_the_second_engaged() -> None:
    doors = cubicle_doors_from_map(parse_map(ROOM))
    assert [d.open for d in doors] == [1.0, 0.0, 0.0, 0.0]
    assert [d.locked for d in doors] == [False, True, False, False]


def _use_door(world_map: str, index: int) -> tuple[list[float], list[Event]]:
    world = new_world(parse_map(world_map), "coop")
    player = spawn_player(world, 1, "p")
    door = world.cubicle_doors[index]
    cx, cy = door.centre()
    player.x, player.y = cx, cy + 0.5  # in front of it, facing it (north)
    face = PlayerInput(angle=3 * math.pi / 2)
    step(world, {1: PlayerInput(angle=face.angle, use=True)})
    events = drain_events(world)
    for _ in range(round(DOOR_SECONDS / world.config.dt) + 2):
        step(world, {1: face})
    return [d.open for d in world.cubicle_doors], events


def test_use_opens_a_vacant_cubicle_door() -> None:
    opens, events = _use_door(ROOM, 2)
    assert opens[2] == 1.0
    used = [e for e in events if isinstance(e, CubicleDoorUsed)]
    assert used and used[0].opening and not used[0].engaged


def test_an_engaged_door_stays_shut_and_says_so() -> None:
    opens, events = _use_door(ROOM, 1)
    assert opens[1] == 0.0
    (used,) = [e for e in events if isinstance(e, CubicleDoorUsed)]
    assert used.engaged
    assert message_for(used) == "Engaged: someone's in there"


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

    sizes = {"two": 2, "three": 3, "four": 4}
    assert count("cubicles_", "gents", sizes) == 4
    assert count("cubicles_", "ladies", sizes) == 8
    assert count("urinals_", "gents", sizes) == 8
    assert count("basins_", "gents", sizes) == count("basins_", "ladies", sizes) == 4
    dryers = office.things_of_type("hand_dryer")
    for kind in ("gents", "ladies"):
        x0, x1 = rooms[kind]
        assert sum(x0 <= d.x < x1 + 1 for d in dryers) == 4

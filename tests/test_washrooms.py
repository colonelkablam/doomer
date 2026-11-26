"""Washrooms: the washroom area, its fittings, and the ground floor's two."""

import math

import pytest

from doomer.client.areas import AREA_LOOKS
from doomer.client.boxes import BOX_MODELS, Box
from doomer.shared.mapfile import WASHROOM, format_map, load_map, parse_map
from doomer.shared.things import (
    CUBICLE_DEPTH,
    CUBICLE_WIDTH,
    THING_TYPES,
)

FITTINGS = [
    "cubicles_two",
    "cubicles_three",
    "cubicles_four",
    "urinals_two",
    "urinals_three",
    "basins_two",
    "basins_three",
]


def test_the_washroom_area_has_a_look_and_round_trips() -> None:
    assert WASHROOM in AREA_LOOKS
    m = parse_map("[walls]\n###\n#.#\n###\n\n[areas]\n...\n.w.\n...\n")
    assert m.area_at(1, 1) == WASHROOM
    assert parse_map(format_map(m)) == m


@pytest.mark.parametrize("name", FITTINGS)
def test_a_fittings_footprint_holds_its_model(name: str) -> None:
    fp = THING_TYPES[name].footprints[0]  # (cubicles: then their open door's)
    boxes = BOX_MODELS[name]
    straight = [b for b in boxes if b.turn == 0.0]
    assert min(b.x[0] for b in straight) >= fp.x[0] - 1e-9
    assert max(b.x[1] for b in straight) <= fp.x[1] + 0.011  # the headrail's lip
    assert min(b.y[0] for b in straight) >= fp.y[0] - 1e-9
    assert max(b.y[1] for b in straight) <= fp.y[1] + 1e-9


def _corners(box: Box) -> list[tuple[float, float]]:
    """A box's footprint corners in its model's frame, `turn` applied."""
    c, s = math.cos(box.turn), math.sin(box.turn)
    return [(x * c - y * s, x * s + y * c) for x in box.x for y in box.y]


def test_the_ajar_cubicle_door_hangs_from_its_hinge() -> None:
    # The first cubicle's door, turned out about the hinge at the front of
    # the partition between the first and second cubicles.
    (door,) = [b for b in BOX_MODELS["cubicles_three"] if b.turn != 0.0]
    corners = _corners(door)
    half = 3 * CUBICLE_WIDTH / 2
    hinge = (CUBICLE_DEPTH, half - CUBICLE_WIDTH)
    assert min(math.dist(c, hinge) for c in corners) < 0.02
    assert max(c[0] for c in corners) > CUBICLE_DEPTH + 0.15  # swung well out into the room
    # ... and it blocks you where it stands.
    _, swung = THING_TYPES["cubicles_three"].footprints
    for x, y in corners:
        assert swung.x[0] - 0.02 <= x <= swung.x[1] + 0.02
        assert swung.y[0] - 0.02 <= y <= swung.y[1] + 0.02


def test_the_ground_floor_has_gents_and_ladies_off_the_corridor() -> None:
    office = load_map("maps/office.map")
    plates = {p.prop("kind"): p for p in office.things_of_type("poster")}
    assert {"gents", "ladies", "wash_hands"} <= set(plates)
    for kind in ("gents", "ladies"):
        plate = plates[kind]
        # On the corridor's north wall, beside a door into a washroom.
        door = min(
            office.things_of_type("door_hinged"),
            key=lambda d: math.hypot(d.x - plate.x, d.y - plate.y),
        )
        assert math.hypot(door.x - plate.x, door.y - plate.y) < 1.5
        assert office.area_at(math.floor(door.x), math.floor(door.y) - 1) == WASHROOM
    for name in ("cubicles_two", "cubicles_four", "urinals_three", "basins_two", "basins_three"):
        (t,) = office.things_of_type(name)
        assert office.area_at(math.floor(t.x), math.floor(t.y)) == WASHROOM

"""Placing things against walls with `at=` (doomer/shared/placement.py)."""

import math

import pytest

from doomer.shared.furniture import placed_footprints
from doomer.shared.mapfile import MapError, Thing, format_map, load_map, parse_map

ROOM = "[walls]\n#####\n#...#\n#...#\n#...#\n#####\n\n[things]\n"


def _thing(line: str) -> Thing:
    return parse_map(ROOM + line + "\n").things[0]


def test_a_corner_thing_sits_just_off_both_walls() -> None:
    t = _thing("filing_cabinet 1.5 1.5 0 at=nw")
    assert t.x == pytest.approx(1 + 0.01 + 0.16) and t.y == pytest.approx(1 + 0.01 + 0.16)
    assert t.angle == 0.0  # a corner keeps its written angle (filing cabinets aren't jittered)


def test_a_wall_thing_faces_out_of_the_wall_and_keeps_its_place_along_it() -> None:
    t = _thing("server 3.5 2.3 0 at=e")
    assert t.angle == pytest.approx(math.pi)
    assert t.y == 2.3
    (fp,) = placed_footprints(t)
    assert fp.x1 == pytest.approx(3.99) and fp.y0 >= 2.0


@pytest.mark.parametrize("at", ["n", "e", "s", "w"])
def test_a_turned_footprint_still_fits_against_each_wall(at: str) -> None:
    cx, cy = {"n": (2, 1), "e": (3, 2), "s": (2, 3), "w": (1, 2)}[at]
    t = _thing(f"breaker {cx + 0.5} {cy + 0.5} 0 at={at}")
    fps = placed_footprints(t)
    x0, x1 = min(p.x0 for p in fps), max(p.x1 for p in fps)
    y0, y1 = min(p.y0 for p in fps), max(p.y1 for p in fps)
    face = {"w": x0 - cx, "e": cx + 1 - x1, "n": y0 - cy, "s": cy + 1 - y1}[at]
    assert face == pytest.approx(0.01)


def test_flat_things_go_flat_on_the_wall() -> None:
    t = _thing("poster 2.4 1.5 0 kind=plan at=n")
    assert t.x == 2.4 and t.y == pytest.approx(1.01)
    assert t.angle == pytest.approx(math.pi / 2)


def test_plants_in_corners_are_turned_a_little_and_the_same_each_time() -> None:
    a, b = _thing("ficus 1.5 3.5 0 at=sw"), _thing("ficus 1.5 3.5 0 at=sw")
    assert a.angle == b.angle != 0.0
    assert abs(a.angle) <= math.radians(25)


def test_no_wall_there_is_an_error() -> None:
    with pytest.raises(MapError, match="no wall to the east of cell"):
        _thing("filing_cabinet 2.5 2.5 0 at=e")
    with pytest.raises(MapError, match="expected one of"):
        _thing("filing_cabinet 1.5 1.5 0 at=up")


def test_doors_are_not_placed() -> None:
    with pytest.raises(MapError, match="has no property 'at'"):
        _thing("door_hinged 1.5 1.5 0 at=n")


def test_the_file_keeps_what_was_written() -> None:
    m = parse_map(ROOM + "filing_cabinet 1.5 1.5 0 at=nw\nficus 3.5 3.5 10 at=se\n")
    assert parse_map(format_map(m)) == m
    assert "filing_cabinet 1.5 1.5 0 at=nw" in format_map(m)
    office = load_map("maps/office.map")
    assert parse_map(format_map(office)) == office

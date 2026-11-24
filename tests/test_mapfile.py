import math
import re
from pathlib import Path

import pytest

from doomer.shared.mapfile import (
    DEFAULT_CEILING,
    DEFAULT_FLOOR,
    MapError,
    format_map,
    load_map,
    parse_map,
)
from doomer.shared.modes import missing_requirements, supported_modes

MAPS = Path(__file__).resolve().parent.parent / "maps"

SMALL = """\
name: Small
# comment before the grid

[walls]
#####
#..2#
#...#
#####

[things]
# type x y angle
coop_start 1.5 1.5 90
exit 3.5 2.5 0
"""


def test_parses_header_grid_and_things() -> None:
    m = parse_map(SMALL)
    assert m.name == "Small"
    assert (m.width, m.height) == (5, 4)
    assert m.walls[1] == (1, 0, 0, 2, 1)  # '#' is an alias for wall type 1
    assert m.things[0].type == "coop_start"
    assert m.things[0].angle == pytest.approx(math.pi / 2)


def test_floor_and_ceiling_default_when_absent() -> None:
    m = parse_map(SMALL)
    assert all(c == DEFAULT_FLOOR for row in m.floor for c in row)
    assert all(c == DEFAULT_CEILING for row in m.ceiling for c in row)


def test_ragged_rows_are_padded_with_walls_and_outside_is_solid() -> None:
    m = parse_map("[walls]\n###\n#.\n###\n")
    assert m.walls[1] == (1, 0, 1)
    assert m.is_solid(-1, 0) and m.is_solid(0, 99)
    assert not m.is_solid(1, 1)


def test_partial_height_grid_is_padded_with_default() -> None:
    m = parse_map("[walls]\n###\n#.#\n###\n\n[floor]\n000\n020\n")
    assert m.floor[1] == (0, 2, 0)
    assert m.floor[2] == (DEFAULT_FLOOR,) * 3


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ("name: x\n", "missing [walls]"),
        ("[walls]\n#x#\n", "bad character 'x'"),
        ("[walls]\n###\n\n###\n", "blank line inside [walls]"),
        ("[walls]\n###\n#.#\n###\n[things]\nbarrel 1.5 1.5 0\n", "unknown thing type"),
        ("[walls]\n###\n#.#\n###\n[things]\nexit 0.5 0.5 0\n", "inside a wall"),
        ("[walls]\n###\n#.#\n###\n[things]\nexit 9 9 0\n", "outside the map"),
        ("[walls]\n###\n#.#\n###\n[things]\nexit 1.5 1.5\n", "expected 'type x y angle'"),
        ("[walls]\n#\n[walls]\n#\n", "duplicate section"),
        ("[sprites]\n", "unknown section"),
        ("hello\n", "expected 'name: ...'"),
    ],
)
def test_errors(text: str, message: str) -> None:
    with pytest.raises(MapError, match=re.escape(message)):
        parse_map(text)


def test_error_includes_line_number() -> None:
    with pytest.raises(MapError, match="line 3"):
        parse_map("[walls]\n###\n#?#\n")


def test_round_trip() -> None:
    original = parse_map(SMALL + "\n[floor]\n00000\n01200\n")
    assert parse_map(format_map(original)) == original


def test_round_trip_test_map() -> None:
    m = load_map(MAPS / "test.map")
    assert parse_map(format_map(m)) == m


def test_mode_readiness() -> None:
    m = parse_map(SMALL)
    assert supported_modes(m) == ["coop"]
    assert missing_requirements(m, "deathmatch") == ["needs 4 more DM spawns"]
    bare = parse_map("[walls]\n###\n#.#\n###\n[things]\ndm_spawn 1.5 1.5 0\n")
    assert missing_requirements(bare, "coop") == ["needs a coop_start", "needs an exit"]
    assert missing_requirements(bare, "deathmatch") == ["needs 3 more DM spawns"]


def test_shipped_maps_load_and_support_both_modes() -> None:
    for path in MAPS.glob("*.map"):
        assert set(supported_modes(load_map(path))) == {"coop", "deathmatch"}, path.name

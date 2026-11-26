"""Load and save the plain-text map format described in docs/DESIGN.md.

A GameMap is static data: it never changes during play. Anything that changes
(doors opening, pickups taken) belongs in the World, not here.
"""

import math
import re
from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path

from doomer.shared.placement import ANCHORS, PlacementError, place
from doomer.shared.things import THING_TYPES

EMPTY = 0
DEFAULT_WALL = 1
# Compass directions, clockwise, as used by [facing] and by ray hits' faces.
NORTH, EAST, SOUTH, WEST = range(4)
DEFAULT_FACING = SOUTH
_FACING_CHARS = "nesw"
# Areas: what kind of space each cell is, for its look (floor, ceiling, and
# the plain walls facing into it): open-plan office, executive suite,
# entrance lobby, service (plant rooms, back corridors), inside a lift car.
# [areas] letters.
OFFICE, EXECUTIVE, LOBBY, SERVICE, LIFT_CAR = range(5)
DEFAULT_AREA = OFFICE
AREA_CHARS = "oelsc"

# Wall types with built-in shape (the rest are just textures, chosen by the
# client): a stair cell's floor slopes up across the cell, towards the way
# its [facing] points, by STAIR_RISE units; its [floor] value is the height at
# the top edge. Two stair cells climb a full storey.
STAIRS_WALL_TYPE = 17  # 'h'
STAIR_RISE = 4
# A balcony floor: a raised floor with a railing along every edge that drops
# to a lower floor (not where stairs or another raised cell meet it).
RAILING_WALL_TYPE = 18  # 'i'
RAIL_HEIGHT = 0.4  # walls (~1.1 m)

# Heights are in eighths of a wall (~34 cm with a 2.7 m office ceiling).
WALL_UNITS = 8
DEFAULT_FLOOR = 0
DEFAULT_CEILING = WALL_UNITS

_SECTION = re.compile(r"^\[(\w+)\]\s*$")
_GRID_SECTIONS = ("walls", "floor", "ceiling", "facing", "areas")
_KNOWN_SECTIONS = (*_GRID_SECTIONS, "things")
_KNOWN_HEADERS = ("name",)

Grid = tuple[tuple[int, ...], ...]


class MapError(ValueError):
    """A map file is malformed. The message includes the line number."""


@dataclass(frozen=True, slots=True)
class Thing:
    type: str
    x: float
    y: float
    angle: float  # radians
    # `key=value` properties after the angle (e.g. a door's lock), in file order.
    props: tuple[tuple[str, str], ...] = ()
    # With `at=` (doomer/shared/placement.py), the x, y and angle as written
    # (radians), which format_map writes back; x, y and angle are where it went.
    written: tuple[float, float, float] | None = None

    def prop(self, key: str) -> str | None:
        return next((v for k, v in self.props if k == key), None)


@dataclass(frozen=True, slots=True)
class GameMap:
    name: str
    width: int
    height: int
    walls: Grid  # walls[y][x]: 0 = empty, 1-35 = wall type
    floor: Grid
    ceiling: Grid
    things: tuple[Thing, ...]
    # Which way each cell's front faces (desks): NORTH/EAST/SOUTH/WEST = 0-3.
    facing: Grid = ()
    # Each cell's area (OFFICE, EXECUTIVE, LOBBY, SERVICE, LIFT_CAR).
    areas: Grid = ()

    def in_bounds(self, x: int, y: int) -> bool:
        return 0 <= x < self.width and 0 <= y < self.height

    def wall_at(self, x: int, y: int) -> int:
        """Wall type at a cell. Outside the map counts as solid."""
        if not self.in_bounds(x, y):
            return DEFAULT_WALL
        return self.walls[y][x]

    def is_solid(self, x: int, y: int) -> bool:
        """A wall cell: a full wall or a low block (which you can only get onto
        from above). Height-aware movement uses floor_height instead."""
        return self.wall_at(x, y) != EMPTY

    def is_full_wall(self, x: int, y: int) -> bool:
        return self.is_solid(x, y) and not self.is_low_block(x, y)

    def floor_height(self, x: int, y: int) -> float:
        """Height of the surface you'd stand on in a cell, in walls (1 = one
        storey, 2.7 m). Empty cells: their [floor] (0 unless raised). Low blocks
        (desks, counters, raised floors): their top. Full walls: infinite."""
        if not self.in_bounds(x, y):
            return math.inf
        units = self.floor[y][x]
        if self.walls[y][x] == EMPTY:
            return units / WALL_UNITS
        return units / WALL_UNITS if 0 < units < self.ceiling[y][x] else math.inf

    def is_stairs(self, x: int, y: int) -> bool:
        return self.is_low_block(x, y) and self.walls[y][x] == STAIRS_WALL_TYPE

    def floor_at(self, x: float, y: float) -> float:
        """Floor height at a point, in walls: the same as floor_height for its
        cell, except on stairs, where it slopes across the cell."""
        cx, cy = math.floor(x), math.floor(y)
        if not self.is_stairs(cx, cy):
            return self.floor_height(cx, cy)
        fx, fy = x - cx, y - cy
        progress = (1.0 - fy, fx, fy, 1.0 - fx)[self.facing_at(cx, cy)]  # 0 low .. 1 high
        top = self.floor[cy][cx]
        low = max(0, top - STAIR_RISE)
        return (low + (top - low) * progress) / WALL_UNITS

    def ceiling_height(self, x: int, y: int) -> float:
        """Height of a cell's ceiling, in walls. Default 1 (one storey)."""
        if not self.in_bounds(x, y):
            return 1.0
        return self.ceiling[y][x] / WALL_UNITS

    def top_height(self, x: int, y: int) -> float:
        """Height of a cell's top for rendering: its floor height, but 1 for a
        full wall (which reaches the default ceiling)."""
        return min(self.floor_height(x, y), 1.0)

    def facing_at(self, x: int, y: int) -> int:
        """Which way a cell's front faces (NORTH/EAST/SOUTH/WEST)."""
        if not self.facing or not self.in_bounds(x, y):
            return DEFAULT_FACING
        return self.facing[y][x]

    def area_at(self, x: int, y: int) -> int:
        """A cell's area (OFFICE outside the map or with no [areas])."""
        if not self.areas or not self.in_bounds(x, y):
            return DEFAULT_AREA
        return self.areas[y][x]

    def is_low_block(self, x: int, y: int) -> bool:
        """A wall cell raised only part way: [floor] between 1 and its ceiling."""
        if not self.in_bounds(x, y) or self.walls[y][x] == EMPTY:
            return False
        return 0 < self.floor[y][x] < self.ceiling[y][x]

    def things_of_type(self, type_name: str) -> list[Thing]:
        return [t for t in self.things if t.type == type_name]


def load_map(path: str | Path) -> GameMap:
    return parse_map(Path(path).read_text(encoding="utf-8"))


def parse_map(text: str) -> GameMap:
    headers: dict[str, str] = {}
    sections: dict[str, list[tuple[int, str]]] = {}
    current: str | None = None

    for lineno, raw in enumerate(text.splitlines(), start=1):
        line = raw.rstrip()
        match = _SECTION.match(line)
        if match:
            current = match.group(1)
            if current not in _KNOWN_SECTIONS:
                raise MapError(f"line {lineno}: unknown section [{current}]")
            if current in sections:
                raise MapError(f"line {lineno}: duplicate section [{current}]")
            sections[current] = []
            continue
        if current in _GRID_SECTIONS:
            # '#' is a wall here, so no comments inside grids.
            sections[current].append((lineno, line))
            continue
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if current is None:
            key, sep, value = stripped.partition(":")
            if not sep or key.strip() not in _KNOWN_HEADERS:
                raise MapError(f"line {lineno}: expected 'name: ...' or a [section]")
            headers[key.strip()] = value.strip()
        else:
            sections[current].append((lineno, stripped))

    if "walls" not in sections:
        raise MapError("missing [walls] section")

    wall_rows = _trim_blank(sections["walls"])
    if not wall_rows:
        raise MapError("[walls] section is empty")
    height = len(wall_rows)
    width = max(len(row) for _, row in wall_rows)

    walls = _parse_grid(wall_rows, width, height, _wall_cell, DEFAULT_WALL, "walls")
    floor = _optional_grid(sections, "floor", width, height, DEFAULT_FLOOR)
    ceiling = _optional_grid(sections, "ceiling", width, height, DEFAULT_CEILING)
    facing = _parse_grid(
        _trim_blank(sections.get("facing", [])),
        width,
        height,
        _facing_cell,
        DEFAULT_FACING,
        "facing",
    )
    areas = _parse_grid(
        _trim_blank(sections.get("areas", [])), width, height, _area_cell, DEFAULT_AREA, "areas"
    )

    game_map = GameMap(
        name=headers.get("name", "Untitled"),
        width=width,
        height=height,
        walls=walls,
        floor=floor,
        ceiling=ceiling,
        things=(),
        facing=facing,
        areas=areas,
    )
    _check_heights(game_map)
    things = tuple(_parse_thing(n, line, game_map) for n, line in sections.get("things", []))
    return replace(game_map, things=things)


def _check_heights(m: GameMap) -> None:
    for y in range(m.height):
        for x in range(m.width):
            floor, ceiling = m.floor[y][x], m.ceiling[y][x]
            if m.walls[y][x] == EMPTY:
                if floor:
                    raise MapError(
                        f"cell ({x}, {y}): a raised floor needs a wall type in [walls] "
                        "(it textures the sides)"
                    )
                if ceiling <= floor:
                    raise MapError(f"cell ({x}, {y}): ceiling must be above the floor")


def format_map(game_map: GameMap) -> str:
    """Serialise a map. parse_map(format_map(m)) == m."""
    out = [f"name: {game_map.name}", "", "[walls]"]
    out += ["".join(wall_char(c) for c in row) for row in game_map.walls]
    for section, grid, default in (
        ("floor", game_map.floor, DEFAULT_FLOOR),
        ("ceiling", game_map.ceiling, DEFAULT_CEILING),
    ):
        if any(c != default for row in grid for c in row):
            out += ["", f"[{section}]"]
            out += ["".join(height_char(c) for c in row) for row in grid]
    if any(c != DEFAULT_FACING for row in game_map.facing for c in row):
        out += ["", "[facing]"]
        out += ["".join(_FACING_CHARS[c] for c in row) for row in game_map.facing]
    if any(c != DEFAULT_AREA for row in game_map.areas for c in row):
        out += ["", "[areas]"]
        out += ["".join(AREA_CHARS[c] for c in row) for row in game_map.areas]
    if game_map.things:
        out += ["", "[things]", "# type x y angle [key=value ...]"]
        for t in game_map.things:
            x, y, angle = t.written or (t.x, t.y, t.angle)
            deg = round(math.degrees(angle), 6) % 360
            props = "".join(f" {k}={v}" for k, v in t.props)
            out.append(f"{t.type} {x:g} {y:g} {deg:g}{props}")
    return "\n".join(out) + "\n"


def _trim_blank(rows: list[tuple[int, str]]) -> list[tuple[int, str]]:
    start, end = 0, len(rows)
    while start < end and not rows[start][1].strip():
        start += 1
    while end > start and not rows[end - 1][1].strip():
        end -= 1
    return rows[start:end]


_LETTERS = "abcdefghijklmnopqrstuvwxyz"
MAX_WALL_TYPE = 9 + len(_LETTERS)  # 1-9 as digits, then a-z = 10-35


def wall_char(wall_type: int) -> str:
    """The [walls] character for a wall type: '.', '1'-'9', or 'a'-'z' (10-35)."""
    if wall_type == EMPTY:
        return "."
    if 1 <= wall_type <= 9:
        return str(wall_type)
    if 10 <= wall_type <= MAX_WALL_TYPE:
        return _LETTERS[wall_type - 10]
    raise ValueError(f"wall type {wall_type} has no map character")


def _wall_cell(ch: str) -> int | None:
    if ch == ".":
        return EMPTY
    if ch == "#":
        return DEFAULT_WALL
    if ch in "123456789":
        return int(ch)
    if ch in _LETTERS:
        return 10 + _LETTERS.index(ch)
    return None


def _facing_cell(ch: str) -> int | None:
    if ch == ".":
        return DEFAULT_FACING
    return _FACING_CHARS.index(ch) if ch in _FACING_CHARS else None


def _area_cell(ch: str) -> int | None:
    if ch == ".":
        return DEFAULT_AREA
    return AREA_CHARS.index(ch) if ch in AREA_CHARS else None


def height_char(units: int) -> str:
    """The [floor]/[ceiling] character for a height: '0'-'9', then 'a'-'z' (10-35)."""
    if 0 <= units <= 9:
        return str(units)
    if 10 <= units <= 35:
        return _LETTERS[units - 10]
    raise ValueError(f"height {units} has no map character")


def _height_cell(ch: str) -> int | None:
    if ch in "0123456789":
        return int(ch)
    if ch in _LETTERS:
        return 10 + _LETTERS.index(ch)
    return None


def _parse_grid(
    rows: list[tuple[int, str]],
    width: int,
    height: int,
    parse: Callable[[str], int | None],
    pad: int,
    section: str,
) -> Grid:
    if len(rows) > height:
        raise MapError(f"line {rows[height][0]}: [{section}] has more rows than [walls]")
    grid: list[tuple[int, ...]] = []
    for lineno, row in rows:
        if not row.strip():
            raise MapError(f"line {lineno}: blank line inside [{section}]")
        if len(row) > width:
            raise MapError(f"line {lineno}: [{section}] row is wider than [walls]")
        values: list[int] = []
        for col, ch in enumerate(row):
            value = parse(ch)
            if value is None:
                raise MapError(f"line {lineno}, column {col + 1}: bad character {ch!r}")
            values.append(value)
        values += [pad] * (width - len(values))
        grid.append(tuple(values))
    grid += [tuple([pad] * width)] * (height - len(grid))
    return tuple(grid)


def _optional_grid(
    sections: dict[str, list[tuple[int, str]]], name: str, width: int, height: int, default: int
) -> Grid:
    rows = _trim_blank(sections.get(name, []))
    return _parse_grid(rows, width, height, _height_cell, default, name)


def _parse_thing(lineno: int, line: str, game_map: GameMap) -> Thing:
    parts = line.split()
    if len(parts) < 4:
        raise MapError(f"line {lineno}: expected 'type x y angle [key=value ...]', got {line!r}")
    type_name = parts[0]
    if type_name not in THING_TYPES:
        raise MapError(f"line {lineno}: unknown thing type {type_name!r}")
    props = _parse_props(lineno, type_name, parts[4:])
    parts = parts[:4]
    try:
        x, y, angle_deg = (float(p) for p in parts[1:])
    except ValueError:
        raise MapError(f"line {lineno}: x, y and angle must be numbers") from None
    cx, cy = math.floor(x), math.floor(y)
    if not game_map.in_bounds(cx, cy):
        raise MapError(f"line {lineno}: {type_name} at ({x:g}, {y:g}) is outside the map")
    if game_map.is_solid(cx, cy) and not game_map.is_low_block(cx, cy):
        raise MapError(f"line {lineno}: {type_name} at ({x:g}, {y:g}) is inside a wall")
    angle = math.radians(angle_deg)
    at = next((v for k, v in props if k == "at"), None)
    if at is None:
        return Thing(type_name, x, y, angle, props)
    try:
        px, py, pa = place(type_name, x, y, angle, at, game_map.is_full_wall)
    except PlacementError as e:
        raise MapError(f"line {lineno}: {type_name} {e}") from None
    if math.floor(px) != cx or math.floor(py) != cy:
        raise MapError(f"line {lineno}: {type_name} at={at}: too big to fit in its cell")
    return Thing(type_name, px, py, pa, props, written=(x, y, angle))


def _parse_props(lineno: int, type_name: str, tokens: list[str]) -> tuple[tuple[str, str], ...]:
    allowed = THING_TYPES[type_name].properties
    props = []
    for token in tokens:
        key, sep, value = token.partition("=")
        if not sep:
            raise MapError(f"line {lineno}: expected key=value after the angle, got {token!r}")
        if key == "at" and not type_name.startswith("door_"):  # any thing but a door
            if value not in ANCHORS:
                choices = ", ".join(ANCHORS)
                raise MapError(
                    f"line {lineno}: {type_name} at={value!r}: expected one of {choices}"
                )
            props.append((key, value))
            continue
        if key not in allowed:
            known = ", ".join(sorted(allowed)) or "none"
            raise MapError(f"line {lineno}: {type_name} has no property {key!r} (has: {known})")
        if value not in allowed[key]:
            choices = ", ".join(allowed[key])
            raise MapError(f"line {lineno}: {type_name} {key}={value!r}: expected one of {choices}")
        props.append((key, value))
    return tuple(props)

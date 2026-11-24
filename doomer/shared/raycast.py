"""Grid raycasting with DDA (digital differential analyser).

Shared because both the renderer (one ray per screen column) and the server
(hitscan weapons, later) need it.

How DDA works: a ray crosses vertical grid lines at regular distances
(`delta_x` apart) and horizontal grid lines at regular distances (`delta_y`
apart). Keep a running "distance to the next vertical line" and "distance to
the next horizontal line", always step to whichever is nearer, and check the
cell you've entered. Each step is a couple of additions, and you never test a
cell the ray doesn't actually pass through.

Heights: every cell has a floor and a ceiling (doomer/shared/mapfile.py). The
ray reports a hit wherever the space it's passing through changes:

- it enters a low block (a desk, a counter, a raised floor): one hit per
  cell, because each has its own faces and textures;
- it enters open floor whose floor or ceiling height differs from where it
  came from (into a double-height atrium, under a bulkhead);
- it reaches a full-height wall, where it stops.

Each hit records the floor and ceiling of the space the ray came from and of
the space it entered: everything a renderer needs for a riser face where the
floor goes up, a bulkhead face where the ceiling comes down, and the floor and
ceiling surfaces of each stretch between hits.
"""

import math
from enum import Enum
from typing import NamedTuple

from doomer.shared.mapfile import EAST, EMPTY, NORTH, SOUTH, WALL_UNITS, WEST, GameMap


class Side(Enum):
    X = "x"  # hit a wall face perpendicular to the x axis (an east or west face)
    Y = "y"  # hit a north or south face


class RayHit(NamedTuple):
    """Where a ray enters a new kind of space. A NamedTuple rather than a frozen
    dataclass: the renderer creates ~1,000 of these per frame, and tuple
    construction is several times cheaper."""

    distance: float  # Euclidean, along the ray. Renderers correct for fisheye themselves.
    cell_x: int
    cell_y: int
    wall_type: int  # 0 when entering open floor with different heights
    side: Side
    # Where along the face the ray hit, 0..1, for texture lookup. Oriented so
    # textures read left-to-right when you face it.
    u: float
    # Heights in walls (1 = a storey). `top` is the entered cell's floor: the
    # top of a low block, the floor of open space, or inf for a full wall.
    # `base` is the floor of the space the ray came from, so a riser face spans
    # base..top (nothing if base >= top).
    top: float = math.inf
    base: float = 0.0
    # Where the ray leaves the low block, or the stretch of open space with
    # these heights: the far edge of its floor and ceiling surfaces.
    exit_distance: float = math.inf
    exit_side: Side = Side.X
    exit_u: float = 0.0
    # Which compass face of the cell the ray entered and left by
    # (mapfile.NORTH/EAST/SOUTH/WEST), for per-side textures.
    face: int = 0
    exit_face: int = 0
    ceiling: float = 1.0  # ceiling of the entered space
    prev_ceiling: float = 1.0  # ceiling of the space the ray came from

    @property
    def full(self) -> bool:
        """A full-height wall: the ray stops here."""
        return math.isinf(self.top)


def cast_ray(
    game_map: GameMap, x: float, y: float, angle: float, max_distance: float = 64.0
) -> list[RayHit]:
    """Cast a ray and return the hits along it, nearest first.

    The list ends at the first full-height wall, or at max_distance.
    """
    dx, dy = math.cos(angle), math.sin(angle)
    cell_x, cell_y = math.floor(x), math.floor(y)

    # Distance along the ray between successive vertical (x) / horizontal (y) grid lines.
    delta_x = abs(1.0 / dx) if dx != 0.0 else math.inf
    delta_y = abs(1.0 / dy) if dy != 0.0 else math.inf

    if dx < 0:
        step_x, next_x = -1, (x - cell_x) * delta_x
    else:
        step_x, next_x = 1, (cell_x + 1 - x) * delta_x if dx != 0.0 else math.inf
    if dy < 0:
        step_y, next_y = -1, (y - cell_y) * delta_y
    else:
        step_y, next_y = 1, (cell_y + 1 - y) * delta_y if dy != 0.0 else math.inf

    # Hot loop (320+ rays a frame): local variables and inlined grid lookups
    # instead of method calls. Heights stay in integer units (eighths) until a
    # hit is made, so comparisons are exact.
    walls, floors, ceilings = game_map.walls, game_map.floor, game_map.ceiling
    width, height = game_map.width, game_map.height
    hits: list[RayHit] = []

    # The space the ray starts in (the camera may be standing on a desk), and
    # the ceiling of the last open (non-block) space it was in: stepping off a
    # desk back onto the same floor isn't a change worth reporting.
    floor_now, ceiling_now = start_heights(game_map, cell_x, cell_y)
    open_ceiling = ceiling_now
    open_hit = -1  # index of the open-space hit whose exit distance is still open
    facing = game_map.facing

    while True:
        if next_x < next_y:
            distance, side = next_x, Side.X
            next_x += delta_x
            cell_x += step_x
        else:
            distance, side = next_y, Side.Y
            next_y += delta_y
            cell_y += step_y
        if distance > max_distance:
            return hits

        if 0 <= cell_x < width and 0 <= cell_y < height:
            wall = walls[cell_y][cell_x]
            floor_units, ceiling_units = floors[cell_y][cell_x], ceilings[cell_y][cell_x]
        else:
            wall, floor_units, ceiling_units = 1, 0, WALL_UNITS  # outside the map is solid
        full = wall != EMPTY and not 0 < floor_units < ceiling_units
        low_block = wall != EMPTY and not full
        new_floor = 0 if wall == EMPTY else floor_units

        if not (full or low_block):
            if ceiling_units == open_ceiling:
                floor_now, ceiling_now = new_floor, ceiling_units
                continue  # same kind of open space as before: nothing to report
            open_ceiling = ceiling_units

        if open_hit >= 0:  # close the open stretch we were in
            hits[open_hit] = hits[open_hit]._replace(exit_distance=distance)
            open_hit = -1

        if next_x < next_y:
            exit_distance, exit_side = next_x, Side.X
        else:
            exit_distance, exit_side = next_y, Side.Y
        exit_u = _wall_u(x, y, dx, dy, exit_distance, exit_side)
        exit_face = (_entry_face(exit_side, dx, dy) + 2) % 4  # leave by the opposite kind

        if low_block and hits:
            last = hits[-1]
            if (
                last.wall_type == wall
                and last.exit_distance == distance
                and floor_units == floor_now
                and ceiling_units == ceiling_now
                and facing[last.cell_y][last.cell_x] == facing[cell_y][cell_x]
            ):
                # A run of identical blocks (a desk pod, a mezzanine): the faces
                # between them have zero height, so extend one hit instead.
                hits[-1] = last._replace(
                    exit_distance=exit_distance,
                    exit_side=exit_side,
                    exit_u=exit_u,
                    exit_face=exit_face,
                )
                continue

        u = _wall_u(x, y, dx, dy, distance, side)
        face = _entry_face(side, dx, dy)

        hits.append(
            RayHit(
                distance,
                cell_x,
                cell_y,
                wall,
                side,
                u,
                math.inf if full else new_floor / WALL_UNITS,
                floor_now / WALL_UNITS,
                exit_distance if low_block else math.inf,
                exit_side,
                exit_u,
                face,
                exit_face,
                ceiling_units / WALL_UNITS,
                ceiling_now / WALL_UNITS,
            )
        )
        if full:
            return hits
        if not low_block:
            open_hit = len(hits) - 1  # its exit is where this stretch of space ends
        floor_now, ceiling_now = new_floor, ceiling_units


def start_heights(game_map: GameMap, x: int, y: int) -> tuple[int, int]:
    """(floor, ceiling) in units of the space in cell (x, y): where a ray (or a
    camera) starts. A low block's floor is its top: you're standing on it."""
    if not game_map.in_bounds(x, y):
        return 0, WALL_UNITS
    floor = game_map.floor[y][x] if game_map.is_low_block(x, y) else 0
    return floor, game_map.ceiling[y][x]


def _entry_face(side: Side, dx: float, dy: float) -> int:
    """The compass face of a cell a ray enters through. y points down (south):
    heading east (+x) you enter through a cell's west face; heading south (+y),
    through its north face."""
    if side is Side.X:
        return WEST if dx > 0 else EAST
    return NORTH if dy > 0 else SOUTH


def _wall_u(x: float, y: float, dx: float, dy: float, distance: float, side: Side) -> float:
    # u increases towards the viewer's right. Facing east, right is +y (south);
    # facing west it's -y; facing south it's -x; facing north it's +x.
    if side is Side.X:
        u = (y + dy * distance) % 1.0
        return u if dx > 0 else 1.0 - u
    u = (x + dx * distance) % 1.0
    return 1.0 - u if dy > 0 else u


def cell_exit_distance(x: float, y: float, angle: float) -> float:
    """How far a ray from (x, y) travels before leaving the cell it starts in."""
    dx, dy = math.cos(angle), math.sin(angle)
    cx, cy = math.floor(x), math.floor(y)
    to_x = ((cx + 1 - x) / dx if dx > 0 else (x - cx) / -dx) if dx != 0.0 else math.inf
    to_y = ((cy + 1 - y) / dy if dy > 0 else (y - cy) / -dy) if dy != 0.0 else math.inf
    return min(to_x, to_y)

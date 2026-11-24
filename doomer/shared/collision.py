"""Player collision against walls and blocking things.

The player is an axis-aligned square of half-width `radius`; so is every
blocking thing (an Obstacle). Movement is resolved one axis at a time, which
gives sliding for free: pushing diagonally into a wall cancels only the
blocked axis.
"""

import math
from collections.abc import Sequence
from typing import NamedTuple

from doomer.shared.mapfile import GameMap

_EPS = 1e-9


class Obstacle(NamedTuple):
    x: float
    y: float
    radius: float  # half-width of its square footprint


def move_box(
    game_map: GameMap,
    x: float,
    y: float,
    dx: float,
    dy: float,
    radius: float,
    obstacles: Sequence[Obstacle] = (),
) -> tuple[float, float]:
    """Move a box from (x, y) by (dx, dy), stopping at walls and obstacles.

    Returns the new centre.
    """
    # Each sub-step moves at most `radius`, so the box can't skip over a cell.
    steps = max(1, math.ceil(max(abs(dx), abs(dy)) / radius))
    sx, sy = dx / steps, dy / steps
    for _ in range(steps):
        nx = _move_x(game_map, x, y, sx, radius)
        x = _clamp_to_obstacles(obstacles, x, nx, y, radius, axis=0)
        ny = _move_y(game_map, x, y, sy, radius)
        y = _clamp_to_obstacles(obstacles, y, ny, x, radius, axis=1)
    return x, y


def _clamp_to_obstacles(
    obstacles: Sequence[Obstacle], pos: float, new: float, other: float, r: float, axis: int
) -> float:
    """Stop movement along one axis at the first obstacle in the way.

    `pos`/`new` are the old and proposed coordinates on this axis, `other` the
    coordinate on the other axis. Only blocks if we weren't already overlapping
    on this axis, so a player who somehow spawns inside a thing can walk out.
    """
    for ob in obstacles:
        o_pos, o_other = (ob.x, ob.y) if axis == 0 else (ob.y, ob.x)
        reach = r + ob.radius
        if abs(other - o_other) >= reach - _EPS:
            continue  # not lined up on the other axis
        if new > pos and pos <= o_pos - reach + _EPS:
            new = min(new, o_pos - reach)
        elif new < pos and pos >= o_pos + reach - _EPS:
            new = max(new, o_pos + reach)
    return new


def _span(lo: float, hi: float) -> range:
    """Cell indices that a [lo, hi] interval overlaps (touching an edge doesn't count)."""
    return range(math.floor(lo + _EPS), math.ceil(hi - _EPS))


def _move_x(m: GameMap, x: float, y: float, dx: float, r: float) -> float:
    if dx == 0:
        return x
    nx = x + dx
    rows = _span(y - r, y + r)
    if dx > 0:
        col = math.ceil(nx + r - _EPS) - 1
        if any(m.is_solid(col, row) for row in rows):
            return col - r
    else:
        col = math.floor(nx - r + _EPS)
        if any(m.is_solid(col, row) for row in rows):
            return col + 1 + r
    return nx


def _move_y(m: GameMap, x: float, y: float, dy: float, r: float) -> float:
    if dy == 0:
        return y
    ny = y + dy
    cols = _span(x - r, x + r)
    if dy > 0:
        row = math.ceil(ny + r - _EPS) - 1
        if any(m.is_solid(col, row) for col in cols):
            return row - r
    else:
        row = math.floor(ny - r + _EPS)
        if any(m.is_solid(col, row) for col in cols):
            return row + 1 + r
    return ny

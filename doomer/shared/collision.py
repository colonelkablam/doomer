"""Player collision against walls and blocking things.

The player is an axis-aligned square of half-width `radius`; so is every
blocking thing (an Obstacle). Movement is resolved one axis at a time, which
gives sliding for free: pushing diagonally into a wall cancels only the
blocked axis.
"""

import math
from collections.abc import Callable, Sequence
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
    blocked: Callable[[int, int], bool] | None = None,
) -> tuple[float, float]:
    """Move a box from (x, y) by (dx, dy), stopping at walls and obstacles.

    `blocked(cell_x, cell_y)` says whether a cell stops the box; by default
    any wall cell does. The simulation passes a height-aware test (a desk
    blocks you on the floor but not once you've jumped onto it). Returns the
    new centre.
    """
    is_blocked = blocked if blocked is not None else game_map.is_solid
    # Each sub-step moves at most `radius`, so the box can't skip over a cell.
    steps = max(1, math.ceil(max(abs(dx), abs(dy)) / radius))
    sx, sy = dx / steps, dy / steps
    for _ in range(steps):
        nx = _move_x(is_blocked, x, y, sx, radius)
        x = _clamp_to_obstacles(obstacles, x, nx, y, radius, axis=0)
        ny = _move_y(is_blocked, x, y, sy, radius)
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


def _move_x(blocked: Callable[[int, int], bool], x: float, y: float, dx: float, r: float) -> float:
    if dx == 0:
        return x
    nx = x + dx
    rows = _span(y - r, y + r)
    # Only a column the box is newly entering can block it: if it was already
    # in it (teleported, or the cell became blocking when it crouched or
    # landed), it may move on and out rather than be snapped backwards.
    if dx > 0:
        col = math.ceil(nx + r - _EPS) - 1
        entering = col > math.ceil(x + r - _EPS) - 1
        if entering and any(blocked(col, row) for row in rows):
            return col - r
    else:
        col = math.floor(nx - r + _EPS)
        entering = col < math.floor(x - r + _EPS)
        if entering and any(blocked(col, row) for row in rows):
            return col + 1 + r
    return nx


def _move_y(blocked: Callable[[int, int], bool], x: float, y: float, dy: float, r: float) -> float:
    if dy == 0:
        return y
    ny = y + dy
    cols = _span(x - r, x + r)
    if dy > 0:
        row = math.ceil(ny + r - _EPS) - 1
        entering = row > math.ceil(y + r - _EPS) - 1
        if entering and any(blocked(col, row) for col in cols):
            return row - r
    else:
        row = math.floor(ny - r + _EPS)
        entering = row < math.floor(y - r + _EPS)
        if entering and any(blocked(col, row) for col in cols):
            return row + 1 + r
    return ny

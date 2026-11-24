"""Player-vs-wall collision on the cell grid.

The player is an axis-aligned square of half-width `radius`. Movement is
resolved one axis at a time, which gives wall sliding for free: pushing
diagonally into a wall cancels only the blocked axis.
"""

import math

from doomer.shared.mapfile import GameMap

_EPS = 1e-9


def move_box(
    game_map: GameMap, x: float, y: float, dx: float, dy: float, radius: float
) -> tuple[float, float]:
    """Move a box from (x, y) by (dx, dy), stopping at walls. Returns the new centre."""
    # Each sub-step moves at most `radius`, so the box can't skip over a cell.
    steps = max(1, math.ceil(max(abs(dx), abs(dy)) / radius))
    sx, sy = dx / steps, dy / steps
    for _ in range(steps):
        x = _move_x(game_map, x, y, sx, radius)
        y = _move_y(game_map, x, y, sy, radius)
    return x, y


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

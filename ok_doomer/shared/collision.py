"""Player collision against walls and blocking things.

The player is an axis-aligned square of half-width `radius`; so is every
blocking thing (an Obstacle). Movement is resolved one axis at a time, which
gives sliding for free: pushing diagonally into a wall cancels only the
blocked axis.
"""

import math
from collections.abc import Callable, Sequence
from typing import NamedTuple

from ok_doomer.shared.mapfile import GameMap

_EPS = 1e-9


class Obstacle(NamedTuple):
    """A blocking rectangle (a thing's footprint): centre and half-widths,
    and the height of its top (walls). A low top can be jumped onto and
    stood on, as a desk can (a toilet, a sofa); `inf`: blocks at any height."""

    x: float
    y: float
    half_x: float
    half_y: float
    top: float = math.inf

    @classmethod
    def square(cls, x: float, y: float, radius: float) -> "Obstacle":
        return cls(x, y, radius, radius)


def obstacles_in_the_way(obstacles: Sequence[Obstacle], feet: float, step: float) -> list[Obstacle]:
    """The obstacles that block a body whose feet are at `feet`: those whose
    tops are more than a step above them (the rest it walks or stands on)."""
    return [ob for ob in obstacles if ob.top > feet + step + _EPS]


def obstacle_support(
    obstacles: Sequence[Obstacle], x: float, y: float, radius: float, feet: float, step: float
) -> float:
    """The highest obstacle top under a box at (x, y) that it can be on
    (no more than a step above its feet); -inf if none."""
    best = -math.inf
    for ob in obstacles:
        if ob.top > feet + step + _EPS or ob.top == math.inf:
            continue
        if abs(x - ob.x) < radius + ob.half_x - _EPS and abs(y - ob.y) < radius + ob.half_y - _EPS:
            best = max(best, ob.top)
    return best


class ThinWall(NamedTuple):
    """A thin panel (a cubicle screen): the plane x = `at` (axis 0) or
    y = `at` (axis 1), `half` thick either side, from `lo` to `hi` along the
    other axis."""

    axis: int
    at: float
    lo: float
    hi: float
    half: float


def move_box(
    game_map: GameMap,
    x: float,
    y: float,
    dx: float,
    dy: float,
    radius: float,
    obstacles: Sequence[Obstacle] = (),
    blocked: Callable[[int, int], bool] | None = None,
    walls: Sequence[ThinWall] = (),
) -> tuple[float, float]:
    """Move a box from (x, y) by (dx, dy), stopping at walls and obstacles.

    `blocked(cell_x, cell_y)` says whether a cell stops the box; by default
    any wall cell does. The simulation passes a height-aware test (a desk
    blocks you on the floor but not once you've jumped onto it). `walls` are
    thin panels inside cells. Returns the new centre.
    """
    is_blocked = blocked if blocked is not None else game_map.is_solid
    # Each sub-step moves at most `radius`, so the box can't skip over a cell.
    steps = max(1, math.ceil(max(abs(dx), abs(dy)) / radius))
    sx, sy = dx / steps, dy / steps
    for _ in range(steps):
        nx = _move_x(is_blocked, x, y, sx, radius)
        nx = _clamp_to_obstacles(obstacles, x, nx, y, radius, axis=0)
        x = _clamp_to_walls(walls, x, nx, y, radius, axis=0)
        ny = _move_y(is_blocked, x, y, sy, radius)
        ny = _clamp_to_obstacles(obstacles, y, ny, x, radius, axis=1)
        y = _clamp_to_walls(walls, y, ny, x, radius, axis=1)
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
        half, half_other = (ob.half_x, ob.half_y) if axis == 0 else (ob.half_y, ob.half_x)
        reach = r + half
        if abs(other - o_other) >= r + half_other - _EPS:
            continue  # not lined up on the other axis
        if new > pos and pos <= o_pos - reach + _EPS:
            new = min(new, o_pos - reach)
        elif new < pos and pos >= o_pos + reach - _EPS:
            new = max(new, o_pos + reach)
    return new


def _clamp_to_walls(
    walls: Sequence[ThinWall], pos: float, new: float, other: float, r: float, axis: int
) -> float:
    """Stop movement along one axis at a thin wall across it. As for
    obstacles, only a wall the box isn't already overlapping can block."""
    for wall in walls:
        if wall.axis != axis:
            continue  # parallel to this movement: can't be run into this way
        if other + r <= wall.lo + _EPS or other - r >= wall.hi - _EPS:
            continue  # beside the wall's end
        reach = r + wall.half
        if new > pos and pos <= wall.at - reach + _EPS:
            new = min(new, wall.at - reach)
        elif new < pos and pos >= wall.at + reach - _EPS:
            new = max(new, wall.at + reach)
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

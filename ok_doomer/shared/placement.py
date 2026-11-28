"""Placing things against walls: a thing's `at=` property (docs/LEVELS.md).

`at=n`, `e`, `s` or `w` stands a thing against that wall of its cell: it's
moved along the wall's normal until its footprint is just off the wall's
face, and turned to face out of it (a cooler facing the wall is never right).
Its position along the wall stays as written, so several posters can share a
wall. `at=nw`, `ne`, `sw` or `se` puts it in that corner of its cell, against
both walls; its angle stays as written, and a plant is also turned a little
(seeded by its cell, so a map always looks the same).

The map gives the cell (the written x, y) and the wall; the thing's own size
does the rest. That's what a level generator wants: it picks a cell and a
wall, not coordinates, and resizing a model doesn't mean moving every copy.

Standard library only; the map parser calls it.
"""

import math
import random
from collections.abc import Callable

from ok_doomer.shared.things import THING_TYPES

GAP = 0.01  # cells between a footprint and the wall's face
# Wall: (dx, dy) to the wall's cell, and the angle facing out of it.
WALLS: dict[str, tuple[int, int, float]] = {
    "n": (0, -1, math.pi / 2),
    "e": (1, 0, math.pi),
    "s": (0, 1, 3 * math.pi / 2),
    "w": (-1, 0, 0.0),
}
CORNERS = ("nw", "ne", "sw", "se")
ANCHORS = (*WALLS, *CORNERS)
# Things turned a little at random in a corner: up to this much either way.
JITTER = {"ficus": math.radians(25), "fern": math.radians(25)}


class PlacementError(ValueError):
    """An `at=` that can't be done: there's no wall there."""


def extents(type_name: str, angle: float) -> tuple[float, float, float, float]:
    """A thing's footprint relative to its point, in the world, turned to
    `angle`: (min x, max x, min y, max y). Zero for things with no size
    (posters, signs: they're flat on the wall)."""
    kind = THING_TYPES[type_name]
    if not kind.footprints:
        r = kind.radius  # a square, not turned (as the obstacle is)
        return -r, r, -r, r
    # As ok_doomer/shared/furniture.py: turned to the nearest quarter turn.
    turns = round(angle / (math.pi / 2)) % 4
    cos, sin = [(1, 0), (0, 1), (-1, 0), (0, -1)][turns]
    xs, ys = [], []
    for f in kind.footprints:
        for a in f.x:
            for b in f.y:
                xs.append(a * cos - b * sin)
                ys.append(a * sin + b * cos)
    return min(xs), max(xs), min(ys), max(ys)


def place(
    type_name: str,
    x: float,
    y: float,
    angle: float,
    at: str,
    is_wall: Callable[[int, int], bool],
) -> tuple[float, float, float]:
    """Where a thing written at (x, y, angle) with `at=<anchor>` goes: its
    (x, y, angle). `is_wall(cx, cy)`: is that cell a full-height wall?"""
    cx, cy = math.floor(x), math.floor(y)
    sides = [at] if at in WALLS else list(at)
    for side in sides:
        dx, dy, _ = WALLS[side]
        if not is_wall(cx + dx, cy + dy):
            raise PlacementError(f"at={at}: no wall to the {_NAMES[side]} of cell ({cx}, {cy})")
    if at in WALLS:
        angle = WALLS[at][2]
    elif type_name in JITTER:
        spread = JITTER[type_name]
        angle += random.Random(f"{type_name}:{cx}:{cy}").uniform(-spread, spread)
    x0, x1, y0, y1 = extents(type_name, angle)
    for side in sides:
        if side == "w":
            x = cx + GAP - x0
        elif side == "e":
            x = cx + 1 - GAP - x1
        elif side == "n":
            y = cy + GAP - y0
        else:
            y = cy + 1 - GAP - y1
    return x, y, angle


_NAMES = {"n": "north", "e": "east", "s": "south", "w": "west"}

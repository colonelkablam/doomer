"""Furniture with a real floor plan: footprints placed in the world.

A thing type can describe its floor plan as rectangles with a top height
(things.Footprint), in its own frame. Placed at the thing's position and
angle (snapped to quarter turns: furniture stands square to the walls), they
are what blocks players (collision.Obstacle), and what things placed on the
furniture rest on: the reception desk's ledge holds a pass, its work surface
a computer.
"""

import math
from typing import NamedTuple

from doomer.shared.collision import Obstacle
from doomer.shared.mapfile import GameMap, Thing
from doomer.shared.things import THING_TYPES


class PlacedFootprint(NamedTuple):
    x0: float
    x1: float
    y0: float
    y1: float
    top: float

    def contains(self, x: float, y: float) -> bool:
        return self.x0 <= x <= self.x1 and self.y0 <= y <= self.y1


def placed_footprints(thing: Thing) -> list[PlacedFootprint]:
    """A thing's footprints in the world."""
    turns = round(thing.angle / (math.pi / 2)) % 4
    out = []
    for f in THING_TYPES[thing.type].footprints:
        (a0, a1), (b0, b1) = f.x, f.y
        # The thing's x (its front) points east, south, west or north.
        if turns == 0:
            xs, ys = (a0, a1), (b0, b1)
        elif turns == 1:  # facing south: x -> +y, y -> -x
            xs, ys = (-b1, -b0), (a0, a1)
        elif turns == 2:
            xs, ys = (-a1, -a0), (-b1, -b0)
        else:  # facing north: x -> -y, y -> +x
            xs, ys = (b0, b1), (-a1, -a0)
        out.append(
            PlacedFootprint(
                thing.x + xs[0], thing.x + xs[1], thing.y + ys[0], thing.y + ys[1], f.top
            )
        )
    return out


def footprint_obstacles(thing: Thing) -> list[Obstacle]:
    return [
        Obstacle((p.x0 + p.x1) / 2, (p.y0 + p.y1) / 2, (p.x1 - p.x0) / 2, (p.y1 - p.y0) / 2)
        for p in placed_footprints(thing)
    ]


def resting_height(game_map: GameMap, thing: Thing) -> float:
    """What a thing stands on: the top of a low block in its cell, or the top
    of furniture it's placed on (another thing's footprint), or the floor."""
    cx, cy = math.floor(thing.x), math.floor(thing.y)
    z = game_map.floor_at(thing.x, thing.y) if game_map.is_low_block(cx, cy) else 0.0
    for other in game_map.things:
        if other is thing or not THING_TYPES[other.type].footprints:
            continue
        for p in placed_footprints(other):
            if p.contains(thing.x, thing.y):
                z = max(z, p.top)
    return z

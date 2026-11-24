"""Pickups: things you take by walking over them. For now, ID passes.

A pass belongs to the team once anyone picks it up: every player can then
open doors locked with its colour (doomer/shared/doors.py). That suits
co-op: nobody has to fetch the one player holding the key. State (what's
still lying about, what the team holds) is in the World, so the server will
own it.
"""

import math
from dataclasses import dataclass

from doomer.shared.furniture import resting_height
from doomer.shared.mapfile import GameMap

PASS_PREFIX = "pass_"
# Cells, centre to centre: the player's box (0.25) plus about an arm (0.5), so
# you can take a pass from just inside a counter's edge (you're stopped at the
# edge; anything deeper in is out of reach). Maps put passes near the edge.
REACH = 0.75
HEIGHT_REACH = 0.5  # walls: you can take one from a counter's top standing on the floor


@dataclass(slots=True)
class Pickup:
    id: int
    type: str  # thing type, e.g. "pass_blue"
    x: float
    y: float
    z: float  # the surface it lies on
    taken: bool = False

    @property
    def pass_colour(self) -> str | None:
        return self.type.removeprefix(PASS_PREFIX) if self.type.startswith(PASS_PREFIX) else None


def pickups_from_map(game_map: GameMap) -> list[Pickup]:
    out: list[Pickup] = []
    for t in game_map.things:
        if t.type.startswith(PASS_PREFIX):
            out.append(Pickup(len(out), t.type, t.x, t.y, resting_height(game_map, t)))
    return out


def take_pickups(pickups: list[Pickup], x: float, y: float, z: float) -> list[Pickup]:
    """The pickups a player at (x, y), feet at z, takes this tick."""
    taken = []
    for p in pickups:
        if p.taken:
            continue
        if math.hypot(p.x - x, p.y - y) <= REACH and abs(p.z - z) <= HEIGHT_REACH:
            p.taken = True
            taken.append(p)
    return taken

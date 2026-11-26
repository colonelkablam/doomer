"""Pickups: things you take by walking over them: ID passes, and things
that top up a player's meters (a coffee cup: coffee, the armour; GAME.md,
Pickups). What each gives is data (PICKUP_EFFECTS), so tuning is editing
numbers.

A pass belongs to the team once anyone picks it up: every player can then
open doors locked with its colour (doomer/shared/doors.py). That suits
co-op: nobody has to fetch the one player holding the key. State (what's
still lying about, what the team holds) is in the World, so the server will
own it.
"""

import math
from collections.abc import Callable
from dataclasses import dataclass

from doomer.shared.furniture import resting_height
from doomer.shared.mapfile import GameMap

PASS_PREFIX = "pass_"
MAX_COFFEE = 100
METER_MAX = {"coffee": MAX_COFFEE, "health": 100}  # PlayerState fields and their caps
# What a meter pickup gives, by thing type: {meter: amount}. A player whose
# meter is full leaves it lying (for whoever needs it).
PICKUP_EFFECTS: dict[str, dict[str, int]] = {"coffee_cup": {"coffee": 5}}
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
    angle: float = 0.0  # which way it's turned (a cup's handle)

    @property
    def effects(self) -> dict[str, int]:
        """What it gives a player's meters ({} for a pass)."""
        return PICKUP_EFFECTS.get(self.type, {})

    @property
    def pass_colour(self) -> str | None:
        return self.type.removeprefix(PASS_PREFIX) if self.type.startswith(PASS_PREFIX) else None


def pickups_from_map(game_map: GameMap) -> list[Pickup]:
    out: list[Pickup] = []
    for t in game_map.things:
        if t.type.startswith(PASS_PREFIX) or t.type in PICKUP_EFFECTS:
            z = resting_height(game_map, t)
            out.append(Pickup(len(out), t.type, t.x, t.y, z, angle=t.angle))
    return out


def take_pickups(
    pickups: list[Pickup],
    x: float,
    y: float,
    z: float,
    wants: Callable[[Pickup], bool] = lambda p: True,
) -> list[Pickup]:
    """The pickups a player at (x, y), feet at z, takes this tick: those in
    reach that they `wants` (a full meter doesn't want more)."""
    taken = []
    for p in pickups:
        if p.taken or not wants(p):
            continue
        if math.hypot(p.x - x, p.y - y) <= REACH and abs(p.z - z) <= HEIGHT_REACH:
            p.taken = True
            taken.append(p)
    return taken

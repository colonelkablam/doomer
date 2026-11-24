"""Switches: wall-mounted things you use (E) to set a level flag.

The first is the breaker in the plant room: it sets `power`. A floor whose
map has a breaker starts without power (emergency lighting only, lifts dead)
until someone throws it; a map without one is powered from the start. Flags
live in the World, so the server will own them and every client agrees.

A switch is like a sign: it stands just in front of a wall's face, angle
out of the wall. It's one-way: once thrown, it stays on.
"""

import math
from dataclasses import dataclass

from doomer.shared.mapfile import GameMap

POWER = "power"
SWITCH_FLAGS = {"breaker": POWER}  # thing type -> the flag it sets
USE_REACH = 1.0  # cells from the switch
USE_CONE = math.cos(math.radians(50))  # must be roughly facing it


@dataclass(slots=True)
class Switch:
    id: int
    type: str
    x: float
    y: float
    facing: float  # radians: out of the wall
    flag: str
    on: bool = False


def switches_from_map(game_map: GameMap) -> list[Switch]:
    out: list[Switch] = []
    for t in game_map.things:
        flag = SWITCH_FLAGS.get(t.type)
        if flag is not None:
            out.append(Switch(len(out), t.type, t.x, t.y, t.angle, flag))
    return out


def initial_flags(game_map: GameMap) -> set[str]:
    """Flags set before anyone touches a switch: `power`, unless the map has
    a breaker to restore it."""
    return set() if any(t.type == "breaker" for t in game_map.things) else {POWER}


def try_switch(switches: list[Switch], x: float, y: float, angle: float) -> Switch | None:
    """The switch a player at (x, y) facing `angle` throws, if any: the
    nearest within reach that they're roughly facing and that isn't on yet."""
    best: Switch | None = None
    best_d = USE_REACH
    for sw in switches:
        if sw.on:
            continue
        dx, dy = sw.x - x, sw.y - y
        d = math.hypot(dx, dy)
        if d > best_d or d < 1e-9:
            continue
        if (dx * math.cos(angle) + dy * math.sin(angle)) / d < USE_CONE:
            continue
        best, best_d = sw, d
    if best is not None:
        best.on = True
    return best

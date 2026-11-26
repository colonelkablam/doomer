"""Toilet cubicle doors: use one (E) to open or shut it, unless someone's in.

A run of cubicles (`cubicles_two`, `_three`, `_four`: doomer/shared/things.py)
has a door at the front of each. Doors swing inwards, into the cubicle, as
real ones do: open or shut, they never stick out into the room, so they
change nothing you can bump into (a cubicle is narrower than a player
anyway). The first door of each run starts open; the second is ENGAGED,
locked for good (someone's in there); the rest start shut. A door swings
over DOOR_SECONDS. The state is in the World, so the server will own it.
"""

import math
from dataclasses import dataclass

from doomer.shared.mapfile import GameMap
from doomer.shared.things import CUBICLE_DEPTH, CUBICLE_WIDTH

DOOR_SECONDS = 0.4  # shut to open
OPEN_ANGLE = math.radians(80)  # inwards, at full open
USE_REACH = 0.9  # cells from the door's middle
USE_CONE = math.cos(math.radians(45))  # must be roughly facing it
CUBICLE_COUNTS = {"cubicles_two": 2, "cubicles_three": 3, "cubicles_four": 4}


@dataclass(slots=True)
class CubicleDoor:
    id: int
    run_x: float  # the run's thing: where it stands and faces
    run_y: float
    run_angle: float
    index: int  # which cubicle along the run: 0 is at the run's +y end
    count: int  # cubicles in the run
    locked: bool = False  # ENGAGED: someone's in
    open: float = 0.0  # 0 shut .. 1 open
    opening: bool = False

    def model_centre(self) -> tuple[float, float]:
        """The door's middle in the run's frame (x out of the wall)."""
        half = self.count * CUBICLE_WIDTH / 2
        return CUBICLE_DEPTH, half - (self.index + 0.5) * CUBICLE_WIDTH

    def centre(self) -> tuple[float, float]:
        mx, my = self.model_centre()
        c, s = math.cos(self.run_angle), math.sin(self.run_angle)
        return self.run_x + mx * c - my * s, self.run_y + mx * s + my * c


def cubicle_doors_from_map(game_map: GameMap) -> list[CubicleDoor]:
    out: list[CubicleDoor] = []
    for t in game_map.things:
        count = CUBICLE_COUNTS.get(t.type)
        if count is None:
            continue
        for k in range(count):
            door = CubicleDoor(len(out), t.x, t.y, t.angle, k, count, locked=k == 1)
            if k == 0:
                door.open, door.opening = 1.0, True
            out.append(door)
    return out


def cubicle_door_in_reach(
    doors: list[CubicleDoor], x: float, y: float, angle: float
) -> CubicleDoor | None:
    """The cubicle door a player at (x, y) facing `angle` is using: the
    nearest within reach that they're roughly facing."""
    best: CubicleDoor | None = None
    best_d = USE_REACH
    for door in doors:
        cx, cy = door.centre()
        dx, dy = cx - x, cy - y
        dist = math.hypot(dx, dy)
        if dist > best_d or dist < 1e-9:
            continue
        if (dx * math.cos(angle) + dy * math.sin(angle)) / dist < USE_CONE:
            continue
        best, best_d = door, dist
    return best


def use_cubicle_door(door: CubicleDoor) -> bool:
    """Open it if it's shut (or shutting), shut it if open; False if it's
    locked (ENGAGED), and nothing changes."""
    if door.locked:
        return False
    door.opening = not door.opening
    return True


def update_cubicle_doors(doors: list[CubicleDoor], dt: float) -> None:
    step = dt / DOOR_SECONDS
    for door in doors:
        target = 1.0 if door.opening else 0.0
        if door.open < target:
            door.open = min(target, door.open + step)
        elif door.open > target:
            door.open = max(target, door.open - step)

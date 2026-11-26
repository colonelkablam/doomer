"""Toilet cubicle doors: use one (E) to open or shut it, unless someone's in.

A run of cubicles (`cubicles_two` .. `_five`: doomer/shared/things.py) has a
door at the front of each, between fixed returns (a strip either side, so
the door's narrower than the cubicle), hinged on the cubicle's +y side (the run's own
frame: x out of the wall; on your left as you face the run), its latch on
the -y side. So the first door of a run, swung open, folds back against
the wall at the run's end. Doors swing outwards,
a quarter turn, over DOOR_SECONDS. Shut, a door is a thin wall across the
cubicle's front; open (past half way), its leaf is a thin obstacle standing
out into the room. A door won't shut while someone's where it sweeps.

A cubicle is wide and deep enough for a player: step in, shut the door, and
you're out of sight; stand on the toilet (jump: its lid is above a step) and
your feet don't show under the door either.

The first door of each run starts open; the second is ENGAGED, locked for
good (someone's in there); the rest start shut. The state is in the World,
so the server will own it.
"""

import math
from dataclasses import dataclass

from doomer.shared.collision import Obstacle, ThinWall
from doomer.shared.mapfile import GameMap
from doomer.shared.things import (
    CUBICLE_DEPTH,
    CUBICLE_RUNS,
    CUBICLE_WIDTH,
    PARTITION_HALF,
    RETURN_WIDTH,
)

DOOR_SECONDS = 0.3  # shut to open
OPEN_ANGLE = math.pi / 2  # outwards, at full open
PASSABLE = 0.5  # past this far open, the doorway's clear and the leaf is in the room
DOOR_HALF = 0.006  # a door leaf's half-thickness
LEAF_TOP = 0.76
USE_REACH = 0.9  # cells from the door's middle
USE_CONE = math.cos(math.radians(45))  # must be roughly facing it


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

    @property
    def hinge_y(self) -> float:
        """The hinge, on the run's y axis (its x is CUBICLE_DEPTH); the
        door runs from it towards -y."""
        half = self.count * CUBICLE_WIDTH / 2
        return half - self.index * CUBICLE_WIDTH - PARTITION_HALF - RETURN_WIDTH

    @property
    def width(self) -> float:
        return CUBICLE_WIDTH - 2 * (PARTITION_HALF + RETURN_WIDTH)

    def to_world(self, mx: float, my: float) -> tuple[float, float]:
        """A point in the run's frame, in the world."""
        c, s = math.cos(self.run_angle), math.sin(self.run_angle)
        return self.run_x + mx * c - my * s, self.run_y + mx * s + my * c

    def centre(self) -> tuple[float, float]:
        """The middle of the doorway."""
        return self.to_world(CUBICLE_DEPTH, self.hinge_y - self.width / 2)

    def _box(self, x0: float, x1: float, y0: float, y1: float) -> tuple[float, ...]:
        """A rectangle in the run's frame, in the world (runs face a quarter
        turn, so it stays a rectangle): (x0, x1, y0, y1)."""
        (ax, ay), (bx, by) = self.to_world(x0, y0), self.to_world(x1, y1)
        return min(ax, bx), max(ax, bx), min(ay, by), max(ay, by)

    def sweep(self) -> tuple[float, ...]:
        """Where the leaf swings: the square in front of the doorway."""
        return self._box(
            CUBICLE_DEPTH, CUBICLE_DEPTH + self.width, self.hinge_y - self.width, self.hinge_y
        )


def cubicle_doors_from_map(game_map: GameMap) -> list[CubicleDoor]:
    out: list[CubicleDoor] = []
    for t in game_map.things:
        count = CUBICLE_RUNS.get(t.type)
        if count is None:
            continue
        for k in range(count):
            door = CubicleDoor(len(out), t.x, t.y, t.angle, k, count, locked=k == 1)
            if k == 0:
                door.open, door.opening = 1.0, True
            out.append(door)
    return out


def cubicle_door_walls(doors: list[CubicleDoor]) -> list[ThinWall]:
    """The shut doors (less than PASSABLE open), as thin walls."""
    walls = []
    for door in doors:
        if door.open >= PASSABLE:
            continue
        x0, x1, y0, y1 = door._box(
            CUBICLE_DEPTH, CUBICLE_DEPTH, door.hinge_y - door.width, door.hinge_y
        )
        if x1 - x0 < y1 - y0:  # the front's a line x = const
            walls.append(ThinWall(0, x0, y0, y1, DOOR_HALF))
        else:
            walls.append(ThinWall(1, y0, x0, x1, DOOR_HALF))
    return walls


def cubicle_door_leaves(doors: list[CubicleDoor]) -> list[Obstacle]:
    """The open doors' leaves (PASSABLE or more open), standing out into the
    room from their hinges, as thin obstacles."""
    out = []
    for door in doors:
        if door.open < PASSABLE:
            continue
        x0, x1, y0, y1 = door._box(
            CUBICLE_DEPTH,
            CUBICLE_DEPTH + door.width,
            door.hinge_y - DOOR_HALF,
            door.hinge_y + DOOR_HALF,
        )
        out.append(Obstacle((x0 + x1) / 2, (y0 + y1) / 2, (x1 - x0) / 2, (y1 - y0) / 2, LEAF_TOP))
    return out


def cubicle_door_in_reach(
    doors: list[CubicleDoor], x: float, y: float, angle: float
) -> CubicleDoor | None:
    """The cubicle door a player at (x, y) facing `angle` is using: the
    nearest within reach that they're roughly facing (from in or out)."""
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


def update_cubicle_doors(
    doors: list[CubicleDoor], dt: float, positions: list[tuple[float, float]], radius: float
) -> None:
    """Swing each door towards open or shut. A shutting door waits while a
    player's box is where it sweeps (in the doorway, or in front of it), so
    it never shuts through someone. An opening one doesn't wait: whoever
    it reaches can walk out of its leaf."""
    step = dt / DOOR_SECONDS
    for door in doors:
        target = 1.0 if door.opening else 0.0
        if door.open == target:
            continue
        if door.open > target:
            x0, x1, y0, y1 = door.sweep()
            if any(
                x0 - radius < px < x1 + radius and y0 - radius < py < y1 + radius
                for px, py in positions
            ):
                continue
            door.open = max(target, door.open - step)
        else:
            door.open = min(target, door.open + step)

"""Doors: panels in the middle of a doorway cell that open and close.

A door is a map thing in a doorway: an open cell between two wall cells. Its
angle says which way it faces: across the doorway, the way a hinged door
swings open (sliding doors don't care which way round). The door stands in
the plane through the cell's middle, across that direction; walls either side
hold it, and a lowered ceiling over the doorway (6/8) makes a lintel.

Kinds:
- hinged: one leaf, `HINGED_LEAF` wide, on a hinge at one side of the
  doorway, with fixed jambs filling the rest. Press use near it to open or
  close. It won't close on a player standing in the doorway.
- double: two leaves, each hinged at its own side of the doorway, meeting
  in the middle, both swinging the same way; narrow jambs. Use, as hinged.
- Any of them can be locked (`lock=blue` in the map): it opens only for a
  team holding that ID pass (doomer/shared/pickups.py).
- sliding: two glass leaves, each half the cell, that part into the walls
  either side. Automatic: they open while any player is near.

State is here, in the simulation, so the server owns it and every client
agrees; the client draws doors from it (doomer/client/boxes.py). Collision:
each door presents thin walls (collision.ThinWall) for whatever of the
doorway is still closed.
"""

import math
from collections.abc import Collection
from dataclasses import dataclass

from doomer.shared.collision import ThinWall
from doomer.shared.mapfile import GameMap

HINGED, DOUBLE, SLIDING = "hinged", "double", "sliding"
DOOR_THING_KINDS = {"door_hinged": HINGED, "door_double": DOUBLE, "door_sliding": SLIDING}

HINGED_LEAF = 0.6  # cells (~1.6 m: the player is 0.5 wide)
JAMB = (1.0 - HINGED_LEAF) / 2  # fixed panel either side of a hinged leaf
DOUBLE_JAMB = 0.04  # a double door's frame either side
DOUBLE_LEAF = (1.0 - 2 * DOUBLE_JAMB) / 2  # each of its two leaves
DOOR_HALF = 0.01  # half-thickness, cells (~5 cm doors)
DOOR_HEIGHT = 0.75  # walls (~2 m): up to the lintel (a doorway's ceiling is 6/8)
SWING = math.pi / 2  # a hinged door opens a quarter turn

OPEN_SECONDS = {HINGED: 0.6, DOUBLE: 0.6, SLIDING: 0.5}  # from shut to fully open
USE_REACH = 1.3  # cells from the door's centre
USE_CONE = math.cos(math.radians(50))  # must be roughly facing it
SLIDING_SENSE = 1.6  # cells: sliding doors open for anyone this near
PASSABLE = 0.95  # open beyond this, a hinged door no longer blocks


@dataclass(slots=True)
class Door:
    id: int
    kind: str
    x: float  # centre of the doorway cell
    y: float
    facing: float  # radians: across the doorway, the way a hinged door opens
    open: float = 0.0  # 0 shut .. 1 open
    opening: bool = False  # moving towards open (else towards shut)
    lock: str | None = None  # the pass colour that opens it, if locked

    def unlocked_by(self, passes: Collection[str]) -> bool:
        return self.lock is None or self.lock in passes

    @property
    def across_x(self) -> bool:
        """Does the doorway run across x (the door stands in a plane x = const)?
        True when it faces east or west."""
        return abs(math.cos(self.facing)) > 0.5


def doors_from_map(game_map: GameMap) -> list[Door]:
    """The map's doors, all shut, facing snapped to the nearest compass point."""
    out: list[Door] = []
    for t in game_map.things:
        kind = DOOR_THING_KINDS.get(t.type)
        if kind is None:
            continue
        cx, cy = math.floor(t.x), math.floor(t.y)
        facing = round(t.angle / (math.pi / 2)) % 4 * (math.pi / 2)
        out.append(Door(len(out), kind, cx + 0.5, cy + 0.5, facing, lock=t.prop("lock")))
    return out


def door_walls(door: Door) -> list[ThinWall]:
    """What of the doorway is closed, as thin walls for collision."""
    axis = 0 if door.across_x else 1  # ThinWall axis: the plane x = at (0) or y = at (1)
    at = door.x if door.across_x else door.y
    lo = (door.y if door.across_x else door.x) - 0.5  # the doorway's extent along the plane
    hi = lo + 1.0

    def wall(a: float, b: float) -> ThinWall:
        return ThinWall(axis, at, a, b, DOOR_HALF)

    if door.kind in (HINGED, DOUBLE):
        jamb = JAMB if door.kind == HINGED else DOUBLE_JAMB
        walls = [wall(lo, lo + jamb), wall(hi - jamb, hi)]  # jambs never move
        if door.open < PASSABLE:
            walls.append(wall(lo + jamb, hi - jamb))
        return walls
    # Sliding: each leaf covers its half, less how far it has slid into the wall.
    closed = 0.5 * (1.0 - door.open)
    if closed <= 1e-6:
        return []
    return [wall(lo, lo + closed), wall(hi - closed, hi)]


def in_doorway(door: Door, x: float, y: float, radius: float) -> bool:
    """Is a box of half-width `radius` at (x, y) overlapping the doorway cell?"""
    return abs(x - door.x) < 0.5 + radius and abs(y - door.y) < 0.5 + radius


def try_use(
    doors: list[Door], x: float, y: float, angle: float, passes: Collection[str]
) -> tuple[Door, bool] | None:
    """The door a player at (x, y) facing `angle` is using: the nearest within
    reach, roughly in front of them; and whether they could. A hinged or
    double door the team can open is toggled. A locked one stays as it is
    (False: say which pass it needs). Sliding doors open by themselves, so
    using one only matters when it's locked."""
    best: Door | None = None
    best_d = USE_REACH
    for door in doors:
        if door.kind == SLIDING and door.unlocked_by(passes):
            continue
        dx, dy = door.x - x, door.y - y
        d = math.hypot(dx, dy)
        if d > best_d or d < 1e-9:
            continue
        if (dx * math.cos(angle) + dy * math.sin(angle)) / d < USE_CONE:
            continue
        best, best_d = door, d
    if best is None:
        return None
    if not best.unlocked_by(passes):
        return best, False
    best.opening = not best.opening
    return best, True


def update_doors(
    doors: list[Door],
    players: list[tuple[float, float]],
    radius: float,
    dt: float,
    passes: Collection[str] = (),
) -> None:
    """Advance every door one tick. Sliding doors open for anyone near (if
    the team can open them); a door that's closing stops (and reopens)
    rather than shut on someone."""
    for door in doors:
        if door.kind == SLIDING:
            door.opening = door.unlocked_by(passes) and any(
                math.hypot(px - door.x, py - door.y) < SLIDING_SENSE for px, py in players
            )
        elif (
            not door.opening
            and door.open > 0.0
            and any(in_doorway(door, px, py, radius) for px, py in players)
        ):
            door.opening = True  # don't shut on someone: open again
        step = dt / OPEN_SECONDS[door.kind]
        door.open = min(1.0, door.open + step) if door.opening else max(0.0, door.open - step)

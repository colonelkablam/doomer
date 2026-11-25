"""Doors: panels at one end of a doorway cell that open and close.

A door is a map thing in a doorway: an open cell between two wall cells. Its
angle says which way it faces: across the doorway, the way it opens. It
stands at one end of the doorway, 0.05 cells in from the edge (DOOR_SET from
the cell's middle), flush with the room it serves; the rest of the doorway
is a recess on the other side (a corridor's):
- hinged and double doors stand at the end away from their swing, and swing
  out into their own recess (a leaf, 0.6, is shorter than the recess, 0.95),
  so the swing never reaches the corridor itself or the room's furniture.
  Face them towards the corridor. Someone standing in the swing as it opens
  is nudged back out of it (clear_swing), as you step back pulling a door.
- sliding and lift doors stand at the end they face, flush with that room
  (the office, the lift lobby); their leaves slide inside the walls either
  side rather than over the room's wall face.
Walls either side hold it, and a lowered ceiling over the doorway (6/8) makes
a lintel: its face over the door is in the door's plane.

Kinds:
- hinged: one leaf, `HINGED_LEAF` wide, on a hinge at one side of the
  doorway, with fixed jambs filling the rest. Press use near it to open or
  close. It won't close on a player standing in the doorway.
- double: two leaves, each hinged at its own side of the doorway, meeting
  in the middle, both swinging the same way; narrow jambs. Use, as hinged.
- Any of them can be locked (`lock=blue` in the map): it opens only for a
  team holding that ID pass (doomer/shared/pickups.py).
- lift: two brushed steel leaves, parting like sliding doors: the outer
  wall of a lift car, which is the doorway cell itself (one cell off the
  lobby, walled in behind, its doors facing out). Use
  calls the lift, only with power; then the doors open for anyone near.
  When every player is in a lift car, the doors close and the floor is done.
- sliding: two glass leaves, each half the cell, that part towards the
  walls either side, stopping with a strip of each still showing (SLIDE).
  Automatic: they open while any player is near.

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

HINGED, DOUBLE, SLIDING, LIFT = "hinged", "double", "sliding", "lift"
DOOR_THING_KINDS = {
    "door_hinged": HINGED,
    "door_double": DOUBLE,
    "door_sliding": SLIDING,
    "door_lift": LIFT,
}
PARTING = (SLIDING, LIFT)  # two leaves that part into the walls

HINGED_LEAF = 0.6  # cells (~1.6 m: the player is 0.5 wide)
JAMB = (1.0 - HINGED_LEAF) / 2  # fixed panel either side of a hinged leaf
DOUBLE_JAMB = 0.04  # a double door's frame either side
DOUBLE_LEAF = (1.0 - 2 * DOUBLE_JAMB) / 2  # each of its two leaves
DOOR_HALF = 0.01  # half-thickness, cells (~5 cm doors)
DOOR_HEIGHT = 0.75  # walls (~2 m): up to the lintel (a doorway's ceiling is 6/8)
SWING = math.pi / 2  # a hinged door opens a quarter turn

OPEN_SECONDS = {HINGED: 0.6, DOUBLE: 0.6, SLIDING: 0.5, LIFT: 0.8}  # from shut to fully open
USE_REACH = 1.3  # cells from the door's middle (on its plane)
USE_CONE = math.cos(math.radians(50))  # must be roughly facing it
SLIDING_SENSE = 1.6  # cells: sliding doors open for anyone this near
PASSABLE = 0.95  # open beyond this, a hinged door no longer blocks
DOOR_SET = 0.45  # cells from the doorway's middle to the door, towards its facing
# How far each leaf of a parting door slides when fully open (cells; each
# leaf is half the doorway, 0.5). Glass doors stop with ~32 cm of each leaf
# still showing, as real ones do (they park behind fixed side panels), a 2 m
# opening; lift doors vanish into the frame.
SLIDE = {SLIDING: 0.38, LIFT: 0.5}


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
    called: bool = False  # a lift that's been called (it then opens for anyone near)

    def unlocked_by(self, passes: Collection[str]) -> bool:
        return self.lock is None or self.lock in passes

    @property
    def plane(self) -> float:
        """Where the door stands across its doorway: x (if across_x) or y of
        its plane, DOOR_SET from the doorway's middle: towards its facing
        (sliding, lift), or away from it (hinged, double: they swing out
        into the doorway)."""
        toward = -1 if self.kind in (HINGED, DOUBLE) else 1
        if self.across_x:
            return self.x + toward * DOOR_SET * round(math.cos(self.facing))
        return self.y + toward * DOOR_SET * round(math.sin(self.facing))

    @property
    def centre(self) -> tuple[float, float]:
        """The middle of the door itself (on its plane)."""
        return (self.plane, self.y) if self.across_x else (self.x, self.plane)

    @property
    def car(self) -> tuple[int, int]:
        """For a lift door: its car, the doorway cell itself. A lift is one
        cell off its lobby, its doors the car's outer wall (at the end they
        face: the facing points out of the lift)."""
        return math.floor(self.x), math.floor(self.y)

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
    at = door.plane
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
    # Sliding and lift doors: each leaf covers its half, less how far it has slid.
    closed = 0.5 - SLIDE[door.kind] * door.open
    if closed <= 1e-6:
        return []
    return [wall(lo, lo + closed), wall(hi - closed, hi)]


SWING_REACH = 0.65  # cells: how far a hinged leaf sweeps into the room (0.6 leaf)


def _swing_frame(door: Door, x: float, y: float) -> tuple[float, float, int]:
    """(how far from the door's plane towards its swing, how far along the
    doorway from its middle, +1/-1: the swing's direction along the axis)."""
    across, along = (x, y) if door.across_x else (y, x)
    middle = door.y if door.across_x else door.x
    side = round(math.cos(door.facing)) if door.across_x else round(math.sin(door.facing))
    return (across - door.plane) * side, along - middle, side


def in_doorway(door: Door, x: float, y: float, radius: float) -> bool:
    """Is a box of half-width `radius` at (x, y) where a hinged door's leaves
    swing: from its plane out into its recess, across the doorway?"""
    into, along, _ = _swing_frame(door, x, y)
    return -radius - 0.05 < into < SWING_REACH + radius and abs(along) < 0.5 + radius


def clear_swing(door: Door, x: float, y: float, radius: float) -> tuple[float, float]:
    """Where a player at (x, y) must be to be clear of a hinged or double
    door's leaves as they are now: nudged straight back out of their sweep,
    or where they are. The sweep is as deep as a leaf has turned (leaf x
    sin(angle)), across the whole doorway; the recess is deeper than a
    leaf plus a player, so it never pushes anyone into a wall."""
    if door.kind not in (HINGED, DOUBLE) or door.open <= 0.0:
        return x, y
    into, along, side = _swing_frame(door, x, y)
    leaf = HINGED_LEAF if door.kind == HINGED else DOUBLE_LEAF
    depth = leaf * math.sin(door.open * SWING)
    if not (0.0 <= into < depth + radius) or abs(along) >= 0.5 + radius:
        return x, y
    push = (depth + radius - into) * side
    return (x + push, y) if door.across_x else (x, y + push)


def try_use(
    doors: list[Door],
    x: float,
    y: float,
    angle: float,
    passes: Collection[str],
    power: bool = True,
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
            continue  # opens by itself
        cx, cy = door.centre  # the door itself, at the end of its doorway
        dx, dy = cx - x, cy - y
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
    if best.kind == LIFT:  # the call button: works only with power
        if not power:
            return best, False
        best.called = best.opening = True
        return best, True
    best.opening = not best.opening
    return best, True


def update_doors(
    doors: list[Door],
    players: list[tuple[float, float]],
    radius: float,
    dt: float,
    passes: Collection[str] = (),
    power: bool = True,
    departing: bool = False,
) -> None:
    """Advance every door one tick. Sliding doors open for anyone near (if
    the team can open them); called lifts too, while there's power, until
    everyone's aboard (`departing`: then they close). A hinged door that's
    closing stops (and reopens) rather than shut on someone."""
    for door in doors:
        if door.kind == LIFT:
            door.opening = (
                door.called
                and power
                and not departing
                # From the doorway's middle, not the doors: that covers the car,
                # so they stay open for someone at the back of it.
                and any(math.hypot(px - door.x, py - door.y) < SLIDING_SENSE for px, py in players)
            )
        elif door.kind == SLIDING:
            door.opening = door.unlocked_by(passes) and any(
                math.dist((px, py), door.centre) < SLIDING_SENSE for px, py in players
            )
        elif (
            not door.opening
            and door.open > 0.0
            and any(in_doorway(door, px, py, radius) for px, py in players)
        ):
            door.opening = True  # don't shut on someone: open again
        step = dt / OPEN_SECONDS[door.kind]
        door.open = min(1.0, door.open + step) if door.opening else max(0.0, door.open - step)

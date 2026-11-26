"""Doors: panels at one end of a doorway cell that open and close.

A door is a map thing in a doorway: an open cell between two wall cells. Its
angle says which way it faces: across the doorway, the way it opens. It
stands at one end of the doorway, 0.05 cells in from the edge (DOOR_SET from
the cell's middle), flush with the room it serves; the rest of the doorway
is a recess on the other side (a corridor's):
- hinged and double doors stand at the end away from their swing, and swing
  out into their own recess (a leaf, 0.6, is shorter than the recess, 0.95),
  so the swing never reaches the corridor itself or the room's furniture.
  Face them towards the corridor. A leaf swings through anyone standing in
  its way (it doesn't push them: that made doors one-way); it just won't
  shut on them.
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
  team holding that ID pass (ok_doomer/shared/pickups.py).
- lift: two brushed steel leaves, parting like sliding doors: the outer
  wall of a lift car, which is the doorway cell itself (one cell off the
  lobby, walled in behind, its doors facing out). The car is on some floor
  of the building (`car=B3` in the map; ok_doomer/shared/building.py): use
  at the doors (the call button) and, with power, it comes, a floor every
  FLOOR_SECONDS, the indicator over the doors counting, and opens when it
  arrives. Called while it's here, it opens. Open, it stays open: inside,
  use (the car's panel) shuts the doors, or opens them again. Shutting
  doors open again for anyone standing in them, as real ones do. When every
  player is in a car with its doors shut, the ride starts (ok_doomer/shared/
  sim.py) and the next floor loads. A floor that starts with the team in a
  car (the coop start in it) is an arrival: that car's here, and its doors
  open after ARRIVE_SECONDS. A locked lift (`lock=gold`) needs its pass to
  be called or sent: an executive floor's keyswitch.
- sliding: two glass leaves, each half the cell, that part towards the
  walls either side, stopping with a strip of each still showing (SLIDE).
  Automatic: they open while any player is near.

State is here, in the simulation, so the server owns it and every client
agrees; the client draws doors from it (ok_doomer/client/boxes.py). Collision:
each door presents thin walls (collision.ThinWall) for whatever of the
doorway is still closed.
"""

import math
from collections.abc import Collection
from dataclasses import dataclass

from ok_doomer.shared.building import floor_index
from ok_doomer.shared.collision import ThinWall
from ok_doomer.shared.mapfile import GameMap

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
# Lift doors sit nearer the edge: ~3 cm behind the lobby wall's face, so
# they read as flush with it (a steel architrave covers the gap), their
# leaves still parking inside the car's walls either side, which hide them.
LIFT_SET = 0.47
# How far each leaf of a parting door slides when fully open (cells; each
# leaf is half the doorway, 0.5): they stop with ~32 cm of each leaf still
# showing, as real ones park behind fixed side panels; a 2 m opening.
SLIDE = {SLIDING: 0.38, LIFT: 0.38}
FLOOR_SECONDS = 1.2  # a lift car, from one floor to the next
ARRIVE_SECONDS = 1.0  # an arriving car waits this long before it opens


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
    # Lifts: where the car is (a floor index, between two while moving),
    # this floor's index, whether it's been called here, and how long until
    # an arriving car opens (0: not arriving).
    car_floor: float = 0.0
    here: int = 0
    called: bool = False
    arrive_in: float = 0.0

    @property
    def car_here(self) -> bool:
        return self.car_floor == self.here

    def unlocked_by(self, passes: Collection[str]) -> bool:
        return self.lock is None or self.lock in passes

    @property
    def plane(self) -> float:
        """Where the door stands across its doorway: x (if across_x) or y of
        its plane, DOOR_SET from the doorway's middle: towards its facing
        (sliding, lift), or away from it (hinged, double: they swing out
        into the doorway)."""
        set_ = (-1 if self.kind in (HINGED, DOUBLE) else 1) * self.set_back
        if self.across_x:
            return self.x + set_ * round(math.cos(self.facing))
        return self.y + set_ * round(math.sin(self.facing))

    @property
    def set_back(self) -> float:
        """How far the door stands from its doorway's middle (DOOR_SET; lifts
        LIFT_SET)."""
        return LIFT_SET if self.kind == LIFT else DOOR_SET

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
    """The map's doors, all shut, facing snapped to the nearest compass point;
    lifts with their cars where the map puts them."""
    out: list[Door] = []
    starts = {(math.floor(t.x), math.floor(t.y)) for t in game_map.things_of_type("coop_start")}
    for t in game_map.things:
        kind = DOOR_THING_KINDS.get(t.type)
        if kind is None:
            continue
        cx, cy = math.floor(t.x), math.floor(t.y)
        facing = round(t.angle / (math.pi / 2)) % 4 * (math.pi / 2)
        door = Door(len(out), kind, cx + 0.5, cy + 0.5, facing, lock=t.prop("lock"))
        if kind == LIFT:
            door.here = floor_index(game_map.storey)
            car = t.prop("car")
            door.car_floor = float(floor_index(car) if car else door.here)
            if (cx, cy) in starts:  # the team arrives in this car
                door.car_floor, door.arrive_in = float(door.here), ARRIVE_SECONDS
        out.append(door)
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


def try_use(
    doors: list[Door],
    x: float,
    y: float,
    angle: float,
    passes: Collection[str],
    power: bool = True,
) -> tuple[Door, bool] | None:
    """The door a player at (x, y) facing `angle` is using, and whether they
    could. Standing in a lift's car, it's that lift's panel, whichever way
    they face. Otherwise the nearest door within reach, roughly in front of
    them. A hinged or double door the team can open is toggled; a lift is
    called (or, if its car's here, opened), or from inside, its doors shut
    or opened. A locked one stays as it is (False: say which pass it
    needs), as does a lift without power. Sliding doors open by themselves,
    so using one only matters when it's locked."""
    cell = (math.floor(x), math.floor(y))
    inside = next((d for d in doors if d.kind == LIFT and d.car == cell), None)
    best: Door | None = inside
    best_d = USE_REACH
    for door in doors if inside is None else ():
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
    if best.kind == LIFT:
        if not power:
            return best, False
        if inside is not None:  # the car's panel: shut the doors, or open them
            best.opening = not best.opening
        elif best.car_here:  # the call button, the car already here: open
            best.opening = True
        else:  # the call button: it comes
            best.called = True
        return best, True
    best.opening = not best.opening
    return best, True


def in_lift_doorway(door: Door, x: float, y: float, radius: float) -> bool:
    """Is a box at (x, y) where a lift's doors close: across their plane?"""
    across = (x - door.plane) if door.across_x else (y - door.plane)
    along = (y - door.y) if door.across_x else (x - door.x)
    return abs(across) < radius + DOOR_HALF and abs(along) < 0.5 + radius


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
    the team can open them). A called lift's car travels towards this floor
    (only with power) and opens when it gets here; an arriving one opens
    after its wait; lift doors that are shutting open again for anyone in
    them, unless the car's leaving (`departing`: then they stay shut). A
    hinged door that's closing stops (and reopens) rather than shut on
    someone."""
    for door in doors:
        if door.kind == LIFT:
            if door.called and power and not departing:
                step_ = dt / FLOOR_SECONDS
                if door.car_floor < door.here:
                    door.car_floor = min(float(door.here), door.car_floor + step_)
                else:
                    door.car_floor = max(float(door.here), door.car_floor - step_)
                if door.car_here:
                    door.called, door.opening = False, True
            if door.arrive_in > 0.0:
                door.arrive_in = max(0.0, door.arrive_in - dt)
                if door.arrive_in == 0.0:
                    door.opening = True
            if departing:
                door.opening = False
            elif (
                not door.opening
                and door.open > 0.0
                and any(in_lift_doorway(door, px, py, radius) for px, py in players)
            ):
                door.opening = True  # the safety edge: open again
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

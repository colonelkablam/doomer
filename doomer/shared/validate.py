"""The map checker: can a co-op floor be finished?

From the co-op start, flood the space a player can walk to, by the
simulation's own rules (it calls the same collision and height tests): walls
and furniture block, rises up to a step are climbed, a jump gets you onto a
desk but not a counter, balcony railings stop you walking off, cubicle
screens and door leaves are thin walls. A door is open only if the team can
open it: unlocked, or its pass already in hand; a lift's doors only with
power. Passes within reach of where you can stand are taken, and the breaker
thrown, which can open more doors: flood again, until nothing changes.

Then the floor is completable if every locked door's pass is reachable (so
the door is), the breaker (if any) is reachable, and you can get into a lift
car with the power on, or to an exit. Each problem comes back as a sentence.

Standard library only: the server, tools and the level generator use it.
"""

import math
from collections import deque
from collections.abc import Iterable
from dataclasses import dataclass, replace

from doomer.shared.collision import Obstacle, ThinWall, move_box
from doomer.shared.config import MovementConfig, SimConfig
from doomer.shared.doors import LIFT, Door, door_walls
from doomer.shared.mapfile import GameMap
from doomer.shared.pickups import HEIGHT_REACH, REACH, Pickup
from doomer.shared.player import PlayerState
from doomer.shared.screens import SCREEN_HALF, screens_at_height
from doomer.shared.sim import _drops_away, cell_blocks, on_railed_floor, support_height
from doomer.shared.switches import POWER, USE_REACH, Switch
from doomer.shared.things import EXIT
from doomer.shared.world import ModeNotSupported, World, new_world

SPACING = 0.25  # cells between the positions sampled: a quarter cell
_NEIGHBOURS = [(dx, dy) for dx in (-1, 0, 1) for dy in (-1, 0, 1) if (dx, dy) != (0, 0)]
_EPS = 1e-6


@dataclass(frozen=True, slots=True)
class Report:
    """What the checker found. `ok` when there are no problems."""

    problems: tuple[str, ...]
    passes: frozenset[str]  # the passes the team can get
    power: bool  # whether the power can be on
    cells: frozenset[tuple[int, int]]  # the cells a player can stand in

    @property
    def ok(self) -> bool:
        return not self.problems


@dataclass(slots=True)
class _State:
    passes: set[str]
    power: bool


def check_floor(game_map: GameMap, config: SimConfig | None = None) -> Report:
    """Check a co-op floor: see the module docstring."""
    try:
        world = new_world(game_map, "coop", config)
    except ModeNotSupported as e:
        return Report((str(e),), frozenset(), False, frozenset())
    start = game_map.things_of_type("coop_start")[0]
    state = _State(set(), POWER in world.flags)
    while True:
        spots = _reachable(world, state, start.x, start.y)
        passes = {p.pass_colour for p in world.pickups if p.pass_colour and _within(p, spots)}
        power = state.power or any(_can_throw(sw, spots) for sw in world.switches)
        if passes <= state.passes and power == state.power:
            break
        state = _State(state.passes | passes, power)
    problems = _problems(world, state, spots, start.x, start.y)
    cells = frozenset((math.floor(x), math.floor(y)) for x, y, _ in spots)
    return Report(tuple(problems), frozenset(state.passes), state.power, cells)


def _problems(
    world: World, state: _State, spots: list[tuple[float, float, float]], sx: float, sy: float
) -> list[str]:
    out = []
    for door in world.doors:
        colour = door.lock
        if colour is None or colour in state.passes:
            continue
        where = f"({door.x:g}, {door.y:g})"
        holders = [p for p in world.pickups if p.pass_colour == colour]
        if not holders:
            out.append(f"the {colour}-locked door at {where} can't open: no {colour} pass")
            continue
        # Would the pass be reachable if its own door were open? Then it's
        # locked in behind the door it opens.
        unlocked = _State(state.passes | {colour}, state.power)
        if any(_within(p, _reachable(world, unlocked, sx, sy)) for p in holders):
            out.append(
                f"the {colour} pass is only reachable through the {colour}-locked door at {where}"
            )
        else:
            p = holders[0]
            out.append(f"the {colour} pass at ({p.x:g}, {p.y:g}) can't be reached")
    for sw in world.switches:
        if not _can_throw(sw, spots):
            out.append(f"the {sw.type} at ({sw.x:g}, {sw.y:g}) can't be reached")
    cars = {door.car for door in world.doors if door.kind == LIFT}
    exits = {(math.floor(t.x), math.floor(t.y)) for t in world.map.things_of_type(EXIT.name)}
    standing = {(math.floor(x), math.floor(y)) for x, y, _ in spots}
    if not (state.power and standing & cars) and not standing & exits:
        if cars and not state.power:
            out.append("no way out: the lifts need power, and it can't be restored")
        else:
            out.append("no way out: no lift car or exit can be reached")
    return out


def _within(p: Pickup, spots: Iterable[tuple[float, float, float]]) -> bool:
    """Can a player standing at one of `spots` take pickup `p`?"""
    return any(
        math.hypot(p.x - x, p.y - y) <= REACH and abs(p.z - z) <= HEIGHT_REACH for x, y, z in spots
    )


def _can_throw(sw: Switch, spots: Iterable[tuple[float, float, float]]) -> bool:
    """Can a player standing at one of `spots` throw switch `sw` (within reach:
    they can always turn to face it)?"""
    return any(_EPS < math.hypot(sw.x - x, sw.y - y) <= USE_REACH for x, y, _ in spots)


def _reachable(
    world: World, state: _State, sx: float, sy: float
) -> list[tuple[float, float, float]]:
    """The positions (x, y, feet z) a player can get to from (sx, sy), on a
    grid SPACING apart anchored there, given what the team has."""
    move = world.config.movement
    doors = [replace(d, open=1.0 if _opens(d, state) else 0.0) for d in world.doors]
    door_panels = [wall for door in doors for wall in door_walls(door)]
    game_map, obstacles = world.map, world.obstacles

    def position(i: int, j: int) -> tuple[float, float]:
        return sx + i * SPACING, sy + j * SPACING

    start_z = support_height(game_map, sx, sy, move.radius)
    seen = {(0, 0): start_z}
    queue = deque([(0, 0)])
    while queue:
        i, j = queue.popleft()
        x, y = position(i, j)
        z = seen[(i, j)]
        for di, dj in _NEIGHBOURS:
            key = (i + di, j + dj)
            if key in seen:
                continue
            tx, ty = position(*key)
            arrived = _step(game_map, obstacles, door_panels, world, move, x, y, z, tx, ty)
            if arrived is not None:
                seen[key] = arrived
                queue.append(key)
    return [(*position(i, j), z) for (i, j), z in seen.items()]


def _opens(door: Door, state: _State) -> bool:
    if not door.unlocked_by(state.passes):
        return False
    return door.kind != LIFT or state.power


def _step(
    game_map: GameMap,
    obstacles: tuple[Obstacle, ...],
    door_panels: list[ThinWall],
    world: World,
    move: MovementConfig,
    x: float,
    y: float,
    z: float,
    tx: float,
    ty: float,
) -> float | None:
    """Can a player standing at (x, y), feet at z, walk (or jump) to (tx, ty)?
    Their feet height there, or None."""
    for feet in (z, z + move.jump_apex):  # walking, then jumping
        player = PlayerState(0, "", x, y, 0.0, z=feet)
        railed = on_railed_floor(game_map, player, move.radius)
        walls = door_panels + [
            ThinWall(sc.axis, sc.at, sc.lo, sc.hi, SCREEN_HALF)
            for sc in screens_at_height(world.screens, feet)
        ]

        def blocked(
            cx: int, cy: int, f: float = feet, p: PlayerState = player, r: bool = railed
        ) -> bool:
            if cell_blocks(game_map, cx, cy, f, move.height, move.step_height, x, y):
                return True
            return r and _drops_away(game_map, cx, cy, p, move.step_height)

        nx, ny = move_box(game_map, x, y, tx - x, ty - y, move.radius, obstacles, blocked, walls)
        if abs(nx - tx) < _EPS and abs(ny - ty) < _EPS:
            return support_height(game_map, tx, ty, move.radius)
    return None


def main(argv: list[str] | None = None) -> int:
    """`python -m doomer.shared.validate maps/office.map ...`: check floors,
    say what's wrong; exit status 1 if any can't be finished."""
    import argparse

    from doomer.shared.mapfile import load_map

    parser = argparse.ArgumentParser(description="Check co-op floors can be finished.")
    parser.add_argument("maps", nargs="+", help="map files")
    args = parser.parse_args(argv)
    failed = 0
    for path in args.maps:
        report = check_floor(load_map(path))
        if report.ok:
            print(f"{path}: OK (passes: {', '.join(sorted(report.passes)) or 'none'})")
        else:
            failed += 1
            print(f"{path}: can't be finished")
            for problem in report.problems:
                print(f"  - {problem}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())

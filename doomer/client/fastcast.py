"""The renderer's ray casting, compiled with numba: every column of a frame in
one call, straight into arrays.

doomer/shared/raycast.py is the reference: the same DDA, written for clarity,
returning a list of RayHit per ray (the server uses it too, so it stays pure
Python). Called once per screen column, it cost the renderer ~40% of a frame,
nearly all of it interpreter overhead (a few dozen bytecodes per grid step,
an object per hit). Here the same algorithm, line for line, is compiled to
machine code by numba on first use (cached on disk after that), and writes
struct-of-arrays output the numpy passes read directly. tests/test_fastcast.py
checks it agrees with cast_ray exactly, ray for ray.

What it adds over cast_ray, which the renderer used to do per column in
Python:
- the camera's own space (standing on a desk, under a high ceiling) as up to
  two pseudo-spans from distance 0, before the ray's own hits;
- the full wall the ray stops at, taken out into per-column wall arrays;
- per span, the atlas slots of its faces and whether a block's back face
  shows (it's left into open floor, not straight into another span).

Set NUMBA_DISABLE_JIT=1 to run it as plain Python (slow, but debuggable).
"""

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, TypeVar, cast

import numba
import numpy as np
import numpy.typing as npt

from doomer.shared.mapfile import (
    DEFAULT_FACING,
    EAST,
    EMPTY,
    NORTH,
    SOUTH,
    STAIR_RISE,
    STAIRS_WALL_TYPE,
    WALL_UNITS,
    WEST,
    GameMap,
)

FloatArray = npt.NDArray[np.float64]
Int64Array = npt.NDArray[np.int64]

_F = TypeVar("_F", bound=Callable[..., Any])


def _jit(f: _F) -> _F:
    # error_model="numpy": float division by zero gives inf, as numpy does,
    # rather than raising (and the compiled code doesn't check for it).
    return cast(_F, numba.njit(cache=True, error_model="numpy")(f))


# Per-span fields: rows of the float and int arrays.
DISTANCE, U, TOP, BASE, EXIT_DISTANCE, EXIT_U, CEILING, PREV_CEILING, EXIT_TOP = range(9)
N_FLOAT = 9
(
    CELL_X,
    CELL_Y,
    WALL_TYPE,
    Y_SIDE,
    EXIT_Y_SIDE,
    FACE,
    EXIT_FACE,
    EXITS_TO_FLOOR,
    SLOT,
    BACK_SLOT,
    UPPER_SLOT,
) = range(11)
N_INT = 11
# Per-column fields of the full wall each ray ends at.
W_DIST, W_U, W_BASE, W_TOP = range(4)
W_SLOT, W_UPPER, W_Y_SIDE = range(3)

_EPS = 1e-9


@dataclass(frozen=True, slots=True)
class MapArrays:
    """A map's grids as numpy arrays, and each cell's atlas slots: [face 0-3,
    upper] (for a cell outside the map: `outside`). Built once per map."""

    walls: Int64Array
    floors: Int64Array
    ceilings: Int64Array
    facing: Int64Array
    slots: Int64Array  # (height, width, 5)
    outside: Int64Array  # (5,)
    bulkhead: int  # the upper slot over any open span (a dropped ceiling's face)


def map_arrays(
    game_map: GameMap, slots: Int64Array, outside: Int64Array, bulkhead: int
) -> MapArrays:
    def grid(rows: tuple[tuple[int, ...], ...]) -> Int64Array:
        return np.array(rows, dtype=np.int64).reshape(game_map.height, game_map.width)

    facing = (
        grid(game_map.facing)
        if game_map.facing
        else np.full((game_map.height, game_map.width), DEFAULT_FACING, dtype=np.int64)
    )
    return MapArrays(
        grid(game_map.walls),
        grid(game_map.floor),
        grid(game_map.ceiling),
        facing,
        slots,
        outside,
        bulkhead,
    )


@dataclass(frozen=True, slots=True)
class Columns:
    """One frame's rays. Span k of column c (nearest first, the camera's own
    space first of all) is at [field, k, c], for k < counts[c]."""

    floats: FloatArray  # (N_FLOAT, spans, columns)
    ints: Int64Array  # (N_INT, spans, columns)
    counts: Int64Array  # (columns,)
    wall_floats: FloatArray  # (4, columns): distance inf where no wall in range
    wall_ints: Int64Array  # (3, columns)

    @property
    def layers(self) -> int:
        return int(self.floats.shape[1])


def cast_columns(
    arrays: MapArrays, x: float, y: float, angles: FloatArray, max_distance: float = 64.0
) -> Columns:
    """Cast one ray per angle from (x, y)."""
    # A ray crosses at most ~1.42 grid lines per cell travelled, each giving
    # at most one span; plus the camera's two.
    capacity = int(max_distance * 1.5) + 4
    floats, ints, counts, wall_floats, wall_ints = _cast_columns(
        arrays.walls,
        arrays.floors,
        arrays.ceilings,
        arrays.facing,
        arrays.slots,
        arrays.outside,
        arrays.bulkhead,
        x,
        y,
        angles,
        max_distance,
        capacity,
    )
    k = int(np.max(counts, initial=0))
    return Columns(floats[:, :k], ints[:, :k], counts, wall_floats, wall_ints)


# --- compiled ------------------------------------------------------------------------
# Mirrors doomer/shared/raycast.py and GameMap's height lookups; keep them in step.


@_jit
def _in_map(walls: Int64Array, cx: int, cy: int) -> bool:
    return bool(0 <= cx < walls.shape[1] and 0 <= cy < walls.shape[0])


@_jit
def _is_low_block(
    walls: Int64Array, floors: Int64Array, ceilings: Int64Array, cx: int, cy: int
) -> bool:
    if not _in_map(walls, cx, cy) or walls[cy, cx] == EMPTY:
        return False
    return bool(0 < floors[cy, cx] < ceilings[cy, cx])


@_jit
def _floor_at(
    walls: Int64Array,
    floors: Int64Array,
    ceilings: Int64Array,
    facing: Int64Array,
    x: float,
    y: float,
) -> float:
    """GameMap.floor_at."""
    cx, cy = math.floor(x), math.floor(y)
    if not _in_map(walls, cx, cy):
        return math.inf
    units = int(floors[cy, cx])
    if walls[cy, cx] == EMPTY:
        return units / WALL_UNITS
    if not 0 < units < ceilings[cy, cx]:
        return math.inf
    if walls[cy, cx] != STAIRS_WALL_TYPE:
        return units / WALL_UNITS
    fx, fy = x - cx, y - cy
    f = int(facing[cy, cx])
    if f == 0:
        progress = 1.0 - fy
    elif f == 1:
        progress = fx
    elif f == 2:
        progress = fy
    else:
        progress = 1.0 - fx
    low = max(0, units - STAIR_RISE)
    return (low + (units - low) * progress) / WALL_UNITS


@_jit
def _floor_in_cell(
    walls: Int64Array,
    floors: Int64Array,
    ceilings: Int64Array,
    facing: Int64Array,
    x: float,
    y: float,
    cx: int,
    cy: int,
) -> float:
    return _floor_at(
        walls,
        floors,
        ceilings,
        facing,
        min(max(x, cx + _EPS), cx + 1 - _EPS),
        min(max(y, cy + _EPS), cy + 1 - _EPS),
    )


@_jit
def _entry_face(y_side: bool, dx: float, dy: float) -> int:
    if not y_side:
        return WEST if dx > 0 else EAST
    return NORTH if dy > 0 else SOUTH


@_jit
def _wall_u(x: float, y: float, dx: float, dy: float, distance: float, y_side: bool) -> float:
    if not y_side:
        u = (y + dy * distance) % 1.0
        return u if dx > 0 else 1.0 - u
    u = (x + dx * distance) % 1.0
    return 1.0 - u if dy > 0 else u


@_jit
def _cast_columns(
    walls: Int64Array,
    floors: Int64Array,
    ceilings: Int64Array,
    facing: Int64Array,
    slots: Int64Array,
    outside: Int64Array,
    bulkhead: int,
    x: float,
    y: float,
    angles: FloatArray,
    max_distance: float,
    capacity: int,
) -> tuple[FloatArray, Int64Array, Int64Array, FloatArray, Int64Array]:
    width, height = walls.shape[1], walls.shape[0]
    ncols = angles.shape[0]
    fl = np.empty((N_FLOAT, capacity, ncols))
    it = np.zeros((N_INT, capacity, ncols), dtype=np.int64)
    counts = np.zeros(ncols, dtype=np.int64)
    wf = np.empty((4, ncols))
    wi = np.zeros((3, ncols), dtype=np.int64)

    # The space the camera is in (raycast.start_heights), the same for every ray.
    start_x, start_y = math.floor(x), math.floor(y)
    on_block = _is_low_block(walls, floors, ceilings, start_x, start_y)
    if _in_map(walls, start_x, start_y):
        floor0 = _floor_at(walls, floors, ceilings, facing, x, y) if on_block else 0.0
        ceiling0 = ceilings[start_y, start_x]
    else:
        floor0, ceiling0 = 0.0, WALL_UNITS
    own_space = floor0 != 0.0 or ceiling0 != WALL_UNITS
    n_own = 0
    if own_space:
        n_own = (1 if ceiling0 != WALL_UNITS else 0) + (1 if on_block else 0)
    start_on_stairs_cell = on_block and walls[start_y, start_x] == STAIRS_WALL_TYPE

    for col in range(ncols):
        angle = angles[col]
        dx, dy = math.cos(angle), math.sin(angle)
        cell_x, cell_y = start_x, start_y
        delta_x = abs(1.0 / dx) if dx != 0.0 else math.inf
        delta_y = abs(1.0 / dy) if dy != 0.0 else math.inf
        if dx < 0:
            step_x, next_x = -1, (x - cell_x) * delta_x
        else:
            step_x = 1
            next_x = (cell_x + 1 - x) * delta_x if dx != 0.0 else math.inf
        if dy < 0:
            step_y, next_y = -1, (y - cell_y) * delta_y
        else:
            step_y = 1
            next_y = (cell_y + 1 - y) * delta_y if dy != 0.0 else math.inf

        wf[W_DIST, col] = math.inf
        wf[W_U, col] = 0.0
        wf[W_BASE, col] = 0.0
        wf[W_TOP, col] = 1.0
        wi[W_SLOT, col] = 0
        wi[W_UPPER, col] = 0
        wi[W_Y_SIDE, col] = 0

        floor_now, ceiling_now = floor0, ceiling0
        open_ceiling = ceiling_now
        open_hit = -1
        start_on_stairs = start_on_stairs_cell
        n = n_own  # the ray's own hits go after the camera's spans

        while True:
            if next_x < next_y:
                distance, y_side = next_x, False
                next_x += delta_x
                cell_x += step_x
            else:
                distance, y_side = next_y, True
                next_y += delta_y
                cell_y += step_y
            if start_on_stairs:
                floor_now = _floor_in_cell(
                    walls,
                    floors,
                    ceilings,
                    facing,
                    x + dx * distance,
                    y + dy * distance,
                    start_x,
                    start_y,
                )
                start_on_stairs = False
            if distance > max_distance:
                break

            if 0 <= cell_x < width and 0 <= cell_y < height:
                wall = walls[cell_y, cell_x]
                floor_units, ceiling_units = floors[cell_y, cell_x], ceilings[cell_y, cell_x]
            else:
                wall, floor_units, ceiling_units = 1, 0, WALL_UNITS
            full = wall != EMPTY and not 0 < floor_units < ceiling_units
            low_block = wall != EMPTY and not full
            stairs = low_block and wall == STAIRS_WALL_TYPE

            if not (full or low_block):
                if ceiling_units == open_ceiling:
                    floor_now, ceiling_now = 0.0, ceiling_units
                    continue
                open_ceiling = ceiling_units

            if open_hit >= 0 and not low_block:
                fl[EXIT_DISTANCE, open_hit, col] = distance
                open_hit = -1

            if next_x < next_y:
                exit_distance, exit_y_side = next_x, False
            else:
                exit_distance, exit_y_side = next_y, True
            exit_u = _wall_u(x, y, dx, dy, exit_distance, exit_y_side)
            exit_face = (_entry_face(exit_y_side, dx, dy) + 2) % 4

            if full:
                top = exit_top = math.inf
            elif stairs:
                top = _floor_in_cell(
                    walls,
                    floors,
                    ceilings,
                    facing,
                    x + dx * distance,
                    y + dy * distance,
                    cell_x,
                    cell_y,
                )
                exit_top = _floor_in_cell(
                    walls,
                    floors,
                    ceilings,
                    facing,
                    x + dx * exit_distance,
                    y + dy * exit_distance,
                    cell_x,
                    cell_y,
                )
            else:
                top = exit_top = (floor_units if low_block else 0) / WALL_UNITS

            if low_block and n > n_own:
                last = n - 1
                lx, ly = it[CELL_X, last, col], it[CELL_Y, last, col]
                if (
                    it[WALL_TYPE, last, col] == wall
                    and fl[EXIT_DISTANCE, last, col] == distance
                    and floors[ly, lx] == floor_units
                    and ceiling_units == ceiling_now
                    and facing[ly, lx] == facing[cell_y, cell_x]
                    and fl[EXIT_TOP, last, col] == top
                ):
                    # A run of identical blocks: extend the last span.
                    fl[EXIT_DISTANCE, last, col] = exit_distance
                    it[EXIT_Y_SIDE, last, col] = exit_y_side
                    fl[EXIT_U, last, col] = exit_u
                    it[EXIT_FACE, last, col] = exit_face
                    fl[EXIT_TOP, last, col] = exit_top
                    floor_now = exit_top
                    continue

            u = _wall_u(x, y, dx, dy, distance, y_side)
            face = _entry_face(y_side, dx, dy)

            if full:
                # The wall the ray stops at: its own arrays.
                if 0 <= cell_x < width and 0 <= cell_y < height:
                    wi[W_SLOT, col] = slots[cell_y, cell_x, face]
                    wi[W_UPPER, col] = slots[cell_y, cell_x, 4]
                else:
                    wi[W_SLOT, col] = outside[face]
                    wi[W_UPPER, col] = outside[4]
                wf[W_DIST, col] = distance
                wf[W_U, col] = u
                wf[W_BASE, col] = floor_now
                wf[W_TOP, col] = ceiling_now / WALL_UNITS
                wi[W_Y_SIDE, col] = y_side
                break

            fl[DISTANCE, n, col] = distance
            fl[U, n, col] = u
            fl[TOP, n, col] = top
            fl[BASE, n, col] = floor_now
            fl[EXIT_DISTANCE, n, col] = exit_distance if low_block else math.inf
            fl[EXIT_U, n, col] = exit_u
            fl[CEILING, n, col] = ceiling_units / WALL_UNITS
            fl[PREV_CEILING, n, col] = ceiling_now / WALL_UNITS
            fl[EXIT_TOP, n, col] = exit_top
            it[CELL_X, n, col] = cell_x
            it[CELL_Y, n, col] = cell_y
            it[WALL_TYPE, n, col] = wall
            it[Y_SIDE, n, col] = y_side
            it[EXIT_Y_SIDE, n, col] = exit_y_side
            it[FACE, n, col] = face
            it[EXIT_FACE, n, col] = exit_face
            if not low_block:
                open_hit = n
            n += 1
            floor_now, ceiling_now = exit_top, ceiling_units

        # The camera's own space, as up to two pseudo-spans from distance 0:
        # its ceiling (if not the default) runs to the next change of ceiling
        # or the wall, carrying on over any desks; the block it's standing on
        # (a desk, a gallery, stairs) ends where the ray leaves that cell.
        end_wall = wf[W_DIST, col]
        if own_space:
            k = 0
            if ceiling0 != WALL_UNITS:
                open_change = end_wall
                for j in range(n_own, n):
                    if it[WALL_TYPE, j, col] == 0:
                        open_change = fl[DISTANCE, j, col]
                        break
                _own_span(fl, it, k, col, 0.0, ceiling0 / WALL_UNITS, open_change, 0.0, 0)
                it[CELL_X, k, col] = 0
                it[CELL_Y, k, col] = 0
                k += 1
            if on_block:
                to_x = (
                    ((start_x + 1 - x) / dx if dx > 0 else (x - start_x) / -dx)
                    if dx != 0.0
                    else math.inf
                )
                to_y = (
                    ((start_y + 1 - y) / dy if dy > 0 else (y - start_y) / -dy)
                    if dy != 0.0
                    else math.inf
                )
                edge = min(to_x, to_y)
                first = fl[DISTANCE, n_own, col] if n > n_own else end_wall
                end = min(first, edge)
                leave = _floor_in_cell(
                    walls, floors, ceilings, facing, x + dx * end, y + dy * end, start_x, start_y
                )
                _own_span(fl, it, k, col, floor0, 1.0, end, leave, walls[start_y, start_x])
                it[CELL_X, k, col] = start_x
                it[CELL_Y, k, col] = start_y
                if to_x < to_y:
                    fl[EXIT_U, k, col] = _wall_u(x, y, dx, dy, to_x, False)
                else:
                    fl[EXIT_U, k, col] = _wall_u(x, y, dx, dy, to_y, True)

        # What the renderer needs per span: face slots, and whether a block's
        # back face can show (it's left into open floor, not another span).
        counts[col] = n
        for j in range(n):
            block = it[WALL_TYPE, j, col] != 0
            after = fl[DISTANCE, j + 1, col] if j + 1 < n else end_wall
            it[EXITS_TO_FLOOR, j, col] = block and after > fl[EXIT_DISTANCE, j, col] + 1e-9
            if not block:
                it[SLOT, j, col] = 0
                it[BACK_SLOT, j, col] = 0
                it[UPPER_SLOT, j, col] = bulkhead  # a bulkhead's face above it
                continue
            cx, cy = it[CELL_X, j, col], it[CELL_Y, j, col]
            if 0 <= cx < width and 0 <= cy < height:
                it[SLOT, j, col] = slots[cy, cx, it[FACE, j, col]]
                it[BACK_SLOT, j, col] = slots[cy, cx, it[EXIT_FACE, j, col]]
                it[UPPER_SLOT, j, col] = slots[cy, cx, 4]
            else:
                it[SLOT, j, col] = outside[it[FACE, j, col]]
                it[BACK_SLOT, j, col] = outside[it[EXIT_FACE, j, col]]
                it[UPPER_SLOT, j, col] = outside[4]
    return fl, it, counts, wf, wi


@_jit
def _own_span(
    fl: FloatArray,
    it: Int64Array,
    k: int,
    col: int,
    floor: float,
    ceiling: float,
    exit_distance: float,
    exit_top: float,
    wall_type: int,
) -> None:
    """render._own_space, written into span k."""
    fl[DISTANCE, k, col] = 0.0
    fl[U, k, col] = 0.0
    fl[TOP, k, col] = floor
    fl[BASE, k, col] = floor
    fl[EXIT_DISTANCE, k, col] = exit_distance
    fl[EXIT_U, k, col] = 0.0
    fl[CEILING, k, col] = ceiling
    fl[PREV_CEILING, k, col] = ceiling
    fl[EXIT_TOP, k, col] = exit_top
    it[WALL_TYPE, k, col] = wall_type
    it[Y_SIDE, k, col] = 0
    it[EXIT_Y_SIDE, k, col] = 0
    it[FACE, k, col] = 0
    it[EXIT_FACE, k, col] = 0

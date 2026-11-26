"""The compiled caster (ok_doomer/client/fastcast.py) agrees exactly with the
reference one (ok_doomer/shared/raycast.py), ray for ray, and adds the camera's
own space as the renderer expects."""

import math
import random

import numpy as np
import pytest

from ok_doomer.client import fastcast as fc
from ok_doomer.client.config import RenderConfig
from ok_doomer.client.render import Renderer
from ok_doomer.shared.mapfile import STAIRS_WALL_TYPE, GameMap, load_map, parse_map
from ok_doomer.shared.raycast import RayHit, Side, cast_ray, start_heights

OFFICE = load_map("maps/office.map")


def _columns(game_map: GameMap, x: float, y: float, angles: list[float]) -> fc.Columns:
    arrays = Renderer(RenderConfig())._map_arrays_for(game_map)
    return fc.cast_columns(arrays, x, y, np.array(angles))


def _span(c: fc.Columns, k: int, col: int) -> RayHit:
    f, i = c.floats[:, k, col], c.ints[:, k, col]
    return RayHit(
        float(f[fc.DISTANCE]),
        int(i[fc.CELL_X]),
        int(i[fc.CELL_Y]),
        int(i[fc.WALL_TYPE]),
        Side.Y if i[fc.Y_SIDE] else Side.X,
        float(f[fc.U]),
        float(f[fc.TOP]),
        float(f[fc.BASE]),
        float(f[fc.EXIT_DISTANCE]),
        Side.Y if i[fc.EXIT_Y_SIDE] else Side.X,
        float(f[fc.EXIT_U]),
        int(i[fc.FACE]),
        int(i[fc.EXIT_FACE]),
        float(f[fc.CEILING]),
        float(f[fc.PREV_CEILING]),
        float(f[fc.EXIT_TOP]),
    )


def _own_spans(game_map: GameMap, x: float, y: float) -> int:
    floor, ceiling = start_heights(game_map, x, y)
    if (floor, ceiling) == (0.0, 8):
        return 0
    return (ceiling != 8) + game_map.is_low_block(math.floor(x), math.floor(y))


def _views() -> list[tuple[float, float]]:
    """Points all over the office, on desks, stairs and balconies too, and
    some exactly on grid lines."""
    rng = random.Random(3)
    points = [(4.0, 5.0), (3.5, 3.0), (6.0, 6.0)]
    while len(points) < 120:
        x, y = rng.uniform(0.5, OFFICE.width - 0.5), rng.uniform(0.5, OFFICE.height - 0.5)
        if not OFFICE.is_full_wall(math.floor(x), math.floor(y)):
            points.append((x, y))
    return points


@pytest.mark.parametrize("x, y", _views())
def test_every_ray_matches_the_reference_caster(x: float, y: float) -> None:
    angles = [i * math.pi / 32 for i in range(-32, 32)]  # includes the axes exactly
    columns = _columns(OFFICE, x, y, angles)
    own = _own_spans(OFFICE, x, y)
    for col, angle in enumerate(angles):
        hits = cast_ray(OFFICE, x, y, angle)
        if hits and hits[-1].full:
            wall = hits.pop()
            assert columns.wall_floats[fc.W_DIST, col] == wall.distance
            assert columns.wall_floats[fc.W_U, col] == wall.u
            assert columns.wall_floats[fc.W_BASE, col] == wall.base
            assert columns.wall_floats[fc.W_TOP, col] == wall.prev_ceiling
            assert bool(columns.wall_ints[fc.W_Y_SIDE, col]) == (wall.side is Side.Y)
        else:
            assert math.isinf(columns.wall_floats[fc.W_DIST, col])
        assert columns.counts[col] == own + len(hits)
        for k, hit in enumerate(hits):
            assert _span(columns, own + k, col) == hit


def test_standing_on_stairs_the_own_span_ends_at_the_cell_edge() -> None:
    (x, y) = next(
        (cx + 0.5, cy + 0.3)
        for cy in range(OFFICE.height)
        for cx in range(OFFICE.width)
        if OFFICE.is_stairs(cx, cy)
    )
    angles = [i * math.pi / 8 for i in range(16)]
    columns = _columns(OFFICE, x, y, angles)
    floor, _ = start_heights(OFFICE, x, y)
    last = _own_spans(OFFICE, x, y) - 1  # the block comes after any ceiling span
    for col in range(len(angles)):
        own = _span(columns, last, col)
        assert own.distance == 0.0 and own.wall_type == STAIRS_WALL_TYPE
        assert own.top == own.base == floor
        assert (own.cell_x, own.cell_y) == (math.floor(x), math.floor(y))
        assert 0.0 < own.exit_distance <= math.hypot(1.0, 1.0)


def test_under_a_high_ceiling_the_own_span_runs_to_the_next_change() -> None:
    room = parse_map("[walls]\n#######\n#.....#\n#######\n\n[ceiling]\n8888888\n8gg8888\n8888888\n")
    columns = _columns(room, 1.5, 1.5, [0.0])
    own = _span(columns, 0, 0)
    assert own.ceiling == room.ceiling_height(1, 1) and own.wall_type == 0
    assert own.exit_distance == pytest.approx(1.5)  # into the normal-height cells at x = 3
    assert columns.counts[0] == 2

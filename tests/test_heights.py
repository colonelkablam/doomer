"""Rendering with heights: tall walls, bulkheads, raised ceilings, raised
floors, camera eye height and looking up/down. Expected pixels come from the
projection formula  row = horizon + (eye - height) * projection / depth."""

import math

import numpy as np

from doomer.client.config import RenderConfig
from doomer.client.render import EYE_HEIGHT, MAX_PITCH, Camera, Renderer
from doomer.shared.mapfile import parse_map

CFG = RenderConfig()
R = Renderer(CFG)
COL = CFG.width // 2

# A corridor: x 1..10 open, wall at x = 11. Cells x 4..7 are double height.
ATRIUM = parse_map(
    "[walls]\n############\n#..........#\n############\n\n[ceiling]\n888888888888\n8888gggg8888\n"
)
# The same corridor, double height all the way to the end wall.
TALL_END = parse_map(
    "[walls]\n############\n#..........#\n############\n\n[ceiling]\n888888888888\n8ggggggggggg\n"
)


def row(
    height: float, depth: float, eye: float = EYE_HEIGHT, horizon: float | None = None
) -> float:
    h = CFG.height / 2 if horizon is None else horizon
    return h + (eye - height) * R.projection / depth


def test_wall_in_a_tall_space_reaches_its_ceiling() -> None:
    R.render_indices(TALL_END, Camera(1.5, 1.5, 0.0))
    depth = 9.5  # wall face at x = 11
    # Halfway between heights 1 and 2: wall, not ceiling.
    y = round((row(1.0, depth) + row(2.0, depth)) / 2)
    assert math.isclose(R.zbuffer[COL, y], depth, rel_tol=1e-6)
    # In a normal-height corridor the same pixel is ceiling (infinitely far).
    plain = parse_map("[walls]\n############\n#..........#\n############\n")
    R.render_indices(plain, Camera(1.5, 1.5, 0.0))
    assert math.isinf(R.zbuffer[COL, y])


def test_bulkhead_where_the_ceiling_comes_down() -> None:
    # Camera inside the tall stretch looking east: at x = 8 the ceiling drops
    # from 2 back to 1, so a bulkhead face spans heights 1..2 at depth 8 - 4.5.
    R.render_indices(ATRIUM, Camera(4.5, 1.5, 0.0))
    depth = 3.5
    y = round((row(1.0, depth) + row(2.0, depth)) / 2)
    assert math.isclose(R.zbuffer[COL, y], depth, rel_tol=1e-6)


def test_raised_ceiling_surface_has_the_right_depth() -> None:
    R.render_indices(ATRIUM, Camera(4.5, 1.5, 0.0))
    # Near the top of the screen the ray meets the height-2 ceiling before the
    # bulkhead 3.5 away. (Nearer than ~2.9 the ceiling is above the screen.)
    for y in (0, 5, 10):
        expected = (2.0 - EYE_HEIGHT) * R.projection / (CFG.height / 2 - (y + 0.5))
        assert expected < 3.5  # still inside the tall stretch
        assert math.isclose(R.zbuffer[COL, y], expected, rel_tol=1e-6)


def test_looking_up_moves_the_horizon_down_by_tan_pitch() -> None:
    plain = parse_map("[walls]\n############\n#..........#\n############\n")
    pitch = math.radians(15)
    R.render_indices(plain, Camera(1.5, 1.5, 0.0, pitch=pitch))
    horizon = CFG.height / 2 + math.tan(pitch) * R.projection
    assert math.isclose(R.horizon, horizon)
    # The end wall's bottom edge moves down by the same amount.
    depth = 9.5
    y = round(row(0.05, depth, horizon=horizon))
    assert math.isclose(R.zbuffer[COL, y], depth, rel_tol=1e-6)
    # Pitch is clamped.
    R.render_indices(plain, Camera(1.5, 1.5, 0.0, pitch=math.radians(80)))
    assert math.isclose(R.horizon, CFG.height / 2 + math.tan(MAX_PITCH) * R.projection)


def test_lower_eye_moves_the_floor_line_up() -> None:
    plain = parse_map("[walls]\n############\n#..........#\n############\n")
    depth = 9.5
    for eye in (EYE_HEIGHT, 0.35):  # standing, crouching
        R.render_indices(plain, Camera(1.5, 1.5, 0.0, z=eye))
        bottom = row(0.0, depth, eye=eye)
        assert math.isclose(R.zbuffer[COL, math.floor(bottom) - 1], depth, rel_tol=1e-6)
        assert math.isinf(R.zbuffer[COL, math.ceil(bottom) + 1])  # floor below it


def test_standing_on_a_raised_floor_draws_it_only_to_its_edge() -> None:
    # Camera on a raised run (type e, height 4/8, x = 2..5) looking east; it
    # drops to the open floor at x = 6, and the end wall is at x = 11.
    stage = parse_map(
        "[walls]\n############\n#.eeee.....#\n############\n\n[floor]\n000000000000\n004444000000\n"
    )
    eye = 0.5 + EYE_HEIGHT
    frame = R.render_indices(stage, Camera(2.5, 1.5, 0.0, z=eye))
    # 2.0 away the raised floor is under you: its surface, at that depth.
    assert math.isclose(R.zbuffer[COL, math.floor(row(0.5, 2.0, eye=eye))], 2.0, rel_tol=0.05)
    # The row where height 0.5 would be 4.0 away passes over the edge (3.5
    # away) and reaches the lower floor beyond: the default floor, infinitely far.
    assert np.isinf(R.zbuffer[COL, round(row(0.5, 4.0, eye=eye))])
    assert frame.shape == (CFG.width, CFG.height)

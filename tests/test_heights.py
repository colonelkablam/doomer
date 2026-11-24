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
    ceiling_depth = (1.0 - EYE_HEIGHT) * R.projection / (CFG.height / 2 - (y + 0.5))
    assert math.isclose(R.zbuffer[COL, y], ceiling_depth, rel_tol=1e-6)


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


# A double-height room (ceiling 16/8) with a desk pair in the middle (x = 5).
DESK_IN_ATRIUM = parse_map(
    "[walls]\n############\n#....d.....#\n############\n\n"
    "[floor]\n000000000000\n000002000000\n\n"
    "[ceiling]\ngggggggggggg\ngggggggggggg\n"
)


def test_high_ceiling_carries_on_beyond_a_desk() -> None:
    # From outside the tall room it'd be an open stretch closed by the desk;
    # from inside it's the camera's own space. Either way, ceiling pixels whose
    # height-2 distance lies beyond the desk (x > 6) must be the high ceiling.
    for camera in (Camera(1.5, 1.5, 0.0), Camera(2.5, 1.5, 0.0)):
        R.render_indices(DESK_IN_ATRIUM, camera)
        end_wall = 11 - camera.x
        for y in range(0, CFG.height // 2 - 1, 3):
            depth = (2.0 - EYE_HEIGHT) * R.projection / (CFG.height / 2 - (y + 0.5))
            if 6 - camera.x < depth < end_wall:
                assert math.isclose(R.zbuffer[COL, y], depth, rel_tol=1e-6), (camera, y)


def test_from_on_a_desk_the_high_ceiling_still_carries_on() -> None:
    eye = 0.25 + EYE_HEIGHT
    camera = Camera(5.5, 1.5, 0.0, z=eye)  # standing on the desk, looking east
    R.render_indices(DESK_IN_ATRIUM, camera)
    for y in range(0, CFG.height // 2 - 1, 3):
        depth = (2.0 - eye) * R.projection / (CFG.height / 2 - (y + 0.5))
        if depth < 11 - camera.x:
            assert math.isclose(R.zbuffer[COL, y], depth, rel_tol=1e-6), y


def test_ceiling_light_panel_is_drawn_into_the_ceiling() -> None:
    from doomer.client.palette import CREAM

    lit = parse_map("[walls]\n#######\n#.....#\n#######\n\n[things]\nceiling_light 3.5 1.5 0\n")
    # Standing at the light's cell centre, looking along x and up: the column
    # straight ahead passes under the panel's long axis.
    camera = Camera(2.6, 1.5, 0.0, pitch=math.radians(25))
    frame = R.render_indices(lit, camera)
    column = frame[COL, : int(R.horizon)]
    diffuser = (CREAM << 4) | 15
    hits = [y for y in range(len(column)) if column[y] == diffuser]
    assert hits, "no light panel pixels"
    # Those pixels are on the ceiling at x within the panel: 3.5 +/- 0.2.
    for y in hits:
        up = R.horizon - (y + 0.5)
        depth = (1.0 - EYE_HEIGHT) * R.projection / up
        assert abs(camera.x + depth - 3.5) < 0.23
    # A room without the light thing has no panel.
    plain = parse_map("[walls]\n#######\n#.....#\n#######\n")
    assert diffuser not in R.render_indices(plain, camera)[COL].tolist()


def test_low_ceiling_hides_the_high_one_beyond_it() -> None:
    # Camera in the normal-height corridor at x = 1.5, looking east into the
    # tall stretch that starts at x = 4. Rows whose default-ceiling point is
    # nearer than x = 4 must show that low ceiling, not the high one beyond.
    R.render_indices(ATRIUM, Camera(1.5, 1.5, 0.0))
    checked = 0
    for y in range(0, CFG.height // 2 - 1):
        low = (1.0 - EYE_HEIGHT) * R.projection / (CFG.height / 2 - (y + 0.5))
        if low < 2.5 - 0.05:  # the boundary is 2.5 away
            assert math.isclose(R.zbuffer[COL, y], low, rel_tol=1e-6), y
            checked += 1
    assert checked > 5


def test_low_ceiling_stops_at_the_boundary_even_when_a_step_is_there() -> None:
    # The tall stretch starts at x = 4 with a step (a raised block, not open
    # floor), as at the lobby door. Beyond x = 4 the default ceiling doesn't
    # exist: no pixel may show it there.
    stepped = parse_map(
        "[walls]\n############\n#...e......#\n############\n\n"
        "[floor]\n000000000000\n000010000000\n\n"
        "[ceiling]\n888888888888\n8888gggggggg\n"
    )
    R.render_indices(stepped, Camera(1.5, 1.5, 0.0))
    for y in range(0, CFG.height // 2 - 1):
        low = (1.0 - EYE_HEIGHT) * R.projection / (CFG.height / 2 - (y + 0.5))
        z = float(R.zbuffer[COL, y])
        if low < 2.5 - 0.05:
            assert math.isclose(z, low, rel_tol=1e-6), y  # our own low ceiling
        elif low > 2.5 + 0.05:
            assert not math.isclose(z, low, rel_tol=1e-6), y  # not a phantom low ceiling

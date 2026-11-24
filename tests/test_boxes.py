"""Box models: small furniture drawn as real textured boxes."""

import math

from doomer.client.boxes import BOX_MODELS
from doomer.client.config import RenderConfig
from doomer.client.render import EYE_HEIGHT, Camera, Renderer, SpriteInstance
from doomer.shared.mapfile import parse_map
from doomer.shared.things import THING_TYPES

OPEN = parse_map("[walls]\n###########\n#.........#\n#.........#\n#.........#\n###########\n")
PILLAR = parse_map("[walls]\n###########\n#.........#\n#..#......#\n#.........#\n###########\n")
R = Renderer(RenderConfig())
CAM = Camera(1.5, 2.5, 0.0)  # looking east along the middle row
MAC, _KEYBOARD, _MOUSE = BOX_MODELS["mac"]
CENTRE = 160  # a column whose ray is (almost) straight ahead


def mac(x: float, facing: float) -> SpriteInstance:
    return SpriteInstance("mac", x, 2.5, 0.0, facing)


def row_for(height: float, depth: float) -> int:
    return int(R.horizon + (EYE_HEIGHT - height) * R.projection / depth)


def test_every_box_model_is_a_registered_thing() -> None:
    assert set(BOX_MODELS) <= set(THING_TYPES)


def test_front_face_and_top_are_drawn_at_their_depths() -> None:
    R.render_indices(OPEN, CAM, [mac(4.5, math.pi)])  # facing the camera
    near, far = 3.0 - MAC.x[1], 3.0 - MAC.x[0]
    top = MAC.z[1]
    front_row = row_for(top / 2, near)
    assert abs(R.zbuffer[CENTRE, front_row] - near) < 0.01
    # Just above the front face, you look down onto the top: deeper, but
    # within the box.
    top_row = row_for(top, (near + far) / 2)
    assert near < R.zbuffer[CENTRE, top_row] < far


def test_which_side_you_see_depends_on_its_angle() -> None:
    facing = R.render_indices(OPEN, CAM, [mac(4.5, math.pi)])
    away = R.render_indices(OPEN, CAM, [mac(4.5, 0.0)])
    plain = R.render_indices(OPEN, CAM)
    assert (facing != plain).any() and (away != plain).any()
    assert (facing != away).any()  # screen in front, vents behind


def test_box_models_hide_behind_walls() -> None:
    plain = R.render_indices(PILLAR, CAM)
    assert (R.render_indices(PILLAR, CAM, [mac(5.5, math.pi)]) == plain).all()


def test_nothing_drawn_when_the_camera_is_inside_one() -> None:
    camera = Camera(4.5, 2.5, 0.0, z=MAC.z[1] / 2)
    plain = R.render_indices(OPEN, camera)
    assert (R.render_indices(OPEN, camera, [mac(4.5, math.pi)]) == plain).all()


def test_screened_desk_pairs_get_a_middle_screen_and_one_per_cell_edge() -> None:
    from doomer.client.boxes import SCREEN_HEIGHT, desk_screens

    # Three screened pairs (j) facing north at x 2..4, and one facing east.
    pods = parse_map(
        "[walls]\n#########\n#.......#\n#.jjj.j.#\n#.......#\n#########\n\n"
        "[floor]\n000000000\n000000000\n002220200\n\n"
        "[facing]\n.........\n.........\n..nnn.e..\n"
    )
    screens = desk_screens(pods)
    row = [p for p in screens if 2 <= p.x <= 5]
    across_x = sorted(p.x for p in row if p.box.x[1] - p.box.x[0] < 0.1)  # thin in x
    along = [p for p in row if p.box.y[1] - p.box.y[0] < 0.1]  # thin in y
    assert across_x == [2.0, 3.0, 4.0, 5.0]  # each edge once, row ends too
    assert sorted(p.x for p in along) == [2.5, 3.5, 4.5] and all(p.y == 2.5 for p in along)
    # The east-facing pair: middle screen along x = 6.5, edges at y = 2 and 3.
    single = [p for p in screens if p.x >= 6]
    assert sorted((p.x, p.y) for p in single) == [(6.5, 2.0), (6.5, 2.5), (6.5, 3.0)]
    assert all(p.z == 0.25 and p.box.z == (0.0, SCREEN_HEIGHT) for p in screens)


def test_plain_desk_pairs_have_no_screens() -> None:
    from doomer.client.boxes import desk_screens

    plain = parse_map("[walls]\n#####\n#.d.#\n#####\n\n[floor]\n00000\n00200\n")
    assert desk_screens(plain) == []

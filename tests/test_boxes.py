"""Box models: small furniture drawn as real textured boxes."""

import math

import numpy as np
import numpy.typing as npt
import pytest

from doomer.client.boxes import BOX_MODELS, PlacedBox
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
    from doomer.client.boxes import desk_screens
    from doomer.shared.screens import SCREEN_HEIGHT

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


def test_furniture_is_box_models_not_sprites() -> None:
    from doomer.client.sprites import SPRITES

    for name in ("filing_cabinet", "water_cooler", "ficus", "mac"):
        assert name in BOX_MODELS and name not in SPRITES


def test_box_models_fit_their_collision_footprints() -> None:
    for name in ("filing_cabinet", "water_cooler"):
        radius = THING_TYPES[name].radius
        for box in BOX_MODELS[name]:
            if box.turn:
                continue  # an octagon's turned half lies within its unturned half
            assert max(abs(v) for v in (*box.x, *box.y)) <= radius + 0.031, name  # taps


def test_water_cooler_bottle_is_an_octagon() -> None:
    # Seen face on, the bottle's nearest point is its inradius away from its
    # axis; turned 22.5 degrees, a corner faces you: inradius / cos(22.5).
    top = max(b.z[1] for b in BOX_MODELS["water_cooler"])
    body = [b for b in BOX_MODELS["water_cooler"] if b.z[1] == top]
    inradius = max(b.x[1] for b in body)
    height = (body[0].z[0] + top) / 2
    for turn, reach in ((0.0, inradius), (math.pi / 8, inradius / math.cos(math.pi / 8))):
        R.render_indices(OPEN, CAM, [SpriteInstance("water_cooler", 4.5, 2.5, 0.0, turn)])
        row = row_for(height, 3.0 - reach)
        assert math.isclose(R.zbuffer[CENTRE, row], 3.0 - reach, abs_tol=0.003), turn


def test_ficus_leaves_are_see_through_between_the_clusters() -> None:
    from doomer.client.boxes import BOX_OPAQUE

    plain = R.render_indices(OPEN, CAM).copy()
    frame = R.render_indices(OPEN, CAM, [SpriteInstance("ficus", 3.5, 2.5)])
    # The crown, 2 cells away: between ~0.95 and ~1.65 m up, the middle columns.
    rows = slice(row_for(0.62, 2.0), row_for(0.4, 2.0))
    crown = frame[150:170, rows] != plain[150:170, rows]
    assert crown.any() and not crown.all()  # leaves, and gaps showing the wall behind
    leaf_textures = {
        tex for box in BOX_MODELS["ficus"] for tex in box.faces if not BOX_OPAQUE[tex].all()
    }
    assert len(leaf_textures) >= 4


def test_testing_only_each_boxs_columns_draws_the_same_as_testing_all(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from doomer.client.render import sprites_from_things
    from doomer.shared.mapfile import load_map

    office = load_map("maps/office.map")
    things = sprites_from_things(office)
    views = [
        Camera(3.0, 5.0, math.radians(-20)),
        Camera(11.5, 4.3, math.radians(-90), pitch=-0.3),
        Camera(6.3, 1.5, math.radians(60), z=1.6, pitch=-0.4),
        Camera(11.68, 1.75, math.radians(90), z=0.42),
    ]
    culled = [R.render_indices(office, cam, things).copy() for cam in views]

    def every_column(
        self: Renderer, boxes: list[PlacedBox], camera: Camera
    ) -> tuple[list[PlacedBox], npt.NDArray[np.intp], npt.NDArray[np.intp]]:
        n = len(boxes)
        return boxes, np.zeros(n, dtype=np.intp), np.full(n, self.config.width, dtype=np.intp)

    monkeypatch.setattr(Renderer, "_in_view", every_column)
    for cam, image in zip(views, culled, strict=True):
        assert (R.render_indices(office, cam, things) == image).all()


def test_ficus_leaf_cards_show_leaves_only_on_their_broad_faces() -> None:
    from doomer.client.boxes import BOX_OPAQUE, FRONT, TOP

    for box in BOX_MODELS["ficus"]:
        sizes = (box.x[1] - box.x[0], box.y[1] - box.y[0], box.z[1] - box.z[0])
        if min(sizes) > 0.01:
            continue  # the pot and trunk, not a card
        thin = sizes.index(min(sizes))
        broad = {0: (FRONT, FRONT + 2), 1: (FRONT + 1, FRONT + 3), 2: (TOP,)}[thin]
        for face, tex in enumerate(box.faces):
            # Edges a few mm wide would squeeze the whole texture into a line.
            assert BOX_OPAQUE[tex].any() == (face in broad), (box, face)


def test_glowing_boxes_ignore_lighting_and_others_dont() -> None:
    from doomer.client.config import LightingConfig
    from doomer.client.render import Renderer as _Renderer

    room = parse_map(  # two storeys high: the sign hangs above one
        "[walls]\n###########\n#.........#\n#.........#\n#.........#\n###########\n"
        "[ceiling]\n" + "ggggggggggg\n" * 5
    )
    sign = SpriteInstance("welcome_sign", 9.99, 2.5, 0.0, math.pi)  # on the east wall
    camera = Camera(4.0, 2.5, 0.0, pitch=0.43)
    frames = {}
    for name, ambient in (("dark", 0.1), ("bright", 1.0)):
        r = _Renderer(RenderConfig(), LightingConfig(ambient=ambient, panel_intensity=0.0))
        frames[name] = r.render_indices(room, camera, [sign]).copy()
    plain = _Renderer(RenderConfig(), LightingConfig(ambient=1.0, panel_intensity=0.0))
    empty = plain.render_indices(room, camera).copy()
    drawn = frames["bright"] != empty
    same = frames["dark"] == frames["bright"]
    # The letters (glowing) look the same in the dark; the panel behind them doesn't.
    assert (drawn & same).sum() > 20
    assert (drawn & ~same).sum() > 20  # (a near-black panel: few shades to lose)

"""Faces: one portrait style, built from stock parts that swap."""

import dataclasses

import numpy as np

from doomer.client.faces import (
    CAST,
    EYEWEAR,
    FACIAL_HAIR,
    HAIR_COLOURS,
    HAIR_STYLES,
    JACKETS,
    MOUTHS,
    SKINS,
    TIES,
    Face,
    draw_portrait,
)
from doomer.client.palette import BLUE
from doomer.client.pixelart import Canvas


def _portrait(face: Face, size: int = 60) -> np.ndarray:
    c = Canvas()
    draw_portrait(c, 0, 0, size, face, (BLUE, 8.0))  # 60: three texels a pixel
    return c.index.copy()


def test_every_stock_part_changes_the_face() -> None:
    base = Face()
    picture = _portrait(base)
    options = {
        "skin": SKINS,
        "hair": HAIR_STYLES,
        "hair_colour": HAIR_COLOURS,
        "eyewear": EYEWEAR,
        "facial_hair": FACIAL_HAIR,
        "mouth": MOUTHS,
        "jacket": JACKETS,
        "tie": TIES,
    }
    for part, values in options.items():
        for value in values:
            if value == getattr(base, part):
                continue
            other = _portrait(dataclasses.replace(base, **{part: value}))
            assert (other != picture).any(), (part, value)


def test_a_portrait_stays_in_its_box_and_fills_it() -> None:
    c = Canvas()
    draw_portrait(c, 10, 20, 30, CAST["gary"], (BLUE, 8.0))
    painted = c.opaque
    assert painted[10:40, 20:50].all()  # the background fills the box
    assert not painted[:10].any() and not painted[40:].any()
    assert not painted[:, :20].any() and not painted[:, 50:].any()


def test_the_cast_are_all_different_people() -> None:
    pictures = [_portrait(face).tobytes() for face in CAST.values()]
    assert len(set(pictures)) == len(CAST)


def test_a_tie_comes_with_a_collar_and_no_tie_with_an_open_neck() -> None:
    from doomer.client.palette import CREAM

    collar = (CREAM << 4) | 15
    tied = _portrait(Face(tie="red"))
    open_ = _portrait(Face(tie="none"))
    # The collar's left point: pixel (14, 8), just below the neck, left of the knot.
    assert tied[43, 25] == collar
    assert open_[43, 25] != collar


def test_faces_look_straight_ahead() -> None:
    from doomer.client.faces import FEATURES, HEAD

    eyes = [col for col, ch in enumerate(FEATURES[6]) if ch == "E"]
    assert eyes == [7, 12]  # five apart
    head = [col for col, ch in enumerate(HEAD[6]) if ch != "."]
    assert (head[0] + head[-1]) / 2 == sum(eyes) / 2  # centred between the ears


def test_pixels_are_solid_blocks() -> None:
    picture = _portrait(CAST["gary"])  # three texels to a pixel
    blocks = picture[:60, :60].reshape(20, 3, 20, 3)
    assert (blocks == blocks[:, :1, :, :1]).all()


def test_glasses_are_closed_rims_round_each_eye_and_nothing_below() -> None:
    from doomer.client.faces import EYEWEAR_TEMPLATES, FEATURES

    for name, top, bottom in (("big_glasses", "R", "R"), ("aviators", "F", "K")):
        glasses = EYEWEAR_TEMPLATES[name]
        for eye in (7, 12):
            assert FEATURES[6][eye] == "E"
            for col in (eye - 1, eye, eye + 1):  # no gaps top or bottom
                assert glasses[5][col] == top and glasses[7][col] == bottom, (name, col)
        assert all(set(row) == {"."} for row in glasses[8:]), name  # nothing hanging below
    assert EYEWEAR_TEMPLATES["big_glasses"][6][7] == "."  # the eye shows through clear glass


def test_the_body_is_square_on() -> None:
    from doomer.client.faces import BODIES

    for name, body in BODIES.items():
        for row in body:
            shape = "".join("." if ch == "." else "x" for ch in row)
            assert shape == shape[::-1], (name, row)  # the same outline both sides
            middle = row.replace("L", "B").replace("S", "B")  # shading aside
            assert middle == middle[::-1], (name, row)  # lapels, collar, tie centred


def test_hair_is_symmetrical_without_gaps() -> None:
    from doomer.client.faces import HAIR_BACKS, HAIR_TOPS, PARTING_GAP

    for name, template in [*HAIR_TOPS.items(), *HAIR_BACKS.items()]:
        for r, row in enumerate(template):
            shape = "".join("." if ch == "." else "x" for ch in row)
            if name == "side_part" and r == PARTING_GAP[0]:  # the parting's one break
                shape = shape[: PARTING_GAP[1]] + "x" + shape[PARTING_GAP[1] + 1 :]
            assert shape == shape[::-1], (name, row)  # the same both sides
    for row in HAIR_TOPS["mullet"][:4]:  # the crown: solid, no holes
        hair = row.strip(".")
        assert "." not in hair, row


def test_a_side_parting_has_a_dark_parting_left_of_centre() -> None:
    from doomer.client.faces import HAIR_TOPS

    parting = [col for row in HAIR_TOPS["side_part"][:3] for col, ch in enumerate(row) if ch == "D"]
    assert parting == [7, 7, 7]  # a line down the crown, left of the middle (9.5)
    assert HAIR_TOPS["side_part"][3][7] == "."  # skin shows just below it
    assert "D" not in "".join(HAIR_TOPS["mullet"])  # the mullet's crown is plain

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


def _portrait(face: Face, size: int = 64) -> np.ndarray:
    c = Canvas()
    draw_portrait(c, 0, 0, size, face, (BLUE, 8.0))
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

"""Sprite art: things drawn as flat pictures that always face the camera.

A sprite image is 64x64 and spans one cell wide and one wall tall, standing on
the floor. So texels are the same size as on walls, and something meant to
hang from the ceiling (a light) is simply drawn at the top of the image.

Sprites are keyed by thing type name (doomer/shared/things.py). Things with no
sprite (spawn points) are invisible.
"""

from dataclasses import dataclass

import numpy as np

from doomer.client.palette import (
    BLUE,
    BRICK,
    CREAM,
    GREEN,
    GREY,
    ORANGE,
    SAND,
    WARM_GREY,
    WOOD,
    U8Array,
)
from doomer.client.pixelart import BoolArray, Canvas, noise


@dataclass(frozen=True, slots=True)
class Sprite:
    index: U8Array  # [u][v] palette indices
    opaque: BoolArray  # [u][v] True where the sprite is drawn
    fullbright: bool = False  # emits light: not darkened by distance


def _finish(canvas: Canvas, fullbright: bool = False) -> Sprite:
    return Sprite(
        np.ascontiguousarray(canvas.index.T), np.ascontiguousarray(canvas.opaque.T), fullbright
    )


def desk() -> Sprite:
    c = Canvas()
    rng = np.random.default_rng(11)
    grain = 7 + noise(rng, 16) * 1.5
    c.paint(c.rect(40, 62, 4, 60), WOOD, grain)  # front panel
    c.paint(c.rect(40, 62, 4, 6) | c.rect(40, 62, 58, 60), WOOD, 5.0)  # edges
    c.paint(c.rect(46, 47, 36, 58) | c.rect(54, 55, 36, 58), WOOD, 4.0)  # drawer lines
    c.paint(c.rect(50, 51, 45, 50) | c.rect(58, 59, 45, 50), GREY, 12.0)  # handles
    c.paint(c.rect(37, 40, 2, 62), WOOD, 10.0)  # desktop
    # Beige PC: CRT monitor on a system unit, keyboard in front.
    c.paint(c.rect(31, 37, 14, 50), SAND, 10.0)  # system unit
    c.paint(c.rect(33, 34, 38, 46), GREY, 3.0)  # floppy slot
    c.paint(c.rect(10, 31, 18, 46), SAND, 11.0)  # monitor case
    c.paint(c.rect(10, 31, 44, 46), SAND, 8.0)  # side shading
    c.paint(c.rect(13, 27, 21, 41), GREEN, 1.0)  # screen
    for row, length in ((15, 12), (17, 16), (19, 9), (21, 14), (23, 6)):
        c.paint(c.rect(row, row + 1, 23, 23 + length), GREEN, 13.0)  # green text
    c.paint(c.rect(35, 37, 20, 44), WARM_GREY, 11.0)  # keyboard
    return _finish(c)


def water_cooler() -> Sprite:
    c = Canvas()
    c.paint(c.rect(30, 63, 23, 41), CREAM, 9.0)  # cabinet
    c.paint(c.rect(30, 63, 38, 41), CREAM, 7.0)  # side shade
    c.paint(c.rect(38, 41, 27, 30), BLUE, 9.0)  # cold tap
    c.paint(c.rect(38, 41, 34, 37), BRICK, 10.0)  # hot tap
    bottle = c.ellipse(17, 32, 13, 10) | c.rect(26, 30, 28, 36)
    c.paint(bottle, BLUE, 11.0)
    c.paint(bottle & c.rect(0, 64, 26, 29), BLUE, 14.0)  # highlight
    c.paint(bottle & c.rect(0, 64, 37, 43), BLUE, 8.0)  # shadow side
    c.paint(c.rect(10, 11, 23, 41) & bottle, BLUE, 13.0)  # ridge
    return _finish(c)


def filing_cabinet() -> Sprite:
    c = Canvas()
    c.paint(c.rect(14, 63, 18, 46), GREY, 8.0)
    c.paint(c.rect(14, 63, 43, 46), GREY, 6.0)
    for top in (16, 28, 40, 52):
        c.paint(c.rect(top, top + 1, 19, 43), GREY, 4.0)  # drawer gap
        c.paint(c.rect(top + 4, top + 6, 26, 36), GREY, 13.0)  # handle
        c.paint(c.rect(top + 7, top + 9, 28, 34), CREAM, 10.0)  # label
    return _finish(c)


def ficus() -> Sprite:
    c = Canvas()
    rng = np.random.default_rng(14)
    pot = c.rect(48, 64, 22, 42) & (np.abs(c.u + 0.5 - 32) <= 10 - (63 - c.v) * 0.15)
    c.paint(pot, ORANGE, np.where(c.u < 30, 8.0, 7.0))
    c.paint(c.rect(48, 50, 21, 43), ORANGE, 9.0)  # rim
    c.paint(c.rect(32, 48, 31, 33), WOOD, 5.0)  # trunk
    # Leaf clusters: coarse noise gives clumps, fine noise breaks up the edges.
    leaves = noise(rng, 8) * 0.6 + noise(rng, 32) * 0.4
    crown = c.ellipse(26, 32, 22, 21) | c.ellipse(16, 24, 10, 10) | c.ellipse(18, 42, 9, 10)
    c.paint(crown & (leaves > 0.3), GREEN, 3 + leaves * 9)
    return _finish(c)


def ceiling_light() -> Sprite:
    c = Canvas()
    c.paint(c.rect(0, 4, 6, 58), GREY, 8.0)  # housing
    c.paint(c.rect(1, 3, 8, 56), CREAM, 15.0)  # diffuser
    return _finish(c, fullbright=True)


def exit_sign() -> Sprite:
    c = Canvas()
    c.paint(c.rect(0, 8, 31, 33), GREY, 6.0)  # hanging rod
    c.paint(c.rect(8, 24, 13, 51), GREEN, 9.0)  # sign
    c.paint(c.rect(8, 9, 13, 51) | c.rect(23, 24, 13, 51), GREEN, 6.0)  # edges
    c.text("EXIT", 11, 17, CREAM, 15.0, scale=2)
    return _finish(c, fullbright=True)


SPRITES: dict[str, Sprite] = {
    "desk": desk(),
    "water_cooler": water_cooler(),
    "filing_cabinet": filing_cabinet(),
    "ficus": ficus(),
    "ceiling_light": ceiling_light(),
    "exit": exit_sign(),
}

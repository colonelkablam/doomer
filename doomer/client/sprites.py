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
    on_ceiling: bool = False  # hangs from the ceiling, so ignores desks below it


def _finish(canvas: Canvas, fullbright: bool = False, on_ceiling: bool = False) -> Sprite:
    return Sprite(
        np.ascontiguousarray(canvas.index.T),
        np.ascontiguousarray(canvas.opaque.T),
        fullbright,
        on_ceiling,
    )


# Scale: the image spans floor to ceiling, and an office ceiling is ~2.7 m, so
# one pixel is ~4 cm of height. Heights are real-world; widths are exaggerated
# ~1.3x (as Doom did) so small things stay readable. Collision footprints in
# doomer/shared/things.py match these widths.


def desk() -> Sprite:
    """Desk 0.75 m high, beige PC on top reaching ~1.4 m."""
    c = Canvas()
    rng = np.random.default_rng(11)
    grain = 7 + noise(rng, 16) * 1.5
    c.paint(c.rect(47, 64, 12, 52), WOOD, grain)  # front panel
    c.paint(c.rect(47, 64, 12, 14) | c.rect(47, 64, 50, 52), WOOD, 5.0)  # edges
    c.paint(c.rect(52, 53, 34, 50) | c.rect(58, 59, 34, 50), WOOD, 4.0)  # drawer lines
    c.paint(c.rect(55, 56, 40, 44) | c.rect(61, 62, 40, 44), GREY, 12.0)  # handles
    c.paint(c.rect(45, 47, 10, 54), WOOD, 10.0)  # desktop
    # Beige PC: CRT monitor on a system unit, keyboard in front.
    c.paint(c.rect(41, 45, 19, 45), SAND, 10.0)  # system unit
    c.paint(c.rect(42, 43, 36, 42), GREY, 3.0)  # floppy slot
    c.paint(c.rect(29, 41, 22, 42), SAND, 11.0)  # monitor case
    c.paint(c.rect(29, 41, 40, 42), SAND, 8.0)  # side shading
    c.paint(c.rect(31, 39, 24, 38), GREEN, 1.0)  # screen
    for row, length in ((32, 9), (34, 12), (36, 6)):
        c.paint(c.rect(row, row + 1, 25, 25 + length), GREEN, 13.0)  # green text
    c.paint(c.rect(44, 45, 22, 40), WARM_GREY, 11.0)  # keyboard
    return _finish(c)


def terminal() -> Sprite:
    """Just the beige PC, for standing on a desk-pod cell (drawn from its base up)."""
    c = Canvas()
    c.paint(c.rect(59, 64, 19, 45), SAND, 10.0)  # system unit
    c.paint(c.rect(60, 61, 36, 42), GREY, 3.0)  # floppy slot
    c.paint(c.rect(47, 59, 22, 42), SAND, 11.0)  # monitor case
    c.paint(c.rect(47, 59, 40, 42), SAND, 8.0)  # side shading
    c.paint(c.rect(49, 57, 24, 38), GREEN, 1.0)  # screen
    for row, length in ((50, 9), (52, 12), (54, 6)):
        c.paint(c.rect(row, row + 1, 25, 25 + length), GREEN, 13.0)  # green text
    return _finish(c)


def water_cooler() -> Sprite:
    """Water cooler ~1.15 m to the top of the bottle."""
    c = Canvas()
    c.paint(c.rect(46, 64, 26, 38), CREAM, 9.0)  # cabinet
    c.paint(c.rect(46, 64, 36, 38), CREAM, 7.0)  # side shade
    c.paint(c.rect(50, 52, 28, 30), BLUE, 9.0)  # cold tap
    c.paint(c.rect(50, 52, 33, 35), BRICK, 10.0)  # hot tap
    bottle = c.ellipse(40, 32, 5, 6) | c.rect(44, 46, 30, 34)
    c.paint(bottle, BLUE, 11.0)
    c.paint(bottle & c.rect(0, 64, 27, 30), BLUE, 14.0)  # highlight
    c.paint(bottle & c.rect(0, 64, 35, 39), BLUE, 8.0)  # shadow side
    return _finish(c)


def filing_cabinet() -> Sprite:
    """Five-drawer filing cabinet, ~1.6 m."""
    c = Canvas()
    c.paint(c.rect(26, 64, 22, 42), GREY, 8.0)
    c.paint(c.rect(26, 64, 40, 42), GREY, 6.0)
    for top in (27, 34, 41, 48, 55):
        c.paint(c.rect(top, top + 1, 23, 40), GREY, 4.0)  # drawer gap
        c.paint(c.rect(top + 2, top + 3, 28, 34), GREY, 13.0)  # handle
        c.paint(c.rect(top + 4, top + 6, 29, 33), CREAM, 10.0)  # label
    return _finish(c)


def ficus() -> Sprite:
    """Potted ficus, ~1.9 m: the tallest furniture."""
    c = Canvas()
    rng = np.random.default_rng(14)
    pot = c.rect(53, 64, 25, 39) & (np.abs(c.u + 0.5 - 32) <= 7 - (63 - c.v) * 0.12)
    c.paint(pot, ORANGE, np.where(c.u < 30, 8.0, 7.0))
    c.paint(c.rect(52, 54, 24, 40), ORANGE, 9.0)  # rim
    c.paint(c.rect(40, 52, 31, 33), WOOD, 5.0)  # trunk
    # Leaf clusters: coarse noise gives clumps, fine noise breaks up the edges.
    leaves = noise(rng, 8) * 0.6 + noise(rng, 32) * 0.4
    crown = c.ellipse(32, 32, 14, 14) | c.ellipse(25, 26, 7, 7) | c.ellipse(27, 39, 7, 7)
    c.paint(crown & (leaves > 0.3), GREEN, 3 + leaves * 9)
    return _finish(c)


def exit_sign() -> Sprite:
    c = Canvas()
    c.paint(c.rect(0, 8, 31, 33), GREY, 6.0)  # hanging rod
    c.paint(c.rect(8, 24, 13, 51), GREEN, 9.0)  # sign
    c.paint(c.rect(8, 9, 13, 51) | c.rect(23, 24, 13, 51), GREEN, 6.0)  # edges
    c.text("EXIT", 11, 17, CREAM, 15.0, scale=2)
    return _finish(c, fullbright=True, on_ceiling=True)


SPRITES: dict[str, Sprite] = {
    "desk": desk(),
    "terminal": terminal(),
    "water_cooler": water_cooler(),
    "filing_cabinet": filing_cabinet(),
    "ficus": ficus(),
    "exit": exit_sign(),
}

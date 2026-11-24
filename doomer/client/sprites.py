"""Sprite art: things drawn as flat pictures that always face the camera.

A sprite image is 64x64 and spans one cell wide and one wall tall, standing on
the floor. So texels are the same size as on walls, and something meant to
hang from the ceiling (a light) is simply drawn at the top of the image.

Sprites are keyed by thing type name (doomer/shared/things.py). Things with no
sprite (spawn points) are invisible.

Most sprites are one picture, the same from every side. A sprite can instead
have VIEWS frames round the thing (Doom's had 8), baked
from a voxel model (doomer/client/models.py); the renderer picks the frame
for the direction the thing is seen from, using its map angle.
"""

import math
from dataclasses import dataclass

import numpy as np

from doomer.client.models import VoxelModel, bake, view_for
from doomer.client.palette import (
    BLUE,
    BRICK,
    CREAM,
    GREEN,
    GREY,
    OCHRE,
    SAND,
    WARM_GREY,
    WOOD,
    U8Array,
)
from doomer.client.pixelart import BoolArray, Canvas, noise


@dataclass(frozen=True, slots=True)
class Sprite:
    index: U8Array  # [frame][u][v] palette indices; 1 frame, or one per view
    opaque: BoolArray  # [frame][u][v] True where the sprite is drawn
    fullbright: bool = False  # emits light: not darkened by distance
    on_ceiling: bool = False  # hangs from the ceiling, so ignores desks below it
    # Baked views (models.py) are seen from above: `sink` image rows hang below
    # the base line, and the model reaches `reach` cells towards you, so it's
    # depth-tested from its near side, not its centre (or the desk top in front
    # of the centre would hide the keyboard).
    sink: int = 0
    reach: float = 0.0

    def frame(self, viewer_angle: float, facing: float) -> int:
        """The frame to show to a viewer in direction `viewer_angle` from the
        thing (radians, as map angles), for a thing facing `facing`."""
        if len(self.index) == 1:
            return 0
        return view_for((viewer_angle - facing) % math.tau)


def _finish(canvas: Canvas, fullbright: bool = False, on_ceiling: bool = False) -> Sprite:
    return Sprite(
        np.ascontiguousarray(canvas.index.T[None]),
        np.ascontiguousarray(canvas.opaque.T[None]),
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
    """A beige 1980s PC on a desk: a desktop system unit with two floppy
    drives, a green-screen CRT monitor on it, and a keyboard in front. Drawn
    from its base up (it stands on a desk top); baked from a model.

    Model axes (models.py): x out of the screen, y to the PC's right, z up.
    Exaggerated ~1.5x in width, like the other sprites; about 24 deep, so two
    terminals back to back on a desk pod (0.4 cells apart) don't overlap."""
    m = VoxelModel()
    # System unit, with a fan grille at the back and two drive bays in front.
    m.box((-9, 7), (-12, 12), (0, 5), SAND, 10.0)
    m.box((6, 7), (1, 11), (1, 4), SAND, 7.0)  # drive bay surround
    for y0 in (2, 7):
        m.box((6, 7), (y0, y0 + 3), (2, 3), GREY, 2.0)  # floppy slots
    m.box((6, 7), (-10, -8), (1, 2), BRICK, 9.0)  # power light
    m.box((-9, -8), (-9, -3), (1, 4), GREY, 3.0)  # fan grille
    # Monitor: a boxy case with a tapering CRT behind, a bezel in front.
    m.box((-6, 6), (-9, 9), (5, 19), SAND, 11.0)
    m.box((-10, -6), (-6, 6), (7, 17), SAND, 10.0)  # tube housing
    for z0 in (9, 11, 13, 15):
        m.box((-10, -9), (-4, 4), (z0, z0 + 1), SAND, 5.0)  # vent slots
    m.clear((5, 6), (-7, 7), (8, 17))  # bezel opening
    m.box((4, 5), (-7, 7), (8, 17), GREEN, 1.5, glow=True)  # dark phosphor glass
    # Green text, read left to right from the front: that's from the PC's +y.
    for z0, length in ((15, 10), (13, 12), (11, 7), (9, 4)):
        m.box((4, 5), (6 - length, 6), (z0, z0 + 1), GREEN, 13.0, glow=True)
    # Keyboard in front, with rows of keys on top.
    m.box((8, 15), (-11, 11), (0, 2), WARM_GREY, 11.0)
    for x0 in (9, 11, 13):
        m.box((x0, x0 + 1), (-10, 10), (1, 2), WARM_GREY, 7.0)
    baked = bake(m)
    return Sprite(baked.index, baked.opaque, sink=baked.sink, reach=baked.reach)


# The ramp each ID pass colour is drawn in (doomer/shared/things.py PASS_COLOURS).
PASS_RAMPS = {"blue": BLUE, "red": BRICK, "gold": OCHRE}


def id_pass(ramp: int) -> Sprite:
    """An ID pass on a lanyard, standing on whatever it lies on: ~3x real
    size (as Doom's keys were oversized) so it's easy to spot, and full-bright
    so it shows up in a dark office."""
    c = Canvas()
    for v in range(47, 56):  # the lanyard: two strands meeting at the clip
        spread = round((56 - v) * 0.35)
        c.paint(c.rect(v, v + 1, 31 - spread, 32 - spread), ramp, 10.0)
        c.paint(c.rect(v, v + 1, 32 + spread, 33 + spread), ramp, 10.0)
    c.paint(c.rect(55, 57, 31, 33), GREY, 12.0)  # clip
    c.paint(c.rect(57, 64, 27, 37), CREAM, 14.0)  # the card
    c.paint(c.rect(57, 59, 27, 37), ramp, 12.0)  # colour band
    c.paint(c.rect(60, 63, 28, 31), WARM_GREY, 8.0)  # photo
    c.paint(c.rect(60, 61, 32, 36) | c.rect(62, 63, 32, 35), GREY, 5.0)  # name, number
    return _finish(c, fullbright=True)


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
    "exit": exit_sign(),
    **{f"pass_{colour}": id_pass(ramp) for colour, ramp in PASS_RAMPS.items()},
}

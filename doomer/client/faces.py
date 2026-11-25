"""People's faces: one style for every portrait in the game (ID cards, the
Employee of the Month, the founder's oil painting; later the NPCs).

The style (docs/GAME.md, "People"): early LucasArts adventure-game sprites
(think Monkey Island). Few, chunky pixels: a portrait is a 20 x 20 grid of
them, the head about 9 wide and 11 tall, each grid pixel a solid block on
the texture. No outlines: shapes stand against the background by colour,
their edges just a darker shade. Single-pixel features: dark pixels for
eyes (no whites), a lighter strip for the nose, a short dark line for the
mouth. Warm, saturated two- or three-tone shading in blocks; hair a messy
silhouette with a stray lock or two. A shirt collar round the neck with a
tie, an open neck without.

A face is a `Face` of stock parts (skin, hair style and colour, glasses,
facial hair, mouth, jacket, tie), swapped to make new people; add a part
by adding an entry to its table. `draw_portrait` paints one into a Canvas,
head and shoulders, in a square box: crisp when the box is a multiple of
20 texels (each grid pixel a whole number of texels).
"""

from dataclasses import dataclass

import numpy as np

from doomer.client.palette import (
    BLUE,
    BRICK,
    CREAM,
    GREY,
    OCHRE,
    ORANGE,
    PINK,
    PURPLE,
    SAND,
    STEEL,
    WOOD,
)
from doomer.client.pixelart import BoolArray, Canvas

GRID = 20  # pixels across a portrait

Colour = tuple[int, float]  # (ramp, shade)


@dataclass(frozen=True, slots=True)
class Tones:
    """A colour's shades, lightest to darkest: they may cross ramps (warm
    shadows on skin), as the palette has one hue per ramp."""

    light: Colour
    base: Colour
    shadow: Colour
    deep: Colour


SKINS = {
    "fair": Tones((SAND, 15.0), (SAND, 13.5), (SAND, 11.0), (BRICK, 6.5)),
    "tan": Tones((ORANGE, 14.0), (ORANGE, 12.0), (ORANGE, 9.5), (BRICK, 5.5)),
    "dark": Tones((WOOD, 10.0), (WOOD, 8.0), (WOOD, 6.0), (WOOD, 3.0)),
}
HAIR_COLOURS = {
    "brown": Tones((ORANGE, 10.0), (WOOD, 7.0), (WOOD, 5.0), (WOOD, 3.0)),
    "black": Tones((GREY, 4.0), (GREY, 2.0), (GREY, 1.0), (GREY, 0.0)),
    "blond": Tones((OCHRE, 15.0), (OCHRE, 13.0), (OCHRE, 10.0), (OCHRE, 7.0)),
    "ginger": Tones((ORANGE, 14.0), (ORANGE, 11.0), (BRICK, 9.0), (BRICK, 6.0)),
    "white": Tones((CREAM, 15.0), (CREAM, 13.0), (GREY, 10.0), (GREY, 7.0)),
}
HAIR_STYLES = ("side_part", "slicked", "perm", "mullet", "bob", "bald")
EYEWEAR = ("none", "big_glasses", "aviators")
FACIAL_HAIR = ("none", "moustache", "beard")
MOUTHS = ("grin", "flat", "stern")
JACKETS = {
    "brown": Tones((WOOD, 8.0), (WOOD, 6.0), (WOOD, 4.0), (WOOD, 2.0)),
    "navy": Tones((PURPLE, 6.0), (BLUE, 4.0), (BLUE, 2.5), (BLUE, 1.0)),
    "charcoal": Tones((GREY, 6.0), (GREY, 4.0), (GREY, 2.5), (GREY, 1.0)),
    "grey": Tones((STEEL, 10.0), (STEEL, 8.0), (STEEL, 6.0), (STEEL, 4.0)),
    "pink": Tones((PINK, 13.0), (PINK, 11.0), (PINK, 8.5), (PINK, 6.0)),
}
TIES = {"red": (BRICK, 10.0), "blue": (BLUE, 9.0), "gold": (OCHRE, 12.0), "none": (CREAM, 14.0)}


@dataclass(frozen=True, slots=True)
class Face:
    skin: str = "fair"
    hair: str = "side_part"
    hair_colour: str = "brown"
    eyewear: str = "none"
    facial_hair: str = "none"
    mouth: str = "grin"
    jacket: str = "brown"
    tie: str = "red"


# The cast so far. New people: a new Face of stock parts.
CAST = {
    # Employee of the Month, every month: big glasses, a moustache, a grin.
    "gary": Face("fair", "side_part", "brown", "big_glasses", "moustache", "grin", "brown", "red"),
    # The founder, in oils: white hair round a bald crown, stern.
    "founder": Face("fair", "bald", "white", "none", "none", "stern", "charcoal", "red"),
    # The faces on the ID passes.
    "pass_blue": Face("tan", "perm", "black", "none", "none", "grin", "pink", "none"),
    "pass_red": Face("dark", "side_part", "black", "none", "beard", "flat", "grey", "blue"),
    "pass_gold": Face("fair", "slicked", "blond", "aviators", "none", "grin", "navy", "gold"),
}

_MOUTH = [(10, 8), (10, 9), (10, 10), (10, 11)]


def draw_portrait(
    c: Canvas, v0: int, u0: int, size: int, face: Face, background: Colour | None
) -> None:
    """Head and shoulders of `face` in the size x size box at (v0, u0) on
    `c`, on a flat `background` (None: leave what's there)."""
    k = size / GRID
    # Each texel's grid pixel (row, column), as its centre: shapes are
    # tested at pixel centres, so every pixel is one solid block.
    gv = np.floor((c.v + 0.5 - v0) / k) + 0.5
    gu = np.floor((c.u + 0.5 - u0) / k) + 0.5
    box = (c.v >= v0) & (c.v < v0 + size) & (c.u >= u0) & (c.u < u0 + size)
    skin, hair = SKINS[face.skin], HAIR_COLOURS[face.hair_colour]
    jacket = JACKETS[face.jacket]
    d = np.abs(gu - 10)  # across from the middle

    def at(cells: list[tuple[int, int]]) -> BoolArray:
        """Exact pixels, (row, column)."""
        mask = np.zeros_like(box)
        for r, col in cells:
            mask |= (gv == r + 0.5) & (gu == col + 0.5)
        return mask & box

    def rows(r0: int, r1: int, c0: int, c1: int) -> BoolArray:
        """Pixels in rows r0..r1-1, columns c0..c1-1."""
        mask: BoolArray = (gv > r0) & (gv < r1) & (gu > c0) & (gu < c1) & box
        return mask

    def oval(cv: float, cu: float, rv: float, ru: float) -> BoolArray:
        mask: BoolArray = (((gv - cv) / rv) ** 2 + ((gu - cu) / ru) ** 2 <= 1) & box
        return mask

    def paint(mask: BoolArray, colour: Colour) -> None:
        c.paint(mask, *colour)

    if background is not None:
        paint(box, background)

    # Long hair hangs behind: a bob beside the face, a mullet at the back.
    if face.hair == "bob":
        long_ = oval(7, 10, 6.5, 6.5) | rows(6, 14, 3, 17)
        paint(long_, hair.base)
        paint(long_ & (gu < 6), hair.light)
        paint(long_ & (gu > 15), hair.shadow)
    elif face.hair == "mullet":
        long_ = rows(9, 16, 4, 16)
        paint(long_, hair.base)
        paint(long_ & ((gu < 5) | (gu > 15)), hair.shadow)

    # Shoulders: a jacket in blocks of shade, lapels down to the shirt.
    body = (gv > 14) & (d < 4 + (gv - 14) * 1.6) & box
    paint(body, jacket.base)
    paint(body & (gu < 4), jacket.light)
    paint(body & (gu > 15), jacket.shadow)
    shirt = body & (d < (gv - 13.5) * 0.65)
    paint(shirt, (CREAM, 14.0))
    paint(shirt & (gu > 10), (GREY, 11.0))  # its shaded side
    paint(body & ~shirt & (d < (gv - 13.5) * 0.65 + 1), jacket.deep)  # the lapels' edges

    # Neck and head: a long face, the jaw narrowing; lit from the left.
    neck = rows(12, 15, 7, 13)
    paint(neck, skin.shadow)
    head = (oval(7.5, 10, 6, 4.6) & (gv < 13.5)) | rows(10, 13, 7, 13)
    ears = at([(7, 5), (8, 5), (7, 14), (8, 14)])
    paint(ears, skin.shadow)
    paint(head, skin.base)
    paint(head & (gu < 9) & (gv < 10), skin.light)
    paint(head & (gu > 12), skin.shadow)
    paint(at([(12, 8), (12, 9), (12, 10), (12, 11)]), skin.shadow)  # under the chin

    # Collar and tie, or an open neck.
    tie_colour = TIES[face.tie]
    if face.tie != "none":
        paint(at([(14, 7), (14, 8), (14, 11), (14, 12), (15, 8), (15, 11)]), (CREAM, 15.0))
        paint((gv > 14) & (d < 1) & box, tie_colour)
        paint(at([(14, 9), (14, 10)]), (tie_colour[0], tie_colour[1] - 3.0))  # the knot
    else:
        paint(shirt & (d < (gv - 13.5) * 0.4) & (gv < 17), skin.shadow)

    # Features: single pixels. Brows, eyes, a nose strip, a mouth.
    paint(at([(5, 7), (5, 8), (5, 11), (5, 12)]), hair.shadow)
    paint(at([(6, 8), (6, 11)]), (GREY, 0.0))  # the eyes
    paint(at([(6, 9), (7, 9), (8, 9)]), skin.light)  # the nose's ridge
    paint(at([(8, 10), (6, 12), (7, 12)]), skin.shadow)  # its side, under the brow
    if face.mouth == "grin":
        paint(at([*_MOUTH, (9, 7), (9, 12)]), skin.deep)  # turned up
    elif face.mouth == "flat":
        paint(at(_MOUTH), skin.deep)
    else:  # stern: turned down
        paint(at([*_MOUTH, (11, 7), (11, 12)]), skin.deep)

    if face.facial_hair == "moustache":
        paint(at([(9, 8), (9, 9), (9, 10), (9, 11)]), hair.base)
        paint(at([(9, 7), (9, 12)]), hair.shadow)
    elif face.facial_hair == "beard":
        beard = head & (gv > 8.6) & ~at(_MOUTH)
        paint(beard, hair.base)
        paint(beard & (gu > 12), hair.shadow)

    # Hair on top: a messy silhouette, clumps of shade, a stray lock.
    top = np.zeros_like(box)
    if face.hair in ("side_part", "mullet"):
        top = rows(0, 4, 5, 15)
        top |= at([(1, 4), (2, 4), (3, 4), (2, 15), (3, 15), (4, 6), (4, 13), (4, 14)])
        top |= at([(4, 5), (5, 5), (5, 14), (6, 14)])  # sideburns
        top |= at([(4, 3), (5, 3), (6, 2)] if face.hair == "side_part" else [(3, 3), (5, 16)])
    elif face.hair == "slicked":
        top = rows(1, 4, 5, 15) | at([(2, 4), (3, 4), (2, 15), (3, 15), (4, 5), (4, 14)])
    elif face.hair == "perm":
        top = oval(4, 10, 4.6, 7) | at([(0, 7), (0, 12), (2, 2), (3, 17), (7, 3), (7, 16)])
        top &= ~(head & (gv > 4.6) & (gu > 6) & (gu < 14))
        top |= rows(4, 9, 3, 6) | rows(4, 9, 14, 17)
    elif face.hair == "bob":
        top = rows(1, 5, 4, 16) | at([(1, 3), (5, 4), (5, 15)])
    elif face.hair == "bald":
        top = rows(4, 8, 4, 6) | rows(4, 8, 14, 16) | at([(3, 5), (3, 14)])
        paint(at([(2, 8), (2, 9)]), skin.light)  # shine
    top &= ~ears
    paint(top, hair.base)
    paint(top & (gu < 9) & (gv < 3), hair.light)
    paint(top & (gv < 3) & (((gu * 2 + gv * 3).astype(int) % 5) == 0), hair.shadow)  # clumps
    paint(top & (gv > 3) & (gu > 12), hair.shadow)
    if face.hair == "side_part":
        paint(at([(1, 7), (2, 7)]), hair.deep)  # the parting
    if face.hair == "slicked":
        paint(at([(2, 6), (2, 7), (2, 8), (2, 9), (2, 10)]), hair.light)

    if face.eyewear == "big_glasses":  # dark rims, a pale lens beside each eye
        rims = at(
            [*((5, u) for u in range(7, 13)), (7, 7), (7, 8), (7, 11), (7, 12), (6, 9), (6, 10)]
        )
        paint(rims, (WOOD, 4.5))  # tortoiseshell: the black eyes stay distinct
        paint(at([(6, 7), (6, 12)]), (STEEL, 13.0))
    elif face.eyewear == "aviators":
        paint(at([(6, 7), (6, 8), (6, 11), (6, 12), (7, 8), (7, 11)]), (GREY, 2.0))
        paint(at([(5, u) for u in range(7, 13)]), (OCHRE, 12.0))
        paint(at([(6, 7), (6, 11)]), (CREAM, 11.0))  # glints

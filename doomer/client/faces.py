"""People's faces: one style for every portrait in the game (ID cards, the
Employee of the Month, the founder's oil painting; later the NPCs).

The style (docs/GAME.md, "People"): early-90s VGA adventure-game portraits.
Big heads, long faces and strong chins; a dark outline round the head;
skin in three tones lit from the upper left, plus a highlight; eyes with
whites and dark pupils under heavy brows; a big shaded nose; hair as one
solid shape with a lighter streak. Flat colours, no noise.

A face is a `Face` of stock parts (skin, hair style and colour, glasses,
facial hair, mouth, jacket, tie), swapped to make new people; add a part
by adding an entry to its table. `draw_portrait` paints one into a Canvas,
head and shoulders, in a box of any size: it's designed on a 32 x 32 grid
and scaled, the grid's texels a little taller than wide on every face it's
used on (cards and frames are taller than wide), so heads come out long.
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
    SAND,
    STEEL,
    WOOD,
)
from doomer.client.pixelart import BoolArray, Canvas


@dataclass(frozen=True, slots=True)
class Tones:
    """A ramp and its shades: highlight, base, shadow, outline."""

    ramp: int
    light: float
    base: float
    shadow: float
    outline: float


SKINS = {
    "fair": Tones(SAND, 14.0, 12.5, 10.0, 4.0),
    "tan": Tones(ORANGE, 12.5, 10.5, 8.0, 3.0),
    "dark": Tones(WOOD, 9.5, 7.5, 5.5, 2.0),
}
HAIR_COLOURS = {
    "brown": Tones(WOOD, 7.0, 4.5, 3.0, 1.0),
    "black": Tones(GREY, 4.5, 2.0, 1.0, 0.0),
    "blond": Tones(OCHRE, 14.0, 12.0, 9.0, 4.0),
    "ginger": Tones(ORANGE, 12.0, 9.0, 6.5, 3.0),
    "white": Tones(CREAM, 15.0, 13.5, 10.5, 5.0),
}
HAIR_STYLES = ("side_part", "slicked", "perm", "mullet", "bob", "bald")
EYEWEAR = ("none", "big_glasses", "aviators")
FACIAL_HAIR = ("none", "moustache", "beard")
MOUTHS = ("grin", "flat", "stern")
JACKETS = {
    "brown": Tones(WOOD, 7.0, 5.0, 3.5, 1.5),
    "navy": Tones(BLUE, 5.0, 3.5, 2.5, 1.0),
    "charcoal": Tones(GREY, 5.0, 3.5, 2.5, 1.0),
    "grey": Tones(STEEL, 9.0, 7.0, 5.0, 2.0),
    "pink": Tones(PINK, 12.0, 10.0, 7.5, 3.0),
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


def draw_portrait(
    c: Canvas, v0: int, u0: int, size: int, face: Face, background: tuple[int, float] | None
) -> None:
    """Head and shoulders of `face` in the size x size box at (v0, u0) on
    `c`, on a flat `background` (ramp, shade; None: leave what's there)."""
    k = size / 32.0
    lv, lu = (c.v + 0.5 - v0) / k, (c.u + 0.5 - u0) / k  # the 32 x 32 design grid
    box = (lv >= 0) & (lv < 32) & (lu >= 0) & (lu < 32)
    skin, hair = SKINS[face.skin], HAIR_COLOURS[face.hair_colour]
    jacket = JACKETS[face.jacket]

    def ellipse(cv: float, cu: float, rv: float, ru: float) -> BoolArray:
        mask: BoolArray = ((lv - cv) / rv) ** 2 + ((lu - cu) / ru) ** 2 <= 1
        return mask & box

    def rect(v0_: float, v1: float, u0_: float, u1: float) -> BoolArray:
        mask: BoolArray = (lv >= v0_) & (lv < v1) & (lu >= u0_) & (lu < u1)
        return mask & box

    def outline(mask: BoolArray) -> BoolArray:
        """The texels just outside `mask` (in canvas texels)."""
        grown = mask.copy()
        grown[1:, :] |= mask[:-1, :]
        grown[:-1, :] |= mask[1:, :]
        grown[:, 1:] |= mask[:, :-1]
        grown[:, :-1] |= mask[:, 1:]
        return grown & ~mask & box

    if background is not None:
        c.paint(box, *background)

    # Shoulders: a jacket, lit left, shadowed right; the shirt's V and tie.
    body = ellipse(37, 16, 11, 16) & (lv >= 22)
    c.paint(body, jacket.ramp, jacket.base)
    c.paint(body & (lu < 9), jacket.ramp, jacket.light)
    c.paint(body & (lu > 24), jacket.ramp, jacket.shadow)
    shirt = body & (np.abs(lu - 16) < (lv - 22) * 0.45)
    c.paint(shirt, CREAM, 14.0)
    tie_ramp, tie_shade = TIES[face.tie]
    if face.tie != "none":
        c.paint(
            shirt & (np.abs(lu - 16) < 1.2 + (lv - 24) * 0.08) & (lv >= 24), tie_ramp, tie_shade
        )
        c.paint(rect(24, 25.5, 15, 17), tie_ramp, tie_shade + 1.5)  # the knot
    c.paint(outline(body) & (lv >= 23), jacket.ramp, jacket.outline)

    # Long hair falls behind the head and neck: a bob beside the face, a
    # mullet only at the back (short on top), down to the shoulders.
    if face.hair in ("mullet", "bob"):
        if face.hair == "bob":
            long_ = ellipse(14, 16, 10.5, 10.5) | rect(14, 23, 5.5, 26.5)
        else:
            long_ = rect(15, 26, 9, 23) | ellipse(25, 16, 3, 8)
        c.paint(long_, hair.ramp, hair.base)
        c.paint(long_ & (lu < 11), hair.ramp, hair.light)
        c.paint(outline(long_), hair.ramp, hair.outline)

    # Neck, then the head: a long face, a strong chin, ears.
    neck = rect(20, 26, 12.5, 19.5)
    c.paint(neck, skin.ramp, skin.shadow)
    head = ellipse(12.5, 16, 9.5, 7.5) | ellipse(17, 16, 5, 6.5)
    ears = ellipse(13.5, 8.5, 2.2, 1.4) | ellipse(13.5, 23.5, 2.2, 1.4)
    c.paint(ears, skin.ramp, skin.shadow)
    c.paint(head, skin.ramp, skin.base)
    c.paint(head & ellipse(10, 13, 5, 3.5), skin.ramp, skin.light)  # lit from the upper left
    c.paint(head & (lu > 20.5), skin.ramp, skin.shadow)
    c.paint(outline(head | ears | neck) & ~body, skin.ramp, skin.outline)

    # Eyes (whites, pupils looking a little left), brows, a big nose.
    for eu in (12.5, 18.5):
        c.paint(rect(12, 13.6, eu - 1.6, eu + 1.6), CREAM, 15.0)
        c.paint(rect(12, 13.6, eu - 1.0, eu + 0.2), GREY, 0.0)
        c.paint(rect(11.2, 12, eu - 1.8, eu + 1.8), skin.ramp, skin.outline)  # upper lid
        c.paint(rect(9.8, 10.8, eu - 2.0, eu + 2.0), hair.ramp, hair.shadow)
    c.paint(rect(13, 17, 15.2, 17.2), skin.ramp, skin.light)  # the nose's lit ridge
    c.paint(rect(13, 17.4, 17.2, 18.4), skin.ramp, skin.shadow)  # its shadow side
    c.paint(rect(17, 18, 14.5, 18.5), skin.ramp, skin.outline)  # nostrils
    mouth_v = 19.6
    if face.mouth == "grin":
        c.paint(rect(mouth_v, mouth_v + 1, 13, 19.5), BRICK, 5.0)
        c.paint(
            rect(mouth_v - 0.8, mouth_v, 12, 13) | rect(mouth_v - 0.8, mouth_v, 19.5, 20.5),
            BRICK,
            5.0,
        )
    elif face.mouth == "flat":
        c.paint(rect(mouth_v, mouth_v + 1, 13.5, 18.5), BRICK, 5.0)
    else:  # stern: turned down
        c.paint(rect(mouth_v, mouth_v + 1, 13.5, 18.5), BRICK, 4.0)
        c.paint(
            rect(mouth_v + 1, mouth_v + 1.8, 12.5, 13.5)
            | rect(mouth_v + 1, mouth_v + 1.8, 18.5, 19.5),
            BRICK,
            4.0,
        )

    if face.facial_hair == "moustache":
        c.paint(rect(18.2, 19.6, 12.5, 19.5), hair.ramp, hair.base)
        c.paint(rect(18.2, 18.9, 12.5, 16), hair.ramp, hair.light)
    elif face.facial_hair == "beard":
        beard = head & (lv >= 17.5) & ~rect(mouth_v, mouth_v + 1, 13, 19)
        c.paint(beard, hair.ramp, hair.base)
        c.paint(beard & (lu < 13), hair.ramp, hair.light)

    # Hair on top: a shape over the crown with a fringe swept across, a
    # lighter streak, sideburns down to the ears, a dark outline.
    sideburns = (rect(9.5, 13, 7.6, 9.4) | rect(9.5, 13, 22.6, 24.4)) & ~ears
    fringe = lv < 9.0 + (lu - 9) * 0.13  # lower to the right: swept across
    top = np.zeros_like(box)
    if face.hair in ("side_part", "mullet", "bob"):
        top = (ellipse(8.5, 16, 6.8, 9.6) & fringe) | sideburns
        if face.hair == "bob":
            top |= rect(7, 10, 8.5, 23.5)  # a straight fringe
    elif face.hair == "slicked":
        top = (ellipse(8.5, 16, 6.2, 9) & (lv < 8.5)) | sideburns
    elif face.hair == "perm":
        for cv, cu in (
            (5, 9),
            (3.5, 13),
            (3, 18.5),
            (4.5, 23),
            (7.5, 7),
            (7.5, 25),
            (10.5, 6.5),
            (10.5, 25.5),
        ):
            top |= ellipse(cv, cu, 3.2, 3.2)
        top &= ~(head & (lv >= 8.5) & (lu > 9) & (lu < 23))
    elif face.hair == "bald":
        top = (rect(9, 15, 7.4, 10) | rect(9, 15, 22, 24.6)) & ~ears
        c.paint(head & ellipse(5.5, 13, 1.5, 2.5), skin.ramp, skin.light + 1.0)  # shine
    c.paint(top, hair.ramp, hair.base)
    c.paint(top & (lu < 13) & (lv < 7), hair.ramp, hair.light)
    if face.hair == "slicked":  # combed straight back: lines of shine
        c.paint(top & (np.abs(lv - 4.5 - (lu - 10) * 0.15) < 0.6), hair.ramp, hair.light)
    if face.hair == "side_part":
        c.paint(top & rect(2.5, 6.5, 11.4, 12.4), hair.ramp, hair.outline)  # the parting
    c.paint(outline(top) & ~head, hair.ramp, hair.outline)
    c.paint(outline(top) & head & (lv < 11), hair.ramp, hair.shadow)  # its edge on the brow

    # Glasses over the eyes.
    if face.eyewear == "big_glasses":
        for eu in (12.5, 18.5):
            lens = rect(10.6, 15.2, eu - 2.8, eu + 2.8)
            c.paint(lens & ~rect(11.4, 14.4, eu - 2.0, eu + 2.0), GREY, 1.0)
        c.paint(rect(11.6, 12.4, 15.3, 15.7), GREY, 1.0)
    elif face.eyewear == "aviators":
        for eu in (12.5, 18.5):
            lens = rect(11, 15, eu - 2.6, eu + 2.6)
            c.paint(lens, GREY, 2.0)
            c.paint(lens & rect(11.4, 12.4, eu - 1.8, eu - 0.2), CREAM, 12.0)  # a glint
            c.paint(outline(lens), OCHRE, 12.0)

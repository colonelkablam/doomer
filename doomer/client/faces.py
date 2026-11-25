"""People's faces: one style for every portrait in the game (ID cards, the
Employee of the Month, the founder's oil painting; later the NPCs).

The style (docs/GAME.md, "People"): early-90s LucasArts-style cartoon
portraits. Big round heads on small shoulders, a short neck; a dark ink
outline round everything; skin in three tones lit from the upper left;
big round outlined eyes with dot pupils, arched brows; a bulbous nose
with a shine; a wide (toothy, if grinning) mouth; hair as one solid shape
with a lighter streak. A collar round the neck with a tie, an open neck
without. Flat colours, no noise.

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
    "fair": Tones(SAND, 14.5, 13.0, 10.5, 4.0),
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
    ink = (GREY, 1.0)  # the cartoon outline round everything

    # Long hair falls behind the head and neck: a bob beside the face, a
    # mullet only at the back (short on top), down to the shoulders.
    if face.hair in ("mullet", "bob"):
        if face.hair == "bob":
            long_ = ellipse(13, 16, 11, 12) | rect(13, 23, 4, 28)
        else:
            long_ = rect(15, 26.5, 8.5, 23.5) | ellipse(25.5, 16, 3, 8)
        c.paint(long_, hair.ramp, hair.base)
        c.paint(long_ & (lu < 10), hair.ramp, hair.light)
        c.paint(outline(long_), *ink)

    # Small shoulders (cartoon bodies are small, heads big), lit left.
    body = ellipse(38.5, 16, 12.5, 13.5) & (lv >= 25)
    c.paint(body, jacket.ramp, jacket.base)
    c.paint(body & (lu < 8.5), jacket.ramp, jacket.light)
    c.paint(body & (lu > 23.5), jacket.ramp, jacket.shadow)
    d = np.abs(lu - 16)  # across from the middle
    tie_ramp, tie_shade = TIES[face.tie]
    # A short neck; with a tie, a buttoned collar round it, its points
    # turned down either side of the knot; without, an open neck.
    neck = rect(20, 26.5, 12.6, 19.4)
    c.paint(neck, skin.ramp, skin.shadow)
    if face.tie != "none":
        shirt = body & (d < (lv - 24.5) * 0.42)
        c.paint(shirt, CREAM, 14.5)
        tie = shirt & (d < 1.0 + (lv - 26.5) * 0.1) & (lv >= 26.5)
        c.paint(tie, tie_ramp, tie_shade)
        c.paint(outline(tie) & shirt, tie_ramp, tie_shade - 4.0)
        knot = ellipse(25.8, 16, 1.1, 1.3)
        c.paint(knot, tie_ramp, tie_shade + 1.5)
        collar = (lv >= 24.2) & (d >= 0.7) & (d <= 4.2) & (lv <= 25.2 + (4.2 - d) * 0.6) & box
        c.paint(collar, CREAM, 15.0)
        c.paint(collar & (lu > 16), CREAM, 12.5)  # the right point in shade
        c.paint(outline(collar) & ~knot, *ink)
        c.paint(outline(knot) & ~collar, *ink)
    else:
        shirt = body & (d < 1.5 + (lv - 24.5) * 0.5)
        c.paint(shirt, CREAM, 14.5)
        c.paint(shirt & (d < (lv - 24.5) * 0.35) & (lv < 29), skin.ramp, skin.shadow)  # open neck
    c.paint(outline(body) & (lv >= 25.5), *ink)
    c.paint(outline(shirt) & body & ~shirt, jacket.ramp, jacket.outline)  # the lapels' edges

    # The head: big and round, a firm jaw, ears; lit from the upper left.
    head = ellipse(12.5, 16, 10, 9) | ellipse(17.5, 16, 5.5, 7.2)
    ears = ellipse(13.5, 7, 2.6, 1.7) | ellipse(13.5, 25, 2.6, 1.7)
    c.paint(ears, skin.ramp, skin.base)
    c.paint(
        ears & ellipse(13.5, 7, 1.2, 0.7) | ears & ellipse(13.5, 25, 1.2, 0.7),
        skin.ramp,
        skin.shadow,
    )
    c.paint(head, skin.ramp, skin.base)
    c.paint(head & ellipse(9, 12, 5, 4), skin.ramp, skin.light)
    c.paint(head & (lu > 21.5), skin.ramp, skin.shadow)
    c.paint(head & (lv > 19.5) & (lu > 18), skin.ramp, skin.shadow)  # under the jaw
    c.paint(outline(head | ears) & ~(body | rect(19, 27, 12.6, 19.4)), *ink)
    c.paint(outline(neck) & ~head & ~body & (lv < 25.5) & ~ellipse(25.5, 16, 2, 6), *ink)

    # Big round eyes, outlined, dot pupils looking a little left; arched
    # brows; a bulbous nose with a shine; a wide mouth.
    for eu in (12.2, 19.8):
        white = ellipse(12.3, eu, 2.2, 2.3)
        c.paint(white, CREAM, 15.0)
        c.paint(rect(11.8, 13.4, eu - 1.1, eu + 0.1), GREY, 0.0)
        c.paint(outline(white) & ~ellipse(14.2, eu, 1.0, 2.5), *ink)
        brow = rect(8.6, 9.8, eu - 2.4, eu + 2.4) & ~(
            rect(8.6, 9.2, eu - 2.4, eu - 1.6) | rect(8.6, 9.2, eu + 1.6, eu + 2.4)
        )
        c.paint(brow, hair.ramp, min(hair.shadow, hair.base))
    nose = ellipse(16.3, 16.6, 2.3, 2.6) | rect(12, 16, 15.4, 17.6)
    c.paint(nose, skin.ramp, skin.base)
    c.paint(ellipse(15.6, 15.8, 0.9, 0.9), skin.ramp, skin.light + 1.0)  # its shine
    c.paint(nose & (lu > 17.6), skin.ramp, skin.shadow)
    c.paint(outline(nose) & (lv > 15) & ~rect(12, 15, 0, 32), *ink)
    mouth_v = 20.2
    if face.mouth == "grin":  # wide and toothy
        grin = rect(mouth_v - 0.4, mouth_v + 1.4, 11.5, 20.5) & (
            lv <= mouth_v + 1.4 - (d / 4.5) ** 2 * 1.6
        )
        c.paint(grin, CREAM, 15.0)
        c.paint(outline(grin), BRICK, 3.0)
    elif face.mouth == "flat":
        c.paint(rect(mouth_v, mouth_v + 0.9, 13, 19), BRICK, 3.0)
    else:  # stern: turned down at the corners
        c.paint(rect(mouth_v, mouth_v + 0.9, 13, 19), BRICK, 3.0)
        low = (mouth_v + 0.9, mouth_v + 1.8)
        c.paint(rect(*low, 12, 13) | rect(*low, 19, 20), BRICK, 3.0)

    if face.facial_hair == "moustache":  # bushy, under the nose
        tache = ellipse(19.1, 14, 1.0, 2.6) | ellipse(19.1, 18.6, 1.0, 2.6)
        c.paint(tache, hair.ramp, hair.base)
        c.paint(tache & (lu < 14), hair.ramp, hair.light)
        c.paint(outline(tache) & ~nose, *ink)
    elif face.facial_hair == "beard":
        beard = head & (lv >= 18.4) & ~rect(mouth_v - 0.5, mouth_v + 1.5, 11.5, 20.5)
        c.paint(beard, hair.ramp, hair.base)
        c.paint(beard & (lu < 12.5), hair.ramp, hair.light)

    # Hair on top: a shape over the crown with a fringe swept across, a
    # lighter streak, sideburns down to the ears, outlined.
    sideburns = (rect(9.5, 13, 6.9, 8.6) | rect(9.5, 13, 23.4, 25.1)) & ~ears
    fringe = lv < 7.8 + (lu - 8) * 0.13  # lower to the right: swept across
    top = np.zeros_like(box)
    if face.hair in ("side_part", "mullet", "bob"):
        top = (ellipse(7.8, 16, 6.2, 10.4) & fringe) | sideburns
        if face.hair == "bob":
            top |= rect(6, 8.6, 7.5, 24.5)  # a straight fringe
    elif face.hair == "slicked":
        top = (ellipse(8, 16, 5.8, 9.8) & (lv < 7.5)) | sideburns
    elif face.hair == "perm":
        curls = ((4.5, 8.5), (2.8, 13), (2.3, 18.5), (3.8, 23.5), (7, 6), (7, 26), (10.5, 5.5))
        for cv, cu in (*curls, (10.5, 26.5)):
            top |= ellipse(cv, cu, 3.4, 3.4)
        top &= ~(head & (lv >= 7.5) & (lu > 8.5) & (lu < 23.5))
    elif face.hair == "bald":
        top = (rect(8.5, 14.5, 6.6, 9.4) | rect(8.5, 14.5, 22.6, 25.4)) & ~ears
        c.paint(head & ellipse(4.6, 12.5, 1.4, 2.6), skin.ramp, skin.light + 1.5)  # shine
    c.paint(top, hair.ramp, hair.base)
    c.paint(top & (lu < 13) & (lv < 6), hair.ramp, hair.light)
    if face.hair == "slicked":  # combed straight back: a line of shine
        c.paint(top & (np.abs(lv - 4 - (lu - 10) * 0.15) < 0.6), hair.ramp, hair.light)
    if face.hair == "side_part":
        c.paint(top & rect(1.5, 5.5, 11, 12), hair.ramp, hair.outline)  # the parting
    c.paint(outline(top) & ~head, *ink)
    c.paint(outline(top) & head & (lv < 11), hair.ramp, hair.outline)  # its edge on the brow

    # Glasses over the eyes.
    if face.eyewear == "big_glasses":
        for eu in (12.2, 19.8):
            frame = rect(9.6, 15.4, eu - 3.4, eu + 3.4) & ~rect(10.4, 14.6, eu - 2.6, eu + 2.6)
            c.paint(frame, *ink)
        c.paint(rect(11.4, 12.2, 15.6, 16.4), *ink)
    elif face.eyewear == "aviators":
        for eu in (12.2, 19.8):
            lens = (
                rect(10.4, 15, eu - 3, eu + 3)
                & ~rect(14.2, 15, eu - 3, eu - 2)
                & ~rect(14.2, 15, eu + 2, eu + 3)
            )
            c.paint(lens, GREY, 2.0)
            c.paint(lens & rect(10.8, 11.8, eu - 2.2, eu - 0.4), CREAM, 12.0)  # a glint
            c.paint(outline(lens), OCHRE, 12.0)

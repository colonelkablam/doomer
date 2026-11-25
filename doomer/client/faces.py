"""People's faces: one style for every portrait in the game (ID cards, the
Employee of the Month, the founder's oil painting; later the NPCs).

The style (docs/GAME.md, "People"): early LucasArts adventure-game sprites
(think Monkey Island). A portrait is a 20 x 20 grid of chunky pixels, the
head 10 wide, facing straight ahead; no outlines, edges a darker shade;
single-pixel features, well apart (eyes five apart, a small mouth in the
middle); warm two- or three-tone shading, lit from the left; simple,
clean hair shapes.

Faces are built from hand-drawn templates (pixel art at this size is drawn,
not computed): each part is a 20 x 20 grid of letters, one per pixel, `.`
for none. A letter is a tone (L light, B base, S shadow, D deep) or a fixed
colour (E an eye, W a white shirt...), coloured from the face's parts when
it's drawn. `draw_portrait` stacks the layers back to front: hair behind,
shoulders, head, features, mouth, facial hair, hair on top, glasses. A new
look: a new template in a part's table; a new person: a new `Face`.
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
from doomer.client.pixelart import Canvas

GRID = 20  # pixels across a portrait

Colour = tuple[int, float]  # (ramp, shade)
Template = tuple[str, ...]  # GRID rows of GRID letters


@dataclass(frozen=True, slots=True)
class Tones:
    """A colour's shades, lightest to darkest: they may cross ramps (warm
    shadows on skin), as the palette has one hue per ramp."""

    light: Colour
    base: Colour
    shadow: Colour
    deep: Colour

    def letters(self) -> dict[str, Colour]:
        return {"L": self.light, "B": self.base, "S": self.shadow, "D": self.deep}


_BLANK = "." * GRID


def _t(*rows: str) -> Template:
    """A template from its rows, top first, padded to GRID rows."""
    padded = rows + (_BLANK,) * (GRID - len(rows))
    assert len(padded) == GRID and all(len(r) == GRID for r in padded), rows
    return padded


def _below(n: int, *rows: str) -> Template:
    """A template whose rows start `n` rows down."""
    return _t(*((_BLANK,) * n), *rows)


# --- the templates (columns: 01234567890123456789) ----------------------------------

HEAD = _t(  # straight ahead: ten wide, ears either side, the jaw narrowing
    "....................",
    ".......BBBBBB.......",
    "......LBBBBBBB......",
    ".....LLBBBBBBBS.....",
    ".....LLBBBBBBBS.....",
    ".....LBBBBBBBBS.....",
    "....SLBBBBBBBBSS....",
    "....SLBBBBBBBBSS....",
    ".....LBBBBBBBBS.....",
    ".....BBBBBBBBBS.....",
    "......BBBBBBBS......",
    "......BBBBBBBS......",
    ".......SBBBBS.......",
    "........SSSS........",
)
# Brows (W), eyes (E) five apart with skin outside each, the nose between
# them: its lit ridge (N) and shaded side (n).
FEATURES = _below(
    5,
    "......WW....WW......",
    ".......E.N..E.......",
    ".........Nn.........",
    ".........Nn.........",
    ".........nn.........",
)
MOUTHS = {  # small and in the middle: never across the face
    "grin": _below(10, "........M..M........", ".........MM........."),
    "flat": _below(11, "........MMMM........"),
    "stern": _below(10, ".........MM.........", "........M..M........"),
}
# Shoulders: a jacket (L B S, lapels D) over a white shirt (W, shaded G);
# with a tie (T, knot K) a collar (C); without, an open neck (n).
BODIES = {
    "tie": _below(
        14,
        ".....LBDCKKCDBBS....",
        "...LLBBDWCTTCGDBBSS.",
        "..LLBBBDWWTTGGDBBBSS",
        ".LLBBBBDWWTTGGDBBBSS",
        "LLBBBBBDWWTTGGDBBBSS",
        "LLBBBBBDWWTTGGDBBBSS",
    ),
    "open": _below(
        14,
        ".....LBDWnnWDBBS....",
        "...LLBBDWWnnGGDBBSS.",
        "..LLBBBDWWWGGGDBBBSS",
        ".LLBBBBDWWWGGGDBBBSS",
        "LLBBBBBDWWWGGGDBBBSS",
        "LLBBBBBDWWWGGGDBBBSS",
    ),
}
# Hair: on top of the head, and (long styles) behind it.
_SIDE_PART = (
    "......BBBBBBB.......",
    "....LLLBBBBBBBB.....",
    "...LLLBBBBBBBBBS....",
    "...LBBBBBBBBBBBSS...",
    "...BBB.BBBBBBBBBS...",
    ".....B........BS....",
)
HAIR_TOPS = {
    "side_part": _t(*_SIDE_PART),
    "mullet": _t(*_SIDE_PART),
    "slicked": _below(
        1,
        ".....LLLLLBBBBB.....",
        "....BBBBBBBBBBBS....",
        "....BBBBBBBBBBBS....",
        "....B..........S....",
        "....B..........S....",
    ),
    "perm": _t(
        ".....BBBBBBBBBB.....",
        "...LLBBBBBBBBBBBB...",
        "..LLLBBBBBBBBBBBBS..",
        "..LLBBBBBBBBBBBBBS..",
        "..LBBB........BBBS..",
        "..BBB..........BBS..",
        "..BBB..........BBS..",
        "..BB............BS..",
        "...B............S...",
    ),
    "bob": _t(
        "......BBBBBBB.......",
        "....LLLBBBBBBBB.....",
        "...LLLBBBBBBBBBS....",
        "...LBBBBBBBBBBBSS...",
        "...BBBBBBBBBBBBSS...",
    ),
    "bald": _below(
        3,
        "....B..........S....",
        "....BB........SS....",
        "....BB........SS....",
        "....BB........SS....",
        "....B..........S....",
    ),
}
HAIR_BACKS = {
    "bob": _below(4, *(("..LLBBBBBBBBBBBBSS..",) * 9), "..LLBB........BBSS.."),
    "mullet": _below(9, *(("....BBBBBBBBBBBS....",) * 6)),
}
BALD_SHINE = _below(2, "........LL..........")
FACIAL_HAIR_TEMPLATES = {
    "moustache": _below(10, "........BBBS........"),
    "beard": _below(
        9,
        ".....BB......BS.....",
        "......BB....BS......",
        "......BBB..BBS......",
        ".......BBBBBS.......",
    ),
}
# Glasses: closed square rims (R) round each eye, a bridge between, pale
# lenses (P) either side of the eye; aviators the same shape in gold
# (F, a bridge between), dark
# lenses (A), a gold bar (F), a glint (G).
EYEWEAR_TEMPLATES = {
    "big_glasses": _below(
        5,
        "......RRR..RRR......",
        "......P.PRRP.P......",
        "......RRR..RRR......",
    ),
    "aviators": _below(
        5,
        "......FFFFFFFF......",
        "......GAAFFGAA......",
        "......FFF..FFF......",
    ),
}
GLASSES_COLOURS = {
    "R": (WOOD, 4.5),
    "P": (STEEL, 13.0),
    "A": (GREY, 2.0),
    "F": (OCHRE, 12.0),
    "G": (CREAM, 11.0),
}

# --- the parts --------------------------------------------------------------------

SKINS = {
    "fair": Tones((SAND, 15.0), (SAND, 13.5), (SAND, 11.0), (BRICK, 6.5)),
    "tan": Tones((ORANGE, 14.0), (ORANGE, 12.0), (ORANGE, 9.5), (BRICK, 5.5)),
    "dark": Tones((WOOD, 10.0), (WOOD, 8.0), (WOOD, 6.0), (WOOD, 3.0)),
}
HAIR_COLOURS = {
    "brown": Tones((ORANGE, 9.0), (WOOD, 6.5), (WOOD, 4.5), (WOOD, 3.0)),
    "black": Tones((GREY, 4.0), (GREY, 2.0), (GREY, 1.0), (GREY, 0.0)),
    "blond": Tones((OCHRE, 15.0), (OCHRE, 13.0), (OCHRE, 10.0), (OCHRE, 7.0)),
    "ginger": Tones((ORANGE, 14.0), (ORANGE, 11.0), (BRICK, 9.0), (BRICK, 6.0)),
    "white": Tones((CREAM, 15.0), (CREAM, 13.0), (GREY, 10.0), (GREY, 7.0)),
}
HAIR_STYLES = tuple(HAIR_TOPS)
EYEWEAR = ("none", *EYEWEAR_TEMPLATES)
FACIAL_HAIR = ("none", *FACIAL_HAIR_TEMPLATES)
MOUTH_SHAPES = tuple(MOUTHS)
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


def draw_portrait(
    c: Canvas, v0: int, u0: int, size: int, face: Face, background: Colour | None
) -> None:
    """Head and shoulders of `face` in the size x size box at (v0, u0) on
    `c`, on a flat `background` (None: leave what's there). Crisp when
    `size` is a multiple of GRID (each pixel a whole number of texels)."""
    k = size / GRID
    box = (c.v >= v0) & (c.v < v0 + size) & (c.u >= u0) & (c.u < u0 + size)
    row = np.clip(np.floor((c.v + 0.5 - v0) / k).astype(np.intp), 0, GRID - 1)
    col = np.clip(np.floor((c.u + 0.5 - u0) / k).astype(np.intp), 0, GRID - 1)

    def layer(template: Template, colours: dict[str, Colour]) -> None:
        letters = np.array([list(r) for r in template])[row, col]
        for letter, colour in colours.items():
            c.paint(box & (letters == letter), *colour)

    skin, hair = SKINS[face.skin], HAIR_COLOURS[face.hair_colour]
    tie = TIES[face.tie]
    if background is not None:
        c.paint(box, *background)
    if face.hair in HAIR_BACKS:
        layer(HAIR_BACKS[face.hair], hair.letters())
    body = {
        **JACKETS[face.jacket].letters(),
        "W": (CREAM, 14.0),
        "G": (GREY, 11.0),
        "C": (CREAM, 15.0),
        "T": tie,
        "K": (tie[0], tie[1] - 3.0),
        "n": skin.shadow,
    }
    layer(BODIES["open" if face.tie == "none" else "tie"], body)
    layer(HEAD, skin.letters())
    layer(FEATURES, {"W": hair.shadow, "E": (GREY, 0.0), "N": skin.light, "n": skin.shadow})
    layer(MOUTHS[face.mouth], {"M": skin.deep})
    if face.facial_hair in FACIAL_HAIR_TEMPLATES:
        layer(FACIAL_HAIR_TEMPLATES[face.facial_hair], hair.letters())
    layer(HAIR_TOPS[face.hair], hair.letters())
    if face.hair == "bald":
        layer(BALD_SHINE, skin.letters())
    if face.eyewear in EYEWEAR_TEMPLATES:
        layer(EYEWEAR_TEMPLATES[face.eyewear], GLASSES_COLOURS)

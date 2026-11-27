"""Helpers for drawing 64x64 palette-indexed art with numpy.

Shared by wall textures and sprites. Art is drawn in (v, u) = (row, column)
order, the way people think about images; callers transpose to [u][v] for the
renderer, which reads one screen column as one contiguous run of memory.
"""

import numpy as np
import numpy.typing as npt

from ok_doomer.client.palette import SHADES, U8Array

ART_BITS = 6
ART_SIZE = 1 << ART_BITS  # 64: a power of two, so coordinates wrap with & 63
ART_MASK = ART_SIZE - 1

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]
BoolArray = npt.NDArray[np.bool_]
Shade = float | FloatArray


def indices(ramp: int | IntArray, shade: Shade) -> U8Array:
    """Palette indices from ramp(s) and (fractional) shade(s): (ramp << 4) | shade."""
    shade_i = np.clip(np.round(shade), 0, SHADES - 1).astype(np.int64)
    result: U8Array = ((np.asarray(ramp, dtype=np.int64) << 4) | shade_i).astype(np.uint8)
    return result


def grid() -> tuple[IntArray, IntArray]:
    """(v, u) coordinate arrays, each (64, 64)."""
    v, u = np.mgrid[0:ART_SIZE, 0:ART_SIZE]
    return v.astype(np.int64), u.astype(np.int64)


def noise(rng: np.random.Generator, cells: int) -> FloatArray:
    """Tileable smooth value noise in [0, 1): bilinear interpolation of a random grid."""
    lattice = rng.random((cells, cells))
    pos = np.arange(ART_SIZE) * cells / ART_SIZE
    i0 = pos.astype(np.int64)
    i1 = (i0 + 1) % cells  # wrap, so textures tile seamlessly
    t = pos - i0
    rows = lattice[i0] * (1 - t)[:, None] + lattice[i1] * t[:, None]
    out: FloatArray = rows[:, i0] * (1 - t)[None, :] + rows[:, i1] * t[None, :]
    return out


class Canvas:
    """A 64x64 drawing surface of palette indices plus an opacity mask."""

    def __init__(self) -> None:
        self.index = np.zeros((ART_SIZE, ART_SIZE), dtype=np.uint8)
        self.opaque = np.zeros((ART_SIZE, ART_SIZE), dtype=np.bool_)
        self.v, self.u = grid()

    def paint(self, where: BoolArray, ramp: int, shade: Shade) -> None:
        """Paint ramp/shade wherever `where` is true. `shade` may be a full array."""
        colours = np.broadcast_to(indices(ramp, shade), self.index.shape)
        self.index[where] = colours[where]
        self.opaque |= where

    def rect(self, v0: int, v1: int, u0: int, u1: int) -> BoolArray:
        """Mask of the half-open box [v0, v1) x [u0, u1)."""
        mask: BoolArray = (self.v >= v0) & (self.v < v1) & (self.u >= u0) & (self.u < u1)
        return mask

    def ellipse(self, cv: float, cu: float, rv: float, ru: float) -> BoolArray:
        mask: BoolArray = ((self.v + 0.5 - cv) / rv) ** 2 + ((self.u + 0.5 - cu) / ru) ** 2 <= 1
        return mask

    def text(
        self,
        message: str,
        v0: int,
        u0: int,
        ramp: int,
        shade: float,
        scale: int = 1,
        scale_u: int | None = None,
    ) -> None:
        """Draw text in a tiny 3x5 pixel font. `scale_u`: a different scale
        across (for a face stretched wider than it's tall)."""
        su = scale if scale_u is None else scale_u
        for i, ch in enumerate(message):
            rows = FONT_3X5[ch]
            for gy, row in enumerate(rows):
                for gx, bit in enumerate(row):
                    if bit == "#":
                        top = v0 + gy * scale
                        left = u0 + (i * 4 + gx) * su
                        self.paint(self.rect(top, top + scale, left, left + su), ramp, shade)

    def text_width(self, message: str, scale_u: int = 1) -> int:
        """Texels across `message` takes (4 a glyph, less the last gap)."""
        return (len(message) * 4 - 1) * scale_u


FONT_3X5: dict[str, tuple[str, ...]] = {
    "E": ("###", "#..", "##.", "#..", "###"),
    "X": ("#.#", "#.#", ".#.", "#.#", "#.#"),
    "I": ("###", ".#.", ".#.", ".#.", "###"),
    "T": ("###", ".#.", ".#.", ".#.", ".#."),
    # Lift buttons and indicators.
    "G": ("###", "#..", "#.#", "#.#", "###"),
    "O": ("###", "#.#", "#.#", "#.#", "###"),
    "S": ("###", "#..", "###", "..#", "###"),
    "B": ("##.", "#.#", "##.", "#.#", "##."),
    "P": ("##.", "#.#", "##.", "#..", "#.."),
    "1": (".#.", "##.", ".#.", ".#.", "###"),
    "2": ("##.", "..#", ".#.", "#..", "###"),
    "3": ("##.", "..#", ".#.", "..#", "##."),
    "4": ("#.#", "#.#", "###", "..#", "..#"),
    "5": ("###", "#..", "##.", "..#", "##."),
    # The rest, for posters and signs.
    " ": ("...", "...", "...", "...", "..."),
    "A": (".#.", "#.#", "###", "#.#", "#.#"),
    "C": (".##", "#..", "#..", "#..", ".##"),
    "D": ("##.", "#.#", "#.#", "#.#", "##."),
    "F": ("###", "#..", "##.", "#..", "#.."),
    "H": ("#.#", "#.#", "###", "#.#", "#.#"),
    "J": ("..#", "..#", "..#", "#.#", ".#."),
    "K": ("#.#", "#.#", "##.", "#.#", "#.#"),
    "L": ("#..", "#..", "#..", "#..", "###"),
    "M": ("#.#", "###", "###", "#.#", "#.#"),
    "N": ("##.", "#.#", "#.#", "#.#", "#.#"),
    "Q": (".#.", "#.#", "#.#", "##.", ".##"),
    "R": ("##.", "#.#", "##.", "#.#", "#.#"),
    "U": ("#.#", "#.#", "#.#", "#.#", "###"),
    "V": ("#.#", "#.#", "#.#", "#.#", ".#."),
    "W": ("#.#", "#.#", "###", "###", "#.#"),
    "Y": ("#.#", "#.#", ".#.", ".#.", ".#."),
    "Z": ("###", "..#", ".#.", "#..", "###"),
    "0": ("###", "#.#", "#.#", "#.#", "###"),
    "6": (".##", "#..", "###", "#.#", "###"),
    "7": ("###", "..#", ".#.", ".#.", ".#."),
    "8": ("###", "#.#", "###", "#.#", "###"),
    "9": ("###", "#.#", "###", "..#", "##."),
    ".": ("...", "...", "...", "...", ".#."),
    "!": (".#.", ".#.", ".#.", "...", ".#."),
    "+": ("...", ".#.", "###", ".#.", "..."),
    "=": ("...", "###", "...", "###", "..."),
}


# A 5x7 font, capitals (as old terminals had), digits and the punctuation
# the HUD's messages use. Started for the lobby's brass letters (boxes.py);
# those glyphs are unchanged. One column between letters when set.
FONT_5X7: dict[str, tuple[str, ...]] = {
    "A": (".###.", "#...#", "#...#", "#####", "#...#", "#...#", "#...#"),
    "B": ("####.", "#...#", "#...#", "####.", "#...#", "#...#", "####."),
    "C": (".####", "#....", "#....", "#....", "#....", "#....", ".####"),
    "D": ("####.", "#...#", "#...#", "#...#", "#...#", "#...#", "####."),
    "E": ("#####", "#....", "#....", "####.", "#....", "#....", "#####"),
    "F": ("#####", "#....", "#....", "####.", "#....", "#....", "#...."),
    "G": (".####", "#....", "#....", "#..##", "#...#", "#...#", ".####"),
    "H": ("#...#", "#...#", "#...#", "#####", "#...#", "#...#", "#...#"),
    "I": ("#####", "..#..", "..#..", "..#..", "..#..", "..#..", "#####"),
    "J": ("..###", "...#.", "...#.", "...#.", "...#.", "#..#.", ".##.."),
    "K": ("#...#", "#..#.", "#.#..", "##...", "#.#..", "#..#.", "#...#"),
    "L": ("#....", "#....", "#....", "#....", "#....", "#....", "#####"),
    "M": ("#...#", "##.##", "#.#.#", "#.#.#", "#...#", "#...#", "#...#"),
    "N": ("#...#", "##..#", "#.#.#", "#..##", "#...#", "#...#", "#...#"),
    "O": (".###.", "#...#", "#...#", "#...#", "#...#", "#...#", ".###."),
    "P": ("####.", "#...#", "#...#", "####.", "#....", "#....", "#...."),
    "Q": (".###.", "#...#", "#...#", "#...#", "#.#.#", "#..#.", ".##.#"),
    "R": ("####.", "#...#", "#...#", "####.", "#.#..", "#..#.", "#...#"),
    "S": (".####", "#....", "#....", ".###.", "....#", "....#", "####."),
    "T": ("#####", "..#..", "..#..", "..#..", "..#..", "..#..", "..#.."),
    "U": ("#...#", "#...#", "#...#", "#...#", "#...#", "#...#", ".###."),
    "V": ("#...#", "#...#", "#...#", "#...#", "#...#", ".#.#.", "..#.."),
    "W": ("#...#", "#...#", "#...#", "#.#.#", "#.#.#", "##.##", "#...#"),
    "X": ("#...#", "#...#", ".#.#.", "..#..", ".#.#.", "#...#", "#...#"),
    "Y": ("#...#", "#...#", ".#.#.", "..#..", "..#..", "..#..", "..#.."),
    "Z": ("#####", "....#", "...#.", "..#..", ".#...", "#....", "#####"),
    "0": (".###.", "#...#", "#..##", "#.#.#", "##..#", "#...#", ".###."),
    "1": ("..#..", ".##..", "..#..", "..#..", "..#..", "..#..", ".###."),
    "2": (".###.", "#...#", "....#", "...#.", "..#..", ".#...", "#####"),
    "3": ("#####", "...#.", "..#..", "...#.", "....#", "#...#", ".###."),
    "4": ("...#.", "..##.", ".#.#.", "#..#.", "#####", "...#.", "...#."),
    "5": ("#####", "#....", "####.", "....#", "....#", "#...#", ".###."),
    "6": ("..##.", ".#...", "#....", "####.", "#...#", "#...#", ".###."),
    "7": ("#####", "....#", "...#.", "..#..", ".#...", ".#...", ".#..."),
    "8": (".###.", "#...#", "#...#", ".###.", "#...#", "#...#", ".###."),
    "9": (".###.", "#...#", "#...#", ".####", "....#", "...#.", ".##.."),
    " ": (".....",) * 7,
    ".": (".....", ".....", ".....", ".....", ".....", ".##..", ".##.."),
    ",": (".....", ".....", ".....", ".....", ".##..", "..#..", ".#..."),
    ":": (".....", ".##..", ".##..", ".....", ".##..", ".##..", "....."),
    "'": ("..#..", "..#..", ".#...", ".....", ".....", ".....", "....."),
    "!": ("..#..", "..#..", "..#..", "..#..", "..#..", ".....", "..#.."),
    "?": (".###.", "#...#", "....#", "...#.", "..#..", ".....", "..#.."),
    "+": (".....", "..#..", "..#..", "#####", "..#..", "..#..", "....."),
    "-": (".....", ".....", ".....", "#####", ".....", ".....", "....."),
    "/": (".....", "....#", "...#.", "..#..", ".#...", "#....", "....."),
    "(": ("...#.", "..#..", ".#...", ".#...", ".#...", "..#..", "...#."),
    ")": (".#...", "..#..", "...#.", "...#.", "...#.", "..#..", ".#..."),
}

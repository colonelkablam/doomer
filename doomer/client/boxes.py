"""Box models: small things built from textured boxes, drawn in true 3D.

A sprite is a flat picture, so it can only fake a solid object, and swaps
pictures as you walk round it. Small box-shaped furniture (a computer, its
keyboard, a mouse) can instead be real geometry: a few axis-aligned boxes in
the thing's own frame, turned to its map angle, textured on each face. The
renderer intersects each column's ray with each box (the slab test) and
draws the face it enters by and the top it sees, so it's right from any
angle and any height. Cost is per box per column, so keep models to a few
boxes and use them for small things; big furniture is a map cell.

Model axes (as for voxel models): x out of the thing's front, y to its right,
z up from its base (the floor or desk top it stands on). x and y are in
cells, z in walls (one storey), all relative to the thing's position.

Each face has a 64x64 texture (palette indices, [u][v]) stretched over it.
Texture orientation: sides as seen from outside, upright; the top as seen
from above with the front at the bottom of the image (v = 63).
"""

import math
from collections.abc import Iterable
from dataclasses import dataclass
from typing import NamedTuple

import numpy as np

from doomer.client.palette import (
    BLUE,
    BRICK,
    CREAM,
    GREEN,
    GREY,
    OCHRE,
    ORANGE,
    PURPLE,
    SAND,
    STEEL,
    WARM_GREY,
    WOOD,
    U8Array,
)
from doomer.client.pixelart import BoolArray, Canvas, noise
from doomer.shared.doors import DOOR_HALF, DOOR_HEIGHT, HINGED, HINGED_LEAF, JAMB, SWING, Door
from doomer.shared.mapfile import STAIR_RISE, STAIRS_WALL_TYPE, WALL_UNITS, GameMap
from doomer.shared.screens import SCREEN_HALF, SCREEN_HEIGHT
from doomer.shared.screens import desk_screens as shared_desk_screens

FRONT, RIGHT, BACK, LEFT, TOP = range(5)  # face order in Box.faces


@dataclass(frozen=True, slots=True)
class Box:
    x: tuple[float, float]  # cells, along the thing's facing
    y: tuple[float, float]  # cells, to its right
    z: tuple[float, float]  # walls, above its base
    faces: tuple[int, int, int, int, int]  # texture ids: front, right, back, left, top
    # Left and right faces textured in world units (tiling 64 texels per cell
    # and per storey, like walls) instead of stretched once over the face, so
    # a row of boxes (a staircase's steps) reads as one continuous side.
    tiled_sides: bool = False
    # Front and back faces only this tall, at the top (0 = the whole face), with
    # their texture over that height: a step's riser. Below it the step in
    # front hides the face, so it isn't drawn at all.
    riser: float = 0.0
    # Turned this far (radians) about the model's origin, on top of the
    # thing's angle (for an octagon's diagonal faces, see _octagon).
    turn: float = 0.0


class PlacedBox(NamedTuple):
    """A box in the world: at (x, y), its base at height z (walls), turned
    to `angle` (radians, as map angles)."""

    x: float
    y: float
    z: float
    angle: float
    box: Box


_TEXTURES: list[U8Array] = []
_OPAQUE: list[BoolArray] = []


def _texture(canvas: Canvas) -> int:
    """Register a face texture; returns its id. Texels the canvas didn't
    paint are see-through (leaves)."""
    _TEXTURES.append(np.ascontiguousarray(canvas.index.T))
    _OPAQUE.append(np.ascontiguousarray(canvas.opaque.T))
    return len(_TEXTURES) - 1


# --- Apple Macintosh SE/30 (1989): platinum all-in-one, 9" mono screen --------------
# Real size 24.4 x 27.4 x 34.5 cm (w x d x h); built ~1.6x, as furniture
# widths are exaggerated so it reads at a distance.

PLATINUM = WARM_GREY
_MAC_W, _MAC_D, _MAC_H = 0.145, 0.162, 0.204
_KEYS_W, _KEYS_D, _KEYS_H = 0.2, 0.077, 0.018
_MOUSE_W, _MOUSE_D, _MOUSE_H = 0.033, 0.056, 0.014


def _platinum(shade: float = 12.0) -> Canvas:
    c = Canvas()
    c.paint(c.rect(0, 64, 0, 64), PLATINUM, shade)
    c.paint(c.rect(0, 1, 0, 64), PLATINUM, shade + 1.5)  # top edge catches the light
    c.paint(c.rect(0, 64, 0, 1) | c.rect(0, 64, 63, 64), PLATINUM, shade - 1.5)  # corners
    return c


def _foot(c: Canvas) -> None:
    """The recessed plinth all round the bottom."""
    c.paint(c.rect(57, 64, 0, 64), PLATINUM, 8.0)
    c.paint(c.rect(57, 58, 0, 64), PLATINUM, 6.0)


def mac_front() -> int:
    c = _platinum(13.0)
    c.paint(c.rect(4, 34, 6, 58), PLATINUM, 9.0)  # screen bezel, recessed
    c.paint(c.rect(5, 33, 7, 57), PLATINUM, 7.0)
    # The screen: a System 6 desktop. Grey desktop, white menu bar, a window.
    c.paint(c.rect(6, 31, 9, 55), GREY, 9.0)
    c.paint(c.rect(6, 8, 9, 55), GREY, 14.0)  # menu bar
    c.paint(c.rect(8, 9, 9, 55), GREY, 1.0)
    for u0, u1 in ((11, 13), (16, 21), (24, 28), (31, 36)):  # menu titles
        c.paint(c.rect(6, 7, u0, u1), GREY, 2.0)
    c.paint(c.rect(11, 28, 13, 45), GREY, 1.0)  # window frame
    c.paint(c.rect(12, 27, 14, 44), GREY, 14.0)
    c.paint(c.rect(12, 14, 14, 44), GREY, 5.0)  # striped title bar
    for v0, length in ((16, 20), (19, 26), (22, 14), (25, 22)):  # text
        c.paint(c.rect(v0, v0 + 1, 16, 16 + length), GREY, 3.0)
    c.paint(c.rect(11, 15, 48, 53), GREY, 14.0)  # hard disk icon
    c.paint(c.rect(13, 14, 49, 52), GREY, 2.0)
    # Floppy drive slot, right of centre.
    c.paint(c.rect(40, 44, 33, 53), PLATINUM, 10.0)
    c.paint(c.rect(41, 43, 34, 52), GREY, 1.0)
    # Six-colour Apple logo and the model name, bottom left.
    for i, ramp in enumerate((GREEN, OCHRE, ORANGE, BRICK, PURPLE, BLUE)):
        width = (3, 4, 4, 4, 4, 3)[i]
        c.paint(c.rect(46 + i, 47 + i, 9, 9 + width), ramp, 11.0)
    c.paint(c.rect(45, 46, 11, 12), GREEN, 9.0)  # leaf
    c.paint(c.rect(49, 50, 16, 34), PLATINUM, 8.0)  # "Macintosh SE/30"
    _foot(c)
    return _texture(c)


def mac_side(vents_at_left: bool) -> int:
    """A side panel; the vents are towards the back (left on the right-hand
    side, as seen from outside)."""
    c = _platinum(11.0)
    u0, u1 = (6, 30) if vents_at_left else (34, 58)
    for v0 in range(5, 26, 3):
        c.paint(c.rect(v0, v0 + 1, u0, u1), PLATINUM, 6.0)
    _foot(c)
    return _texture(c)


def mac_back() -> int:
    c = _platinum(11.0)
    for v0 in range(4, 25, 3):  # vents
        c.paint(c.rect(v0, v0 + 1, 8, 56), PLATINUM, 6.0)
    c.paint(c.rect(30, 39, 7, 15), GREY, 2.0)  # power socket
    c.paint(c.rect(31, 36, 18, 21), GREY, 3.0)  # power switch
    for u0, width in ((8, 10), (22, 4), (30, 4), (40, 6), (50, 6)):  # SCSI, ADB, serial
        c.paint(c.rect(44, 49, u0, u0 + width), GREY, 3.0)
    _foot(c)
    return _texture(c)


def mac_top() -> int:
    c = _platinum(13.0)
    c.paint(c.rect(5, 12, 12, 52), PLATINUM, 7.0)  # carry handle recess, at the back
    c.paint(c.rect(11, 12, 12, 52), PLATINUM, 14.0)
    for v0 in range(20, 40, 3):  # vents
        c.paint(c.rect(v0, v0 + 1, 14, 50), PLATINUM, 8.0)
    return _texture(c)


def keyboard_top() -> int:
    """Apple Keyboard (M0116): rows of grey keys in a platinum case. The
    texture is stretched over a shallow box (wide, not deep), so its rows
    are tall in texels."""
    c = _platinum(12.0)
    keys = PLATINUM
    rows = ((6, 15), (16, 25), (26, 35), (36, 45), (46, 56))
    for r, (v0, v1) in enumerate(rows):
        if r == 4:  # bottom row: modifiers and a long space bar
            spans = [(4, 10), (11, 17), (18, 46), (47, 53), (54, 60)]
        else:
            offset = (0, 2, 3, 5)[r]
            spans = [(u, u + 3) for u in range(4 + offset, 59, 4)]
        for u0, u1 in spans:
            c.paint(c.rect(v0, v1, u0, u1), keys, 9.0)
            c.paint(c.rect(v0, v0 + 2, u0, u1), keys, 11.0)  # lit key tops
    return _texture(c)


def mouse_top() -> int:
    """Apple Desktop Bus mouse, seen from above: the button at the far end."""
    c = _platinum(13.0)
    c.paint(c.rect(0, 24, 0, 64), PLATINUM, 11.0)  # the button
    c.paint(c.rect(24, 27, 0, 64), PLATINUM, 6.0)  # its seam
    return _texture(c)


def plain(shade: float) -> int:
    c = _platinum(shade)
    c.paint(c.rect(56, 64, 0, 64), PLATINUM, shade - 3.0)  # underside shadow
    return _texture(c)


def _mac_se30() -> tuple[Box, ...]:
    front, back, top = mac_front(), mac_back(), mac_top()
    mac = Box(
        (-_MAC_D / 2, _MAC_D / 2),
        (-_MAC_W / 2, _MAC_W / 2),
        (0.0, _MAC_H),
        (front, mac_side(vents_at_left=True), back, mac_side(vents_at_left=False), top),
    )
    case = plain(11.0)
    gap = 0.02
    k0 = _MAC_D / 2 + gap
    keyboard = Box(
        (k0, k0 + _KEYS_D),
        (-_KEYS_W / 2, _KEYS_W / 2),
        (0.0, _KEYS_H),
        (case, case, case, case, keyboard_top()),
    )
    # The mouse to the right of the keyboard (the sitter's right: -y, as they
    # face the screen, looking along -x).
    m1 = -_KEYS_W / 2 - gap
    mouse = Box(
        (k0, k0 + _MOUSE_D),
        (m1 - _MOUSE_W, m1),
        (0.0, _MOUSE_H),
        (case, case, case, case, mouse_top()),
    )
    return mac, keyboard, mouse


# --- Stairs ----------------------------------------------------------------------
# Stair cells (wall type 'h') slope smoothly for movement (doomer/shared), but
# are drawn as real steps: one box per riser. A cell rises STAIR_RISE units
# (half a storey) in 8 steps of 1/16 storey (~17 cm) with 1/8-cell (34 cm)
# treads. Each step's top is level with the slope at the step's back edge, so
# feet on the slope are at most one riser below the tread they're over.

STEPS_PER_STOREY = 16


def stair_tread() -> int:
    """A tread from above: blue carpet, a metal nosing strip along the front
    edge (the downhill side: the top of the image is a box's -x side)."""
    rng = np.random.default_rng(31)
    c = Canvas()
    c.paint(c.rect(0, 64, 0, 64), BLUE, 6.0 + (noise(rng, 16) - 0.5) * 1.2)
    c.paint(c.rect(0, 6, 0, 64), GREY, 10.0)
    c.paint(c.rect(0, 1, 0, 64), GREY, 13.0)
    return _texture(c)


def stair_riser() -> int:
    """A riser: carpet, darker than the tread, under the nosing's lip."""
    rng = np.random.default_rng(32)
    c = Canvas()
    c.paint(c.rect(0, 64, 0, 64), BLUE, 4.5 + (noise(rng, 16) - 0.5) * 1.2)
    c.paint(c.rect(0, 10, 0, 64), GREY, 7.0)
    return _texture(c)


def stair_side() -> int:
    """The side of the staircase, tiled in world units: painted plaster with
    skirting at the floor (as raised floors' edges)."""
    rng = np.random.default_rng(14)
    c = Canvas()
    c.paint(c.rect(0, 64, 0, 64), SAND, 8.5 + (noise(rng, 8) - 0.5))
    c.paint(c.rect(60, 64, 0, 64), WOOD, 5.0)
    return _texture(c)


_TREAD, _RISER, _SIDE = stair_tread(), stair_riser(), stair_side()


def stair_boxes(game_map: GameMap) -> list[PlacedBox]:
    """The steps of every stair cell in the map. Box x points uphill (the
    cell's facing), so each step's riser is its -x (back) face."""
    out = []
    for cy, row in enumerate(game_map.walls):
        for cx, wall in enumerate(row):
            if wall != STAIRS_WALL_TYPE or not game_map.is_low_block(cx, cy):
                continue
            top = game_map.floor[cy][cx] / WALL_UNITS
            low = max(0, game_map.floor[cy][cx] - STAIR_RISE) / WALL_UNITS
            steps = max(1, round((top - low) * STEPS_PER_STOREY))
            angle = (game_map.facing_at(cx, cy) - 1) * math.pi / 2  # NORTH = 0 -> -y
            for i in range(steps):
                box = Box(
                    (-0.5 + i / steps, -0.5 + (i + 1) / steps),
                    (-0.5, 0.5),
                    (0.0, low + (top - low) * (i + 1) / steps),
                    (_SIDE, _SIDE, _RISER, _SIDE, _TREAD),
                    tiled_sides=True,
                    riser=(top - low) / steps,
                )
                out.append(PlacedBox(cx + 0.5, cy + 0.5, 0.0, angle, box))
    return out


# --- Cubicle screens on desk pairs ('j') ---------------------------------------
# A fabric screen down the middle of each pair (between the two sitters) and
# one across the pair at each cell edge, so every sitter has a three-sided
# booth. Where they are, their size and their collision: doomer/shared/screens.py.


def screen_fabric() -> int:
    """Grey-blue woven fabric with an aluminium cap; stretched over a whole
    cell's length and the screen's height, so texels are ~4 x 1 cm."""
    rng = np.random.default_rng(41)
    c = Canvas()
    weave = ((c.v // 2) % 2) * 0.4
    c.paint(c.rect(0, 64, 0, 64), STEEL, 7.0 + weave + (noise(rng, 32) - 0.5) * 1.2)
    c.paint(c.rect(0, 7, 0, 64), GREY, 11.0)  # cap
    c.paint(c.rect(0, 2, 0, 64), GREY, 13.0)
    c.paint(c.rect(60, 64, 0, 64), GREY, 6.0)  # foot, in shadow
    return _texture(c)


def screen_trim() -> int:
    """The aluminium edge (screen ends and the top of the cap)."""
    c = Canvas()
    c.paint(c.rect(0, 64, 0, 64), GREY, 11.0)
    return _texture(c)


_FABRIC, _TRIM = screen_fabric(), screen_trim()
# Screens are world-aligned (angle 0): thin across x (big faces east/west) or
# thin across y (big faces north/south).
_THIN_X = (_FABRIC, _TRIM, _FABRIC, _TRIM, _TRIM)
_THIN_Y = (_TRIM, _FABRIC, _TRIM, _FABRIC, _TRIM)


def desk_screens(game_map: GameMap) -> list[PlacedBox]:
    """The map's cubicle screens (doomer/shared/screens.py) as boxes."""
    out = []
    for sc in shared_desk_screens(game_map):
        middle, length = (sc.lo + sc.hi) / 2, (sc.hi - sc.lo) / 2
        thin, long = (-SCREEN_HALF, SCREEN_HALF), (-length, length)
        if sc.axis == 0:  # the plane x = at: thin across x
            box = Box(thin, long, (0.0, SCREEN_HEIGHT), _THIN_X)
            out.append(PlacedBox(sc.at, middle, sc.base, 0.0, box))
        else:
            box = Box(long, thin, (0.0, SCREEN_HEIGHT), _THIN_Y)
            out.append(PlacedBox(middle, sc.at, sc.base, 0.0, box))
    return out


# --- Five-drawer steel filing cabinet --------------------------------------------
# ~1.5 m tall; footprint 0.32 x 0.3 cells to match its collision square
# (doomer/shared/things.py, half-width 0.16). Front = its map angle.

_CAB_H = 0.55  # walls
_CAB_DRAWERS = 5


def cabinet_front() -> int:
    """Five drawers, each with a pull and a cream label card; kick plate."""
    c = Canvas()
    c.paint(c.rect(0, 64, 0, 64), GREY, 8.0)
    c.paint(c.rect(0, 64, 0, 2) | c.rect(0, 64, 62, 64), GREY, 6.0)  # frame edges
    pitch = 60 // _CAB_DRAWERS  # rows per drawer; the bottom 4 rows are the plinth
    for i in range(_CAB_DRAWERS):
        top = i * pitch
        c.paint(c.rect(top, top + 1, 0, 64), GREY, 3.0)  # gap above the drawer
        c.paint(c.rect(top + 1, top + 2, 2, 62), GREY, 10.0)  # lit top edge
        c.paint(c.rect(top + 4, top + 6, 14, 50), CREAM, 10.0)  # label card
        c.paint(c.rect(top + 4, top + 6, 13, 14) | c.rect(top + 4, top + 6, 50, 51), GREY, 11.0)
        c.paint(c.rect(top + 7, top + 9, 22, 42), GREY, 13.0)  # pull
        c.paint(c.rect(top + 9, top + 10, 22, 42), GREY, 4.0)  # its shadow
    c.paint(c.rect(60, 64, 0, 64), GREY, 4.0)  # recessed kick plate
    return _texture(c)


def cabinet_side() -> int:
    c = Canvas()
    c.paint(c.rect(0, 64, 0, 64), GREY, 7.5)
    c.paint(c.rect(0, 64, 0, 1) | c.rect(0, 64, 63, 64), GREY, 6.0)
    c.paint(c.rect(60, 64, 0, 64), GREY, 4.0)
    return _texture(c)


def cabinet_top() -> int:
    c = Canvas()
    c.paint(c.rect(0, 64, 0, 64), GREY, 9.0)
    c.paint(c.rect(0, 2, 0, 64) | c.rect(62, 64, 0, 64), GREY, 7.0)
    return _texture(c)


def _filing_cabinet() -> tuple[Box, ...]:
    side = cabinet_side()
    return (
        Box(
            (-0.15, 0.15),
            (-0.16, 0.16),
            (0.0, _CAB_H),
            (cabinet_front(), side, side, side, cabinet_top()),
        ),
    )


# --- Water cooler ----------------------------------------------------------------
# A cream cabinet with hot and cold taps, under an upturned blue bottle; ~1.3 m
# to the top. Footprint 0.24 cells, matching its collision (half-width 0.12).
# The bottle is round, so it's an octagonal prism (`_octagon`, using `turn`).
# Each face is lit by which way it faces, so the eight shade round like a
# cylinder.


def cooler_front() -> int:
    """The cabinet's front: a recess with the two taps' drip tray."""
    c = Canvas()
    c.paint(c.rect(0, 64, 0, 64), CREAM, 9.0)
    c.paint(c.rect(0, 2, 0, 64), CREAM, 11.0)
    c.paint(c.rect(8, 30, 10, 54), CREAM, 6.0)  # recess for the taps
    c.paint(c.rect(26, 30, 12, 52), GREY, 5.0)  # drip tray grille
    for u0 in range(14, 52, 4):
        c.paint(c.rect(26, 30, u0, u0 + 1), GREY, 9.0)
    c.paint(c.rect(60, 64, 0, 64), CREAM, 5.0)  # plinth
    return _texture(c)


def cooler_side() -> int:
    c = Canvas()
    c.paint(c.rect(0, 64, 0, 64), CREAM, 8.0)
    c.paint(c.rect(0, 2, 0, 64), CREAM, 10.0)
    for v0 in range(36, 56, 4):  # vents for the chiller
        c.paint(c.rect(v0, v0 + 2, 16, 48), CREAM, 5.0)
    c.paint(c.rect(60, 64, 0, 64), CREAM, 5.0)
    return _texture(c)


def cooler_top() -> int:
    c = Canvas()
    c.paint(c.rect(0, 64, 0, 64), CREAM, 10.0)
    return _texture(c)


def bottle() -> int:
    """Water, moulded ribs, and the air gap at the top. (The bottle stands
    upside down with its neck in the cabinet, so the air is at the top.)"""
    rng = np.random.default_rng(51)
    c = Canvas()
    c.paint(c.rect(0, 64, 0, 64), BLUE, 10.0 + (noise(rng, 8) - 0.5))
    c.paint(c.rect(0, 10, 0, 64), BLUE, 13.0)  # air above the water line
    c.paint(c.rect(10, 12, 0, 64), BLUE, 14.0)  # meniscus glint
    for v0 in (28, 46):
        c.paint(c.rect(v0, v0 + 3, 0, 64), BLUE, 12.0)  # moulded ribs
    return _texture(c)


def tap(ramp: int) -> int:
    c = Canvas()
    c.paint(c.rect(0, 64, 0, 64), ramp, 10.0)
    c.paint(c.rect(0, 20, 0, 64), ramp, 12.0)
    return _texture(c)


def _octagon(radius: float, z: tuple[float, float], faces: tuple[int, ...]) -> tuple[Box, ...]:
    """An octagonal prism of inradius `radius` centred on the model origin, as
    a union of boxes. (Two squares, one turned 45 degrees, would make an
    eight-pointed star: a union keeps the corners.) Instead, four strips, each
    from one face to the opposite one and one face wide (2 r tan 22.5): each
    lies inside the octagon, together they cover it, and their ends are its
    eight faces."""
    half = radius * math.tan(math.pi / 8)
    r, w = (-radius, radius), (-half, half)
    sides = (faces[0], faces[0], faces[0], faces[0], faces[1])
    return tuple(
        Box(x, y, z, sides, turn=turn) for turn in (0.0, math.pi / 4) for x, y in ((r, w), (w, r))
    )


def _water_cooler() -> tuple[Box, ...]:
    side = cooler_side()
    water, cap = bottle(), cooler_top()
    cabinet = Box(
        (-0.12, 0.12), (-0.12, 0.12), (0.0, 0.29), (cooler_front(), side, side, side, cap)
    )
    cold, hot = tap(BLUE), tap(BRICK)
    taps = (
        Box((0.12, 0.15), (0.02, 0.05), (0.2, 0.22), (cold, cold, cold, cold, cold)),
        Box((0.12, 0.15), (-0.05, -0.02), (0.2, 0.22), (hot, hot, hot, hot, hot)),
    )
    return (
        cabinet,
        *taps,
        # The bottle stands upside down: neck down in the cabinet's collar,
        # shoulder, then the body up to its flat base at the top (~1.3 m).
        *_octagon(0.03, (0.29, 0.30), (water, water)),  # neck
        *_octagon(0.06, (0.30, 0.325), (water, water)),  # shoulder
        *_octagon(0.085, (0.325, 0.48), (water, water)),  # body
    )


# --- Potted ficus ---------------------------------------------------------------
# ~1.95 m: a terracotta pot (octagonal, flaring to a rim), two braided stems,
# and a crown of crossed leaf cards: thin panels with see-through leaf
# textures, four upright at 45 degrees to each other through the crown's
# axis and two flat ones across it, so it reads as a bush from any side and
# from above. (Solid boxes would look like a topiary cube.) Pot footprint
# matches its collision (half-width 0.16); the crown spreads wider, at head
# height, as a real one does.

_CROWN = (0.30, 0.72)  # walls
_CROWN_R = 0.23  # cells


def leaves(seed: int, flat: bool = False) -> int:
    """A ragged oval of leaf clusters, see-through between them. Lit from
    above (upright cards) and darker inside the clumps. Flat cards are
    sparser: seen nearly edge-on, a dense one squashes into a solid stripe."""
    rng = np.random.default_rng(seed)
    c = Canvas()
    coarse, fine = noise(rng, 8), noise(rng, 32)
    density = 0.6 * coarse + 0.4 * fine
    outline = c.ellipse(32, 32, 31, 31) if flat else c.ellipse(34, 32, 30, 31)
    lit = 0.0 if flat else (32 - c.v) / 32 * 1.5  # tops of the crown catch the light
    leafy = outline & (density > (0.55 if flat else 0.42))
    c.paint(leafy, GREEN, 4.0 + fine * 7.0 + lit)
    c.paint(leafy & (fine < 0.25), GREEN, 2.5)  # shadowed leaves
    return _texture(c)


def pot_side() -> int:
    c = Canvas()
    c.paint(c.rect(0, 64, 0, 64), ORANGE, 7.5)
    c.paint(c.rect(0, 64, 0, 2) | c.rect(0, 64, 62, 64), ORANGE, 6.0)
    return _texture(c)


def soil() -> int:
    rng = np.random.default_rng(61)
    c = Canvas()
    c.paint(c.rect(0, 64, 0, 64), WOOD, 2.5 + noise(rng, 16) * 1.5)
    return _texture(c)


def bark() -> int:
    c = Canvas()
    c.paint(c.rect(0, 64, 0, 64), WOOD, 5.0)
    c.paint(c.rect(0, 64, 0, 16), WOOD, 6.5)
    return _texture(c)


def _ficus() -> tuple[Box, ...]:
    side, earth, stem = pot_side(), soil(), bark()
    pot = (
        *_octagon(0.10, (0.0, 0.12), (side, earth)),
        *_octagon(0.12, (0.12, 0.2), (side, earth)),  # flaring
        *_octagon(0.13, (0.2, 0.22), (side, earth)),  # rim; its top is the soil
    )
    stems = tuple(
        Box((x - 0.012, x + 0.012), (y - 0.012, y + 0.012), (0.22, 0.5), (stem,) * 5)
        for x, y in ((0.02, -0.015), (-0.02, 0.02))
    )
    r = (-_CROWN_R, _CROWN_R)
    card = (-0.003, 0.003)
    # A card shows leaves on its two broad faces only. Its edges are a few mm
    # wide but would still get the whole texture squeezed across them, solid
    # middle columns included: thin green lines through (and past) the crown.
    edge = _texture(Canvas())  # nothing painted: fully see-through
    upright = tuple(
        Box(card, r, _CROWN, (tex, edge, tex, edge, edge), turn=turn)
        for turn, tex in zip(
            (0.0, math.pi / 4, math.pi / 2, 3 * math.pi / 4),
            (leaves(71), leaves(72), leaves(73), leaves(74)),
            strict=True,
        )
    )
    flat = tuple(
        Box(
            (-_CROWN_R * k, _CROWN_R * k),
            (-_CROWN_R * k, _CROWN_R * k),
            (z, z + 0.004),
            (edge, edge, edge, edge, tex),  # leaves on top (and underneath)
            turn=turn,
        )
        for z, k, turn, tex in (
            # Away from standing eye level (0.6), where a flat card is edge-on.
            (0.47, 0.9, 0.3, leaves(75, flat=True)),
            (0.69, 0.6, 1.1, leaves(76, flat=True)),
        )
    )
    return (*pot, *stems, *upright, *flat)


# --- Doors (doomer/shared/doors.py has their state and collision) -------------------
# Drawn from the simulation's door states each frame. A hinged door is two
# fixed jambs and a leaf whose model origin is the hinge, turned by how far
# open it is (so it really swings). Sliding doors are two glass leaves that
# part into the walls either side (which hide them).


def door_leaf(handle_on_right: bool) -> int:
    """One face of a hinged door: wood veneer, a steel kick plate, and the
    handle at the free end (right or left as you look at this face)."""
    rng = np.random.default_rng(81)
    c = Canvas()
    c.paint(c.rect(0, 64, 0, 64), WOOD, 8.0 + (noise(rng, 8) - 0.5) * 1.5)
    c.paint(c.rect(0, 64, 0, 64) & ((c.u * 7 + c.v // 9) % 11 == 0), WOOD, 6.5)  # grain
    c.paint(c.rect(0, 1, 0, 64) | c.rect(0, 64, 0, 1) | c.rect(0, 64, 63, 64), WOOD, 6.0)
    c.paint(c.rect(54, 63, 3, 61), GREY, 10.0)  # kick plate
    c.paint(c.rect(54, 55, 3, 61), GREY, 13.0)
    u0 = 50 if handle_on_right else 6
    c.paint(c.rect(29, 34, u0, u0 + 8), GREY, 12.0)  # lever handle
    c.paint(c.rect(33, 34, u0, u0 + 8), GREY, 6.0)
    c.paint(
        c.rect(27, 36, u0 + (6 if handle_on_right else 0), u0 + (8 if handle_on_right else 2)),
        GREY,
        9.0,
    )  # rose
    return _texture(c)


def door_edge() -> int:
    c = Canvas()
    c.paint(c.rect(0, 64, 0, 64), WOOD, 6.5)
    return _texture(c)


def jamb() -> int:
    """The fixed panel beside a door: painted plaster with a wooden architrave
    along both long edges."""
    rng = np.random.default_rng(82)
    c = Canvas()
    c.paint(c.rect(0, 64, 0, 64), SAND, 8.5 + (noise(rng, 8) - 0.5))
    c.paint(c.rect(0, 64, 0, 5) | c.rect(0, 64, 59, 64), WOOD, 5.5)
    c.paint(c.rect(60, 64, 0, 64), WOOD, 4.5)  # skirting
    return _texture(c)


def glass_leaf() -> int:
    """A sliding glass door: an aluminium frame, a push bar, a frosted band
    at eye height, and glass as a dither (every other texel), the old way to
    draw see-through glass with a palette."""
    c = Canvas()
    frame = (
        c.rect(0, 3, 0, 64) | c.rect(61, 64, 0, 64) | c.rect(0, 64, 0, 3) | c.rect(0, 64, 61, 64)
    )
    glass = ~frame & ((c.u + c.v) % 2 == 0)
    c.paint(glass, BLUE, 12.0)
    c.paint(~frame & c.rect(18, 22, 0, 64), CREAM, 11.0)  # frosted band
    c.paint(frame, GREY, 10.0)
    c.paint(c.rect(0, 1, 0, 64) | c.rect(0, 64, 0, 1), GREY, 13.0)
    c.paint(c.rect(34, 36, 6, 58), GREY, 13.0)  # push bar
    return _texture(c)


def glass_edge() -> int:
    c = Canvas()
    c.paint(c.rect(0, 64, 0, 64), GREY, 10.0)
    return _texture(c)


_LEAF_RIGHT, _LEAF_LEFT, _LEAF_EDGE = door_leaf(True), door_leaf(False), door_edge()
_JAMB, _GLASS, _GLASS_EDGE = jamb(), glass_leaf(), glass_edge()
_JAMB_HALF = 0.03  # jambs a little thicker than the leaf: a frame


def door_boxes(doors: Iterable[Door]) -> list[PlacedBox]:
    """Boxes for doors in their current state."""
    out = []
    h = (0.0, DOOR_HEIGHT)
    t = (-DOOR_HALF, DOOR_HALF)
    for door in doors:
        f = door.facing
        along = (-math.sin(f), math.cos(f))  # the doorway's direction (model +y at angle f)
        if door.kind == HINGED:
            # Jambs: model x across the doorway (thickness), y along it.
            jt = (-_JAMB_HALF, _JAMB_HALF)
            sides = (_JAMB, _JAMB, _JAMB, _JAMB, _JAMB)
            for y in ((-0.5, -0.5 + JAMB), (0.5 - JAMB, 0.5)):
                out.append(PlacedBox(door.x, door.y, 0.0, f, Box(jt, y, h, sides)))
            # The leaf: model x along it from the hinge, closed along the
            # doorway; opening swings its free end towards the facing.
            hinge_x = door.x + along[0] * (-0.5 + JAMB)
            hinge_y = door.y + along[1] * (-0.5 + JAMB)
            angle = f + math.pi / 2 - door.open * SWING
            faces = (_LEAF_EDGE, _LEAF_RIGHT, _LEAF_EDGE, _LEAF_LEFT, _LEAF_EDGE)
            out.append(
                PlacedBox(hinge_x, hinge_y, 0.0, angle, Box((0.0, HINGED_LEAF), t, h, faces))
            )
        else:
            slide = 0.5 * door.open  # each leaf slides into the wall beside it
            faces = (_GLASS, _GLASS_EDGE, _GLASS, _GLASS_EDGE, _GLASS_EDGE)
            for y in ((-0.5 - slide, -slide), (slide, 0.5 + slide)):
                out.append(PlacedBox(door.x, door.y, 0.0, f, Box(t, y, h, faces)))
    return out


# Keyed by thing type name, like SPRITES.
BOX_MODELS: dict[str, tuple[Box, ...]] = {
    "mac": _mac_se30(),
    "filing_cabinet": _filing_cabinet(),
    "water_cooler": _water_cooler(),
    "ficus": _ficus(),
}
BOX_TEXTURES: U8Array = np.stack(_TEXTURES)  # [texture][u][v]
BOX_OPAQUE: BoolArray = np.stack(_OPAQUE)  # [texture][u][v]: False = see-through

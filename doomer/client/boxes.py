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
from collections.abc import Collection, Iterable
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
from doomer.shared.doors import (
    DOOR_HALF,
    DOOR_HEIGHT,
    DOUBLE,
    DOUBLE_JAMB,
    DOUBLE_LEAF,
    HINGED,
    HINGED_LEAF,
    JAMB,
    LIFT,
    SWING,
    Door,
)
from doomer.shared.mapfile import STAIR_RISE, STAIRS_WALL_TYPE, WALL_UNITS, GameMap
from doomer.shared.screens import SCREEN_HALF, SCREEN_HEIGHT
from doomer.shared.screens import desk_screens as shared_desk_screens
from doomer.shared.switches import Switch

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
    # Lights itself (a lit sign): drawn at full brightness, ignoring the light
    # map and the distance fade, as the ceiling panels and exit signs are.
    glow: bool = False
    # Glows only while the floor has power (a lit sign; the breaker's own lamp
    # runs off a battery, so it doesn't need this). Without power it's lit
    # like anything else.
    mains: bool = False


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


def card_reader(ramp: int, unlocked: bool) -> int:
    """A card reader by a locked door: dark plastic, a stripe in the colour of
    the pass it wants, a lamp (red until the team has that pass, then green),
    and the card slot."""
    c = Canvas()
    c.paint(c.rect(0, 64, 0, 64), GREY, 4.0)
    c.paint(c.rect(0, 12, 0, 64), ramp, 11.0)
    c.paint(c.rect(18, 28, 22, 42), GREEN if unlocked else BRICK, 15.0)
    c.paint(c.rect(38, 42, 10, 54), GREY, 1.0)  # slot
    c.paint(c.rect(37, 38, 10, 54), GREY, 7.0)
    return _texture(c)


_PASS_RAMPS = {"blue": BLUE, "red": BRICK, "gold": OCHRE}
_READERS = {
    (colour, unlocked): card_reader(ramp, unlocked)
    for colour, ramp in _PASS_RAMPS.items()
    for unlocked in (False, True)
}
READER_HEIGHT = (0.38, 0.47)  # walls (~1.0-1.25 m): at hand height
READER_ALONG = 0.65  # from the doorway's centre: on the wall just beside it

_LEAF_RIGHT, _LEAF_LEFT, _LEAF_EDGE = door_leaf(True), door_leaf(False), door_edge()
_JAMB, _GLASS, _GLASS_EDGE = jamb(), glass_leaf(), glass_edge()
_JAMB_HALF = 0.03  # jambs a little thicker than the leaf: a frame


def door_boxes(
    doors: Iterable[Door], passes: Collection[str] = (), power: bool = True
) -> list[PlacedBox]:
    """Boxes for doors in their current state; by each locked door, a card
    reader on the wall either side, lit green once the team (`passes`) can
    open it; and by each lift, its call button and floor indicator (lit when
    there's `power`)."""
    out = []
    h = (0.0, DOOR_HEIGHT)
    t = (-DOOR_HALF, DOOR_HALF)
    for door in doors:
        f = door.facing
        along = (-math.sin(f), math.cos(f))  # the doorway's direction (model +y at angle f)
        if door.lock is not None:
            tex = _READERS[(door.lock, door.lock in passes)]
            rx, ry = door.x + along[0] * READER_ALONG, door.y + along[1] * READER_ALONG
            # On the wall faces either side of the doorway (0.5 out across it).
            for across in ((0.5, 0.52), (-0.52, -0.5)):
                reader = Box(across, (-0.035, 0.035), READER_HEIGHT, (tex,) * 5)
                out.append(PlacedBox(rx, ry, 0.0, f, reader))
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
        elif door.kind == DOUBLE:
            jt = (-_JAMB_HALF, _JAMB_HALF)
            sides = (_JAMB, _JAMB, _JAMB, _JAMB, _JAMB)
            for y in ((-0.5, -0.5 + DOUBLE_JAMB), (0.5 - DOUBLE_JAMB, 0.5)):
                out.append(PlacedBox(door.x, door.y, 0.0, f, Box(jt, y, h, sides)))
            faces = (_LEAF_EDGE, _LEAF_RIGHT, _LEAF_EDGE, _LEAF_LEFT, _LEAF_EDGE)
            leaf = Box((0.0, DOUBLE_LEAF), t, h, faces)
            # Each leaf's model x runs from its hinge towards the middle; both
            # free ends swing towards the facing.
            for side, closed, turn in ((-1, f + math.pi / 2, -1), (1, f - math.pi / 2, 1)):
                offset = side * (0.5 - DOUBLE_JAMB)
                hx, hy = door.x + along[0] * offset, door.y + along[1] * offset
                angle = closed + turn * door.open * SWING
                out.append(PlacedBox(hx, hy, 0.0, angle, leaf))
        else:
            slide = 0.5 * door.open  # each leaf slides into the wall beside it
            if door.kind == LIFT:
                faces = (_STEEL_LEAF, _STEEL_EDGE, _STEEL_LEAF, _STEEL_EDGE, _STEEL_EDGE)
                out += _lift_fittings(door, power)
            else:
                faces = (_GLASS, _GLASS_EDGE, _GLASS, _GLASS_EDGE, _GLASS_EDGE)
            for y in ((-0.5 - slide, -slide), (slide, 0.5 + slide)):
                out.append(PlacedBox(door.x, door.y, 0.0, f, Box(t, y, h, faces)))
    return out


def steel_leaf() -> int:
    """A lift door leaf: brushed steel, the brushing vertical."""
    rng = np.random.default_rng(101)
    c = Canvas()
    streaks = (noise(rng, 32)[:1, :] - 0.5) * 1.2  # varies across, constant down
    c.paint(c.rect(0, 64, 0, 64), GREY, 10.0 + np.broadcast_to(streaks, (64, 64)))
    c.paint(c.rect(0, 64, 0, 2) | c.rect(0, 64, 62, 64), GREY, 6.0)  # its edges
    return _texture(c)


def steel_edge() -> int:
    c = Canvas()
    c.paint(c.rect(0, 64, 0, 64), GREY, 7.0)
    return _texture(c)


def call_button(lit: bool) -> int:
    """A lift call button on a steel plate: lit amber once the lift's coming."""
    c = Canvas()
    c.paint(c.rect(0, 64, 0, 64), GREY, 11.0)
    c.paint(c.rect(0, 64, 0, 2) | c.rect(0, 64, 62, 64), GREY, 7.0)
    button = c.ellipse(32, 32, 14, 14)
    c.paint(button, OCHRE if lit else GREY, 15.0 if lit else 6.0)
    c.paint(button & c.rect(26, 30, 28, 36), GREY, 3.0)  # an up arrow
    c.paint(button & c.rect(30, 40, 31, 33), GREY, 3.0)
    return _texture(c)


def floor_indicator(lit: bool) -> int:
    """Above a lift: G (ground), in amber segments on black when powered."""
    c = Canvas()
    c.paint(c.rect(0, 64, 0, 64), GREY, 1.0)
    c.paint(c.rect(0, 2, 0, 64) | c.rect(62, 64, 0, 64), GREY, 9.0)
    glyph = ("#####", "#....", "#....", "#.###", "#...#", "#...#", "#####")
    for row, line in enumerate(glyph):
        for col, bit in enumerate(line):
            if bit == "#":
                v0, u0 = 10 + row * 6, 17 + col * 6
                c.paint(
                    c.rect(v0, v0 + 5, u0, u0 + 5), OCHRE if lit else GREY, 15.0 if lit else 3.0
                )
    return _texture(c)


_STEEL_LEAF, _STEEL_EDGE = steel_leaf(), steel_edge()
_CALL = {lit: call_button(lit) for lit in (False, True)}
_INDICATOR = {lit: floor_indicator(lit) for lit in (False, True)}


def _lift_fittings(door: Door, power: bool) -> list[PlacedBox]:
    """A lift's call button, on the wall beside the doorway, and its floor
    indicator over the lintel: both on the lobby side (the facing)."""
    f = door.facing
    along = (-math.sin(f), math.cos(f))
    # The lobby-side wall faces are 0.5 out from the doorway's centre line.
    bx, by = door.x + along[0] * 0.62, door.y + along[1] * 0.62
    called = power and door.called
    button = Box((0.5, 0.515), (-0.04, 0.04), (0.36, 0.44), (_CALL[called],) * 5, glow=called)
    indicator = Box((0.5, 0.52), (-0.1, 0.1), (0.8, 0.9), (_INDICATOR[power],) * 5, glow=power)
    return [PlacedBox(bx, by, 0.0, f, button), PlacedBox(door.x, door.y, 0.0, f, indicator)]


# --- Reception desk (1980s, two levels) -------------------------------------------
# A transaction ledge at the front (~1.15 m) for visitors to lean on and sign
# in at, over a wood-slatted front panel on a recessed plinth with the
# word WELCOME in brass letters; behind it, a work surface at desk height (~0.73 m)
# over a steel drawer pedestal, with the switchboard phone; returns wrapping
# both ends. Model x points to the visitors' side. Collision and the heights
# things rest at (the ledge, the work surface): RECEPTION_FOOTPRINTS in
# doomer/shared/things.py, which these dimensions match.

_R_LEN = 0.92  # half-length along the desk, cells
_R_FRONT = (0.18, 0.26)  # the front panel, across the desk
_R_LEDGE = (0.14, 0.31)
_R_BACK = -0.30  # the work surface's back edge (the receptionist's side)
_R_PANEL_TOP, _R_LEDGE_TOP, _R_WORK = 0.40, 0.425, 0.27  # walls


def slatted_wood() -> int:
    rng = np.random.default_rng(91)
    c = Canvas()
    c.paint(c.rect(0, 64, 0, 64), WOOD, 8.5 + (noise(rng, 16) - 0.5) * 1.2)
    c.paint(c.rect(0, 64, 0, 64) & (c.u % 6 == 0), WOOD, 5.5)  # vertical slats
    c.paint(c.rect(0, 4, 0, 64), WOOD, 5.0)  # in the ledge's shadow
    return _texture(c)


def plinth() -> int:
    c = Canvas()
    c.paint(c.rect(0, 64, 0, 64), WOOD, 3.0)
    return _texture(c)


def ledge_top() -> int:
    """Cream laminate, faintly mottled, with a chrome strip along the front."""
    rng = np.random.default_rng(92)
    c = Canvas()
    c.paint(c.rect(0, 64, 0, 64), CREAM, 11.0 + (noise(rng, 16) - 0.5) * 0.8)
    c.paint(c.rect(58, 64, 0, 64), GREY, 13.0)
    return _texture(c)


def ledge_edge() -> int:
    c = Canvas()
    c.paint(c.rect(0, 64, 0, 64), GREY, 12.0)
    c.paint(c.rect(0, 16, 0, 64), GREY, 14.0)  # catches the light
    c.paint(c.rect(56, 64, 0, 64), GREY, 6.0)
    return _texture(c)


def work_top() -> int:
    rng = np.random.default_rng(93)
    c = Canvas()
    c.paint(c.rect(0, 64, 0, 64), WARM_GREY, 10.0 + (noise(rng, 16) - 0.5) * 0.8)
    return _texture(c)


def work_edge() -> int:
    c = Canvas()
    c.paint(c.rect(0, 64, 0, 64), WARM_GREY, 8.0)
    return _texture(c)


def pedestal_front() -> int:
    """Three steel drawers, seen from the receptionist's chair."""
    c = Canvas()
    c.paint(c.rect(0, 64, 0, 64), GREY, 8.0)
    for top in (0, 21, 42):
        c.paint(c.rect(top, top + 1, 0, 64), GREY, 3.0)
        c.paint(c.rect(top + 8, top + 11, 22, 42), GREY, 13.0)  # pull
    return _texture(c)


def phone_top() -> int:
    """A switchboard phone from above: the handset along the back, a keypad,
    and a row of line lamps."""
    c = Canvas()
    c.paint(c.rect(0, 64, 0, 64), CREAM, 10.0)
    c.paint(c.rect(4, 18, 6, 58), GREY, 4.0)  # handset
    for v0 in (26, 36, 46):
        for u0 in (14, 26, 38):
            c.paint(c.rect(v0, v0 + 6, u0, u0 + 8), GREY, 9.0)  # keys
    for u0 in range(8, 56, 8):
        c.paint(c.rect(56, 60, u0, u0 + 4), BRICK, 12.0)  # line lamps
    return _texture(c)


def book_top() -> int:
    """The visitors' book, open: ruled pages either side of the spine."""
    c = Canvas()
    c.paint(c.rect(0, 64, 0, 64), BLUE, 5.0)  # cover
    c.paint(c.rect(4, 60, 4, 60), CREAM, 14.0)
    c.paint(c.rect(4, 60, 31, 33), CREAM, 10.0)  # spine
    c.paint(c.rect(4, 60, 4, 60) & (c.v % 6 == 0), BLUE, 10.0)  # rules
    return _texture(c)


def book_edge() -> int:
    c = Canvas()
    c.paint(c.rect(0, 64, 0, 64), BLUE, 5.0)
    return _texture(c)


def _reception_desk() -> tuple[Box, ...]:
    wood, dark, edge = slatted_wood(), plinth(), ledge_edge()
    work, work_side = work_top(), work_edge()
    steel = pedestal_front()
    length = (-_R_LEN, _R_LEN)
    inner = (-_R_LEN + 0.06, _R_LEN - 0.06)  # between the returns
    book_side = book_edge()
    boxes = [
        # The front: slatted panel on a recessed plinth, the ledge above it.
        Box(_R_FRONT, inner, (0.03, _R_PANEL_TOP), (wood, wood, wood, wood, wood)),
        Box((0.17, 0.235), inner, (0.0, 0.03), (dark,) * 5),
        Box(_R_LEDGE, length, (_R_PANEL_TOP, _R_LEDGE_TOP), (edge, edge, edge, edge, ledge_top())),
        # Returns at both ends, from the back of the work surface to the front.
        Box((_R_BACK, _R_FRONT[1]), (-_R_LEN, -_R_LEN + 0.06), (0.0, _R_PANEL_TOP), (wood,) * 5),
        Box((_R_BACK, _R_FRONT[1]), (_R_LEN - 0.06, _R_LEN), (0.0, _R_PANEL_TOP), (wood,) * 5),
        # The work surface, and a drawer pedestal under its right-hand end.
        Box((_R_BACK, _R_FRONT[0]), inner, (_R_WORK - 0.015, _R_WORK), (work_side,) * 4 + (work,)),
        Box(
            (_R_BACK + 0.03, _R_FRONT[0] - 0.03),
            (0.5, _R_LEN - 0.07),
            (0.0, _R_WORK - 0.015),
            (steel, steel, steel, steel, steel),
        ),
        # The phone, the visitors' book.
        Box(
            (-0.12, -0.02),
            (-0.55, -0.42),
            (_R_WORK, _R_WORK + 0.03),
            (work_side,) * 4 + (phone_top(),),
        ),
        Box(
            (0.17, 0.27),
            (0.35, 0.52),
            (_R_LEDGE_TOP, _R_LEDGE_TOP + 0.01),
            (book_side,) * 4 + (book_top(),),
        ),
    ]
    # "WELCOME" in brass letters across the middle of the front; not lit like
    # the wall sign, just polished (they take the room's light).
    boxes += _letters("WELCOME", 0.0, 0.08, 0.09, 0.17, 0.025, face=_R_FRONT[1], glow=False)
    return tuple(boxes)


# --- "Welcome to Boomer Industries": brass letters on the lobby wall ---------------
# Each letter is its own thin box with its own texture, see-through round the
# letter (like real fixed letters), so it isn't smeared: a 64x64 texture
# stretched across a 6 m sign would be. A dark panel behind them. Model x
# points out of the wall; the thing stands at the wall's face.

_FONT_5X7: dict[str, tuple[str, ...]] = {
    "B": ("####.", "#...#", "#...#", "####.", "#...#", "#...#", "####."),
    "C": (".####", "#....", "#....", "#....", "#....", "#....", ".####"),
    "D": ("####.", "#...#", "#...#", "#...#", "#...#", "#...#", "####."),
    "E": ("#####", "#....", "#....", "####.", "#....", "#....", "#####"),
    "I": ("#####", "..#..", "..#..", "..#..", "..#..", "..#..", "#####"),
    "L": ("#....", "#....", "#....", "#....", "#....", "#....", "#####"),
    "M": ("#...#", "##.##", "#.#.#", "#.#.#", "#...#", "#...#", "#...#"),
    "N": ("#...#", "##..#", "#.#.#", "#..##", "#...#", "#...#", "#...#"),
    "O": (".###.", "#...#", "#...#", "#...#", "#...#", "#...#", ".###."),
    "R": ("####.", "#...#", "#...#", "####.", "#.#..", "#..#.", "#...#"),
    "S": (".####", "#....", "#....", ".###.", "....#", "....#", "####."),
    "T": ("#####", "..#..", "..#..", "..#..", "..#..", "..#..", "..#.."),
    "U": ("#...#", "#...#", "#...#", "#...#", "#...#", "#...#", ".###."),
    "W": ("#...#", "#...#", "#...#", "#.#.#", "#.#.#", "##.##", "#...#"),
}


def brass_letter(ch: str) -> int:
    """One letter filling the texture, polished brass: lighter along the top
    of each stroke, darker along the bottom."""
    c = Canvas()
    rows = _FONT_5X7[ch]
    col = (c.u * 5) // 64
    row = (c.v * 7) // 64
    glyph = np.array([[bit == "#" for bit in line] for line in rows])
    on = glyph[row, col]
    within = c.v - (row * 64 + 6) // 7  # texels down from the top of this glyph row
    shade = np.where(within < 2, 14.0, np.where(within > 6, 9.0, 12.0))
    c.paint(on, OCHRE, shade)
    return _texture(c)


def sign_panel() -> int:
    c = Canvas()
    c.paint(c.rect(0, 64, 0, 64), GREY, 2.5)
    c.paint(c.rect(0, 2, 0, 64) | c.rect(62, 64, 0, 64), OCHRE, 9.0)  # brass trim
    return _texture(c)


def _letters(
    text: str,
    centre: float,
    width: float,
    height: float,
    base: float,
    gap: float,
    face: float = 0.012,
    glow: bool = True,
) -> list[Box]:
    """A line of letters standing out from a surface at model x = `face`,
    reading left to right as you face it: towards the model's -y (your
    right, as you look back along -x)."""
    clear = _texture(Canvas())
    total = len(text) * width + (len(text) - 1) * gap
    out = []
    for i, ch in enumerate(text):
        if ch == " ":
            continue
        y1 = centre + total / 2 - i * (width + gap)
        if ch not in _LETTERS:  # one texture per letter, shared by every use
            _LETTERS[ch] = brass_letter(ch)
        tex = _LETTERS[ch]
        out.append(
            Box(
                (face, face + 0.008),
                (y1 - width, y1),
                (base, base + height),
                (tex, clear, tex, clear, clear),
                glow=glow,
                mains=glow,
            )
        )
    return out


_LETTERS: dict[str, int] = {}


def _welcome_sign() -> tuple[Box, ...]:
    """ "Boomer Industries" in lit brass letters (they glow, as if spotlit) on
    a dark panel, high on the wall: legible from the entrance, ~8 cells away."""
    panel, dark = sign_panel(), plinth()
    letters = _letters("BOOMER INDUSTRIES", 0.0, 0.19, 0.21, 1.165, 0.045)
    half = (17 * 0.19 + 16 * 0.045) / 2 + 0.12
    backing = Box((0.0, 0.012), (-half, half), (1.085, 1.455), (panel, dark, dark, dark, dark))
    return (backing, *letters)


# --- The plant room's breaker (doomer/shared/switches.py has its state) ---------
# A steel switchgear cabinet against the wall: louvres, a hazard band, a big
# lever (down: off; thrown up: on) and a status lamp (red, then green, lit).
# Drawn from the switch's state each frame, like doors. Model x points out of
# the wall; the thing stands at the wall's face.

BREAKER_DEPTH, BREAKER_HALF_WIDTH, BREAKER_TOP = 0.12, 0.3, 0.62
_PIVOT, _LEVER = 0.36, 0.11  # walls: the lever's pivot height and length


def switchgear_front() -> int:
    c = Canvas()
    c.paint(c.rect(0, 64, 0, 64), GREY, 8.0)
    c.paint(c.rect(0, 64, 0, 2) | c.rect(0, 64, 62, 64), GREY, 5.0)
    for v0 in range(6, 22, 3):  # louvres
        c.paint(c.rect(v0, v0 + 1, 8, 40), GREY, 3.0)
    stripes = c.rect(54, 60, 2, 62)
    c.paint(stripes, OCHRE, 12.0)  # hazard band
    c.paint(stripes & ((c.u + c.v) % 8 < 4), GREY, 1.0)
    c.paint(c.rect(28, 40, 10, 22), OCHRE, 12.0)  # a warning label
    c.paint(c.rect(30, 38, 15, 17), GREY, 1.0)
    return _texture(c)


def switchgear_side() -> int:
    c = Canvas()
    c.paint(c.rect(0, 64, 0, 64), GREY, 6.5)
    return _texture(c)


def lever_arm() -> int:
    c = Canvas()
    c.paint(c.rect(0, 64, 0, 64), GREY, 11.0)
    return _texture(c)


def lever_handle() -> int:
    c = Canvas()
    c.paint(c.rect(0, 64, 0, 64), BRICK, 11.0)
    c.paint(c.rect(0, 20, 0, 64), BRICK, 13.0)
    return _texture(c)


def lamp(ramp: int) -> int:
    c = Canvas()
    c.paint(c.rect(0, 64, 0, 64), ramp, 15.0)
    return _texture(c)


_GEAR_FRONT, _GEAR_SIDE = switchgear_front(), switchgear_side()
_ARM, _HANDLE = lever_arm(), lever_handle()
_LAMP_OFF, _LAMP_ON = lamp(BRICK), lamp(GREEN)


def switch_boxes(switches: Iterable[Switch]) -> list[PlacedBox]:
    """Boxes for switches (the breaker) in their current state."""
    out = []
    for sw in switches:
        cabinet = Box(
            (0.0, BREAKER_DEPTH),
            (-BREAKER_HALF_WIDTH, BREAKER_HALF_WIDTH),
            (0.02, BREAKER_TOP),
            (_GEAR_FRONT, _GEAR_SIDE, _GEAR_SIDE, _GEAR_SIDE, _GEAR_SIDE),
        )
        # The lever on the right of the front: hangs down when off, up when on.
        z = (_PIVOT, _PIVOT + _LEVER) if sw.on else (_PIVOT - _LEVER, _PIVOT)
        knob = (z[1] - 0.03, z[1]) if sw.on else (z[0], z[0] + 0.03)
        lever_x = (BREAKER_DEPTH, BREAKER_DEPTH + 0.03)
        arm = Box(lever_x, (0.17, 0.2), z, (_ARM,) * 5)
        handle = Box((BREAKER_DEPTH, BREAKER_DEPTH + 0.05), (0.15, 0.22), knob, (_HANDLE,) * 5)
        lit = _LAMP_ON if sw.on else _LAMP_OFF
        light = Box(
            (BREAKER_DEPTH, BREAKER_DEPTH + 0.008),
            (-0.2, -0.14),
            (0.5, 0.54),
            (lit,) * 5,
            glow=True,
        )
        for box in (cabinet, arm, handle, light):
            out.append(PlacedBox(sw.x, sw.y, 0.0, sw.facing, box))
    return out


# Keyed by thing type name, like SPRITES.
BOX_MODELS: dict[str, tuple[Box, ...]] = {
    "mac": _mac_se30(),
    "filing_cabinet": _filing_cabinet(),
    "water_cooler": _water_cooler(),
    "ficus": _ficus(),
    "reception_desk": _reception_desk(),
    "welcome_sign": _welcome_sign(),
}
BOX_TEXTURES: U8Array = np.stack(_TEXTURES)  # [texture][u][v]
BOX_OPAQUE: BoolArray = np.stack(_OPAQUE)  # [texture][u][v]: False = see-through

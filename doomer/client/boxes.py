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
from collections.abc import Callable, Collection, Iterable
from dataclasses import dataclass
from typing import NamedTuple

import numpy as np
import numpy.typing as npt

from doomer.client.areas import AREA_LOOKS
from doomer.client.faces import CAST, draw_portrait
from doomer.client.palette import (
    BLUE,
    BRICK,
    CREAM,
    GREEN,
    GREY,
    MOSS,
    OCHRE,
    ORANGE,
    PALETTE,
    PINK,
    PURPLE,
    SAND,
    STEEL,
    WARM_GREY,
    WOOD,
    U8Array,
)
from doomer.client.pixelart import FONT_3X5, BoolArray, Canvas, noise
from doomer.shared.cubicles import OPEN_ANGLE, CubicleDoor
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
    SLIDE,
    SWING,
    Door,
)
from doomer.shared.mapfile import STAIR_RISE, STAIRS_WALL_TYPE, WALL_UNITS, GameMap
from doomer.shared.screens import SCREEN_HALF, SCREEN_HEIGHT
from doomer.shared.screens import desk_screens as shared_desk_screens
from doomer.shared.switches import Switch
from doomer.shared.things import (
    BASIN_DEPTH,
    BASIN_PITCH,
    BASIN_TOP,
    CUBICLE_DEPTH,
    CUBICLE_TOP,
    CUBICLE_WIDTH,
    EXIT_SIGN_HEIGHTS,
    POSTER_KINDS,
    URINAL_DEPTH,
    URINAL_PITCH,
)

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
    # A lamp on a panel: drawn only while it's lit, the dark socket painted
    # behind it showing otherwise. Unlit when it's on the mains (`mains`)
    # and there's no power, or in the off half of its blink.
    indicator: bool = False
    # Its faces when its circuit is dead (a screen: blank glass); drawn then
    # without its glow. None: the same either way.
    off_faces: tuple[int, int, int, int, int] | None = None
    # Blinking (an indicator): seconds per on-off cycle (0: steady), and
    # where in the cycle it starts (0..1); things of a model are offset by
    # where they stand, so a row of them doesn't blink in step.
    blink: float = 0.0
    blink_phase: float = 0.0


def lit_now(box: Box, x: float, y: float, now: float, power: bool) -> bool:
    """Whether an indicator lamp (a box at (x, y)) is lit at time `now`."""
    if box.mains and not power:
        return False
    if box.blink <= 0.0:
        return True
    return (now / box.blink + box.blink_phase + 0.37 * x + 0.61 * y) % 1.0 < 0.5


class PlacedBox(NamedTuple):
    """A box in the world: at (x, y), its base at height z (walls), turned
    to `angle` (radians, as map angles)."""

    x: float
    y: float
    z: float
    angle: float
    box: Box
    emergency: bool = False  # on the emergency circuit: powered in a power cut


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
    # The screen itself is a box of its own (mac_screen), just proud of this:
    # behind it, dark glass.
    c.paint(c.rect(6, 31, 9, 55), GREY, 2.0)
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


_MAC_SCREEN_UV = (6, 31, 9, 55)  # v0, v1, u0, u1: the screen on the front texture


def mac_screen() -> int:
    """The screen, powered: a System 6 desktop. Grey desktop, white menu bar,
    a window with a striped title bar and lines of text, the hard disk icon."""
    c = Canvas()
    c.paint(c.rect(0, 64, 0, 64), GREY, 9.0)
    c.paint(c.rect(0, 6, 0, 64), GREY, 14.0)  # menu bar
    c.paint(c.rect(6, 8, 0, 64), GREY, 1.0)
    for u0, u1 in ((3, 6), (10, 17), (21, 27), (31, 38)):  # menu titles
        c.paint(c.rect(2, 4, u0, u1), GREY, 2.0)
    c.paint(c.rect(13, 56, 6, 50), GREY, 1.0)  # window frame
    c.paint(c.rect(15, 54, 8, 48), GREY, 14.0)
    c.paint(c.rect(15, 20, 8, 48), GREY, 5.0)  # striped title bar
    for v0, length in ((25, 28), (32, 36), (39, 20), (46, 30)):  # text
        c.paint(c.rect(v0, v0 + 2, 11, 11 + length), GREY, 3.0)
    c.paint(c.rect(12, 22, 53, 61), GREY, 14.0)  # hard disk icon
    c.paint(c.rect(17, 19, 54, 60), GREY, 2.0)
    return _texture(c)


def blank_screen() -> int:
    """A screen with no power: dark glass, a faint reflection across it."""
    c = Canvas()
    c.paint(c.rect(0, 64, 0, 64), GREY, 2.0)
    c.paint(c.rect(0, 64, 0, 64) & (np.abs(c.u - c.v - 10) < 4), GREY, 3.0)
    return _texture(c)


_BLANK_SCREEN = blank_screen()


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
    # The screen: lit (full brightness) with power, blank glass without.
    v0, v1, u0, u1 = _MAC_SCREEN_UV
    half = _MAC_W / 2
    edge = (plain(9.0),) * 4
    screen = Box(
        (_MAC_D / 2, _MAC_D / 2 + 0.002),
        (half - u1 / 64 * _MAC_W, half - u0 / 64 * _MAC_W),
        (_MAC_H * (1 - v1 / 64), _MAC_H * (1 - v0 / 64)),
        (mac_screen(), *edge),
        glow=True,
        mains=True,
        off_faces=(_BLANK_SCREEN, *edge),
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
    return mac, screen, keyboard, mouse


# --- Posters and pictures ---------------------------------------------------------------
# Flat things on walls: one thin box each (the thing at the wall's face,
# model +x out of it), its picture on the front. `poster` things pick one
# by `kind=`; the fire plan (kind=plan) is drawn from the map itself, with
# a red dot where it hangs (see poster_plans). Text in the 3x5 font, big
# enough to read from a cell or two.

_POSTER_DEPTH = 0.012  # cells out from the wall


def _centred(c: Canvas, text: str, v0: int, ramp: int, shade: float, scale: int = 1) -> None:
    su = scale if c.text_width(text, scale) <= 58 else max(1, scale - 1)
    c.text(text, v0, (64 - c.text_width(text, su)) // 2, ramp, shade, scale=scale, scale_u=su)


def _motivational(word: str, caption: str, picture: Callable[[Canvas], None]) -> int:
    """The classic: a black border, a moody photo, one big word, a caption."""
    c = Canvas()
    c.paint(c.rect(0, 64, 0, 64), GREY, 1.0)
    c.paint(c.rect(3, 41, 5, 59), GREY, 12.0)  # a fine keyline round the photo
    picture(c)
    _centred(c, word, 45, CREAM, 15.0, scale=2)
    _centred(c, caption, 57, CREAM, 10.0)
    return _texture(c)


def _photo(c: Canvas) -> BoolArray:
    return c.rect(4, 40, 6, 58)


def _sunset_eagle(c: Canvas) -> None:
    photo = _photo(c)
    for v0, ramp, shade in (
        (4, PURPLE, 4.0),
        (12, ORANGE, 7.0),
        (20, ORANGE, 10.0),
        (26, OCHRE, 13.0),
    ):
        c.paint(photo & (c.v >= v0), ramp, shade)
    c.paint(photo & c.ellipse(31, 32, 7, 8) & (c.v < 31), OCHRE, 15.0)  # the sun, setting
    c.paint(photo & (c.v >= 31), BLUE, 3.0)  # the sea
    c.paint(photo & (c.v >= 31) & (c.v % 3 == 0) & (np.abs(c.u - 32) < 8), OCHRE, 11.0)
    for k in range(11):  # an eagle, wings up
        lift = k // 3
        c.paint(
            c.rect(13 - lift, 15 - lift, 31 - k, 32 - k)
            | c.rect(13 - lift, 15 - lift, 32 + k, 33 + k),
            GREY,
            0.0,
        )
    c.paint(c.rect(13, 17, 31, 33), GREY, 0.0)


def _rowing_crew(c: Canvas) -> None:
    photo = _photo(c)
    c.paint(photo, STEEL, 11.0)  # a pale dawn sky
    c.paint(photo & (c.v >= 22), BLUE, 6.0)
    c.paint(photo & (c.v >= 22) & (c.v % 4 == 0), BLUE, 8.0)  # ripples
    c.paint(c.rect(28, 30, 12, 52), GREY, 1.0)  # the hull
    for u0 in (18, 26, 34, 42):  # four rowers, oars in the water
        c.paint(c.rect(23, 28, u0, u0 + 2), GREY, 1.0)
        for k in range(6):
            c.paint(c.rect(28 + k, 29 + k, u0 - k, u0 - k + 1), GREY, 2.0)


def _mountain(c: Canvas) -> None:
    photo = _photo(c)
    c.paint(photo, BLUE, 10.0)
    c.paint(photo & (c.v >= 26), BLUE, 11.0)
    behind = photo & (c.v >= 18 + np.abs(c.u - 18) * 0.9)
    c.paint(behind, GREY, 4.0)
    peak = photo & (c.v >= 9 + np.abs(c.u - 34) * 0.9)
    c.paint(peak, GREY, 6.0)
    c.paint(peak & (c.v < 17), CREAM, 15.0)  # snow
    c.paint(peak & (c.u > 34) & (c.v >= 17), GREY, 5.0)  # its shadowed side


def employee_of_the_month(month: str) -> int:
    """A framed photo (the same chap every month: CAST["gary"]) over a brass
    plate with the month on it."""
    c = Canvas()
    c.paint(c.rect(0, 64, 0, 64), WOOD, 5.0)  # the frame
    c.paint(c.rect(3, 61, 3, 61), CREAM, 13.0)  # the mount
    c.paint(c.rect(6, 46, 8, 56), BLUE, 8.0)  # studio backdrop
    draw_portrait(c, 6, 12, 40, CAST["gary"], None)
    c.paint(c.rect(47, 60, 7, 57), OCHRE, 11.0)  # the brass plate
    _centred(c, "EMPLOYEE", 48, WOOD, 2.0)
    _centred(c, month, 54, WOOD, 2.0)
    return _texture(c)


def founder_portrait() -> int:
    """Oils in a gilt frame: the founder (CAST["founder"]), on a dark,
    varnished background, a brass plaque on the frame."""
    c = Canvas()
    c.paint(c.rect(0, 64, 0, 64), OCHRE, 10.0)  # gilt
    c.paint(c.rect(0, 1, 0, 64) | c.rect(0, 64, 0, 1), OCHRE, 13.0)
    c.paint(c.rect(4, 60, 4, 60), OCHRE, 7.0)
    c.paint(c.rect(6, 58, 6, 58), WOOD, 2.0)
    draw_portrait(c, 8, 12, 40, CAST["founder"], None)
    c.paint(c.rect(50, 57, 14, 50), OCHRE, 13.0)  # the plaque
    _centred(c, "FOUNDER", 51, GREY, 2.0)
    return _texture(c)


def accident_board() -> int:
    """The plant room's safety board: days since the last accident: 0."""
    c = Canvas()
    c.paint(c.rect(0, 64, 0, 64), GREEN, 7.0)
    c.paint(c.rect(3, 61, 3, 61), CREAM, 15.0)
    _centred(c, "DAYS SINCE", 6, GREEN, 5.0)
    _centred(c, "LAST ACCIDENT", 13, GREEN, 5.0)
    c.paint(c.rect(21, 49, 22, 42), GREY, 1.0)  # the flip card
    c.paint(c.rect(34, 35, 22, 42), GREY, 4.0)  # its fold
    c.text("0", 25, 26, BRICK, 13.0, scale=4)
    _centred(c, "THINK SAFE", 53, GREEN, 5.0)
    return _texture(c)


def high_voltage_sign() -> int:
    """A yellow warning triangle with a lightning bolt, DANGER HIGH VOLTAGE."""
    c = Canvas()
    c.paint(c.rect(0, 64, 0, 64), GREY, 1.0)
    c.paint(c.rect(1, 63, 1, 63), CREAM, 15.0)
    outer = (c.v >= 4 + np.abs(c.u - 32) * 1.1) & (c.v < 31)
    inner = (c.v >= 9 + np.abs(c.u - 32) * 1.1) & (c.v < 28)
    c.paint(outer, GREY, 1.0)
    c.paint(inner, OCHRE, 13.0)
    for v0, u0 in ((12, 33), (15, 31), (18, 29), (20, 31), (23, 29)):  # the bolt
        c.paint(c.rect(v0, v0 + 3, u0, u0 + 3), GREY, 1.0)
    _centred(c, "DANGER", 35, BRICK, 11.0, scale=2)
    _centred(c, "HIGH VOLTAGE", 50, GREY, 1.0)
    return _texture(c)


# kind: (texture, half-width (cells), z (walls))


def washroom_plate(ladies: bool) -> int:
    """A door plate in the 1980s style: a bold white pictogram filling a
    coloured enamel plate (blue for the gents, red for the ladies) in an
    aluminium edge. No words: at this resolution they'd be a smudge, and the
    figure and the colour say it."""
    c = Canvas()
    c.paint(c.rect(0, 64, 0, 64), BRICK if ladies else BLUE, 4.0 if ladies else 4.5)
    c.paint(~c.rect(3, 61, 3, 61), STEEL, 12.0)
    c.paint(c.rect(3, 4, 3, 61) | c.rect(3, 61, 3, 4), STEEL, 8.0)  # the edge's inner shadow
    figure = (CREAM, 15.0)
    c.paint(c.ellipse(13, 32, 7, 7), *figure)  # the head
    across = np.abs(c.u + 0.5 - 32)
    if ladies:
        # A dress flaring from narrow shoulders, arms held out from it, legs.
        c.paint((c.v >= 22) & (c.v < 47) & (across < 6 + (c.v - 22) * 0.4), *figure)
        c.paint((c.v >= 23) & (c.v < 38) & (across >= 13) & (across < 17), *figure)
        c.paint((c.v >= 47) & (c.v < 59) & (across >= 2) & (across < 6), *figure)
    else:
        # Square shoulders, the body, arms apart from it, legs with a gap.
        c.paint((c.v >= 22) & (c.v < 44) & (across < 10), *figure)
        c.paint((c.v >= 23) & (c.v < 42) & (across >= 12) & (across < 16), *figure)
        c.paint((c.v >= 44) & (c.v < 59) & (across >= 2) & (across < 9), *figure)
    return _texture(c)


def wash_hands_notice() -> int:
    """WASH HANDS, a printed card: big blue capitals over a tap with water
    falling from it into a pair of cupped hands."""
    c = Canvas()
    c.paint(c.rect(0, 64, 0, 64), CREAM, 14.5)
    c.paint(~c.rect(3, 61, 3, 61), BLUE, 6.0)
    for word, v0 in (("WASH", 6), ("HANDS", 18)):
        c.text(word, v0, (64 - c.text_width(word, 2)) // 2, BLUE, 4.0, scale=2)
    c.paint(c.rect(31, 35, 14, 36), STEEL, 9.0)  # the tap's spout, from the wall (left)
    c.paint(c.rect(31, 32, 14, 36), STEEL, 12.0)  # lit along its top
    c.paint(c.rect(35, 38, 31, 36), STEEL, 8.0)  # its nozzle
    c.paint(c.rect(39, 47, 32, 35), BLUE, 10.0)  # water falling
    c.paint(c.ellipse(52, 26, 6, 10) | c.ellipse(52, 40, 6, 10), SAND, 11.0)  # cupped hands
    c.paint((c.ellipse(52, 26, 6, 10) | c.ellipse(52, 40, 6, 10)) & (c.v >= 55), SAND, 8.5)
    c.paint(c.rect(48, 57, 33, 34), SAND, 7.0)  # where they meet
    return _texture(c)


_POSTER_KINDS: dict[str, tuple[int, float, tuple[float, float]]] = {
    "synergy": (_motivational("SYNERGY", "1 + 1 = 3", _sunset_eagle), 0.15, (0.33, 0.73)),
    "teamwork": (_motivational("TEAMWORK", "ROW AS ONE", _rowing_crew), 0.15, (0.33, 0.73)),
    "excellence": (_motivational("EXCELLENCE", "AIM HIGHER", _mountain), 0.15, (0.33, 0.73)),
    "employee_mar": (employee_of_the_month("MAR 1987"), 0.12, (0.4, 0.72)),
    "employee_apr": (employee_of_the_month("APR 1987"), 0.12, (0.4, 0.72)),
    "employee_may": (employee_of_the_month("MAY 1987"), 0.12, (0.4, 0.72)),
    "founder": (founder_portrait(), 0.25, (0.38, 1.0)),
    "accident": (accident_board(), 0.18, (0.45, 0.75)),
    "high_voltage": (high_voltage_sign(), 0.1, (0.45, 0.73)),
    "gents": (washroom_plate(ladies=False), 0.1, (0.46, 0.68)),
    "ladies": (washroom_plate(ladies=True), 0.1, (0.46, 0.68)),
    "wash_hands": (wash_hands_notice(), 0.15, (0.4, 0.7)),
}
assert set(_POSTER_KINDS) | {"plan"} == set(POSTER_KINDS), "posters: shared/things.py"


def _poster(texture: int, half: float, z: tuple[float, float]) -> tuple[Box, ...]:
    edge = (_POSTER_EDGE,) * 4
    return (Box((0.0, _POSTER_DEPTH), (-half, half), z, (texture, *edge)),)


def _poster_edge() -> int:
    c = Canvas()
    c.paint(c.rect(0, 64, 0, 64), GREY, 3.0)
    return _texture(c)


_POSTER_EDGE = _poster_edge()
_PLAN_HALF, _PLAN_Z = 0.2, (0.42, 0.72)
_PLANS: dict[tuple[str, float, float], int] = {}


def plan_texture(game_map: GameMap, x: float, y: float, angle: float) -> int:
    """A fire evacuation plan of `game_map`: walls dark, floor pale, desks
    and counters grey, exits green (exit signs and lifts), a red dot where
    the plan hangs (a little out from its wall at (x, y), facing `angle`).
    Cells drawn about square on the (wider than tall) plan."""
    key = (game_map.name, round(x, 3), round(y, 3))
    if key in _PLANS:
        return _PLANS[key]
    c = Canvas()
    c.paint(c.rect(0, 64, 0, 64), CREAM, 15.0)
    c.paint(c.rect(0, 9, 0, 64), BRICK, 9.0)
    _centred(c, "FIRE PLAN", 2, CREAM, 15.0)
    w, h = game_map.width, game_map.height
    aspect = (2 * _PLAN_HALF) / (_PLAN_Z[1] - _PLAN_Z[0])  # texel height / width, in the world
    cw = max(1, min(60 // w, int(52 / (h * aspect))))
    ch = max(1, round(cw * aspect))
    u0, v0 = (64 - cw * w) // 2, 10 + (54 - ch * h) // 2

    def cell(cx: float, cy: float) -> BoolArray:
        u, v = u0 + math.floor(cx) * cw, v0 + math.floor(cy) * ch
        return c.rect(v, v + ch, u, u + cw)

    for cy in range(h):
        for cx in range(w):
            if game_map.is_full_wall(cx, cy):
                c.paint(cell(cx, cy), GREY, 3.0)
            elif game_map.is_solid(cx, cy):
                c.paint(cell(cx, cy), GREY, 10.0)
    for t in game_map.things:
        if t.type in ("exit_sign", "door_lift"):
            ex, ey = (
                (t.x, t.y)
                if t.type == "door_lift"
                else (t.x - 0.5 * math.cos(t.angle), t.y - 0.5 * math.sin(t.angle))
            )
            c.paint(cell(ex, ey), GREEN, 9.0)
    here = cell(x + 0.3 * math.cos(angle), y + 0.3 * math.sin(angle))
    c.paint(here, BRICK, 13.0)
    _PLANS[key] = _texture(c)
    return _PLANS[key]


def poster_plans(game_map: GameMap) -> list[PlacedBox]:
    """The map's fire plans (posters, kind=plan): each its own texture."""
    out = []
    for t in game_map.things:
        if t.type == "poster" and t.prop("kind") == "plan":
            tex = plan_texture(game_map, t.x, t.y, t.angle)
            for box in _poster(tex, _PLAN_HALF, _PLAN_Z):
                out.append(PlacedBox(t.x, t.y, 0.0, t.angle, box))
    return out


# --- Office swivel chair ---------------------------------------------------------------
# The cheap 1980s kind: a black five-star base on casters, a gas column, a
# charcoal fabric seat and a low backrest on a plastic shell. Four boxes, as
# a pod row has a dozen: the base is one flat box with the star painted on
# its top (the rest see-through), so the arms cost nothing. Model +x is the
# way the sitter faces (at the desk); the back is at -x. Real seat ~46 cm
# wide at ~45 cm, the back to ~95 cm; built a little wide (~1.2x) so it reads.

_CHAIR_SEAT = (-0.085, 0.09, 0.1, (0.15, 0.175))  # x0, x1, half-width, z
_CHAIR_BACK = (-0.11, -0.08, 0.085, (0.2, 0.36))
_CHAIR_BASE = 0.13  # half-span of the star, cells
_CHAIR_COLUMN = 0.018  # half-width


def chair_base_top() -> int:
    """The five-star base from above: five black arms from the hub, each
    ending in a caster; the rest see-through."""
    c = Canvas()
    v, u = c.v + 0.5 - 32.0, c.u + 0.5 - 32.0
    arms = np.zeros((64, 64), dtype=np.bool_)
    for k in range(5):
        a = 2 * math.pi * k / 5
        along = u * math.cos(a) + v * math.sin(a)
        across = -u * math.sin(a) + v * math.cos(a)
        arms |= (along > 0) & (along < 28) & (np.abs(across) < 2.5)
        cu, cv = 32 + 28 * math.cos(a), 32 + 28 * math.sin(a)
        c.paint(c.ellipse(cv, cu, 3.5, 3.5), GREY, 1.0)  # the caster
    c.paint(arms, GREY, 2.0)
    c.paint(c.ellipse(32, 32, 5, 5), GREY, 3.0)  # the hub
    return _texture(c)


def chair_fabric(edge: bool = True) -> int:
    """Charcoal tweed, flat (no noise close up): a piped edge round it."""
    c = Canvas()
    c.paint(c.rect(0, 64, 0, 64), STEEL, 4.0)
    if edge:
        c.paint(c.rect(0, 4, 0, 64) | c.rect(60, 64, 0, 64), STEEL, 2.5)
        c.paint(c.rect(0, 64, 0, 4) | c.rect(0, 64, 60, 64), STEEL, 2.5)
    return _texture(c)


def chair_shell() -> int:
    """The black plastic behind the back and under the seat, its edges."""
    c = Canvas()
    c.paint(c.rect(0, 64, 0, 64), GREY, 2.0)
    c.paint(c.rect(0, 6, 0, 64), GREY, 3.5)  # the top edge catches the light
    return _texture(c)


def chair_column() -> int:
    """The gas column: chrome, a black sleeve over its lower half."""
    c = Canvas()
    c.paint(c.rect(0, 64, 0, 64), GREY, 11.0)
    c.paint(c.rect(0, 64, 0, 16), GREY, 13.0)  # a highlight down one side
    c.paint(c.rect(34, 64, 0, 64), GREY, 2.0)
    return _texture(c)


def _chair() -> tuple[Box, ...]:
    clear = _texture(Canvas())
    fabric, plain, shell, column = (
        chair_fabric(),
        chair_fabric(edge=False),
        chair_shell(),
        chair_column(),
    )
    x0, x1, hw, seat_z = _CHAIR_SEAT
    bx0, bx1, bw, back_z = _CHAIR_BACK
    b, k = _CHAIR_BASE, _CHAIR_COLUMN
    return (
        Box((-b, b), (-b, b), (0.0, 0.02), (clear, clear, clear, clear, chair_base_top())),
        Box((-k, k), (-k, k), (0.02, seat_z[0]), (column,) * 5),
        # The seat: fabric on top, the shell's edge round it.
        Box((x0, x1), (-hw, hw), seat_z, (shell, shell, shell, shell, fabric)),
        # The back: fabric towards the sitter (+x), the shell behind.
        Box((bx0, bx1), (-bw, bw), back_z, (fabric, plain, shell, plain, shell)),
    )


# --- Lobby seating: a modular sofa, a coffee table, a fern ------------------------------
# A 1970s-80s modular sofa (as the reference: a lobby's corner group): seat
# units in rust velvet, button-tufted, on a brushed steel plinth with a dark
# wood rim. Each unit is three boxes (plinth, seat cushion, back cushion);
# runs of two and three, and a corner unit with backs on two sides, make an
# L. Model +x is the way the sitter faces, the backs at -x; a run's seats
# side by side along y. Sizes match the footprints in shared/things.py.

SOFA_SEAT_W, SOFA_HALF_D = 0.28, 0.16  # cells: a unit's width, half its depth
_SOFA_PLINTH, _SOFA_SEAT, _SOFA_BACK = 0.055, 0.155, 0.29  # walls: their tops
_SOFA_BACK_D = 0.08  # cells: the back cushion's depth
VELVET = ORANGE


def _tufted(base: float, buttons: tuple[tuple[int, int], ...]) -> int:
    """Rust velvet: piping round the edge, buttons pulling the cushion in."""
    c = Canvas()
    c.paint(c.rect(0, 64, 0, 64), VELVET, base)
    c.paint(~c.rect(3, 61, 3, 61), VELVET, base - 2.0)  # piping
    for v, u in buttons:
        c.paint(c.ellipse(v, u, 7, 7), VELVET, base - 1.0)  # the dimple
        c.paint(c.ellipse(v, u, 2.5, 2.5), VELVET, base - 3.0)  # the button
    return _texture(c)


def sofa_plinth() -> int:
    c = Canvas()
    c.paint(c.rect(0, 64, 0, 64), GREY, 11.0)  # brushed steel
    c.paint(c.rect(0, 64, 0, 64) & (c.u % 6 == 0), GREY, 12.0)
    c.paint(c.rect(0, 14, 0, 64), WOOD, 3.0)  # the dark wood rim
    return _texture(c)


_SEAT_TOP = _tufted(11.0, ((32, 20), (32, 44)))
_SEAT_SIDE = _tufted(10.0, ())
_BACK_FRONT = _tufted(11.0, ((22, 20), (22, 44)))
_BACK_TOP = _tufted(12.0, ())
_PLINTH = sofa_plinth()


def _sofa_unit(y: float, backs: tuple[str, ...] = ("-x",)) -> list[Box]:
    """One seat unit centred at y across the run; its back cushions on the
    sides named (-x: behind; -y: to one side, for a corner)."""
    d, w = SOFA_HALF_D, SOFA_SEAT_W / 2
    ys = (y - w, y + w)
    plinth_faces = (_PLINTH, _PLINTH, _PLINTH, _PLINTH, _PLINTH)
    out = [Box((-d, d), ys, (0.0, _SOFA_PLINTH), plinth_faces)]
    seat_x = (-d + _SOFA_BACK_D if "-x" in backs else -d, d)
    seat_y = (ys[0] + _SOFA_BACK_D, ys[1]) if "-y" in backs else ys
    seat = (_SEAT_SIDE, _SEAT_SIDE, _SEAT_SIDE, _SEAT_SIDE, _SEAT_TOP)
    out.append(Box(seat_x, seat_y, (_SOFA_PLINTH, _SOFA_SEAT), seat))
    back = (_BACK_FRONT, _SEAT_SIDE, _SEAT_SIDE, _SEAT_SIDE, _BACK_TOP)
    if "-x" in backs:
        out.append(Box((-d, -d + _SOFA_BACK_D), ys, (_SOFA_PLINTH, _SOFA_BACK), back))
    if "-y" in backs:  # its front (+x face) can't face the sitter: tufts on all sides
        side_back = (_BACK_FRONT,) * 4 + (_BACK_TOP,)
        out.append(
            Box((-d, d), (ys[0], ys[0] + _SOFA_BACK_D), (_SOFA_PLINTH, _SOFA_BACK), side_back)
        )
    return out


def _sofa(seats: int) -> tuple[Box, ...]:
    offset = (seats - 1) / 2
    return tuple(b for k in range(seats) for b in _sofa_unit((k - offset) * SOFA_SEAT_W))


# The coffee table: an amber glass top in a dark frame on four chrome legs,
# a fan of magazines at one end (BOOMER NEWS on top).
TABLE_HALF = (0.25, 0.15)  # cells: half its length (x), half its width (y)
TABLE_TOP = 0.15  # walls
_LEG = 0.012


def _cover(
    c: Canvas,
    left: int,
    top: int,
    width: int,
    height: int,
    paper: tuple[int, float],
    masthead: tuple[int, float],
    picture: Callable[[Canvas, BoolArray, BoolArray], None],
) -> None:
    """A magazine lying on a box top, upright to someone looking from its +y
    side (the coffee table's open side): on a top face, texel rows (v) run
    along the box's x and columns (u) across it from +y, so the cover's top
    is at high u. `left`, `top`: its top-left corner (v, u); `width` along v,
    `height` along u, in texels. A masthead band, a picture, cover lines
    (bars: no lettering). `picture(c, area, ...)` paints in `area`, given
    a mask of its upper half (for skies)."""
    q = c.v - left  # across the cover, left to right
    r = (top - 1) - c.u  # down the cover, top to bottom
    cover = (q >= 0) & (q < width) & (r >= 0) & (r < height)
    c.paint(cover, *paper)
    c.paint(cover & (r < 6), *masthead)
    c.paint(cover & (r >= 2) & (r < 4) & (q >= 2) & (q < width - 3), paper[0], paper[1] + 1.0)
    area = cover & (r >= 8) & (r < height - 8) & (q >= 2) & (q < width - 2)
    picture(c, area, area & (r < (8 + height - 8) // 2))
    for k, length in enumerate((width - 6, width - 9)):  # cover lines
        c.paint(cover & (r == height - 6 + 2 * k) & (q >= 2) & (q < 2 + length), GREY, 3.0)
    c.paint(~cover & (q >= 0) & (q <= width) & (r == height), GREY, 1.0)  # its shadow edge


def _city_picture(c: Canvas, area: BoolArray, sky: BoolArray) -> None:
    c.paint(area, BLUE, 10.0)
    c.paint(area & ~sky, STEEL, 5.0)  # towers
    c.paint(area & ~sky & (c.v % 3 == 0), STEEL, 9.0)


def _portrait_picture(c: Canvas, area: BoolArray, sky: BoolArray) -> None:
    c.paint(area, OCHRE, 12.0)
    c.paint(area & ~sky, GREY, 3.0)  # a suit
    c.paint(area & sky & (c.v % 7 > 1), SAND, 13.0)  # a head, roughly


def _chart_picture(c: Canvas, area: BoolArray, sky: BoolArray) -> None:
    c.paint(area, CREAM, 14.0)
    c.paint(area & (c.v % 4 == 0) & ~sky, GREEN, 8.0)  # rising bars
    c.paint(area & (c.v % 4 == 0) & sky & (c.v % 8 == 0), GREEN, 8.0)


def table_glass() -> int:
    """The coffee table's top: amber glass in a dark frame, and at its -x
    end three magazines, two side by side and one fanned over them, upright
    to someone at its +y side (the fern stands at the other end)."""
    c = Canvas()
    c.paint(c.rect(0, 64, 0, 64), ORANGE, 12.0)  # amber glass
    c.paint(c.rect(0, 64, 0, 64) & ((c.u + c.v) % 40 < 2), ORANGE, 13.0)  # faint reflections
    c.paint(~c.rect(2, 62, 2, 62), GREY, 2.0)  # the frame
    # 64 texels span the table's 0.5 along v and 0.3 across u: texels are
    # ~1.7x longer along v, so a magazine is ~15 wide (v) by ~33 tall (u).
    _cover(c, 4, 46, 15, 33, (CREAM, 15.0), (BRICK, 10.0), _city_picture)
    _cover(c, 20, 44, 15, 33, (CREAM, 14.0), (BLUE, 7.0), _chart_picture)
    _cover(c, 12, 56, 15, 33, (CREAM, 15.0), (GREEN, 7.0), _portrait_picture)  # on top
    return _texture(c)


def _coffee_table() -> tuple[Box, ...]:
    hx, hy = TABLE_HALF
    frame = _plain_texture(GREY, 2.0)
    chrome = _plain_texture(GREY, 13.0)
    top = Box((-hx, hx), (-hy, hy), (TABLE_TOP - 0.02, TABLE_TOP), (frame,) * 4 + (table_glass(),))
    legs = tuple(
        Box((x - _LEG, x + _LEG), (y - _LEG, y + _LEG), (0.0, TABLE_TOP - 0.02), (chrome,) * 5)
        for x in (-hx + 0.02, hx - 0.02)
        for y in (-hy + 0.02, hy - 0.02)
    )
    return (top, *legs)


def _plain_texture(ramp: int, shade: float) -> int:
    c = Canvas()
    c.paint(c.rect(0, 64, 0, 64), ramp, shade)
    return _texture(c)


# A small fern (a bird's nest fern) in a copper bowl, for a table or desk:
# fronds arching up and out, drooping at their tips; crossed see-through
# cards, and one flat card seen from above.
_FERN_R, _FERN_Z = 0.12, (0.05, 0.17)


def fronds(seed: int, flat: bool = False) -> int:
    rng = np.random.default_rng(seed)
    c = Canvas()
    count = 9 if flat else 7
    for k in range(count):
        a = (k / count) * 2 * math.pi if flat else math.radians(-75 + 150 * k / (count - 1))
        a += rng.uniform(-0.15, 0.15)
        for t in np.linspace(0.05, 1.0, 40):
            if flat:  # from the middle, out in every direction
                u, v = 32 + math.cos(a) * t * 30, 32 + math.sin(a) * t * 30
            else:  # from the bowl, up and out, the tip drooping
                u = 32 + math.sin(a) * t * 30
                v = 63 - math.cos(a) * t * 50 + (t**2) * 22 * abs(math.sin(a))
            width = 3.0 * (1 - t) + 1.0
            leaf = c.ellipse(v, u, width, width)
            c.paint(leaf, GREEN, 8.0 + 3.0 * (1 - t) + (1.0 if k % 2 else 0.0))
            c.paint(c.ellipse(v, u, 0.6, 0.6), GREEN, 12.5)  # the midrib
    return _texture(c)


def _fern() -> tuple[Box, ...]:
    copper, earth = _plain_texture(ORANGE, 9.0), soil()
    bowl = (
        *_octagon(0.045, (0.0, 0.04), (copper, earth)),
        *_octagon(0.06, (0.04, 0.06), (_plain_texture(ORANGE, 11.0), earth)),
    )
    edge = _texture(Canvas())  # see-through
    r = (-_FERN_R, _FERN_R)
    cards = tuple(
        Box((-0.003, 0.003), r, _FERN_Z, (tex, edge, tex, edge, edge), turn=turn)
        for turn, tex in zip(
            (0.0, math.pi / 3, 2 * math.pi / 3), (fronds(61), fronds(62), fronds(63)), strict=True
        )
    )
    flat = Box(r, r, (0.09, 0.094), (edge,) * 4 + (fronds(64, flat=True),), turn=0.4)
    return (*bowl, *cards, flat)


# --- A coffee cup (a pickup: +5 coffee) ---------------------------------------------
# A chunky office mug, white with a blue band, full of coffee, its handle out
# to one side. About 1.6x real size (13 cm across), as the desk things are.
_MUG_R, _MUG_H = 0.024, 0.055  # cells (its inradius), walls


def mug_side() -> int:
    c = Canvas()
    c.paint(c.rect(0, 64, 0, 64), CREAM, 14.0)
    c.paint(c.rect(22, 36, 0, 64), BLUE, 8.0)  # the band
    c.paint(c.rect(0, 6, 0, 64), CREAM, 15.0)  # its lit rim
    c.paint(c.rect(56, 64, 0, 64), CREAM, 11.0)  # shadowed foot
    return _texture(c)


def coffee_top() -> int:
    c = Canvas()
    c.paint(c.rect(0, 64, 0, 64), WOOD, 3.0)  # coffee
    c.paint(c.ellipse(24, 26, 10, 14), WOOD, 5.0)  # a bit of crema catching the light
    c.paint(~c.rect(6, 58, 6, 58), CREAM, 14.0)  # the rim round it
    return _texture(c)


def _coffee_cup() -> tuple[Box, ...]:
    side = mug_side()
    body = _octagon(_MUG_R, (0.0, _MUG_H), (side, coffee_top()))
    white = _plain_texture(CREAM, 13.0)
    # The handle: a loop out to one side, as a box (its hole too small to show).
    handle = Box((-0.006, 0.006), (_MUG_R, _MUG_R + 0.018), (0.012, _MUG_H - 0.01), (white,) * 5)
    return (*body, handle)


# --- Late-80s business PC (the boss's): green screen on a flat system unit -----------
# Generic (no maker's badge), in the pearly off-white of the time: a wide
# system unit lying flat, a big 14" CRT standing on it (a deep bezel box,
# the tube's housing tapering behind as a smaller box), a green-phosphor
# screen that glows while there's power, and a long heavy keyboard in
# front. Real sizes ~48 x 40 x 13 cm (unit), 36 x 38 x 33 cm (monitor),
# 49 x 21 cm (keyboard); built ~1.9x (the Mac's 1.6x and a bit: the
# boss's desk is huge). Model +x faces the
# user (the screen), +y is the user's left.

PC_BEIGE = CREAM
_PC_UNIT = (0.28, 0.33, 0.085)  # depth (x), width (y), height (walls)
_PC_BEZEL = (0.1, 0.25, 0.22)
_PC_TUBE = (0.14, 0.18, 0.17)
_PC_KEYS = (0.13, 0.33, 0.03)
_PC_SCREEN = (0.19, 0.15)  # the glass: width, height, centred on the bezel


def _beige(shade: float = 11.0) -> Canvas:
    c = Canvas()
    c.paint(c.rect(0, 64, 0, 64), PC_BEIGE, shade)
    c.paint(c.rect(0, 2, 0, 64), PC_BEIGE, shade + 1.5)  # top edge catches the light
    c.paint(c.rect(60, 64, 0, 64), PC_BEIGE, shade - 3.0)  # shadow underneath
    return c


def pc_unit_front() -> int:
    """The system unit's front (wide and low, stretched over 64 x 64): two
    5.25" floppy drives on the right, a hard disk lamp, a big power switch."""
    c = _beige(11.5)
    for u0 in (34, 49):  # the drives, each a slot and a latch
        c.paint(c.rect(14, 46, u0, u0 + 13), PC_BEIGE, 9.0)
        c.paint(c.rect(28, 32, u0 + 1, u0 + 12), GREY, 2.0)
        c.paint(c.rect(20, 26, u0 + 5, u0 + 8), PC_BEIGE, 7.0)
    c.paint(c.rect(18, 24, 6, 9), GREEN, 11.0)  # power lamp
    c.paint(c.rect(18, 24, 12, 15), ORANGE, 9.0)  # disk lamp
    c.paint(c.rect(4, 8, 6, 30), PC_BEIGE, 9.0)  # a moulded line along the top
    for u0 in range(6, 28, 3):  # vents below
        c.paint(c.rect(34, 50, u0, u0 + 1), PC_BEIGE, 7.0)
    return _texture(c)


def pc_side(vents: bool = True) -> int:
    c = _beige(10.0)
    if vents:
        for v0 in range(10, 50, 4):
            c.paint(c.rect(v0, v0 + 2, 40, 58), PC_BEIGE, 7.0)
    return _texture(c)


def pc_top() -> int:
    c = _beige(12.0)
    return _texture(c)


def pc_bezel() -> int:
    """The monitor's front: a deep beige bezel round the dark screen (the
    glowing glass itself is a box of its own, just proud of this), a row of
    knobs under it."""
    c = _beige(11.5)
    c.paint(c.rect(5, 52, 6, 58), PC_BEIGE, 8.0)  # the bezel's inner step
    c.paint(c.rect(6, 51, 7, 57), GREY, 1.0)
    for u0 in (40, 46, 52):
        c.paint(c.rect(55, 59, u0, u0 + 3), PC_BEIGE, 6.0)  # brightness, contrast, power
    return _texture(c)


def pc_screen() -> int:
    """The green screen: a spreadsheet, green on black. It glows while
    there's power (Box.mains); without, it only takes the room's light."""
    c = Canvas()
    c.paint(c.rect(0, 64, 0, 64), GREEN, 0.0)
    rng = np.random.default_rng(77)
    c.paint(c.rect(3, 9, 3, 61), GREEN, 11.0)  # the inverse menu bar
    for u0, u1 in ((5, 13), (16, 22), (25, 35), (38, 44)):
        c.paint(c.rect(5, 7, u0, u1), GREEN, 1.0)
    c.paint(c.rect(13, 14, 3, 61), GREEN, 7.0)  # column header rule
    for row in range(7):  # rows of figures: a label, then numbers right-aligned
        v0 = 17 + row * 6
        c.paint(c.rect(v0, v0 + 3, 3, 3 + int(rng.integers(6, 14))), GREEN, 12.0)
        for right in (34, 47, 60):
            c.paint(c.rect(v0, v0 + 3, right - int(rng.integers(5, 10)), right), GREEN, 13.0)
    c.paint(c.rect(59, 62, 3, 6), GREEN, 15.0)  # the cursor
    return _texture(c)


def pc_keyboard_top() -> int:
    """A long, heavy keyboard: function keys across the top, the main block,
    the cursor keys and number pad to the right (stretched over a wide,
    shallow box, so its rows are tall in texels)."""
    c = _beige(11.0)
    rows = ((6, 14), (17, 25), (27, 35), (37, 45), (47, 55))
    for r, (v0, v1) in enumerate(rows):
        main = (3, 41) if r else (3, 41)
        step = 3
        spans = [(u, u + 2) for u in range(main[0] + (r % 2), main[1], step)]
        if r == 4:
            spans = [(3, 7), (8, 12), (13, 33), (34, 38), (39, 41)]  # the space bar
        spans += [(u, u + 2) for u in range(51, 61, 3)]  # the number pad
        if r >= 3:
            spans += [(44, 46), (47, 49)] if r == 4 else [(45, 48)]  # cursor keys
        for u0, u1 in spans:
            c.paint(c.rect(v0, v1, u0, u1), GREY, 9.0)
            c.paint(c.rect(v0, v0 + 2, u0, u1), GREY, 11.0)
    return _texture(c)


def _glass_edge() -> int:
    c = Canvas()
    c.paint(c.rect(0, 64, 0, 64), GREY, 1.0)
    return _texture(c)


_PC_GLASS_EDGE = _glass_edge()


def _pc() -> tuple[Box, ...]:
    side, plain_side, top = pc_side(), pc_side(vents=False), pc_top()
    ud, uw, uh = _PC_UNIT
    bd, bw, bh = _PC_BEZEL
    td, tw, th = _PC_TUBE
    kd, kw, kh = _PC_KEYS
    x0 = ud / 2  # the unit's front
    bx = (x0 - bd - 0.01, x0 - 0.01)  # the monitor, set back a touch on the unit
    sw, sh = _PC_SCREEN
    s0 = uh + (bh - sh) / 2 + 0.01
    k0 = x0 + 0.03
    return (
        Box((-ud / 2, x0), (-uw / 2, uw / 2), (0.0, uh), (pc_unit_front(), side, side, side, top)),
        Box(
            bx,
            (-bw / 2, bw / 2),
            (uh, uh + bh),
            (pc_bezel(), plain_side, plain_side, plain_side, top),
        ),
        Box(
            (bx[0] - td, bx[0]),
            (-tw / 2, tw / 2),
            (uh + 0.01, uh + th),
            (side, side, side, side, top),
        ),
        Box(
            (bx[1], bx[1] + 0.002),
            (-sw / 2, sw / 2),
            (s0, s0 + sh),
            (pc_screen(), *(_PC_GLASS_EDGE,) * 4),
            glow=True,
            mains=True,
            off_faces=(_BLANK_SCREEN, *(_PC_GLASS_EDGE,) * 4),
        ),
        Box(
            (k0, k0 + kd), (-kw / 2, kw / 2), (0.0, kh), (side, side, side, side, pc_keyboard_top())
        ),
    )


# --- 1980s desk phone ----------------------------------------------------------------
# Brown plastic, low: on the user's left a raised cradle with the handset
# lying on it front to back (an earpiece and a mouthpiece, wider and deeper,
# joined by a narrower grip: its shape painted on one box);
# on the right the keypad and a column of amber line buttons. Three boxes;
# still, only a few in a map (the boss's desk). Real
# size ~22 x 20 cm; built ~1.6x, as the Mac is, so it reads. Model +x faces
# whoever uses it (as the Mac's screen); +y is the phone's right, the user's
# left (they face it, looking along -x).

_PHONE_W, _PHONE_D = 0.065, 0.06  # half-width (y), half-depth (x), cells
_PHONE_LOW, _PHONE_HIGH = 0.028, 0.033  # walls: the keypad's top, the cradle's (just above)
_CRADLE_Y = 0.015  # the cradle runs from here to the phone's +y edge


def phone_keys() -> int:
    """The keypad side from above: v runs back to front (+x), u along it to
    the user's right (-y). A 3 x 4 keypad, the line buttons in a column on
    the right, amber."""
    c = Canvas()
    c.paint(c.rect(0, 64, 0, 64), WOOD, 4.0)
    c.paint(c.rect(0, 64, 62, 64) | c.rect(60, 64, 0, 64), WOOD, 3.0)
    for row in range(4):
        for col in range(3):
            v0, u0 = 8 + row * 13, 6 + col * 13
            c.paint(c.rect(v0, v0 + 9, u0, u0 + 9), WOOD, 2.0)  # a key
            c.paint(c.rect(v0, v0 + 2, u0, u0 + 9), GREY, 9.0)  # its pale top edge
    for k in range(5):
        v0 = 6 + k * 11
        c.paint(c.rect(v0, v0 + 7, 48, 58), ORANGE, 13.0)  # the line buttons
    return _texture(c)


def phone_body() -> int:
    c = Canvas()
    c.paint(c.rect(0, 64, 0, 64), WOOD, 3.5)
    c.paint(c.rect(0, 8, 0, 64), WOOD, 5.0)  # its top edge catches the light
    return _texture(c)


def handset_side() -> int:
    """The handset's long sides (u along it, v down): an old receiver's
    profile, painted, not cut out: the earpiece and mouthpiece ends full
    height, the grip between them lighter in its upper part with a dark
    shadow under it."""
    c = Canvas()
    c.paint(c.rect(0, 64, 0, 64), WOOD, 6.5)
    c.paint(c.rect(0, 6, 0, 64), WOOD, 8.0)  # the light along its back
    c.paint(c.rect(32, 64, 18, 46), WOOD, 2.0)  # the shadow under the grip
    c.paint(c.rect(56, 64, 0, 18) | c.rect(56, 64, 46, 64), WOOD, 4.0)  # the cups' undersides
    return _texture(c)


def handset_end() -> int:
    """An end of the handset (an ear or mouth cup), seen head on."""
    c = Canvas()
    c.paint(c.rect(0, 64, 0, 64), WOOD, 5.5)
    c.paint(c.rect(0, 10, 0, 64), WOOD, 7.5)
    return _texture(c)


def handset_top() -> int:
    """The handset from above (v along it, u across): a dog-bone, painted:
    the cups at the ends full width, the grip narrower, darker beside it."""
    c = Canvas()
    c.paint(c.rect(0, 64, 0, 64), WOOD, 3.0)  # beside the grip: in shadow
    shape = c.rect(0, 18, 0, 64) | c.rect(46, 64, 0, 64) | c.rect(0, 64, 18, 46)
    c.paint(shape, WOOD, 6.5)
    c.paint(shape & c.rect(0, 64, 26, 38), WOOD, 8.0)  # the light along its back
    return _texture(c)


def cradle_top() -> int:
    """The cradle under the handset: dark, so the handset stands out on it."""
    c = Canvas()
    c.paint(c.rect(0, 64, 0, 64), WOOD, 2.0)
    return _texture(c)


def _phone() -> tuple[Box, ...]:
    keys, body, cradle = phone_keys(), phone_body(), cradle_top()
    side, end, top = handset_side(), handset_end(), handset_top()
    w, d, h = _PHONE_W, _PHONE_D, _PHONE_HIGH
    return (
        Box((-d, d), (-w, _CRADLE_Y), (0.0, _PHONE_LOW), (body, body, body, body, keys)),
        Box((-d, d), (_CRADLE_Y, w), (0.0, h), (body, body, body, body, cradle)),
        # The handset, front to back along the cradle, a little beyond it; its
        # shape (cups and grip) painted on one box.
        Box(
            (-d - 0.008, d + 0.008),
            (_CRADLE_Y + 0.002, w + 0.004),
            (h, h + 0.022),
            (end, side, end, side, top),
        ),
    )


# --- Stairs ----------------------------------------------------------------------
# Stair cells (wall type 'h') slope smoothly for movement (doomer/shared), but
# are drawn as real steps: one box per riser. A cell rises STAIR_RISE units
# (half a storey) in 8 steps of 1/16 storey (~17 cm) with 1/8-cell (34 cm)
# treads. Each step's top is level with the slope at the step's back edge, so
# feet on the slope are at most one riser below the tread they're over.

STEPS_PER_STOREY = 16


def stair_tread() -> int:
    """A tread from above: blue carpet, a metal nosing strip along the front
    edge (the downhill side: the top of the image is a box's -x side).

    A box's top is stretched once over it, and a tread is a cell wide but an
    eighth deep: v is squashed 8:1. So the carpet's speckle (as
    areas.blue_carpet, the balcony's) varies per texel across the step but
    only in 8-texel bands front to back, which comes out square."""
    rng = np.random.default_rng(31)
    c = Canvas()
    speckle = rng.random((8, 64))[c.v // 8, c.u]
    shade = (
        6.0
        + (noise(rng, 8)[:1, :] - 0.5) * 0.6
        + np.where(speckle < 0.1, -1.0, 0.0)
        + np.where(speckle > 0.93, 1.0, 0.0)
    )
    c.paint(c.rect(0, 64, 0, 64), BLUE, shade)
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


_TREAD, _RISER = stair_tread(), stair_riser()


def _texture_of(texture: npt.NDArray[np.uint8]) -> int:
    """Register an opaque (v, u) wall texture as a box face texture."""
    _TEXTURES.append(np.ascontiguousarray(texture.T))
    _OPAQUE.append(np.ones((64, 64), dtype=np.bool_))
    return len(_TEXTURES) - 1


# A staircase's sides look like the walls of the area it stands in (the
# lobby's stone), as its balcony's front does.
_AREA_SIDES = {area: _texture_of(look.wall()) for area, look in AREA_LOOKS.items()}


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
            side = _AREA_SIDES[game_map.area_at(cx, cy)]
            for i in range(steps):
                box = Box(
                    (-0.5 + i / steps, -0.5 + (i + 1) / steps),
                    (-0.5, 0.5),
                    (0.0, low + (top - low) * (i + 1) / steps),
                    (side, side, _RISER, side, _TREAD),
                    tiled_sides=True,
                    riser=(top - low) / steps,
                )
                out.append(PlacedBox(cx + 0.5, cy + 0.5, 0.0, angle, box))
    return out


# --- Cubicle screens on desk pairs ('j') ---------------------------------------
# A fabric screen down the middle of each pair (between the two sitters) and
# one across the pair at each cell edge, so every sitter has a three-sided
# booth. Where they are, their size and their collision: doomer/shared/screens.py.


def screen_fabric(look: int = 0) -> int:
    """Grey-blue woven fabric with an aluminium cap; stretched over a whole
    cell's length and the screen's height, so a texel is ~4.2 cm across by
    ~0.9 cm high: anything pinned on is drawn ~4.7 times taller than wide.

    `look` (SCREEN_LOOKS): 0 bare; 1 two Post-its; 2 a cluster of four (one
    pink); 3 a wall calendar and a Post-it. Things pinned on are flat, so a
    texture suits them (as the desks' paperwork)."""
    rng = np.random.default_rng(41)
    c = Canvas()
    weave = ((c.v // 2) % 2) * 0.4
    c.paint(c.rect(0, 64, 0, 64), STEEL, 7.0 + weave + (noise(rng, 32) - 0.5) * 1.2)
    c.paint(c.rect(0, 7, 0, 64), GREY, 11.0)  # cap
    c.paint(c.rect(0, 2, 0, 64), GREY, 13.0)
    c.paint(c.rect(60, 64, 0, 64), GREY, 6.0)  # foot, in shadow
    pin = np.random.default_rng(140 + look)

    def post_it(v0: int, u0: int, ramp: int = OCHRE) -> None:
        """~10 cm square: 3 texels across, 12 down; shadow below and right."""
        c.paint(c.rect(v0 + 1, v0 + 13, u0 + 1, u0 + 4), STEEL, 4.0)
        c.paint(c.rect(v0, v0 + 12, u0, u0 + 3), ramp, 15.0 if ramp == OCHRE else 13.0)
        c.paint(c.rect(v0, v0 + 3, u0, u0 + 3), ramp, 13.0 if ramp == OCHRE else 11.0)

    def spot(width: int) -> tuple[int, int]:
        return int(pin.integers(12, 44)), int(pin.integers(4, 60 - width))

    if look == 1:
        for _ in range(2):
            post_it(*spot(3))
    elif look == 2:
        v0, u0 = spot(10)
        for k, (dv, du) in enumerate(((0, 0), (2, 4), (15, 1), (13, 6))):
            post_it(min(v0 + dv, 46), u0 + du, PINK if k == 2 else OCHRE)
    elif look == 3:
        # A wall calendar, ~34 x 42 cm: 8 texels across, 47 down. Its month
        # in red, a landscape (sky, hills), then the grid of days.
        u0 = int(pin.integers(2, 7))  # near an end: Macs stand against screens at 14-27, 37-50
        c.paint(c.rect(10, 58, u0 + 1, u0 + 9), STEEL, 4.0)  # shadow
        c.paint(c.rect(9, 57, u0, u0 + 8), CREAM, 14.0)
        c.paint(c.rect(9, 13, u0, u0 + 8), BRICK, 10.0)  # the month
        c.paint(c.rect(13, 22, u0, u0 + 8), BLUE, 11.0)  # sky
        c.paint(c.rect(22, 30, u0, u0 + 8), GREEN, 8.0)  # hills
        c.paint(c.rect(19, 23, u0 + 2, u0 + 6) & (c.v + c.u * 2 > 41 + u0 * 2), GREEN, 9.5)
        days = c.rect(32, 56, u0 + 1, u0 + 7) & ((c.v - 32) % 5 == 0)
        c.paint(days, GREY, 9.0)  # the rows of weeks
        c.paint(c.rect(32, 56, u0, u0 + 8) & (c.u % 2 == 0) & ((c.v - 34) % 5 == 0), GREY, 7.0)
        c.paint(c.rect(9, 10, u0 + 3, u0 + 5), GREY, 4.0)  # the pin
        post_it(int(pin.integers(14, 40)), u0 + 14 + int(pin.integers(0, 12)))
    return _texture(c)


def screen_trim() -> int:
    """The aluminium edge (screen ends and the top of the cap)."""
    c = Canvas()
    c.paint(c.rect(0, 64, 0, 64), GREY, 11.0)
    return _texture(c)


SCREEN_LOOKS = 4
# How often each look appears: mostly bare, the calendar rarest.
_LOOK_WEIGHTS = (0, 0, 0, 0, 0, 1, 1, 1, 2, 2, 3)
_FABRICS, _TRIM = [screen_fabric(k) for k in range(SCREEN_LOOKS)], screen_trim()


def _look(x: float, y: float, side: int) -> int:
    """A screen face's look, scrambled from its position, so it's the same
    every time and neighbours differ."""
    n = int(round(x * 16)) * 7919 + int(round(y * 16)) * 104729 + side * 15485863
    return _LOOK_WEIGHTS[n % 1009 % len(_LOOK_WEIGHTS)]


def desk_screens(game_map: GameMap) -> list[PlacedBox]:
    """The map's cubicle screens (doomer/shared/screens.py) as boxes."""
    out = []
    for sc in shared_desk_screens(game_map):
        middle, length = (sc.lo + sc.hi) / 2, (sc.hi - sc.lo) / 2
        thin, long = (-SCREEN_HALF, SCREEN_HALF), (-length, length)
        # Screens are world-aligned (angle 0): thin across x (big faces east
        # and west) or thin across y (big faces north and south). Each big
        # face gets its own look.
        x, y = (sc.at, middle) if sc.axis == 0 else (middle, sc.at)
        a, b = _FABRICS[_look(x, y, 0)], _FABRICS[_look(x, y, 1)]
        if sc.axis == 0:  # the plane x = at: thin across x
            box = Box(thin, long, (0.0, SCREEN_HEIGHT), (a, _TRIM, b, _TRIM, _TRIM))
        else:
            box = Box(long, thin, (0.0, SCREEN_HEIGHT), (_TRIM, a, _TRIM, b, _TRIM))
        out.append(PlacedBox(x, y, sc.base, 0.0, box))
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
        # shoulder, then the body up to its flat base at the top (~1.45 m).
        # A 19-litre bottle is nearly as wide as the cabinet.
        *_octagon(0.035, (0.29, 0.30), (water, water)),  # neck
        *_octagon(0.075, (0.30, 0.33), (water, water)),  # shoulder
        *_octagon(0.105, (0.33, 0.54), (water, water)),  # body
    )


# --- Potted ficus ---------------------------------------------------------------
# ~1.95 m: a terracotta pot (octagonal, flaring to a rim), a single trunk,
# and a crown of crossed leaf cards: thin panels with see-through leaf
# textures, four upright at 45 degrees to each other through the crown's
# axis and a flat one across it, so it reads as a bush from any side and
# from above. (Solid boxes would look like a topiary cube.) Pot footprint
# matches its collision (half-width 0.16); the crown spreads wider, at head
# height, as a real one does.

_CROWN = (0.30, 0.72)  # walls
_CROWN_R = 0.23  # cells


def leaves(seed: int, flat: bool = False) -> int:
    """A crown of individual leaves (a weeping fig's: small, pointed, glossy),
    see-through between them. Each is a pointed oval at its own angle,
    drooping a little on upright cards, lit on one half with a darker edge,
    the crown's top catching more light. They're scattered inside a ragged
    outline, not a circle, so the crown has no hard round edge. Flat cards
    are sparser: seen nearly edge-on, a dense one squashes into a stripe.
    Exaggerated (~12 cm leaves) so single leaves read at this resolution."""
    rng = np.random.default_rng(seed)
    c = Canvas()
    v, u = c.v.astype(np.float64), c.u.astype(np.float64)
    cv, cu = (32.0, 32.0) if flat else (34.0, 32.0)
    angle = np.arctan2(v - cv, u - cu)
    a, b = rng.uniform(0, 2 * math.pi, 2)
    reach = 30.0 * (1.0 + 0.12 * np.sin(3 * angle + a) + 0.08 * np.sin(5 * angle + b))
    inside = np.hypot(v - cv, u - cu) < reach
    count = 90 if flat else 230
    placed = 0
    while placed < count:
        lv, lu = rng.uniform(0, 64, 2)
        if not inside[int(lv), int(lu)]:
            continue
        placed += 1
        # Pointing anywhere on a flat card; mostly downwards (drooping) on an
        # upright one (v grows downwards).
        theta = rng.uniform(0, 2 * math.pi) if flat else math.pi / 2 + rng.normal(0, 0.9)
        along = (v - lv) * math.sin(theta) + (u - lu) * math.cos(theta)
        across = (v - lv) * math.cos(theta) - (u - lu) * math.sin(theta)
        length, width = rng.uniform(2.8, 3.8), rng.uniform(1.3, 1.8)
        taper = 1.0 - 0.35 * np.clip(along / length, 0.0, 1.0)  # pointed at its tip
        leaf = (along / length) ** 2 + (across / (width * taper)) ** 2 <= 1.0
        edge = leaf & ((along / length) ** 2 + (across / (width * taper)) ** 2 > 0.55)
        top = 0.0 if flat else (64.0 - lv) / 64.0 * 2.0  # the crown's top catches the light
        base = 4.5 + top + rng.uniform(-0.8, 0.8)
        young = rng.random() < 0.15  # a few paler new leaves
        ramp = MOSS if young else GREEN
        shade = np.where(across > 0, base + 1.5, base)  # one half catches the light
        c.paint(leaf, ramp, np.where(edge, base - 1.5, shade))
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
    trunk = Box((-0.016, 0.016), (-0.016, 0.016), (0.22, 0.5), (stem,) * 5)
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
            # Below standing eye level (0.6), where a flat card is edge-on. (A
            # second, near the crown's top, stuck out as a flat slab: gone.)
            (0.47, 0.9, 0.3, leaves(75, flat=True)),
        )
    )
    return (*pot, trunk, *upright, *flat)


# --- Doors (doomer/shared/doors.py has their state and collision) -------------------
# Drawn from the simulation's door states each frame. A hinged door is two
# fixed jambs and a leaf whose model origin is the hinge, turned by how far
# open it is (so it really swings). Sliding doors are two glass leaves that
# part into the walls either side (which hide them).


def door_leaf(handle_on_right: bool) -> int:
    """One face of a hinged door: flush wood veneer (even, a few faint grain
    lines: noise here quantised into dark streaks), a steel kick plate, and
    the handle at the free end (right or left as you look at this face)."""
    c = Canvas()
    c.paint(c.rect(0, 64, 0, 64), WOOD, 8.0)
    c.paint(c.rect(4, 52, 0, 64) & (c.u % 9 == 4), WOOD, 7.0)  # grain
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
    """The fixed panel beside a door: flat painted plaster with a wooden
    architrave along both long edges, a lit edge on each."""
    c = Canvas()
    c.paint(c.rect(0, 64, 0, 64), SAND, 9.0)
    c.paint(c.rect(0, 64, 0, 5) | c.rect(0, 64, 59, 64), WOOD, 6.0)
    c.paint(c.rect(0, 64, 0, 1) | c.rect(0, 64, 59, 60), WOOD, 8.0)
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

# --- ID passes: a spinning card (pickups) ----------------------------------------------
# A Boomer Industries ID card, ~4.5x real size (as Doom's keys were big)
# and full-bright, hovering over whatever it lies on and slowly turning
# (Renderer turns SPINNING models by the clock): motion says "pick me up".
# Mostly the pass's colour, so it reads across a room; its name on it too.

PASS_HALF_WIDTH, PASS_HEIGHT, PASS_HALF_THICK = 0.045, 0.125, 0.004
PASS_HOVER = 0.04  # walls: its foot above the surface it lies on
PASS_SPIN = 1.8  # radians a second
PASS_BOB = 0.008  # walls: up and down as it turns


def _card(c: Canvas, ramp: int) -> BoolArray:
    """The card's shape: rounded corners (see-through), the clip slot."""
    corners = (
        c.rect(0, 2, 0, 2) | c.rect(0, 2, 62, 64) | c.rect(62, 64, 0, 2) | c.rect(62, 64, 62, 64)
    )
    card = ~corners
    c.paint(card, ramp, 9.0)
    c.paint(card & (c.rect(0, 2, 0, 64) | c.rect(0, 64, 0, 2)), ramp, 11.0)  # lit edges
    return card


def pass_front(colour: str, ramp: int) -> int:
    """BOOMER on a white strip, a big photo (CAST["pass_<colour>"], 40
    texels: two to each of its pixels), name lines and a barcode beside it,
    the colour's name on a dark band at the foot."""
    c = Canvas()
    _card(c, ramp)
    c.paint(c.rect(1, 3, 26, 38), GREY, 2.0)  # the clip slot
    c.paint(c.rect(4, 11, 3, 61), CREAM, 15.0)
    _centred(c, "BOOMER", 5, GREY, 2.0)
    c.paint(c.rect(13, 55, 2, 44), GREY, 2.0)  # the photo's border
    draw_portrait(c, 14, 3, 40, CAST[f"pass_{colour}"], (STEEL, 12.0))
    for v0 in (15, 20, 25):  # name, department, number
        c.paint(c.rect(v0, v0 + 2, 47, 61), CREAM, 15.0)
    c.paint(c.rect(30, 55, 47, 61), CREAM, 15.0)
    c.paint(c.rect(31, 54, 48, 60) & np.isin(c.v % 5, (0, 2, 3)), GREY, 1.0)  # a barcode
    c.paint(c.rect(56, 63, 2, 62), ramp, 4.0)
    _centred(c, colour.upper(), 57, CREAM, 15.0)
    return _texture(c)


def pass_back(ramp: int) -> int:
    """The magnetic stripe, a line of small print."""
    c = Canvas()
    _card(c, ramp)
    c.paint(c.rect(10, 20, 0, 64), GREY, 1.0)
    for v0 in (30, 34, 38):
        c.paint(c.rect(v0, v0 + 1, 8, 56), ramp, 6.0)
    return _texture(c)


def pass_edge(ramp: int) -> int:
    c = Canvas()
    c.paint(c.rect(0, 64, 0, 64), ramp, 6.0)
    return _texture(c)


def _pass(colour: str, ramp: int) -> tuple[Box, ...]:
    edge = pass_edge(ramp)
    faces = (pass_front(colour, ramp), edge, pass_back(ramp), edge, edge)
    t, w = PASS_HALF_THICK, PASS_HALF_WIDTH
    return (Box((-t, t), (-w, w), (PASS_HOVER, PASS_HOVER + PASS_HEIGHT), faces, glow=True),)


# Box models the renderer turns by the clock: name -> (radians a second, bob).
SPINNING: dict[str, tuple[float, float]] = {
    f"pass_{colour}": (PASS_SPIN, PASS_BOB) for colour in _PASS_RAMPS
}
PASS_MODELS = {f"pass_{colour}": _pass(colour, ramp) for colour, ramp in _PASS_RAMPS.items()}


def pass_card_rgba(colour: str) -> npt.NDArray[np.uint8]:
    """A pass's front as RGBA, (u, v, 4) (pygame's surfarray order), for the
    HUD to show the passes held as the same cards; alpha 0 at its corners."""
    front = PASS_MODELS[f"pass_{colour}"][0].faces[0]
    rgba = np.zeros((64, 64, 4), dtype=np.uint8)
    rgba[..., :3] = PALETTE[_TEXTURES[front]]
    rgba[..., 3] = np.where(_OPAQUE[front], 255, 0)
    return rgba


_LEAF_RIGHT, _LEAF_LEFT, _LEAF_EDGE = door_leaf(True), door_leaf(False), door_edge()
_JAMB, _GLASS, _GLASS_EDGE = jamb(), glass_leaf(), glass_edge()
_JAMB_HALF = 0.03  # jambs a little thicker than the leaf: a frame


def door_boxes(doors: Iterable[Door], passes: Collection[str] = ()) -> list[PlacedBox]:
    """Boxes for doors in their current state; by each locked door, a card
    reader on the wall either side, lit green once the team (`passes`) can
    open it; and by each lift, its call button, floor indicator and car
    panel, on the floor's power (the renderer lights them while it's on)."""
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
                out.append(PlacedBox(*door.centre, 0.0, f, Box(jt, y, h, sides)))
            # The leaf: model x along it from the hinge, closed along the
            # doorway; opening swings its free end towards the facing.
            hinge_x = door.centre[0] + along[0] * (-0.5 + JAMB)
            hinge_y = door.centre[1] + along[1] * (-0.5 + JAMB)
            angle = f + math.pi / 2 - door.open * SWING
            faces = (_LEAF_EDGE, _LEAF_RIGHT, _LEAF_EDGE, _LEAF_LEFT, _LEAF_EDGE)
            out.append(
                PlacedBox(hinge_x, hinge_y, 0.0, angle, Box((0.0, HINGED_LEAF), t, h, faces))
            )
        elif door.kind == DOUBLE:
            jt = (-_JAMB_HALF, _JAMB_HALF)
            sides = (_JAMB, _JAMB, _JAMB, _JAMB, _JAMB)
            for y in ((-0.5, -0.5 + DOUBLE_JAMB), (0.5 - DOUBLE_JAMB, 0.5)):
                out.append(PlacedBox(*door.centre, 0.0, f, Box(jt, y, h, sides)))
            faces = (_LEAF_EDGE, _LEAF_RIGHT, _LEAF_EDGE, _LEAF_LEFT, _LEAF_EDGE)
            leaf = Box((0.0, DOUBLE_LEAF), t, h, faces)
            # Each leaf's model x runs from its hinge towards the middle; both
            # free ends swing towards the facing.
            for side, closed, turn in ((-1, f + math.pi / 2, -1), (1, f - math.pi / 2, 1)):
                offset = side * (0.5 - DOUBLE_JAMB)
                hx, hy = door.centre[0] + along[0] * offset, door.centre[1] + along[1] * offset
                angle = closed + turn * door.open * SWING
                out.append(PlacedBox(hx, hy, 0.0, angle, leaf))
        else:
            slide = SLIDE[door.kind] * door.open  # each leaf slides towards the wall beside it
            if door.kind == LIFT:
                faces = (_STEEL_LEAF, _STEEL_EDGE, _STEEL_LEAF, _STEEL_EDGE, _STEEL_EDGE)
                out += _lift_fittings(door)
            else:
                faces = (_GLASS, _GLASS_EDGE, _GLASS, _GLASS_EDGE, _GLASS_EDGE)
            for y in ((-0.5 - slide, -slide), (slide, 0.5 + slide)):
                out.append(PlacedBox(*door.centre, 0.0, f, Box(t, y, h, faces)))
    return out


def steel_leaf() -> int:
    """A lift door leaf: hairline-brushed stainless (the brushing runs up, so
    it varies across, never down: a highlight band read as drips once
    quantised to 16 shades), a darker meeting edge where the leaves close."""
    rng = np.random.default_rng(101)
    c = Canvas()
    streaks = (noise(rng, 64)[:1, :] - 0.5) * 0.5 + ((c.u[:1, :] * 5) % 3 == 0) * 0.3
    c.paint(c.rect(0, 64, 0, 64), GREY, 10.0 + np.broadcast_to(streaks, (64, 64)))
    c.paint(c.rect(0, 64, 0, 2) | c.rect(0, 64, 62, 64), GREY, 5.0)  # its edges
    c.paint(c.rect(0, 2, 0, 64), GREY, 6.0)
    return _texture(c)


def steel_edge() -> int:
    c = Canvas()
    c.paint(c.rect(0, 64, 0, 64), GREY, 7.0)
    return _texture(c)


def polished_steel() -> int:
    """The doorway's lining and architrave: polished stainless, flat (a
    gradient banded into steps), a lit edge and a shadowed one."""
    c = Canvas()
    c.paint(c.rect(0, 64, 0, 64), GREY, 11.0)
    c.paint(c.rect(0, 64, 0, 4), GREY, 13.0)
    c.paint(c.rect(0, 64, 60, 64), GREY, 8.0)
    return _texture(c)


def sill() -> int:
    """The doors' threshold: aluminium, grooved across for the door guides."""
    c = Canvas()
    c.paint(c.rect(0, 64, 0, 64), GREY, 11.0)
    c.paint(c.rect(0, 64, 0, 64) & (c.v % 8 < 2), GREY, 5.0)
    return _texture(c)


def call_plate() -> int:
    """A lift's call plate: plain steel, a darker border."""
    c = Canvas()
    c.paint(c.rect(0, 64, 0, 64), GREY, 11.0)
    c.paint(c.rect(0, 64, 0, 64) & ~c.rect(4, 60, 6, 58), GREY, 8.0)
    return _texture(c)


def call_button(lit: bool) -> int:
    """The call button: a plain round button, amber once the lift's coming."""
    c = Canvas()
    c.paint(c.rect(0, 64, 0, 64), GREY, 11.0)
    c.paint(c.ellipse(32, 32, 28, 28), OCHRE if lit else GREY, 15.0 if lit else 7.0)
    return _texture(c)


def floor_indicator(lit: bool) -> int:
    """Above a lift: a black strip, the floor (G) in amber when powered."""
    c = Canvas()
    c.paint(c.rect(0, 64, 0, 64), GREY, 1.0)
    c.paint(c.rect(0, 64, 0, 64) & ~c.rect(3, 61, 2, 62), GREY, 10.0)  # thin steel frame
    if lit:
        c.text("G", 22, 26, OCHRE, 15.0, scale=4)
    return _texture(c)


def car_panel(power: bool) -> int:
    """Inside the car by the door: steel, a column of round floor buttons,
    the lowest (G: you're here) lit amber with power."""
    c = Canvas()
    c.paint(c.rect(0, 64, 0, 64), GREY, 11.0)
    c.paint(c.rect(0, 64, 0, 64) & ~c.rect(3, 61, 4, 60), GREY, 8.0)
    for k in range(4):
        here = power and k == 3
        c.paint(c.ellipse(12 + k * 13, 32, 5, 9), OCHRE if here else GREY, 15.0 if here else 7.0)
    return _texture(c)


_STEEL_LEAF, _STEEL_EDGE = steel_leaf(), steel_edge()
_POLISHED, _SILL, _CALL_PLATE = polished_steel(), sill(), call_plate()
_CALL = {lit: call_button(lit) for lit in (False, True)}
_INDICATOR = {lit: floor_indicator(lit) for lit in (False, True)}
_CAR_PANEL = {lit: car_panel(lit) for lit in (False, True)}


def _lift_fittings(door: Door) -> list[PlacedBox]:
    """Everything round a lift door but its leaves, in the door's model
    space (x across the doorway, + towards the lobby; y along it). The lift
    is one cell, its car the doorway cell itself, its doors at the lobby
    end: a steel architrave on the lobby wall round them, the sill under
    them, the floor indicator over them and the call plate beside them; and
    inside the car, on its side wall by the door, its button panel. The
    indicator, the call button and the car's panel are on the main circuit:
    lit while the floor has power, dark (their off faces) without."""
    f = door.facing
    along = (-math.sin(f), math.cos(f))
    steel = (_POLISHED,) * 5
    out = []

    def at(box: Box, x: float = door.x, y: float = door.y) -> None:
        out.append(PlacedBox(x, y, 0.0, f, box))

    top = DOOR_HEIGHT
    for side in (-1, 1):
        lo, hi = sorted((side * 0.5, side * 0.56))
        at(Box((0.5, 0.515), (lo, hi), (0.0, top + 0.04), steel))  # architrave
    at(Box((0.5, 0.515), (-0.56, 0.56), (top, top + 0.04), steel))  # its head
    sill = (door.set_back - 0.05, door.set_back + 0.05)
    at(Box(sill, (-0.5, 0.5), (0.0, 0.006), (_SILL,) * 5))  # under the doors
    # Only the fronts carry the display and buttons; their thin sides are steel.
    sides = (_POLISHED,) * 4
    indicator = Box(
        (0.5, 0.52),
        (-0.12, 0.12),
        (top + 0.07, top + 0.15),
        (_INDICATOR[True], *sides),
        glow=True,
        mains=True,
        off_faces=(_INDICATOR[False], *sides),
    )
    at(indicator)
    # The call plate, on the wall beside the doors, and its one button.
    bx, by = door.x + along[0] * 0.66, door.y + along[1] * 0.66
    at(Box((0.5, 0.51), (-0.035, 0.035), (0.36, 0.44), (_CALL_PLATE, *(_POLISHED,) * 4)), bx, by)
    button = Box(
        (0.5, 0.516),
        (-0.018, 0.018),
        (0.39, 0.41),
        (_CALL[door.called], *sides),
        glow=door.called,  # once called (and only with power)
        mains=True,
        off_faces=(_CALL[False], *sides),
    )
    at(button, bx, by)
    # The car's panel, on its side wall (y = 0.5) just inside the doors.
    # Its buttons on the face into the car: -y, the box's left face.
    lit_panel = (_POLISHED, _POLISHED, _POLISHED, _CAR_PANEL[True], _POLISHED)
    dark_panel = (_POLISHED, _POLISHED, _POLISHED, _CAR_PANEL[False], _POLISHED)
    at(Box((0.05, 0.22), (0.485, 0.5), (0.3, 0.52), lit_panel, mains=True, off_faces=dark_panel))
    return out


# --- Reception desk (1980s, two levels) -------------------------------------------
# A transaction ledge at the front (~1.15 m) for visitors to lean on and sign
# in at, over a wood-slatted front panel on a recessed plinth with the
# word WELCOME in brass letters; behind it, a work surface at desk height (~0.73 m)
# over a steel drawer pedestal (its phone a `phone` thing on it); returns wrapping
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
        # The visitors' book (the phone is a thing of its own: `phone`).
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


# --- The boss's desk ---------------------------------------------------------------
# Absurd on purpose: two cells long (~5.4 m), a thick top overhanging two
# carved pedestals, green leather inlaid in it, a brass nameplate, and a
# high-backed oxblood leather chair. Model x points to the visitors' side;
# the boss sits at -x. Box faces stretch their texture once over the face,
# so each texture is drawn for its face's proportions. Collision and the
# heights things rest at: EXECUTIVE_DESK in doomer/shared/things.py.

_X_TOP = (0.29, 0.33)  # walls: the top slab (~0.8-0.9 m)
_X_HALF_LEN, _X_HALF_DEPTH = 1.0, 0.46  # cells


def _mahogany(c: Canvas, where: BoolArray, shade: float = 6.0, along_u: bool = True) -> None:
    """Dark red-brown mahogany, its grain running along u (or v)."""
    rng = np.random.default_rng(171)
    run = c.v if along_u else c.u
    grain = np.sin(run * 0.9 + noise(rng, 8) * 5) * 0.35
    c.paint(where, BRICK, shade + grain)


def exec_top() -> int:
    """The top, from above: u along the desk's length (2 cells), v across it
    (0.92): texels ~2:1. A mahogany border, a gold-tooled line, green leather."""
    c = Canvas()
    _mahogany(c, c.rect(0, 64, 0, 64), 6.5)
    leather = c.rect(8, 56, 4, 60)
    rng = np.random.default_rng(172)
    c.paint(leather, GREEN, 4.0 + (noise(rng, 16) - 0.5) * 0.5)
    tooling = leather & ~c.rect(10, 54, 5, 59)
    c.paint(tooling, OCHRE, 11.0)
    return _texture(c)


def exec_edge() -> int:
    """The top's thick edge: mahogany with a brass inlay strip."""
    c = Canvas()
    _mahogany(c, c.rect(0, 64, 0, 64), 7.0)
    c.paint(c.rect(0, 8, 0, 64), BRICK, 8.5)  # catches the light
    c.paint(c.rect(28, 36, 0, 64), OCHRE, 12.0)
    c.paint(c.rect(56, 64, 0, 64), BRICK, 4.0)
    return _texture(c)


def _raised_panels(c: Canvas, columns: int, rows: int = 1, margin: int = 6) -> None:
    """Raised-and-fielded panels in a mahogany frame: lit top and left
    bevels, shadowed bottom and right."""
    _mahogany(c, c.rect(0, 64, 0, 64), 6.0, along_u=False)
    w, h = (64 - margin * (columns + 1)) // columns, (64 - margin * (rows + 1)) // rows
    for i in range(columns):
        for j in range(rows):
            u0, v0 = margin + i * (w + margin), margin + j * (h + margin)
            c.paint(c.rect(v0, v0 + h, u0, u0 + w), BRICK, 5.0)
            c.paint(c.rect(v0 + 3, v0 + h - 3, u0 + 3, u0 + w - 3), BRICK, 7.0)  # the field
            c.paint(c.rect(v0, v0 + 1, u0, u0 + w) | c.rect(v0, v0 + h, u0, u0 + 1), BRICK, 9.0)
            c.paint(
                c.rect(v0 + h - 1, v0 + h, u0, u0 + w) | c.rect(v0, v0 + h, u0 + w - 1, u0 + w),
                BRICK,
                3.0,
            )


def exec_front() -> int:
    """The visitors' side, between the pedestals: two grand raised panels."""
    c = Canvas()
    _raised_panels(c, 2)
    return _texture(c)


def exec_pedestal_panel() -> int:
    c = Canvas()
    _raised_panels(c, 1)
    return _texture(c)


def exec_drawers() -> int:
    """The boss's side of a pedestal: three drawers, brass pulls."""
    c = Canvas()
    _mahogany(c, c.rect(0, 64, 0, 64), 6.5, along_u=True)
    for k in range(3):
        v0 = 3 + k * 20
        c.paint(c.rect(v0 + 18, v0 + 20, 0, 64), BRICK, 3.0)  # the gap under it
        c.paint(c.rect(v0 + 7, v0 + 11, 24, 40), OCHRE, 12.0)  # the pull
        c.paint(c.rect(v0 + 10, v0 + 11, 24, 40), OCHRE, 8.0)
    return _texture(c)


def exec_dark() -> int:
    c = Canvas()
    c.paint(c.rect(0, 64, 0, 64), BRICK, 2.5)
    return _texture(c)


def _brass_canvas() -> Canvas:
    c = Canvas()
    c.paint(c.rect(0, 64, 0, 64), OCHRE, 12.0)
    return c


def nameplate() -> int:
    """Polished brass, BOSS engraved in black (a wedge on the desk's front edge)."""
    c = Canvas()
    c.paint(c.rect(0, 64, 0, 64), OCHRE, 13.0)
    c.paint(c.rect(0, 64, 0, 64) & ~c.rect(4, 60, 3, 61), OCHRE, 10.0)
    for i, ch in enumerate("BOSS"):
        for gy, row in enumerate(FONT_3X5[ch]):
            for gx, bit in enumerate(row):
                if bit == "#":
                    u0, v0 = 9 + i * 12 + gx * 3, 14 + gy * 7
                    c.paint(c.rect(v0, v0 + 7, u0, u0 + 3), GREY, 1.0)
    return _texture(c)


def leather(tufted: bool) -> int:
    """Oxblood leather; buttoned (tufted) on the chair's back."""
    rng = np.random.default_rng(173)
    c = Canvas()
    c.paint(c.rect(0, 64, 0, 64), BRICK, 4.5 + (noise(rng, 16) - 0.5) * 0.5)
    if tufted:
        buttons = ((c.v + (c.u // 12 % 2) * 6) % 12 == 6) & (c.u % 12 == 6)
        c.paint(buttons, BRICK, 2.0)
        c.paint(c.rect(0, 4, 0, 64), BRICK, 6.0)  # its rolled top
    return _texture(c)


def _executive_desk() -> tuple[Box, ...]:
    top, edge = exec_top(), exec_edge()
    front, panel, drawers, dark = exec_front(), exec_pedestal_panel(), exec_drawers(), exec_dark()
    plate = nameplate()
    brass = _texture(_brass_canvas())
    seat, back_rest = leather(False), leather(True)
    steel = steel_edge()
    lx, ly = _X_HALF_DEPTH, _X_HALF_LEN
    boxes = [
        Box((-lx, lx), (-ly, ly), _X_TOP, (edge, edge, edge, edge, top)),
        # The front between the pedestals, set back under the overhang.
        Box((0.30, 0.36), (-0.55, 0.55), (0.06, _X_TOP[0]), (front, dark, dark, dark, dark)),
        # The nameplate, on the front edge, facing visitors.
        Box((0.36, 0.42), (-0.16, 0.16), (_X_TOP[1], _X_TOP[1] + 0.035), (plate, *(brass,) * 4)),
        # The chair: a star base, a column, the seat, a high buttoned back.
        Box((-0.93, -0.65), (-0.16, 0.16), (0.0, 0.02), (steel,) * 5),
        Box((-0.83, -0.75), (-0.04, 0.04), (0.02, 0.14), (steel,) * 5),
        Box((-0.96, -0.62), (-0.21, 0.21), (0.14, 0.2), (seat,) * 5),
        Box((-1.0, -0.93), (-0.22, 0.22), (0.2, 0.55), (back_rest, seat, seat, seat, seat)),
    ]
    for side in (-1, 1):
        lo, hi = sorted((side * 0.55, side * 0.95))
        # A pedestal: carved panels outside, drawers on the boss's side; a
        # dark plinth under it.
        boxes.append(
            Box((-0.40, 0.38), (lo, hi), (0.025, _X_TOP[0]), (panel, panel, drawers, panel, dark))
        )
        boxes.append(Box((-0.38, 0.36), (lo + 0.02, hi - 0.02), (0.0, 0.025), (dark,) * 5))
    return tuple(boxes)


# --- Exit signs: the emergency lights ---------------------------------------------
# A green box over a main door, "EXIT" in white, lit from inside: on its own
# battery, so lit with or without power. It's also what lights the floor in
# a power cut (doomer/client/lighting.py: a dim emergency light at the sign).
# Model x points out of the wall; the thing stands at the wall's face.

EXIT_SIGN_HALF_HEIGHT = 0.045  # walls: centred on EXIT_SIGN_HEIGHTS (shared/things.py)
EXIT_SIGN_HALF_WIDTH = 0.09  # cells (~49 cm wide, ~24 cm high)


def exit_sign_face() -> int:
    """White EXIT on green, a white border. Stretched over a box face about
    twice as wide as it is high, so texels are ~2:1 and the letters read
    wide (as real exit signs' do)."""
    c = Canvas()
    c.paint(c.rect(0, 64, 0, 64), GREEN, 11.0)
    edge = c.rect(0, 64, 0, 64) & ~c.rect(4, 60, 2, 62)
    c.paint(edge, GREY, 15.0)
    # The tiny font, each of its pixels 3 texels across and 8 down: with the
    # face's stretch that's bold letters ~15 cm tall, like a real sign's.
    for i, ch in enumerate("EXIT"):
        for gy, row in enumerate(FONT_3X5[ch]):
            for gx, bit in enumerate(row):
                if bit == "#":
                    u0, v0 = 8 + i * 13 + gx * 3, 12 + gy * 8
                    c.paint(c.rect(v0, v0 + 8, u0, u0 + 3), GREY, 15.0)
    return _texture(c)


def exit_sign_side() -> int:
    c = Canvas()
    c.paint(c.rect(0, 64, 0, 64), GREY, 12.0)
    return _texture(c)


def _exit_sign(middle: float) -> tuple[Box, ...]:
    face, side = _EXIT_FACE, _EXIT_SIDE
    w, h = EXIT_SIGN_HALF_WIDTH, EXIT_SIGN_HALF_HEIGHT
    z = (middle - h, middle + h)
    return (Box((0.0, 0.03), (-w, w), z, (face, side, side, side, side), glow=True),)


_EXIT_FACE, _EXIT_SIDE = exit_sign_face(), exit_sign_side()


# --- The plant room's breaker (doomer/shared/switches.py has its state) ---------
# An industrial isolator in a hazardous area: a heavy steel box mounted on
# a solid wall at chest height, a yellow and black hazard border, bolts at
# its corners, a DANGER 415V plate; a big red lever in a slot on a yellow
# backplate (down at OFF, thrown up to ON), a padlock hasp; two lamps, red
# blinking until it's thrown, then steady green (its own battery).
# Galvanised conduit runs up through a junction box into the ceiling and
# down into the floor; a hazard-striped keep-clear mat lies in front.
# Drawn from the switch's state each frame, like doors. Model x points out
# of the wall; the thing stands at the wall's face. Sizes match its
# footprint in doomer/shared/things.py.

BREAKER_DEPTH, BREAKER_HALF_WIDTH = 0.16, 0.22
BREAKER_Z = (0.26, 0.8)  # walls: its foot and top
_PIVOT, _LEVER = 0.46, 0.1  # walls: the lever's pivot height and length
# The front, in texels: a left column (lamps, red over green; the padlock
# hasp), the yellow backplate and its slot in the middle, the ON and OFF
# labels in a right column, each level with its end of the slot.
_LAMP_U = 12  # the lamps' column
_LAMP_V = {"red": (26, 31), "green": (34, 39)}
_PLATE_U = (22, 42)
_LABEL_U = 46
_CONDUIT_HALF = 0.03
_JUNCTION_Z = (0.88, 0.95)
BREAKER_BLINK = 1.0  # seconds per blink of its red lamp


def _front_v(z: float, z0: float, z1: float) -> int:
    """The texel row (v) of height z on a front face from z0 up to z1."""
    return round((z1 - z) / (z1 - z0) * 64)


def _front_u(y: float, half: float) -> int:
    """The texel column (u) of y across a front face -half..half (mirrored
    on the box: keep layouts symmetric, or place by u)."""
    return round((y + half) / (2 * half) * 64)


def _z_of_v(v: float, z0: float, z1: float) -> float:
    return z1 - v / 64 * (z1 - z0)


def _y_of_u(u: float, half: float) -> float:
    """Model y of texel column u on a box's front face: u runs from +y to
    -y (left to right as you face it)."""
    return half - (u + 0.5) / 64 * (2 * half)


def _hazard(c: Canvas, mask: BoolArray, step: int = 8) -> None:
    """Yellow and black diagonal stripes."""
    c.paint(mask, OCHRE, 12.5)
    c.paint(mask & ((c.u + c.v) % step < step // 2), GREY, 1.0)


def switchgear_front() -> int:
    z0, z1 = BREAKER_Z
    c = Canvas()
    c.paint(c.rect(0, 64, 0, 64), GREY, 6.0)
    _hazard(c, ~c.rect(4, 60, 4, 60))  # the hazard border
    c.paint(c.rect(4, 60, 4, 5) | c.rect(4, 5, 4, 60), GREY, 8.0)  # its lit inner edge
    for v, u in ((6, 6), (6, 56), (56, 6), (56, 56)):
        c.paint(c.rect(v, v + 2, u, u + 2), GREY, 11.0)  # bolts
    c.paint(c.rect(7, 18, 9, 55), CREAM, 15.0)  # the danger plate
    c.paint(c.rect(7, 12, 9, 55), BRICK, 10.0)
    _centred(c, "DANGER", 7, CREAM, 15.0)
    _centred(c, "415V", 13, BRICK, 10.0)
    for v0, v1 in _LAMP_V.values():  # the lamps' dark sockets (lit lamps sit proud of them)
        c.paint(c.rect(v0, v1, _LAMP_U - 3, _LAMP_U + 3), GREY, 0.5)
    top = _front_v(_PIVOT + _LEVER + 0.03, z0, z1)
    bottom = _front_v(_PIVOT - _LEVER - 0.03, z0, z1)
    c.paint(c.rect(top - 2, bottom + 2, *_PLATE_U), OCHRE, 12.5)  # the yellow backplate
    c.paint(c.rect(top, bottom, 30, 34), GREY, 0.0)  # the slot the lever runs in
    c.text("ON", top, _LABEL_U, CREAM, 14.0)  # level with the slot's ends, on the grey
    c.text("OFF", bottom - 5, _LABEL_U, CREAM, 14.0)
    c.paint(c.rect(bottom - 6, bottom - 1, _LAMP_U - 2, _LAMP_U + 3), GREY, 11.0)  # padlock hasp
    return _texture(c)


def switchgear_side() -> int:
    c = Canvas()
    c.paint(c.rect(0, 64, 0, 64), GREY, 5.5)
    _hazard(c, ~c.rect(6, 58, 0, 64))
    return _texture(c)


def conduit() -> int:
    """Galvanised conduit: a lit side, couplings every half storey."""
    c = Canvas()
    c.paint(c.rect(0, 64, 0, 64), GREY, 10.5)
    c.paint(c.rect(0, 64, 0, 16), GREY, 12.5)
    c.paint(c.rect(0, 64, 0, 64) & (c.v % 32 < 4), GREY, 7.0)
    return _texture(c)


def junction_box() -> int:
    c = Canvas()
    c.paint(c.rect(0, 64, 0, 64), GREY, 9.0)
    c.paint(c.rect(6, 58, 6, 58), GREY, 10.5)
    for v, u in ((8, 8), (8, 54), (54, 8), (54, 54)):
        c.paint(c.rect(v, v + 3, u, u + 3), GREY, 5.0)
    return _texture(c)


def keep_clear_mat() -> int:
    c = Canvas()
    _hazard(c, c.rect(0, 64, 0, 64), step=16)
    c.paint(~c.rect(3, 61, 3, 61), GREY, 1.0)
    return _texture(c)


def lever_arm() -> int:
    c = Canvas()
    c.paint(c.rect(0, 64, 0, 64), GREY, 12.0)
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


def _plain(ramp: int, shade: float) -> int:
    c = Canvas()
    c.paint(c.rect(0, 64, 0, 64), ramp, shade)
    return _texture(c)


_GEAR_FRONT, _GEAR_SIDE, _GEAR_TOP = switchgear_front(), switchgear_side(), _plain(GREY, 7.0)
_CONDUIT, _JUNCTION, _MAT = conduit(), junction_box(), keep_clear_mat()
_ARM, _HANDLE = lever_arm(), lever_handle()
_LAMP_OFF, _LAMP_ON = lamp(BRICK), lamp(GREEN)


def _breaker_fixed() -> list[Box]:
    """The parts that don't move: the box, its conduit, the mat."""
    z0, z1 = BREAKER_Z
    h, c = BREAKER_HALF_WIDTH, _CONDUIT_HALF
    pipe = (_CONDUIT,) * 5
    return [
        Box((0.0, BREAKER_DEPTH), (-h, h), (z0, z1), (_GEAR_FRONT, *(_GEAR_SIDE,) * 3, _GEAR_TOP)),
        Box((0.0, 0.07), (-c, c), (z1, _JUNCTION_Z[0]), pipe),  # up to the junction box
        Box((0.0, 0.1), (-0.07, 0.07), _JUNCTION_Z, (_JUNCTION,) * 5),
        Box((0.0, 0.07), (-c, c), (_JUNCTION_Z[1], 1.0), pipe),  # on into the ceiling
        Box((0.0, 0.07), (-c, c), (0.0, z0), pipe),  # down into the floor
        Box((0.02, 0.62), (-0.32, 0.32), (0.0, 0.004), (_MAT,) * 5),  # keep clear
    ]


_BREAKER_FIXED = _breaker_fixed()


def switch_boxes(switches: Iterable[Switch]) -> list[PlacedBox]:
    """Boxes for switches (the breaker) in their current state."""
    out = []
    d = BREAKER_DEPTH
    for sw in switches:
        # The lever in its slot: hangs down at OFF, thrown up to ON.
        z = (_PIVOT, _PIVOT + _LEVER) if sw.on else (_PIVOT - _LEVER, _PIVOT)
        end = z[1] if sw.on else z[0]
        arm = Box((d, d + 0.04), (-0.014, 0.014), z, (_ARM,) * 5)
        grip = Box((d + 0.015, d + 0.08), (-0.07, 0.07), (end - 0.025, end + 0.025), (_HANDLE,) * 5)
        # One lamp lit: red, blinking, until it's thrown; then green, steady.
        v0, v1 = _LAMP_V["green" if sw.on else "red"]
        y = _y_of_u(_LAMP_U, BREAKER_HALF_WIDTH)
        light = Box(
            (d, d + 0.01),
            (y - 0.022, y + 0.022),
            (_z_of_v(v1, *BREAKER_Z), _z_of_v(v0, *BREAKER_Z)),
            ((_LAMP_ON if sw.on else _LAMP_OFF),) * 5,
            glow=True,
            indicator=True,
            blink=0.0 if sw.on else BREAKER_BLINK,
        )
        for box in (*_BREAKER_FIXED, arm, grip, light):
            out.append(PlacedBox(sw.x, sw.y, 0.0, sw.facing, box))
    return out


# --- Server cabinets (the plant room) ------------------------------------------------
# Late-80s minicomputer cabinets: tall, cream steel, a blue band at the top;
# a tape drive (two reels behind a smoked glass door) or a disk cabinet
# (three drive drawers). A row of lamps on a panel blinks away, each on its
# own timer, while there's power; dark without (indicators on the mains).
# Five boxes each: the cabinet and four lamps. Model x out of the wall, the
# thing at the wall's face; sizes match its footprint (shared/things.py).

SERVER_DEPTH, SERVER_HALF_WIDTH, SERVER_TOP = 0.36, 0.19, 0.74
_SERVER_LAMPS = (  # (y across, blink seconds, phase, colour ramp): symmetric in y
    (-0.11, 0.7, 0.0, GREEN),
    (-0.04, 1.3, 0.4, ORANGE),
    (0.04, 0.45, 0.2, GREEN),
    (0.11, 2.1, 0.7, BRICK),
)
_SERVER_LAMP_Z = {"tape": (0.36, 0.38), "disk": (0.62, 0.64)}


def server_front(kind: str) -> int:
    c = Canvas()
    c.paint(c.rect(0, 64, 0, 64), CREAM, 12.0)
    c.paint(c.rect(0, 64, 0, 2) | c.rect(0, 64, 62, 64), CREAM, 9.0)  # its edges
    c.paint(c.rect(0, 6, 0, 64), BLUE, 7.0)  # the band at the top
    lamp_z = _SERVER_LAMP_Z[kind]
    v0, v1 = _front_v(lamp_z[1], 0.0, SERVER_TOP), _front_v(lamp_z[0], 0.0, SERVER_TOP)
    c.paint(c.rect(v0 - 2, v1 + 2, 6, 58), GREY, 3.0)  # the lamp panel
    for y, _, _, _ in _SERVER_LAMPS:
        u = _front_u(y, SERVER_HALF_WIDTH)
        c.paint(c.rect(v0, v1, u - 2, u + 2), GREY, 1.0)  # dark sockets
    if kind == "tape":
        glass = c.rect(8, 26, 6, 58)
        c.paint(glass, GREY, 2.5)  # smoked glass
        for cu in (20, 44):  # two reels (texels are tall here: ellipses wide)
            c.paint(c.ellipse(17, cu, 7, 11), WOOD, 3.0)  # tape on them
            c.paint(c.ellipse(17, cu, 3.5, 5), GREY, 9.0)  # the hubs
            c.paint(c.ellipse(17, cu, 1.2, 1.8), GREY, 2.0)
        c.paint(c.rect(8, 9, 6, 58), GREY, 6.0)  # a glint along the top
        for v in range(40, 60, 3):  # vents below
            c.paint(c.rect(v, v + 1, 10, 54), CREAM, 8.0)
    else:
        for top in (26, 38, 50):  # three drive drawers, each with a handle
            c.paint(c.rect(top, top + 10, 6, 58), CREAM, 10.5)
            c.paint(c.rect(top, top + 1, 6, 58), CREAM, 14.0)
            c.paint(c.rect(top + 4, top + 6, 26, 38), GREY, 4.0)
    c.paint(c.rect(61, 64, 2, 62), GREY, 2.0)  # the plinth
    return _texture(c)


def server_side() -> int:
    c = Canvas()
    c.paint(c.rect(0, 64, 0, 64), CREAM, 10.5)
    c.paint(c.rect(0, 6, 0, 64), BLUE, 6.0)
    for v in range(14, 50, 4):  # vents
        c.paint(c.rect(v, v + 1, 12, 52), CREAM, 7.0)
    return _texture(c)


def server_top() -> int:
    c = Canvas()
    c.paint(c.rect(0, 64, 0, 64), CREAM, 11.0)
    return _texture(c)


def _server(kind: str) -> tuple[Box, ...]:
    side, top = server_side(), server_top()
    cabinet = Box(
        (0.0, SERVER_DEPTH),
        (-SERVER_HALF_WIDTH, SERVER_HALF_WIDTH),
        (0.0, SERVER_TOP),
        (server_front(kind), side, side, side, top),
    )
    lamps = tuple(
        Box(
            (SERVER_DEPTH, SERVER_DEPTH + 0.005),
            (y - 0.012, y + 0.012),
            _SERVER_LAMP_Z[kind],
            (_SERVER_LAMP_TEXTURES[ramp],) * 5,
            glow=True,
            mains=True,
            indicator=True,
            blink=period,
            blink_phase=phase,
        )
        for y, period, phase, ramp in _SERVER_LAMPS
    )
    return (cabinet, *lamps)


_SERVER_LAMP_TEXTURES = {ramp: lamp(ramp) for ramp in (GREEN, ORANGE, BRICK)}
SERVER_MODELS = {f"server:{kind}": _server(kind) for kind in _SERVER_LAMP_Z}


# --- Vending machine (a health dispenser: doomer/shared/dispensers.py) -------------
# A snack machine against a wall, as they were: a dark charcoal cabinet with
# chrome trim; a big window onto a dark interior, a light strip along its
# top, rows of silver spiral coils holding bright packets (a few slots sold
# out); a narrow panel to the right (a red LED display, a keypad, the coin
# slot); a pale PUSH flap under the window; a louvred kick plate. The window
# is a box of its own, lit from inside while the floor has power (it's on
# the main circuit). Model x out of the wall, the thing at the wall's face;
# sizes match its footprint (shared/things.py).

VEND_DEPTH, VEND_HALF_WIDTH, VEND_TOP = 0.3, 0.18, 0.7
_VEND_WINDOW = (4, 45, 3, 44)  # v0, v1, u0, u1 on the front


def vending_front() -> int:
    c = Canvas()
    c.paint(c.rect(0, 64, 0, 64), GREY, 3.5)  # the charcoal cabinet
    c.paint(c.rect(0, 64, 0, 1) | c.rect(0, 1, 0, 64) | c.rect(0, 64, 63, 64), GREY, 10.0)  # trim
    v0, v1, u0, u1 = _VEND_WINDOW
    c.paint(c.rect(v0 - 1, v1 + 1, u0 - 1, u1 + 1), GREY, 9.0)  # the window's chrome frame
    # The panel to the right: display, keypad, coin slot.
    c.paint(c.rect(v0 + 2, v0 + 5, 48, 60), GREY, 0.5)
    c.paint(c.rect(v0 + 3, v0 + 4, 49, 57), BRICK, 13.0)  # red LED digits
    for row in range(5):
        for col in range(3):
            vk, uk = v0 + 9 + row * 4, 49 + col * 4
            c.paint(c.rect(vk, vk + 2, uk, uk + 3), GREY, 11.0)
    c.paint(c.rect(v0 + 31, v0 + 36, 51, 57), GREY, 8.0)  # the coin slot plate
    c.paint(c.rect(v0 + 32, v0 + 35, 53, 55), GREY, 0.0)
    c.paint(c.rect(47, 53, u0, u1), GREY, 12.0)  # the PUSH flap
    c.paint(c.rect(47, 48, u0, u1), GREY, 14.0)
    _centred(c, "PUSH", 48, GREY, 4.0)
    for v in range(55, 63, 2):  # the kick plate's louvres
        c.paint(c.rect(v, v + 1, 3, 61), GREY, 1.0)
        c.paint(c.rect(v + 1, v + 2, 3, 61), GREY, 6.0)
    return _texture(c)


def vending_window() -> int:
    """Behind the glass: a dark interior, a light strip along the top, six
    rows of spiral coils, most holding a bright packet."""
    c = Canvas()
    c.paint(c.rect(0, 64, 0, 64), GREY, 1.0)  # dark inside
    c.paint(c.rect(0, 2, 0, 64), CREAM, 15.0)  # the light strip
    colours = (BRICK, OCHRE, BLUE, GREEN, ORANGE, PURPLE)
    sold_out = {(1, 3), (3, 0), (4, 2)}
    for row in range(6):
        top = 4 + row * 10
        for slot in range(4):
            u0 = 2 + slot * 15
            coil = c.rect(top + 6, top + 8, u0, u0 + 12) & (c.u % 3 != 0)
            c.paint(coil, GREY, 11.0)  # the spiral, its loops catching the light
            if (row, slot) in sold_out:
                continue
            ramp = colours[(row * 4 + slot * 3) % len(colours)]
            c.paint(c.rect(top, top + 6, u0 + 1, u0 + 11), ramp, 11.0)  # a packet
            c.paint(c.rect(top, top + 1, u0 + 1, u0 + 11), ramp, 14.0)  # its crimped top
            c.paint(c.rect(top + 2, top + 4, u0 + 3, u0 + 9), CREAM, 14.0)  # its label
        c.paint(c.rect(top + 8, top + 9, 0, 64), GREY, 5.0)  # the shelf
    c.paint(c.rect(0, 64, 0, 64) & ((c.u - c.v) % 47 == 0), GREY, 8.0)  # a glint on the glass
    return _texture(c)


def _vending_machine() -> tuple[Box, ...]:
    side = _plain_texture(GREY, 3.0)
    cabinet = Box(
        (0.0, VEND_DEPTH),
        (-VEND_HALF_WIDTH, VEND_HALF_WIDTH),
        (0.0, VEND_TOP),
        (vending_front(), side, side, side, _plain_texture(GREY, 4.0)),
    )
    # The window: lit from inside while there's power; unlit (the room's
    # light on the same shelves) without.
    v0, v1, u0, u1 = _VEND_WINDOW
    edge = (_plain_texture(GREY, 2.0),) * 4
    window = Box(
        (VEND_DEPTH, VEND_DEPTH + 0.003),
        (_y_of_u(u1 - 0.5, VEND_HALF_WIDTH), _y_of_u(u0 - 0.5, VEND_HALF_WIDTH)),
        (_z_of_v(v1, 0.0, VEND_TOP), _z_of_v(v0, 0.0, VEND_TOP)),
        (vending_window(), *edge),
        glow=True,
        mains=True,
    )
    return (cabinet, window)


# --- Washroom fittings ---------------------------------------------------------------
# Sizes from doomer/shared/things.py. Each stands against a wall, model +x
# out of it. White porcelain, chrome, beige laminate (with the avocado
# tiles, the 1980s washroom).

_PORCELAIN = (GREY, 14.0)
_CHROME = (STEEL, 12.0)
# Partitions and doors raised off the floor: a gap under them to see feet.
_PARTITION_Z = (0.11, CUBICLE_TOP)
_DOOR_Z = (0.11, 0.7)
_THICK = 0.012  # a partition's half-thickness


def _laminate(shade: float = 10.0) -> Canvas:
    c = Canvas()
    c.paint(c.rect(0, 64, 0, 64), SAND, shade)
    c.paint(c.rect(0, 2, 0, 64), STEEL, 12.0)  # an aluminium capping along the top
    return c


def cubicle_door(state: str) -> int:
    """A cubicle door from outside: laminate, a chrome handle, and the lock's
    indicator above it: VACANT (green) or ENGAGED (red). `state` "inside":
    the door's plain inner face."""
    c = _laminate(10.5)
    c.paint(c.rect(0, 64, 0, 3) | c.rect(0, 64, 61, 64), SAND, 8.0)  # its edges
    if state != "inside":
        ramp = BRICK if state == "engaged" else GREEN
        c.paint(c.rect(22, 30, 40, 58), GREY, 10.0)  # the indicator's plate, by the lock
        c.paint(c.rect(23, 29, 41, 57), ramp, 11.0)
        c.paint(c.rect(33, 37, 44, 58), STEEL, 13.0)  # the handle
        c.paint(c.rect(37, 38, 44, 58), STEEL, 7.0)
    return _texture(c)


_DOOR_FACES = {state: cubicle_door(state) for state in ("vacant", "engaged", "inside")}
_DOOR_EDGE = _plain_texture(*_CHROME)


def _porcelain(top_ring: bool = False) -> int:
    c = Canvas()
    c.paint(c.rect(0, 64, 0, 64), *_PORCELAIN)
    c.paint(c.rect(56, 64, 0, 64), GREY, 11.0)  # shaded towards the floor
    if top_ring:  # seen from above: the seat round the bowl, lid up
        c.paint(c.ellipse(36, 32, 24, 26), GREY, 12.0)
        c.paint(c.ellipse(36, 32, 17, 19), GREY, 9.0)
        c.paint(c.ellipse(40, 32, 9, 11), STEEL, 6.0)  # water
    return _texture(c)


_PANEL = _texture(_laminate())
_PAN_TOP, _PAN_SIDE = _porcelain(top_ring=True), _porcelain()
_FEET = tuple(
    _plain_texture(*look) for look in ((BRICK, 7.0), (WOOD, 5.0), (GREY, 1.5), (GREY, 6.0))
)


def _feet(kind: str, cy: float) -> list[Box]:
    """Whoever's in the ENGAGED cubicle, as seen under the door. Gents:
    black shoes, grey trousers round the ankles. Ladies: red court shoes
    (a heel each) and a handbag on the floor beside them."""
    out = []
    red, bag, shoe, cloth = _FEET
    if kind == "ladies":
        for dy in (-0.032, 0.032):
            y = (cy + dy - 0.017, cy + dy + 0.017)
            out.append(Box((0.36, 0.47), y, (0.0, 0.02), (red,) * 5))  # the shoe
            out.append(Box((0.35, 0.37), y, (0.0, 0.05), (red,) * 5))  # its heel
        out.append(Box((0.4, 0.48), (cy + 0.07, cy + 0.11), (0.0, 0.07), (bag,) * 5))
    else:
        for dy in (-0.032, 0.032):
            y = (cy + dy - 0.02, cy + dy + 0.02)
            out.append(Box((0.35, 0.48), y, (0.0, 0.035), (shoe,) * 5))
        out.append(Box((0.28, 0.36), (cy - 0.075, cy + 0.075), (0.0, 0.035), (cloth,) * 5))
    return out


def _cubicles(n: int, kind: str) -> tuple[Box, ...]:
    """A run of n cubicles without their doors (cubicle_door_boxes draws
    those, open or shut): partitions between and at the ends, a chrome
    headrail across the front, a toilet in each (a pan, the cistern on the
    wall behind), and feet under the second (ENGAGED: shared/cubicles.py)."""
    w, d = CUBICLE_WIDTH, CUBICLE_DEPTH
    half = n * w / 2
    panel, edge, pan, white = _PANEL, _DOOR_EDGE, _PAN_TOP, _PAN_SIDE
    boxes: list[Box] = []
    for k in range(n + 1):  # partitions, +y (your left) first
        y = half - k * w
        y0, y1 = max(y - _THICK, -half), min(y + _THICK, half)
        boxes.append(Box((0.0, d), (y0, y1), _PARTITION_Z, (edge, panel, edge, panel, edge)))
    boxes.append(Box((d - 0.01, d + 0.01), (-half, half), (0.7, CUBICLE_TOP), (edge,) * 5))
    for k in range(n):
        cy = half - (k + 0.5) * w
        boxes.append(Box((0.06, 0.28), (cy - 0.06, cy + 0.06), (0.0, 0.15), (white,) * 4 + (pan,)))
        boxes.append(Box((0.0, 0.06), (cy - 0.08, cy + 0.08), (0.12, 0.3), (white,) * 5))
        if k == 1:
            boxes += _feet(kind, cy)
    return tuple(boxes)


def _hinged(
    x: float, hinge_y: float, width: float, swing: float, faces: tuple[int, int, int, int, int]
) -> Box:
    """A door hinged at (x, hinge_y), its free end towards +y when shut,
    swung by `swing` radians (positive: out, +x; negative: in). A box turns
    only about the model's origin, so this is the door's rectangle turned
    back by the swing, `turn` taking it round to where it stands."""
    c, s = math.cos(swing), math.sin(swing)
    hx, hy = x * c - hinge_y * s, x * s + hinge_y * c
    return Box((hx - 0.008, hx), (hy, hy + width), _DOOR_Z, faces, turn=-swing)


def cubicle_door_boxes(doors: Iterable[CubicleDoor]) -> list[PlacedBox]:
    """Each cubicle's door as it stands: shut, or swung in (the hinge at
    the cubicle's -y side)."""
    side = _DOOR_EDGE
    out = []
    for door in doors:
        half = door.count * CUBICLE_WIDTH / 2
        y0 = half - (door.index + 1) * CUBICLE_WIDTH + _THICK
        front = _DOOR_FACES["engaged" if door.locked else "vacant"]
        faces = (front, side, _DOOR_FACES["inside"], side, side)
        box = _hinged(CUBICLE_DEPTH, y0, CUBICLE_WIDTH - 2 * _THICK, -door.open * OPEN_ANGLE, faces)
        out.append(PlacedBox(door.run_x, door.run_y, 0.0, door.run_angle, box))
    return out


def urinal_front() -> int:
    """A squarish wall-hung urinal from the front: a white slab with a wide
    flat lip, the bowl's hollow shaded inside it, the drain, and a blue
    deodorising block. The bottom corners rounded off (see-through)."""
    c = Canvas()
    body = ~((c.v >= 52) & ((c.u < 6 - (64 - c.v) // 2) | (c.u >= 58 + (64 - c.v) // 2)))
    body &= ~(((c.u + 0.5 - 32) / 32) ** 2 + ((c.v + 0.5 - 44) / 21) ** 2 > 1) | (c.v < 44)
    c.paint(body, *_PORCELAIN)
    c.paint(body & (c.v < 4), GREY, 15.0)  # the lip, catching the light
    hollow = c.rect(8, 56, 8, 56) & (
        ((c.u + 0.5 - 32) / 24) ** 2 + ((c.v + 0.5 - 40) / 18) ** 2 <= 1
    ) | c.rect(8, 40, 8, 56)
    c.paint(hollow, GREY, 11.0)
    c.paint(hollow & (c.v < 14), GREY, 9.5)  # shadowed under the lip
    c.paint(c.ellipse(50, 32, 3, 6), STEEL, 5.0)  # the drain
    c.paint(c.rect(45, 49, 26, 38), BLUE, 9.0)  # the deodorising block
    return _texture(c)


def urinal_side() -> int:
    """From the side: square at the wall, the lower front corner rounded."""
    c = Canvas()
    body = ~(
        (c.v >= 44) & (((c.u + 0.5) / 30) ** 2 + ((c.v + 0.5 - 44) / 20) ** 2 > 1) & (c.u < 30)
    )
    c.paint(body, *_PORCELAIN)
    c.paint(body & (c.v >= 56), GREY, 11.0)
    return _texture(c)


def urinal_top() -> int:
    c = Canvas()
    c.paint(c.rect(0, 64, 0, 64), *_PORCELAIN)
    c.paint(c.rect(10, 54, 8, 56), GREY, 10.0)  # looking down into the bowl
    return _texture(c)


def _urinals(n: int) -> tuple[Box, ...]:
    """A run of urinals: squarish bowls low on the wall, a chrome flush valve
    and pipe up from each, laminate screens between and at the ends."""
    half = n * URINAL_PITCH / 2
    front, side, top = urinal_front(), urinal_side(), urinal_top()
    chrome = _plain_texture(*_CHROME)
    screen = _texture(_laminate(9.5))
    boxes: list[Box] = []
    for k in range(n):
        cy = half - (k + 0.5) * URINAL_PITCH
        boxes.append(
            Box((0.0, 0.17), (cy - 0.13, cy + 0.13), (0.1, 0.42), (front, side, side, side, top))
        )
        boxes.append(Box((0.0, 0.05), (cy - 0.025, cy + 0.025), (0.42, 0.47), (chrome,) * 5))
        boxes.append(Box((0.0, 0.02), (cy - 0.01, cy + 0.01), (0.47, 0.7), (chrome,) * 5))
    for k in range(n + 1):  # screens between, and at the ends (runs can meet)
        y = half - k * URINAL_PITCH
        boxes.append(
            Box(
                (0.0, URINAL_DEPTH),
                (max(y - 0.006, -half), min(y + 0.006, half)),
                (0.14, 0.6),
                (chrome, screen, chrome, screen, chrome),
            )
        )
    return tuple(boxes)


def vanity_front(n: int) -> int:
    """The vanity unit's front: a brown wood-grain door under each basin,
    a white counter edge over them."""
    c = Canvas()
    c.paint(c.rect(0, 64, 0, 64), WOOD, 6.5)
    c.paint(c.rect(0, 8, 0, 64), GREY, 13.5)  # the counter's edge
    c.paint(c.rect(8, 9, 0, 64), WOOD, 3.0)
    pitch = 64 / n
    for k in range(n):
        u0 = round(k * pitch)
        c.paint(c.rect(9, 64, u0, u0 + 1), WOOD, 3.5)  # the doors' joints
        mid = round((k + 0.5) * pitch)
        c.paint(c.rect(18, 26, mid - 1, mid + 1), STEEL, 12.0)  # a chrome knob
    c.paint(c.rect(58, 64, 0, 64), WOOD, 2.5)  # the plinth, in shadow
    return _texture(c)


def vanity_top(n: int) -> int:
    """The counter from above (front at the bottom): dark marbled laminate,
    n white basins let into it, each bowl shaded, a plughole in each."""
    rng = np.random.default_rng(48)
    c = Canvas()
    c.paint(c.rect(0, 64, 0, 64), WOOD, 4.5 + (noise(rng, 16) - 0.5) * 1.2)
    pitch = 64 / n
    for k in range(n):
        cu = pitch * (k + 0.5)
        c.paint(c.ellipse(36, cu, 22, pitch * 0.42), GREY, 15.0)  # the rim
        c.paint(c.ellipse(37, cu, 17, pitch * 0.33), GREY, 12.0)  # the bowl
        c.paint(c.ellipse(41, cu, 11, pitch * 0.22), GREY, 10.5)
        c.paint(c.ellipse(41, cu, 3, 2.5), STEEL, 5.0)  # plughole
    return _texture(c)


def mirror_glass() -> int:
    """A mirror: it can't reflect (there's nothing behind a wall to show),
    so a cool grey sheen with two diagonal highlights, in a chrome edge."""
    c = Canvas()
    c.paint(c.rect(0, 64, 0, 64), STEEL, 7.0)
    for offset in (14, 22):
        c.paint(np.abs(c.u + c.v - 2 * offset - 20) < 3 + (offset == 14) * 2, STEEL, 10.0)
    c.paint(~c.rect(2, 62, 2, 62), STEEL, 13.0)
    return _texture(c)


def _basins(n: int) -> tuple[Box, ...]:
    """A vanity unit with n basins let into its counter, a chrome pillar tap
    with its spout over each, a mirror the unit's width over it."""
    half = n * BASIN_PITCH / 2
    side = _plain_texture(WOOD, 5.5)
    chrome = _plain_texture(*_CHROME)
    unit = Box(
        (0.0, BASIN_DEPTH),
        (-half, half),
        (0.0, BASIN_TOP),
        (vanity_front(n), side, side, side, vanity_top(n)),
    )
    mirror = Box(
        (0.0, 0.006), (-half, half), (0.42, 0.74), (mirror_glass(), chrome, chrome, chrome, chrome)
    )
    taps: list[Box] = []
    for k in range(n):
        cy = half - (k + 0.5) * BASIN_PITCH
        taps.append(
            Box(
                (0.02, 0.05), (cy - 0.015, cy + 0.015), (BASIN_TOP, BASIN_TOP + 0.07), (chrome,) * 5
            )
        )
        taps.append(
            Box(
                (0.05, 0.11),
                (cy - 0.01, cy + 0.01),
                (BASIN_TOP + 0.05, BASIN_TOP + 0.07),
                (chrome,) * 5,
            )
        )
    return (unit, mirror, *taps)


def hand_dryer_front() -> int:
    c = Canvas()
    c.paint(c.rect(0, 64, 0, 64), GREY, 13.5)
    c.paint(c.rect(56, 64, 0, 64), GREY, 11.0)
    c.paint(c.ellipse(26, 32, 9, 12), STEEL, 12.0)  # the big chrome button
    c.paint(c.ellipse(26, 32, 6, 9), STEEL, 9.0)
    c.text("PRESS", 42, 22, GREY, 5.0)
    return _texture(c)


def _hand_dryer() -> tuple[Box, ...]:
    white = _plain_texture(GREY, 13.0)
    chrome = _plain_texture(*_CHROME)
    body = Box(
        (0.0, 0.07), (-0.055, 0.055), (0.36, 0.46), (hand_dryer_front(), white, white, white, white)
    )
    nozzle = Box((0.02, 0.06), (-0.015, 0.015), (0.33, 0.36), (chrome,) * 5)
    return (body, nozzle)


def wet_floor_front() -> int:
    """CAUTION WET FLOOR: black on yellow, a figure slipping."""
    c = Canvas()
    c.paint(c.rect(0, 64, 0, 64), OCHRE, 12.5)
    c.paint(c.rect(0, 64, 0, 3) | c.rect(0, 64, 61, 64), OCHRE, 9.0)
    _centred(c, "CAUTION", 6, GREY, 1.0)
    # The slipping figure: head, a body tipping back, legs flying.
    c.paint(c.ellipse(20, 30, 3, 3), GREY, 1.0)
    for k in range(9):
        c.paint(c.rect(23 + k, 25 + k, 28 - k // 3, 31 - k // 3), GREY, 1.0)
    c.paint(c.rect(32, 34, 26, 40), GREY, 1.0)
    c.paint(c.rect(36, 37, 14, 50), GREY, 1.0)  # the floor
    _centred(c, "WET", 42, GREY, 1.0)
    _centred(c, "FLOOR", 50, GREY, 1.0)
    return _texture(c)


def wet_floor_side() -> int:
    """Seen from the side, an A: a yellow triangle, see-through round it."""
    c = Canvas()
    c.paint(
        (np.abs(c.u - 31.5) < 1 + c.v * 0.47) & (np.abs(c.u - 31.5) > c.v * 0.47 - 4), OCHRE, 11.0
    )
    return _texture(c)


def _wet_floor_sign() -> tuple[Box, ...]:
    front, side = wet_floor_front(), wet_floor_side()
    top = Canvas()
    top.paint(top.rect(0, 64, 28, 36), OCHRE, 12.0)  # just the hinge along the ridge
    return (
        Box((-0.05, 0.05), (-0.075, 0.075), (0.0, 0.24), (front, side, front, side, _texture(top))),
    )


# Keyed by thing type name, like SPRITES.
BOX_MODELS: dict[str, tuple[Box, ...]] = {
    "mac": _mac_se30(),
    "phone": _phone(),
    "pc": _pc(),
    "chair": _chair(),
    "sofa_two": _sofa(2),
    "sofa_three": _sofa(3),
    "sofa_corner": tuple(_sofa_unit(0.0, backs=("-x", "-y"))),
    "coffee_table": _coffee_table(),
    "fern": _fern(),
    "coffee_cup": _coffee_cup(),
    "vending_machine": _vending_machine(),
    **SERVER_MODELS,
    **PASS_MODELS,
    **{f"poster:{k}": _poster(*v) for k, v in _POSTER_KINDS.items()},
    "filing_cabinet": _filing_cabinet(),
    "water_cooler": _water_cooler(),
    "ficus": _ficus(),
    "reception_desk": _reception_desk(),
    "welcome_sign": _welcome_sign(),
    "exit_sign": _exit_sign(EXIT_SIGN_HEIGHTS["door"]),
    **{f"exit_sign:{k}": _exit_sign(z) for k, z in EXIT_SIGN_HEIGHTS.items()},
    "executive_desk": _executive_desk(),
    **{
        f"cubicles_{w}{suffix}": _cubicles(n, kind)
        for n, w in ((2, "two"), (3, "three"), (4, "four"))
        for kind, suffix in (("gents", ""), ("gents", ":gents"), ("ladies", ":ladies"))
    },
    **{f"urinals_{w}": _urinals(n) for n, w in ((2, "two"), (3, "three"), (4, "four"))},
    **{f"basins_{w}": _basins(n) for n, w in ((2, "two"), (3, "three"), (4, "four"))},
    "hand_dryer": _hand_dryer(),
    "wet_floor_sign": _wet_floor_sign(),
}
BOX_TEXTURES: U8Array = np.stack(_TEXTURES)  # [texture][u][v]
BOX_OPAQUE: BoolArray = np.stack(_OPAQUE)  # [texture][u][v]: False = see-through
_STACKED: list[tuple[U8Array, BoolArray]] = [(BOX_TEXTURES, BOX_OPAQUE)]


def box_textures() -> tuple[U8Array, BoolArray]:
    """(BOX_TEXTURES, BOX_OPAQUE), with any textures made since (a map's fire
    plans): restacked only when there are new ones."""
    if len(_STACKED[0][0]) != len(_TEXTURES):
        _STACKED[0] = (np.stack(_TEXTURES), np.stack(_OPAQUE))
    return _STACKED[0]

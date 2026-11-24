"""Procedurally generated 64x64 office wall textures, as 8-bit palette indices.

Why 64x64: a power of two, so coordinates wrap with a bitmask (u & 63), and at
our 320x200 resolution one texel maps to roughly one or more screen pixels.
Bigger textures would be squashed into fewer pixels than they have texels,
which aliases (shimmers) without mipmapping.

Generators draw in (v, u) order; the atlas stores them transposed as [u][v].
Adding a texture: write a generator returning a (64, 64) uint8 index array and
register it against a wall type with `register_wall_texture`. The wall types
are listed in docs/GAME.md.

Tiling rule: anything framed (panels, trims, borders) must be split evenly
between opposite edges of the 64x64 tile, e.g. 2 texels at each edge rather
than 4 at one. Then the frame lines up with the edges of the wall block, and
two blocks side by side make one full-width stile. Likewise desk legs sit
flush with a face's edges, so the legs seen on two faces meet at the corner.

Masked textures: a texture may come with an opacity mask. Transparent texels
aren't drawn, so whatever is behind shows through. Only low-block faces
(desks) honour it: a full-height wall stops the ray, so there's nothing
behind it to show.
"""

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from doomer.client.palette import (
    BLUE,
    BRICK,
    CREAM,
    GREEN,
    GREY,
    MAGENTA,
    MOSS,
    OCHRE,
    ORANGE,
    SAND,
    STEEL,
    WARM_GREY,
    WOOD,
    U8Array,
)
from doomer.client.pixelart import ART_MASK, ART_SIZE, grid, indices, noise

TEX_SIZE = ART_SIZE  # 64
TEX_MASK = ART_MASK  # u & TEX_MASK == u % 64

BoolArray = npt.NDArray[np.bool_]

# Which side of a wall cell a texture is for, from the cell's own point of view
# (its front faces the way its [facing] says). A face's role is
# (compass face hit - cell facing) mod 4, with compass faces numbered clockwise
# from north; see doomer/shared/mapfile.py.
FRONT, RIGHT, BACK, LEFT = range(4)
ALL_SIDES = (FRONT, RIGHT, BACK, LEFT)
# Not a side: the texture for any part of a wall above one storey (a double-
# height atrium), so the dado rail and skirting don't repeat halfway up.
# Wall types without one repeat their front texture.
UPPER = 4


@dataclass(frozen=True, slots=True)
class _Registration:
    wall_type: int
    sides: tuple[int, ...]
    texture: U8Array
    opaque: BoolArray | None


_REGISTRATIONS: list[_Registration] = []


def register_wall_texture(
    wall_type: int,
    texture: U8Array,
    opaque: BoolArray | None = None,
    sides: tuple[int, ...] = ALL_SIDES,
) -> None:
    """Register a (v, u) texture for some or all sides of a wall type,
    optionally with an opacity mask. Later registrations override earlier
    ones for the sides they name."""
    if texture.shape != (TEX_SIZE, TEX_SIZE) or texture.dtype != np.uint8:
        raise ValueError(f"texture must be ({TEX_SIZE}, {TEX_SIZE}) uint8")
    if opaque is not None and opaque.shape != texture.shape:
        raise ValueError("opacity mask must match the texture's shape")
    mask = None if opaque is None else opaque.astype(np.bool_)
    _REGISTRATIONS.append(_Registration(wall_type, tuple(sides), texture, mask))


def build_atlas() -> tuple[U8Array, npt.NDArray[np.intp]]:
    """Stack all textures for the renderer.

    Returns (atlas, slot_table): atlas[slot, u, v] is a palette index, and
    slot_table[wall_type, side] picks the slot (0 = the missing-texture checker).
    """
    max_type = max((r.wall_type for r in _REGISTRATIONS), default=0)
    slots = [missing_texture()]
    slot_table = np.zeros((max_type + 1, 5), dtype=np.intp)
    upper_given: set[int] = set()
    for reg in _REGISTRATIONS:
        for side in reg.sides:
            slot_table[reg.wall_type, side] = len(slots)
            if side == UPPER:
                upper_given.add(reg.wall_type)
        slots.append(reg.texture)
    for wall_type in range(max_type + 1):
        if wall_type not in upper_given:
            slot_table[wall_type, UPPER] = slot_table[wall_type, FRONT]
    atlas = np.stack([s.T for s in slots])  # (n, u, v)
    return np.ascontiguousarray(atlas), slot_table


def build_opacity() -> BoolArray:
    """(n, u, v) opacity per atlas slot, in the same slot order as build_atlas()."""
    full = np.ones((TEX_SIZE, TEX_SIZE), dtype=np.bool_)
    masks = [full] + [full if r.opaque is None else r.opaque for r in _REGISTRATIONS]
    return np.ascontiguousarray(np.stack([m.T for m in masks]))


def _blocks(
    block_w: int, block_h: int, rng: np.random.Generator
) -> tuple[npt.NDArray[np.bool_], npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Running-bond blocks (each row offset by half a block).

    Returns (mortar mask, per-block random tone in [0, 1), bevel: +1 on top/left
    edges, -1 on bottom/right edges, 0 elsewhere).
    """
    v, u = grid()
    row = v // block_h
    shifted = (u + (row % 2) * (block_w // 2)) % TEX_SIZE
    col = shifted // block_w
    bu, bv = shifted % block_w, v % block_h
    mortar = (bu == 0) | (bv == 0)
    tone = rng.random((TEX_SIZE // block_h, TEX_SIZE // block_w))[row, col]
    bevel = np.where((bv == 1) | (bu == 1), 1.0, 0.0) - np.where(
        (bv == block_h - 1) | (bu == block_w - 1), 1.0, 0.0
    )
    return mortar, tone, bevel


# --- generators --------------------------------------------------------------------


def missing_texture() -> U8Array:
    v, u = grid()
    checker = ((u >> 3) ^ (v >> 3)) & 1
    return indices(MAGENTA, np.where(checker == 1, 15.0, 3.0))


def office_wall_upper(seed: int = 1) -> U8Array:
    """The same striped wallpaper as office_wall, without the dado and skirting,
    for walls taller than a storey."""
    rng = np.random.default_rng(seed)
    v, u = grid()
    stripes = np.where((u % 8) < 4, 0.6, 0.0)
    return indices(SAND, 10 + stripes + (noise(rng, 16) - 0.5) * 1.2)


def raised_floor_edge(seed: int = 14) -> U8Array:
    """Side of a raised floor (mezzanine, steps): painted plaster with a
    carpet-gripper strip at the top edge and skirting at the bottom."""
    rng = np.random.default_rng(seed)
    v, u = grid()
    tex = indices(SAND, 8.5 + (noise(rng, 8) - 0.5))
    tex = np.where(v % 8 == 0, indices(GREY, 10.0), tex)  # nosing at each 1/8 step edge
    return np.where(v >= 60, indices(WOOD, 5.0), tex).astype(np.uint8)


# --- two-storey walls ----------------------------------------------------------------
# Lower textures cover heights 0..1 (v = 63 at the floor, v = 0 at the storey
# line); upper textures cover 1..2. Each draws half of the transom at the storey
# line (lower: v 0..1, upper: v 62..63), so stacked they meet as one, and the
# mullions are split evenly at u 0/63 and centred at 31/32 to tile sideways.


def _glass(
    rng: np.random.Generator, v: npt.NDArray[np.int64], u: npt.NDArray[np.int64], storey: int
) -> U8Array:
    """Tinted glazing: sky gradient over two storeys plus a diagonal reflection."""
    height = (63 - v) / 64 + storey  # 0 at the floor, 2 at the top of the upper storey
    shade = 7 + height * 1.5 + (noise(rng, 8) - 0.5)
    glint = ((u + v + storey * 64) % 48) < 6
    return np.where(glint, indices(STEEL, 13.0), indices(BLUE, shade)).astype(np.uint8)


def _mullions(tex: U8Array, v: npt.NDArray[np.int64], u: npt.NDArray[np.int64]) -> U8Array:
    frame = (u < 2) | (u >= 62) | ((u >= 31) & (u < 33))
    shade = np.where((u == 0) | (u == 31), 12.0, 9.0)
    return np.where(frame, indices(GREY, shade), tex).astype(np.uint8)


def curtain_wall_lower(seed: int = 15) -> U8Array:
    """Ground storey of a glazed curtain wall: glass between aluminium mullions."""
    rng = np.random.default_rng(seed)
    v, u = grid()
    tex = _mullions(_glass(rng, v, u, 0), v, u)
    tex = np.where(v < 2, indices(GREY, 10.0), tex)  # half the transom at the storey line
    return np.where(v >= 60, indices(GREY, 5.0), tex).astype(np.uint8)  # base cill


def curtain_wall_upper(seed: int = 16) -> U8Array:
    """Upper storey of the curtain wall: the same mullions, the other half of the transom."""
    rng = np.random.default_rng(seed)
    v, u = grid()
    tex = _mullions(_glass(rng, v, u, 1), v, u)
    tex = np.where(v >= 62, indices(GREY, 10.0), tex)  # meets the lower transom
    return np.where(v < 2, indices(GREY, 7.0), tex).astype(np.uint8)  # head at the ceiling


def entrance_doors(seed: int = 17) -> U8Array:
    """Main entrance: a pair of closed glass doors in aluminium frames, push bars."""
    rng = np.random.default_rng(seed)
    v, u = grid()
    tex = _glass(rng, v, u, 0)
    frame = (u < 4) | (u >= 60) | ((u >= 30) & (u < 34)) | (v < 4) | (v >= 58)
    tex = np.where(frame, indices(GREY, np.where((u == 31) | (u == 32), 6.0, 10.0)), tex)
    bar = (v >= 34) & (v < 37) & (((u >= 8) & (u < 28)) | ((u >= 36) & (u < 56)))
    tex = np.where(bar, indices(GREY, 14.0), tex)  # push bars
    kick = (v >= 50) & (v < 58) & ~frame
    tex = np.where(kick, indices(GREY, 8.0), tex)  # kick plates
    return tex.astype(np.uint8)


def office_wall(seed: int = 1) -> U8Array:
    """Beige wallpaper above a dado rail, a darker painted panel below, skirting board."""
    rng = np.random.default_rng(seed)
    v, u = grid()
    stripes = np.where((u % 8) < 4, 0.6, 0.0)
    shade = 10 + stripes + (noise(rng, 16) - 0.5) * 1.2
    shade = np.where(v >= 40, 8.5 + (noise(rng, 8) - 0.5), shade)  # lower panel
    tex = indices(SAND, shade)
    rail = (v >= 38) & (v < 41)
    tex = np.where(rail, indices(WOOD, np.where(v == 38, 10.0, 7.0)), tex)
    skirting = v >= 58
    tex = np.where(skirting, indices(WOOD, np.where(v == 58, 8.0, 5.0)), tex)
    return tex.astype(np.uint8)


def wood_panelling(seed: int = 2) -> U8Array:
    """Executive-floor panelling: raised panels in a frame of rails and stiles."""
    rng = np.random.default_rng(seed)
    v, u = grid()
    grain = np.sin(u * 0.7 + noise(rng, 8) * 8) * 0.8
    shade = 7 + grain + noise(rng, 32) * 1.5
    pu, pv = u % 32, v % 32
    # The frame is split evenly between the edges of each 32-texel panel (2 on
    # each side), so it lines up with the edges of the wall block and two blocks
    # side by side make one 4-texel stile.
    frame = (pu < 2) | (pu >= 30) | (pv < 2) | (pv >= 30)
    shade = np.where(frame, 6 + noise(rng, 16), shade)
    lit = ~frame & ((pu == 2) | (pv == 2))  # raised panel, lit from the top left
    shadowed = ~frame & ((pu == 29) | (pv == 29))
    shade = np.where(lit, 9.5, np.where(shadowed, 4.0, shade))
    return indices(WOOD, shade)


def cubicle_partition(seed: int = 3) -> U8Array:
    """Grey-blue fabric partition with an aluminium trim at the top and sides."""
    rng = np.random.default_rng(seed)
    v, u = grid()
    weave = ((u + v) % 2) * 0.5
    shade = 7 + weave + (noise(rng, 32) - 0.5) * 1.5
    tex = indices(STEEL, shade)
    trim = (v < 4) | ((u % 32) < 1) | ((u % 32) >= 31)  # split evenly between edges
    return np.where(trim, indices(GREY, np.where(v == 1, 13.0, 10.0)), tex).astype(np.uint8)


def window_blinds(seed: int = 4) -> U8Array:
    """An outer-wall window: frame, venetian blinds, sky glimpsed between the slats."""
    rng = np.random.default_rng(seed)
    v, u = grid()
    sky = indices(BLUE, 12 - v / 16 + (noise(rng, 4) - 0.5))
    slat = (v % 6) < 4
    slats = indices(CREAM, np.where((v % 6) == 0, 9.0, 7.5))
    tex = np.where(slat, slats, sky)
    frame = (u < 5) | (u >= 59) | (v < 5) | (v >= 56) | (np.abs(u - 31.5) < 1.5)
    tex = np.where(frame, indices(GREY, np.where((u < 2) | (v < 2), 12.0, 9.0)), tex)
    sill = v >= 60
    return np.where(sill, indices(SAND, 7.0), tex).astype(np.uint8)


def lift_doors(seed: int = 5) -> U8Array:
    """Brushed steel lift doors with a centre seam and a lit floor indicator."""
    rng = np.random.default_rng(seed)
    v, u = grid()
    streaks = noise(rng, 32)[0][None, :]
    tex = indices(STEEL, 9 + streaks * 2)
    tex = np.where(np.abs(u - 31.5) < 1, indices(GREY, 2.0), tex)  # seam
    frame = (u < 4) | (u >= 60) | (v < 12)
    tex = np.where(frame, indices(GREY, np.where(u < 2, 6.0, 8.0)), tex)
    panel = (v >= 3) & (v < 9) & (u >= 22) & (u < 42)
    tex = np.where(panel, indices(GREY, 2.0), tex)
    lit = panel & ((u - 22) % 4 < 2) & (v >= 5) & (v < 7)
    tex = np.where(lit & (u < 30), indices(ORANGE, 15.0), tex)
    return tex.astype(np.uint8)


def stairwell_brick(seed: int = 6) -> U8Array:
    rng = np.random.default_rng(seed)
    mortar, tone, bevel = _blocks(16, 8, rng)
    shade = 8 + tone * 4 + bevel + (noise(rng, 32) - 0.5) * 2
    ramps = np.where(mortar, GREY, BRICK)
    return indices(ramps, np.where(mortar, 6.0, shade))


def avocado_tiles(seed: int = 7) -> U8Array:
    """Washroom tiles, in that very 1980s avocado."""
    rng = np.random.default_rng(seed)
    v, u = grid()
    grout = ((u % 16) == 0) | ((v % 16) == 0)
    tone = rng.random((4, 4))[v // 16, u // 16]
    shade = 9 + tone * 2 + (noise(rng, 16) - 0.5) * 1.5
    shade = np.where(((u % 16) == 1) | ((v % 16) == 1), shade + 2, shade)
    return indices(np.where(grout, GREY, MOSS), np.where(grout, 11.0, shade))


def steel_panels(seed: int = 8) -> U8Array:
    rng = np.random.default_rng(seed)
    v, u = grid()
    pu, pv = u % 32, v % 32
    border = (pu < 1) | (pu >= 31) | (pv < 1) | (pv >= 31)  # split evenly between edges
    highlight = (pu == 1) | (pv == 1)
    corner = np.isin(pu, (4, 5, 26, 27)) & np.isin(pv, (4, 5, 26, 27))  # symmetric
    rivet = corner & ~border
    streaks = noise(rng, 32)[0][None, :]
    shade = 8 + streaks * 3 + (noise(rng, 8) - 0.5) * 2
    shade = np.where(highlight, 12.0, shade)
    shade = np.where(rivet, 14.0, shade)
    return indices(STEEL, np.where(border, 4.0, shade))


def mainframe(seed: int = 9) -> U8Array:
    """Server-room cabinet: tape reels on top, a grid of blinking lights below."""
    rng = np.random.default_rng(seed)
    v, u = grid()
    tex = indices(GREY, 10 + (noise(rng, 16) - 0.5))
    for cu in (18.5, 45.5):
        dist = np.hypot(v + 0.5 - 16, u + 0.5 - cu)
        tex = np.where(dist < 11, indices(GREY, 3.0), tex)
        tex = np.where((dist < 9) & (dist >= 4), indices(WOOD, 4 + (dist % 2)), tex)
        tex = np.where(dist < 2, indices(GREY, 12.0), tex)
    lights = (v >= 34) & (v < 56) & (u >= 6) & (u < 58) & (v % 4 < 2) & (u % 4 < 2)
    colours = rng.choice(np.array([GREEN, ORANGE, BRICK, GREY]), size=(16, 16))[v // 4, u // 4]
    on = rng.random((16, 16))[v // 4, u // 4] > 0.35
    tex = np.where(lights, indices(colours, np.where(on, 15.0, 4.0)), tex)
    tex = np.where((v < 2) | (u < 2) | (u >= 62), indices(GREY, 5.0), tex)
    return tex.astype(np.uint8)


# --- open desks (masked) ----------------------------------------------------------
# Designed for desk height 2/8: textures are floor-anchored, so a face of height
# 0.25 shows texel rows v = 48..63. The slab is the top 4 of those; below it,
# only the legs or pedestals are opaque and the rest is open space.

DESK_SLAB = (48, 52)  # texel rows of the desktop's front edge


def office_desk() -> tuple[U8Array, BoolArray]:
    """Beige laminate desktop on tubular steel legs, open underneath."""
    v, u = grid()
    tex = indices(SAND, np.where(v == 48, 13.0, np.where(v == 51, 8.0, 11.0)))
    # Legs flush with each face's edges: at a corner, the leg seen on the front
    # face and the one on the side face meet and read as one square post.
    legs = (u < 3) | (u >= 61)
    leg_shade = np.where((u == 0) | (u == 61), 11.0, np.where((u == 2) | (u == 63), 5.0, 8.0))
    tex = np.where(legs & (v >= DESK_SLAB[1]), indices(GREY, leg_shade), tex)
    foot = legs & (v >= 62)
    tex = np.where(foot, indices(GREY, 4.0), tex)
    opaque = (v < DESK_SLAB[1]) | legs
    return tex.astype(np.uint8), opaque


def executive_desk(seed: int = 12) -> tuple[U8Array, BoolArray]:
    """Wooden kneehole desk: drawer pedestals either side, open in the middle."""
    rng = np.random.default_rng(seed)
    v, u = grid()
    grain = 7 + np.sin(u * 0.7 + noise(rng, 8) * 6) * 0.8
    tex = indices(WOOD, grain)
    slab = (v >= DESK_SLAB[0]) & (v < DESK_SLAB[1])
    tex = np.where(slab, indices(WOOD, np.where(v == 48, 11.0, 9.5)), tex)
    pedestal = (u < 20) | (u >= 44)
    drawer_line = pedestal & (v >= 52) & ((v - 52) % 4 == 3)
    tex = np.where(drawer_line, indices(WOOD, 4.0), tex)
    handle = pedestal & (v >= 52) & ((v - 52) % 4 == 1) & ((u % 22 >= 8) & (u % 22 < 12))
    tex = np.where(handle, indices(OCHRE, 13.0), tex)
    edge = (u == 19) | (u == 44)
    tex = np.where(edge & (v >= 52), indices(WOOD, 5.0), tex)
    opaque = (v < DESK_SLAB[1]) | pedestal
    return tex.astype(np.uint8), opaque


def executive_panel(seed: int = 13) -> tuple[U8Array, BoolArray]:
    """Solid wood: the executive desk's modesty panel (back) and pedestal sides."""
    rng = np.random.default_rng(seed)
    v, u = grid()
    grain = 7 + np.sin(u * 0.7 + noise(rng, 8) * 6) * 0.8
    tex = indices(WOOD, grain)
    slab = (v >= DESK_SLAB[0]) & (v < DESK_SLAB[1])
    tex = np.where(slab, indices(WOOD, np.where(v == 48, 11.0, 9.5)), tex)
    inset = (v >= 54) & (v < 62) & (u >= 4) & (u < 60)
    frame_line = inset & ((v == 54) | (u == 4))
    tex = np.where(frame_line, indices(WOOD, 4.5), tex)  # recessed panel's shadow edge
    tex = np.where(inset & ((v == 61) | (u == 59)), indices(WOOD, 9.5), tex)  # lit edge
    return tex.astype(np.uint8), np.ones_like(v, dtype=np.bool_)


def railing() -> tuple[U8Array, BoolArray]:
    """A balcony railing, top (v = 0) to the balcony floor (v = 63): a polished
    handrail, a bottom rail and steel balusters, see-through in between."""
    v, u = grid()
    handrail = v < 5
    tex = indices(WOOD, np.where(v == 0, 12.0, np.where(v == 4, 6.0, 9.5)))
    bottom = v >= 58
    balusters = (u % 8 == 3) | (u % 8 == 4)
    steel = np.where(u % 8 == 3, 12.0, 8.0)
    tex = np.where(~handrail, indices(GREY, np.where(bottom, 9.0, steel)), tex)
    opaque = handrail | bottom | balusters
    return tex.astype(np.uint8), opaque


# --- 1980s steel desk (wall type 'c'), after a photo of the real thing --------------
# Dark mahogany laminate top on a grey steel frame: an apron rail under the top,
# square steel legs, and one pedestal of drawers at the desk's left end (its own
# left: the right-hand end as you face the drawers). Open underneath otherwise.

STEEL_APRON = (51, 54)  # texel rows of the steel rail under the top
# u ranges of the two square legs on a face: flush with its edges, so at each
# corner the front and side views of a leg meet as one post.
_STEEL_LEG_A = (0, 3)
_STEEL_LEG_B = (61, 64)


def _steel_desk_base() -> tuple[U8Array, BoolArray, npt.NDArray[np.int64], npt.NDArray[np.int64]]:
    """Top, apron and both legs; returns (texture, opaque, v, u) to add to."""
    v, u = grid()
    tex = indices(WOOD, np.where(v == 48, 7.0, 5.0))  # mahogany laminate edge
    apron = (v >= STEEL_APRON[0]) & (v < STEEL_APRON[1])
    apron_shade = np.where(v == STEEL_APRON[0], 11.0, np.where(v == STEEL_APRON[1] - 1, 6.0, 9.0))
    tex = np.where(apron, indices(GREY, apron_shade), tex)
    tex, opaque = _steel_legs(tex, v, u)
    return tex, opaque | (v < STEEL_APRON[1]), v, u


def _steel_legs(
    tex: U8Array, v: npt.NDArray[np.int64], u: npt.NDArray[np.int64]
) -> tuple[U8Array, BoolArray]:
    legs = np.zeros_like(v, dtype=np.bool_)
    for lo, hi in (_STEEL_LEG_A, _STEEL_LEG_B):
        leg = (u >= lo) & (u < hi) & (v >= STEEL_APRON[1])
        shade = np.where(u == lo, 11.0, np.where(u == hi - 1, 6.0, 9.0))
        tex = np.where(leg, indices(GREY, np.where(v == 63, 4.0, shade)), tex)
        legs |= leg
    return tex.astype(np.uint8), legs


def _steel_panel(tex: U8Array, where: BoolArray, shade: float = 9.0) -> U8Array:
    return np.where(where, indices(GREY, shade), tex).astype(np.uint8)


def steel_desk_front() -> tuple[U8Array, BoolArray]:
    """Facing the drawers: pencil drawer in the rail, pedestal on the right."""
    tex, opaque, v, u = _steel_desk_base()
    under = v >= STEEL_APRON[1]
    pedestal = under & (u >= 38) & (u < 61)
    tex = _steel_panel(tex, pedestal)
    tex = _steel_panel(tex, pedestal & (u == 38), 6.0)  # pedestal edge in shadow
    for line in (57, 60):  # gaps between the three drawers
        tex = _steel_panel(tex, pedestal & (v == line), 5.0)
    for row in (55, 58, 61):  # drawer pulls
        tex = _steel_panel(tex, pedestal & (v == row) & (u >= 48) & (u < 53), 13.0)
    rail = (v >= STEEL_APRON[0]) & (v < STEEL_APRON[1])
    tex = _steel_panel(tex, rail & ((u == 14) | (u == 34)), 5.0)  # pencil drawer outline
    tex = _steel_panel(tex, (v == 52) & (u >= 22) & (u < 27), 13.0)  # its pull
    return tex, opaque | pedestal


def steel_desk_back() -> tuple[U8Array, BoolArray]:
    """From behind: the pedestal's plain back panel, now on the left."""
    tex, opaque, v, u = _steel_desk_base()
    pedestal = (v >= STEEL_APRON[1]) & (u >= 3) & (u < 27)
    tex = _steel_panel(tex, pedestal, 8.0)
    tex = _steel_panel(tex, pedestal & (u == 26), 5.0)
    return tex, opaque | pedestal


def steel_desk_pedestal_side() -> tuple[U8Array, BoolArray]:
    """The pedestal end: a solid steel side panel between the legs."""
    tex, opaque, v, u = _steel_desk_base()
    panel = (v >= STEEL_APRON[1]) & (u >= 3) & (u < 61)
    tex = _steel_panel(tex, panel, 8.0)
    tex = _steel_panel(tex, panel & (u == 31), 6.0)  # seam
    return tex, opaque | panel


def steel_desk_open_side() -> tuple[U8Array, BoolArray]:
    """The other end: just the rail and two legs."""
    tex, opaque, _, _ = _steel_desk_base()
    return tex, opaque


# --- a pair of steel desks back to back in one cell (wall type 'd') ---------------
# The two sitters face each other across the pair. Both long sides look like a
# single desk's front (each desk's pedestal is at its own left end, which from
# either long side is the viewer's right), and both ends are half pedestal side
# panel (one desk's pedestal end) and half open legs (the other desk's open end).


def steel_desk_pair_end() -> tuple[U8Array, BoolArray]:
    """End of a back-to-back pair: one desk's pedestal panel, the other's open end."""
    tex, opaque, v, u = _steel_desk_base()
    under = v >= STEEL_APRON[1]
    # The two desks' back legs meet in the middle: legs at u 29..31 and 32..34.
    for lo, hi in ((29, 32), (32, 35)):
        leg = under & (u >= lo) & (u < hi)
        shade = np.where(u == lo, 11.0, np.where(u == hi - 1, 6.0, 9.0))
        tex = np.where(leg, indices(GREY, np.where(v == 63, 4.0, shade)), tex).astype(np.uint8)
        opaque = opaque | leg
    panel = under & (u >= 3) & (u < 29)  # the left-hand desk's pedestal end
    tex = _steel_panel(tex, panel, 8.0)
    tex = _steel_panel(tex, panel & (u == 15), 6.0)  # seam
    return tex, opaque | panel


# Low blocks whose top shows a seam across its middle, perpendicular to the
# cell's facing: two desks pushed together back to back ('d').
TOP_SEAM_WALL_TYPES: frozenset[int] = frozenset({13, 19})


# Colour of a low block's top surface, by wall type: a partition-fronted desk
# pod has a beige laminate top, a panelled counter a wooden one. Types not
# listed fall back to the commonest colour in the top of their texture.
TOP_COLOURS: dict[int, int] = {
    2: indices(WOOD, 9.0).item(),
    3: indices(SAND, 12.0).item(),
    10: indices(SAND, 12.0).item(),  # office desk: beige laminate
    11: indices(WOOD, 10.0).item(),  # executive desk
    12: indices(WOOD, 6.0).item(),  # 1980s steel desk: dark mahogany laminate
    13: indices(WOOD, 6.0).item(),  # pair of steel desks back to back
    19: indices(WOOD, 6.0).item(),  # ... with cubicle screens
    14: indices(BLUE, 6.0).item(),  # raised floor: 1980s corporate blue carpet
    17: indices(BLUE, 6.0).item(),  # stairs: carpeted
    18: indices(BLUE, 6.0).item(),  # balcony: carpeted
}


def lift_car_wall(seed: int = 20) -> U8Array:
    """Inside a lift car: brushed steel panels, a mirror strip, a handrail."""
    rng = np.random.default_rng(seed)
    v, u = grid()
    brushed = 9.0 + (noise(rng, 32)[:, :1] - 0.5) * 0.8 + ((u * 7) % 5 == 0) * 0.6
    tex = indices(GREY, np.broadcast_to(brushed, v.shape))
    tex = np.where((u % 32 == 0) | (u % 32 == 31), indices(GREY, 6.0), tex)  # panel joints
    mirror_shade = (11.0 - (v - 8) * 0.12).astype(np.float64)
    tex = np.where((v >= 8) & (v < 26), indices(STEEL, mirror_shade), tex)  # mirror
    rail = (v >= 40) & (v < 43)
    tex = np.where(rail, indices(GREY, np.where(v == 40, 14.0, 11.0)), tex)  # handrail
    return np.where(v >= 60, indices(GREY, 4.0), tex).astype(np.uint8)


def stone_cladding(seed: int = 21) -> U8Array:
    """The lift lobby's polished stone: pale, faintly veined, in big slabs."""
    rng = np.random.default_rng(seed)
    v, u = grid()
    veins = np.abs(noise(rng, 4) - noise(rng, 16) * 0.5 - 0.25) < 0.02
    tex = indices(WARM_GREY, 12.0 + (noise(rng, 8) - 0.5) * 0.8)
    tex = np.where(veins, indices(WARM_GREY, 10.5), tex)  # faint: veins, not cracks
    tex = np.where((v == 31) | (u == 0), indices(WARM_GREY, 8.0), tex)  # slab joints
    return np.where(v >= 60, indices(WARM_GREY, 5.0), tex).astype(np.uint8)


DEFAULT_WALL_TEXTURES: dict[int, Callable[[], U8Array]] = {
    1: office_wall,
    2: wood_panelling,
    3: cubicle_partition,
    4: window_blinds,
    5: lift_doors,
    6: stairwell_brick,
    7: avocado_tiles,
    8: steel_panels,
    9: mainframe,
}

for _wall_type, _make in DEFAULT_WALL_TEXTURES.items():
    register_wall_texture(_wall_type, _make())

# Above one storey: plain wallpaper over office walls and lift doors (the lift
# shaft wall continues up); other types repeat their own texture.
register_wall_texture(1, office_wall_upper(), sides=(UPPER,))
register_wall_texture(5, office_wall_upper(), sides=(UPPER,))
# 'k' (20): lift car walls: steel inside the car; the back face (BACK, with
# [facing] pointing into the car) is ordinary office wall.
register_wall_texture(20, lift_car_wall())
register_wall_texture(20, office_wall(), sides=(BACK,))
# 'l' (21): the lift lobby's stone cladding round the lift doors.
register_wall_texture(21, stone_cladding())
register_wall_texture(21, office_wall_upper(), sides=(UPPER,))
# 'h' (17): stairs (shape in doomer/shared/mapfile.py). Sides: plain plaster
# stringer with a skirting line; top: the same blue carpet as raised floors.
register_wall_texture(17, raised_floor_edge())
register_wall_texture(17, office_wall_upper(), sides=(UPPER,))
# 'i' (18): balcony floor (railing behaviour in doomer/shared/mapfile.py);
# its sides and top look like any raised floor.
register_wall_texture(18, raised_floor_edge())
register_wall_texture(18, office_wall_upper(), sides=(UPPER,))
# 'f' (15): glazed curtain wall, 'g' (16): main entrance doors; both carry on
# as the curtain wall's upper storey above one storey.
register_wall_texture(15, curtain_wall_lower())
register_wall_texture(15, curtain_wall_upper(), sides=(UPPER,))
register_wall_texture(16, entrance_doors())
register_wall_texture(16, curtain_wall_upper(), sides=(UPPER,))
# 'e' (14): raised floor (mezzanines, steps). Walkable once you're up.
register_wall_texture(14, raised_floor_edge())
register_wall_texture(14, office_wall_upper(), sides=(UPPER,))

# Desks, per side (from the desk's own point of view; its front is where you
# sit). Map characters: 'a' (10) office desk, 'b' (11) executive desk,
# 'c' (12) 1980s steel desk, 'd' (13) two steel desks back to back, 'j' (19)
# the same with cubicle screens on top (the screens are box models, boxes.py).
DESK_TEXTURES: dict[int, dict[int, Callable[[], tuple[U8Array, BoolArray]]]] = {
    10: {FRONT: office_desk, RIGHT: office_desk, BACK: office_desk, LEFT: office_desk},
    11: {
        FRONT: executive_desk,
        RIGHT: executive_panel,
        BACK: executive_panel,
        LEFT: executive_panel,
    },
    12: {
        FRONT: steel_desk_front,
        RIGHT: steel_desk_open_side,
        BACK: steel_desk_back,
        LEFT: steel_desk_pedestal_side,
    },
    13: {
        FRONT: steel_desk_front,
        RIGHT: steel_desk_pair_end,
        BACK: steel_desk_front,
        LEFT: steel_desk_pair_end,
    },
}
DESK_TEXTURES[19] = DESK_TEXTURES[13]
for _wall_type, _by_side in DESK_TEXTURES.items():
    for _side, _make_side in _by_side.items():
        register_wall_texture(_wall_type, *_make_side(), sides=(_side,))

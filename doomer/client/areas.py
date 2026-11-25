"""What each kind of area looks like: its floor, its ceiling, and the plain
walls that face into it.

A map's `[areas]` grid (doomer/shared/mapfile.py) gives each cell an area:
open-plan office, executive suite, entrance lobby or service. Here each area
gets its materials:

- floor and ceiling textures, 64 x 64 texels per cell, looked up per pixel by
  the renderer (floor casting: the world point a pixel shows, then its texel);
- a wall look for plain walls (`#`, wall type 1): each face of a plain wall
  takes the look of the area it faces. A wall between the lobby and a back
  corridor is stone on one side and painted block on the other, with no wall
  letter for each combination. Other wall types (windows, lift doors, desks)
  keep their own textures. Chosen per (cell, face) when the map loads, so it
  costs nothing per frame.

Mipmaps: a floor seen at a slant squeezes many texels into one pixel, and
a 64-texel pattern then shimmers as you move (aliasing). Each surface texture
has pre-shrunk copies, each averaging 2 x 2 texels of the one before, down to
one colour; the renderer picks the copy whose texels are about a pixel wide
(dithering between neighbouring copies so there's no visible seam).
"""

import functools
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from doomer.client.palette import BLUE, CREAM, GREY, PALETTE, STEEL, WARM_GREY, WOOD, U8Array
from doomer.client.pixelart import grid, indices, noise
from doomer.client.textures import (
    ALL_SIDES,
    BACK,
    TEX_SIZE,
    UPPER,
    lift_car_side,
    office_wall,
    office_wall_upper,
    painted_block,
    painted_block_upper,
    register_wall_texture,
    stone_cladding,
    stone_cladding_upper,
    wood_panelling,
)
from doomer.shared.mapfile import EXECUTIVE, LIFT_CAR, LOBBY, OFFICE, SERVICE

MIP_LEVELS = 7  # 64, 32, 16, 8, 4, 2, 1 texels across
# Internal wall types (not in map files, which stop at 35) for the plain-wall
# look of each area: AREA_WALL_TYPE + area.
AREA_WALL_TYPE = 40
# Which sides of which wall types are plain walls, taking the look of the
# area they face: all of `#` (1), a balcony's front (i, 18: it carries on
# the room's walls below it), and the outside of a lift car (k, 20).
AREA_FACES: dict[int, tuple[int, ...]] = {1: ALL_SIDES, 18: ALL_SIDES, 20: (BACK,)}


# --- floors ------------------------------------------------------------------------


def carpet_tiles(seed: int = 30, ramp: int = STEEL, base: float = 6.0) -> U8Array:
    """Carpet tiles, 4 x 4 per cell (~67 cm), laid quarter-turned: the pile
    runs across in one tile and along in the next, so neighbours catch the
    light differently and the floor reads as tiles, not a print."""
    rng = np.random.default_rng(seed)
    v, u = grid()
    tu, tv = u // 16, v // 16
    turned = (tu + tv) % 2
    pile = np.where(turned, (u % 2) * 0.6, (v % 2) * 0.6)
    tone = rng.random((4, 4))[tv, tu] * 0.6 + turned * 0.5
    fleck = np.where(rng.random((TEX_SIZE, TEX_SIZE)) < 0.06, -1.5, 0.0)
    seam = ((u % 16) == 0) | ((v % 16) == 0)
    return indices(ramp, base + pile + tone + fleck + np.where(seam, -1.0, 0.0))


def terrazzo(seed: int = 32) -> U8Array:
    """The lobby: polished terrazzo in big slabs, 2 x 2 per cell."""
    rng = np.random.default_rng(seed)
    v, u = grid()
    chips = rng.random((TEX_SIZE, TEX_SIZE))
    shade = (
        11.0
        + (noise(rng, 8) - 0.5) * 0.8
        + np.where(chips < 0.08, -3.0, 0.0)
        + np.where(chips > 0.95, 2.0, 0.0)
    )
    joint = ((u % 32) == 0) | ((v % 32) == 0)
    return np.where(joint, indices(WARM_GREY, 7.0), indices(WARM_GREY, shade)).astype(np.uint8)


def lino(seed: int = 33) -> U8Array:
    """Service areas: grey flecked vinyl sheet with a welded seam."""
    rng = np.random.default_rng(seed)
    _, u = grid()
    chips = rng.random((TEX_SIZE, TEX_SIZE))
    shade = (
        8.0
        + (noise(rng, 8) - 0.5)
        + np.where(chips < 0.1, -1.5, 0.0)
        + np.where(chips > 0.93, 1.5, 0.0)
    )
    return np.where(u == 0, indices(GREY, 5.0), indices(GREY, shade)).astype(np.uint8)


def studded_rubber(seed: int = 39) -> U8Array:
    """Lift cars: black studded rubber flooring (raised discs, lit on top)."""
    rng = np.random.default_rng(seed)
    v, u = grid()
    shade = 2.5 + (noise(rng, 8) - 0.5) * 0.6
    du, dv = (u % 8) - 3.5, (v % 8) - 3.5
    stud = du * du + dv * dv < 6.0
    lit = stud & (du + dv < -1.0)
    return indices(GREY, np.where(lit, 6.0, np.where(stud, 4.0, shade)))


def blue_carpet(seed: int = 41) -> U8Array:
    """Fitted carpet (balconies, raised floors; the stairs' treads match it):
    1980s corporate blue, a fine pile speckle, no tile seams."""
    rng = np.random.default_rng(seed)
    speckle = rng.random((TEX_SIZE, TEX_SIZE))
    shade = (
        6.0
        + (noise(rng, 8) - 0.5) * 0.6
        + np.where(speckle < 0.1, -1.0, 0.0)
        + np.where(speckle > 0.93, 1.0, 0.0)
    )
    return indices(BLUE, shade)


# Raised surfaces you walk on, by wall type: textured like floors (world-
# mapped, mipmapped) instead of a flat colour. Desk tops stay flat colours.
TOP_TEXTURES: dict[int, Callable[[], U8Array]] = {14: blue_carpet}


# --- ceilings ----------------------------------------------------------------------


def ceiling_tiles(seed: int = 31) -> U8Array:
    """A suspended ceiling: fissured mineral tiles (600 mm, 4 x 4 per cell) in
    a white T-bar grid lined up with the cell's edges. A light panel takes
    the middle 2 x 2 tiles of its cell (render.LIGHT_HALF)."""
    rng = np.random.default_rng(seed)
    v, u = grid()
    tu, tv = u % 16, v % 16
    # 2-texel bars, split evenly between the tiles either side (the tiling
    # rule), so they survive the first mip level instead of averaging away.
    tbar = (tu == 0) | (tu == 15) | (tv == 0) | (tv == 15)
    fissure = rng.random((TEX_SIZE, TEX_SIZE)) < 0.12
    shade = 11.3 + (noise(rng, 16) - 0.5) * 0.6 - fissure * 1.3
    shade = np.where((tu == 1) | (tv == 1), shade - 1.2, shade)  # tile edge in the bar's shadow
    return indices(np.where(tbar, GREY, CREAM), np.where(tbar, 15.0, shade))


def plaster(seed: int = 37) -> U8Array:
    """A smooth painted plaster ceiling (the lobby)."""
    rng = np.random.default_rng(seed)
    return indices(CREAM, 12.6 + (noise(rng, 4) - 0.5) * 0.5 + (noise(rng, 32) - 0.5) * 0.3)


def car_ceiling(seed: int = 40) -> U8Array:
    """A lift car's ceiling: brushed steel panels, perforated (a ventilated
    ceiling), the light panel in the middle."""
    rng = np.random.default_rng(seed)
    v, u = grid()
    shade = 9.5 + (noise(rng, 64)[:1, :] - 0.5) * 0.8
    holes = (u % 4 == 1) & (v % 4 == 1)
    joints = (u % 32 == 0) | (v % 32 == 0)
    return indices(GREY, np.where(joints, 6.0, np.where(holes, 7.5, shade)))


# Soffits: the underside of a ceiling lower than a storey (the lintel over
# a doorway, the tunnel through a thick wall). Not the room's ceiling tiles:
# a lining, per area.


def _border(width: int) -> npt.NDArray[np.bool_]:
    """The texels within `width` of a tile's edge, split evenly between
    opposite edges (the tiling rule), so neighbouring cells make one band."""
    v, u = grid()
    return (u < width) | (u >= TEX_SIZE - width) | (v < width) | (v >= TEX_SIZE - width)


def soffit_plaster(seed: int = 42) -> U8Array:
    """Offices: painted plaster framed by a timber lining where it meets the
    walls (a door frame's head and casings carried round the opening), with
    a shadow line inside it."""
    rng = np.random.default_rng(seed)
    shade = 11.4 + (noise(rng, 8) - 0.5) * 0.4
    tex = indices(CREAM, np.where(_border(6), 9.5, shade))
    wood = indices(WOOD, 8.0 + (noise(rng, 16) - 0.5) * 1.0)
    return np.where(_border(4), wood, tex).astype(np.uint8)


def soffit_wood(seed: int = 43) -> U8Array:
    """The executive suite: timber boards, the grain running along them, in a
    darker moulding where they meet the walls."""
    rng = np.random.default_rng(seed)
    v, u = grid()
    grain = np.sin(v * 0.9 + noise(rng, 8) * 5) * 0.5
    shade = 7.5 + grain + (noise(rng, 16) - 0.5) * 0.8
    shade = np.where(u % 16 == 0, 5.0, shade)
    return indices(WOOD, np.where(_border(4), 4.5, shade))


def soffit_stone(seed: int = 44) -> U8Array:
    """The lobby: pale stone slabs, faintly veined as the walls, in a brushed
    steel edge where they meet the walls."""
    rng = np.random.default_rng(seed)
    v, u = grid()
    veins = np.abs(noise(rng, 4) - noise(rng, 16) * 0.5 - 0.25) < 0.02
    tex = indices(WARM_GREY, np.where(veins, 10.5, 12.5 + (noise(rng, 8) - 0.5) * 0.6))
    tex = np.where((u == 31) | (v == 31), indices(WARM_GREY, 8.5), tex)  # slab joints
    return np.where(_border(3), indices(GREY, 11.0), tex).astype(np.uint8)


def concrete(seed: int = 38) -> U8Array:
    """Bare concrete soffit (service areas), with the marks of the formwork
    boards it was cast against."""
    rng = np.random.default_rng(seed)
    v, _ = grid()
    shade = 9.0 + (noise(rng, 8) - 0.5) * 1.2 + (noise(rng, 32) - 0.5) * 0.8
    shade = np.where(v % 16 == 0, shade - 1.2, shade)
    return indices(GREY, shade)


# --- the areas ---------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class AreaLook:
    floor: Callable[[], U8Array]
    ceiling: Callable[[], U8Array]
    wall: Callable[[], U8Array]  # plain walls facing into the area
    wall_upper: Callable[[], U8Array]  # ... above one storey
    soffit: Callable[[], U8Array]  # ceilings lower than a storey (over doorways)


AREA_LOOKS: dict[int, AreaLook] = {
    OFFICE: AreaLook(carpet_tiles, ceiling_tiles, office_wall, office_wall_upper, soffit_plaster),
    EXECUTIVE: AreaLook(
        lambda: carpet_tiles(36, BLUE, 5.0),
        ceiling_tiles,
        wood_panelling,
        wood_panelling,
        soffit_wood,
    ),
    LOBBY: AreaLook(terrazzo, plaster, stone_cladding, stone_cladding_upper, soffit_stone),
    SERVICE: AreaLook(lino, concrete, painted_block, painted_block_upper, concrete),
    LIFT_CAR: AreaLook(studded_rubber, car_ceiling, lift_car_side, lift_car_side, car_ceiling),
}

for _area, _look in AREA_LOOKS.items():
    register_wall_texture(AREA_WALL_TYPE + _area, _look.wall())
    register_wall_texture(AREA_WALL_TYPE + _area, _look.wall_upper(), sides=(UPPER,))


# Kinds of horizontal surface: columns of Surfaces.table. Then one per
# TOP_TEXTURES entry, from TOP_KIND.
FLOOR, CEILING, SOFFIT = range(3)
TOP_KIND = {wall_type: 3 + k for k, wall_type in enumerate(TOP_TEXTURES)}
TOP_KIND[18] = FLOOR  # a balcony is floored as the area it's in (the lobby's terrazzo)


@dataclass(frozen=True, slots=True)
class Surfaces:
    """Every horizontal surface texture, mipmapped, for the renderer: which
    texture a surface shows is table[area, kind]."""

    mips: U8Array  # (texture, level, v, u): level k averages 2^k x 2^k texels. Read-only.
    table: npt.NDArray[np.int64]  # (area, kind) -> texture

    @property
    def floor(self) -> npt.NDArray[np.int64]:
        return np.ascontiguousarray(self.table[:, FLOOR])

    @property
    def ceiling(self) -> npt.NDArray[np.int64]:
        return np.ascontiguousarray(self.table[:, CEILING])


@functools.cache  # deterministic, ~0.1 s: built once, shared by every Renderer
def build_surfaces() -> Surfaces:
    textures: list[U8Array] = []

    def add(make: Callable[[], U8Array]) -> int:
        textures.append(make())
        return len(textures) - 1

    tops = [add(make) for make in TOP_TEXTURES.values()]  # the same in every area
    table = []
    for area in range(max(AREA_LOOKS) + 1):
        look = AREA_LOOKS.get(area, AREA_LOOKS[OFFICE])
        table.append([add(look.floor), add(look.ceiling), add(look.soffit), *tops])
    mips = np.stack([mipmaps(t) for t in textures])
    return Surfaces(mips, np.array(table, dtype=np.int64))


def mipmaps(texture: U8Array) -> U8Array:
    """(MIP_LEVELS, 64, 64): the texture, then copies averaging 2 x 2, 4 x 4 ...
    texels (in RGB, then the nearest palette colour), each blown back up to
    64 x 64 so every level is looked up the same way."""
    rgb = PALETTE[texture].astype(np.float64)  # (v, u, 3)
    levels = [texture]
    for k in range(1, MIP_LEVELS):
        n = 1 << k
        blocks = rgb.reshape(TEX_SIZE // n, n, TEX_SIZE // n, n, 3).mean(axis=(1, 3))
        small = _nearest_indices(blocks)
        levels.append(np.repeat(np.repeat(small, n, axis=0), n, axis=1))
    return np.stack(levels).astype(np.uint8)


def _nearest_indices(rgb: npt.NDArray[np.float64]) -> U8Array:
    diff = rgb[..., None, :] - PALETTE[None, None, :, :].astype(np.float64)
    result: U8Array = np.argmin((diff * diff).sum(axis=-1), axis=-1).astype(np.uint8)
    return result

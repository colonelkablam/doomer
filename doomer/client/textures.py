"""Procedurally generated 64x64 office wall textures, as 8-bit palette indices.

Why 64x64: a power of two, so coordinates wrap with a bitmask (u & 63), and at
our 320x200 resolution one texel maps to roughly one or more screen pixels.
Bigger textures would be squashed into fewer pixels than they have texels,
which aliases (shimmers) without mipmapping.

Generators draw in (v, u) order; the atlas stores them transposed as [u][v].
Adding a texture: write a generator returning a (64, 64) uint8 index array and
register it against a wall type with `register_wall_texture`. The wall types
are listed in docs/GAME.md.
"""

from collections.abc import Callable

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
    ORANGE,
    SAND,
    STEEL,
    WOOD,
    U8Array,
)
from doomer.client.pixelart import ART_MASK, ART_SIZE, grid, indices, noise

TEX_SIZE = ART_SIZE  # 64
TEX_MASK = ART_MASK  # u & TEX_MASK == u % 64

_WALL_TEXTURES: dict[int, U8Array] = {}


def register_wall_texture(wall_type: int, texture: U8Array) -> None:
    if texture.shape != (TEX_SIZE, TEX_SIZE) or texture.dtype != np.uint8:
        raise ValueError(f"texture must be ({TEX_SIZE}, {TEX_SIZE}) uint8")
    _WALL_TEXTURES[wall_type] = texture


def build_atlas() -> tuple[U8Array, npt.NDArray[np.intp]]:
    """Stack all textures for the renderer.

    Returns (atlas, slot_for_type): atlas[slot, u, v] is a palette index, and
    slot_for_type[wall_type] picks the slot (0 = the missing-texture checker).
    """
    max_type = max(_WALL_TEXTURES, default=0)
    slots = [missing_texture()]
    slot_for_type = np.zeros(max_type + 1, dtype=np.intp)
    for wall_type in sorted(_WALL_TEXTURES):
        slot_for_type[wall_type] = len(slots)
        slots.append(_WALL_TEXTURES[wall_type])
    atlas = np.stack([s.T for s in slots])  # (n, u, v)
    return np.ascontiguousarray(atlas), slot_for_type


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
    frame = (pu < 4) | (pv < 4)
    shade = np.where(frame, 6 + noise(rng, 16), shade)
    shade = np.where(frame & ((pu == 3) | (pv == 3)), 4.0, shade)  # shadow line
    shade = np.where(~frame & ((pu == 4) | (pv == 4)), 9.5, shade)  # highlight
    return indices(WOOD, shade)


def cubicle_partition(seed: int = 3) -> U8Array:
    """Grey-blue fabric partition with an aluminium trim at the top and sides."""
    rng = np.random.default_rng(seed)
    v, u = grid()
    weave = ((u + v) % 2) * 0.5
    shade = 7 + weave + (noise(rng, 32) - 0.5) * 1.5
    tex = indices(STEEL, shade)
    trim = (v < 4) | ((u % 32) < 2)
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
    border = (pu < 2) | (pv < 2)
    highlight = (pu == 2) | (pv == 2)
    rivet = ((pu - 5) % 22 < 2) & ((pv - 5) % 22 < 2) & ~border
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

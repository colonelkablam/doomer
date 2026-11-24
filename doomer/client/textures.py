"""Procedurally generated 64x64 wall textures, as 8-bit palette indices.

Why 64x64: a power of two, so coordinates wrap with a bitmask (u & 63), and at
our 320x200 resolution one texel maps to roughly one or more screen pixels.
Bigger textures would be squashed into fewer pixels than they have texels,
which aliases (shimmers) without mipmapping.

Generators draw in (v, u) = (row, column) order because that's how humans
think about images; the atlas stores them transposed as [u][v] so the renderer
reads one wall column as one contiguous row of memory.

Adding a texture: write a generator returning a (64, 64) uint8 index array and
register it against a wall type with `register_wall_texture`.
"""

from collections.abc import Callable

import numpy as np
import numpy.typing as npt

from doomer.client.palette import (
    BLUE,
    BRICK,
    CREAM,
    GREY,
    MAGENTA,
    MOSS,
    ORANGE,
    PURPLE,
    SHADES,
    STEEL,
    TEAL,
    WOOD,
    U8Array,
)

TEX_BITS = 6
TEX_SIZE = 1 << TEX_BITS  # 64
TEX_MASK = TEX_SIZE - 1  # 0b111111: u & TEX_MASK == u % 64

FloatArray = npt.NDArray[np.float64]

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


# --- helpers ---------------------------------------------------------------------


def _indices(ramp: int | npt.NDArray[np.int64], shade: FloatArray) -> U8Array:
    shade_i = np.clip(np.round(shade), 0, SHADES - 1).astype(np.int64)
    return ((np.asarray(ramp, dtype=np.int64) << 4) | shade_i).astype(np.uint8)


def _noise(rng: np.random.Generator, cells: int) -> FloatArray:
    """Tileable smooth value noise in [0, 1): bilinear interpolation of a random grid."""
    grid = rng.random((cells, cells))
    pos = np.arange(TEX_SIZE) * cells / TEX_SIZE
    i0 = pos.astype(np.int64)
    i1 = (i0 + 1) % cells  # wrap, so the texture tiles seamlessly
    t = pos - i0
    rows = grid[i0] * (1 - t)[:, None] + grid[i1] * t[:, None]
    out: FloatArray = rows[:, i0] * (1 - t)[None, :] + rows[:, i1] * t[None, :]
    return out


def _grid() -> tuple[npt.NDArray[np.int64], npt.NDArray[np.int64]]:
    v, u = np.mgrid[0:TEX_SIZE, 0:TEX_SIZE]
    return v.astype(np.int64), u.astype(np.int64)


def _blocks(
    block_w: int, block_h: int, rng: np.random.Generator
) -> tuple[npt.NDArray[np.bool_], FloatArray, FloatArray]:
    """Running-bond blocks (each row offset by half a block).

    Returns (mortar mask, per-block random tone in [0, 1), bevel: +1 on top/left
    edges, -1 on bottom/right edges, 0 elsewhere).
    """
    v, u = _grid()
    row = v // block_h
    shifted = (u + (row % 2) * (block_w // 2)) % TEX_SIZE
    col = shifted // block_w
    bu, bv = shifted % block_w, v % block_h
    mortar = (bu == 0) | (bv == 0)
    tones = rng.random((TEX_SIZE // block_h, TEX_SIZE // block_w))
    tone = tones[row, col]
    bevel = np.where((bv == 1) | (bu == 1), 1.0, 0.0) - np.where(
        (bv == block_h - 1) | (bu == block_w - 1), 1.0, 0.0
    )
    return mortar, tone, bevel


# --- generators --------------------------------------------------------------------


def missing_texture() -> U8Array:
    v, u = _grid()
    checker = ((u >> 3) ^ (v >> 3)) & 1
    return _indices(MAGENTA, np.where(checker == 1, 15.0, 3.0))


def stone(ramp: int = GREY, seed: int = 1) -> U8Array:
    rng = np.random.default_rng(seed)
    mortar, tone, bevel = _blocks(32, 16, rng)
    shade = 8 + tone * 3 + bevel * 2 + (_noise(rng, 16) - 0.5) * 3
    return _indices(ramp, np.where(mortar, 3.0, shade))


def brick(ramp: int = BRICK, seed: int = 2) -> U8Array:
    rng = np.random.default_rng(seed)
    mortar, tone, bevel = _blocks(16, 8, rng)
    shade = 8 + tone * 4 + bevel + (_noise(rng, 32) - 0.5) * 2
    ramps = np.where(mortar, GREY, ramp)
    return _indices(ramps, np.where(mortar, 6.0, shade))


def metal(ramp: int = STEEL, seed: int = 3) -> U8Array:
    rng = np.random.default_rng(seed)
    v, u = _grid()
    pu, pv = u % 32, v % 32
    border = (pu < 2) | (pv < 2)
    highlight = (pu == 2) | (pv == 2)
    rivet = ((pu - 5) % 22 < 2) & ((pv - 5) % 22 < 2) & ~border
    streaks = _noise(rng, 32)[0][None, :]  # vary across u only: vertical brushed streaks
    shade = 8 + streaks * 3 + (_noise(rng, 8) - 0.5) * 2
    shade = np.where(highlight, 12.0, shade)
    shade = np.where(rivet, 14.0, shade)
    return _indices(ramp, np.where(border, 4.0, shade))


def mossy_stone(seed: int = 4) -> U8Array:
    rng = np.random.default_rng(seed)
    base = stone(GREY, seed)
    moss = _noise(rng, 4) * 0.7 + _noise(rng, 16) * 0.3
    v, _ = _grid()
    moss = moss + v / TEX_SIZE * 0.3  # moss gathers towards the bottom
    shade = 6 + _noise(rng, 32) * 5
    return np.where(moss > 0.8, _indices(MOSS, shade), base).astype(np.uint8)


def wood(ramp: int = WOOD, seed: int = 5) -> U8Array:
    rng = np.random.default_rng(seed)
    v, u = _grid()
    plank = u // 16
    gap = (u % 16) == 0
    offsets = rng.random(TEX_SIZE // 16) * 40
    grain = np.sin((u % 16) * 0.9 + _noise(rng, 8) * 6 + offsets[plank])
    shade = 8 + grain * 1.5 + rng.random(TEX_SIZE // 16)[plank] * 2
    nail = ((v % 32) == 4) & ((u % 16) == 8)
    shade = np.where(nail, 2.0, shade)
    return _indices(ramp, np.where(gap, 2.0, shade))


def tech(seed: int = 6) -> U8Array:
    base = metal(STEEL, seed)
    v, u = _grid()
    strip = (v >= 28) & (v < 36)
    glow = 15 - np.abs(v - 31.5) * 1.5
    lights = (u % 8) < 6
    return np.where(
        strip, np.where(lights, _indices(TEAL, glow), _indices(CREAM, np.full(v.shape, 4.0))), base
    ).astype(np.uint8)


def tiles(ramp: int = BLUE, seed: int = 7) -> U8Array:
    rng = np.random.default_rng(seed)
    v, u = _grid()
    grout = ((u % 16) == 0) | ((v % 16) == 0)
    tone = rng.random((4, 4))[v // 16, u // 16]
    shade = 9 + tone * 3 + (_noise(rng, 16) - 0.5) * 2
    shade = np.where(((u % 16) == 1) | ((v % 16) == 1), shade + 2, shade)
    return _indices(np.where(grout, GREY, ramp), np.where(grout, 10.0, shade))


DEFAULT_WALL_TEXTURES: dict[int, Callable[[], U8Array]] = {
    1: stone,
    2: brick,
    3: metal,
    4: mossy_stone,
    5: wood,
    6: tech,
    7: tiles,
    8: lambda: brick(ORANGE, seed=8),
    9: lambda: stone(PURPLE, seed=9),
}

for _wall_type, _make in DEFAULT_WALL_TEXTURES.items():
    register_wall_texture(_wall_type, _make())

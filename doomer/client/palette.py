"""The 256-colour palette and the light colormap.

Every colour on screen is one byte: an index into this palette. The byte is
laid out as two nibbles:

    index = (ramp << 4) | shade      ramp 0-15 = hue, shade 0-15 = brightness

Shade 15 is the ramp's full colour; each step down is 1/16 darker, so shade
s has brightness (s + 1) / 16. Because brightness is linear in the low nibble,
darkening a colour by a factor f never changes its ramp, and the darkened
index can be computed exactly, with no nearest-colour search.

The colormap (Doom's COLORMAP) precomputes that for every light level:
colormap[level, index] -> darkened index. Shading a whole frame is then one
numpy fancy-index: colormap[levels, texels].
"""

import numpy as np
import numpy.typing as npt

RAMPS = 16
SHADES = 16
LIGHT_LEVELS = 32  # level 0 = full brightness, LIGHT_LEVELS - 1 = black

U8Array = npt.NDArray[np.uint8]

# Full-brightness colour of each ramp. Order is part of the format: textures
# refer to ramps by number, so append rather than reorder.
RAMP_COLOURS: tuple[tuple[int, int, int], ...] = (
    (240, 240, 240),  # 0  neutral grey
    (230, 90, 70),  # 1  brick red
    (250, 150, 60),  # 2  orange
    (240, 210, 90),  # 3  ochre / yellow
    (190, 130, 80),  # 4  wood brown
    (110, 200, 90),  # 5  green
    (150, 170, 90),  # 6  moss / olive
    (80, 210, 200),  # 7  teal
    (90, 140, 240),  # 8  blue
    (160, 180, 210),  # 9  steel
    (180, 110, 220),  # 10 purple
    (250, 130, 190),  # 11 pink
    (230, 190, 160),  # 12 skin / sand
    (200, 180, 160),  # 13 warm grey
    (255, 250, 220),  # 14 cream (lights)
    (255, 0, 255),  # 15 debug magenta
)
assert len(RAMP_COLOURS) == RAMPS

GREY, BRICK, ORANGE, OCHRE, WOOD, GREEN, MOSS, TEAL = range(8)
BLUE, STEEL, PURPLE, PINK, SAND, WARM_GREY, CREAM, MAGENTA = range(8, 16)


def colour_index(ramp: int, shade: int) -> int:
    return (ramp << 4) | shade


def build_palette() -> U8Array:
    """(256, 3) uint8 RGB."""
    base = np.array(RAMP_COLOURS, dtype=np.float64)  # (16, 3)
    brightness = (np.arange(SHADES) + 1) / SHADES  # (16,)
    rgb = base[:, None, :] * brightness[None, :, None]  # (ramp, shade, 3)
    return np.round(rgb).reshape(RAMPS * SHADES, 3).astype(np.uint8)


def build_colormap(levels: int = LIGHT_LEVELS) -> U8Array:
    """(levels, 256) uint8: colormap[level, index] is index darkened to that level."""
    index = np.arange(RAMPS * SHADES)
    ramp, shade = index >> 4, index & 0x0F
    factor = 1.0 - np.arange(levels) / (levels - 1)  # 1.0 .. 0.0
    # Brightness (s + 1) / 16 scaled by f is (s' + 1) / 16 with s' = (s + 1) f - 1.
    exact = np.round((shade[None, :] + 1) * factor[:, None]) - 1
    new_shade = np.clip(exact, 0, SHADES - 1).astype(np.int64)
    result: U8Array = ((ramp[None, :] << 4) | new_shade).astype(np.uint8)
    return result


def light_level(
    factor: npt.NDArray[np.float64], levels: int = LIGHT_LEVELS
) -> npt.NDArray[np.intp]:
    """Map brightness factors in [0, 1] to colormap levels."""
    return np.clip(np.round((1.0 - factor) * (levels - 1)), 0, levels - 1).astype(np.intp)


def nearest_index(palette: U8Array, rgb: tuple[int, ...]) -> int:
    diff = palette.astype(np.int64) - np.array(rgb[:3], dtype=np.int64)
    return int(np.argmin((diff * diff).sum(axis=1)))


PALETTE = build_palette()
COLORMAP = build_colormap()

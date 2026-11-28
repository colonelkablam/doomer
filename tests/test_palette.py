import numpy as np

from ok_doomer.client.palette import (
    BRICK,
    COLORMAP,
    LIGHT_LEVELS,
    PALETTE,
    RAMP_COLOURS,
    colour_index,
    light_level,
    nearest_index,
)


def test_palette_layout_is_ramp_high_nibble_shade_low_nibble() -> None:
    assert PALETTE.shape == (256, 3) and PALETTE.dtype == np.uint8
    assert tuple(PALETTE[colour_index(BRICK, 15)]) == RAMP_COLOURS[BRICK]
    # Brightness halves from shade 15 to shade 7: (7 + 1) / 16 = 0.5.
    assert tuple(PALETTE[colour_index(BRICK, 7)]) == tuple(
        round(c / 2) for c in RAMP_COLOURS[BRICK]
    )


def test_colormap_level_zero_is_identity_and_last_level_is_darkest() -> None:
    assert COLORMAP.shape == (LIGHT_LEVELS, 256)
    assert np.array_equal(COLORMAP[0], np.arange(256))
    assert np.all((COLORMAP[-1] & 0x0F) == 0)


def test_colormap_never_changes_hue_and_never_brightens() -> None:
    index = np.arange(256)
    assert np.all((index >> 4) == (COLORMAP >> 4))
    shades = COLORMAP & 0x0F
    assert np.all(np.diff(shades.astype(int), axis=0) <= 0)


def test_light_level_mapping() -> None:
    levels = light_level(np.array([1.0, 0.5, 0.0]))
    assert list(levels) == [0, round(0.5 * (LIGHT_LEVELS - 1)), LIGHT_LEVELS - 1]


def test_nearest_index_finds_exact_entries() -> None:
    for i in (0, 37, 128, 255):
        assert nearest_index(PALETTE, tuple(int(c) for c in PALETTE[i])) in {
            j for j in range(256) if tuple(PALETTE[j]) == tuple(PALETTE[i])
        }

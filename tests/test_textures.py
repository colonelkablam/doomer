import numpy as np
import pytest

from ok_doomer.client.palette import MAGENTA
from ok_doomer.client.textures import (
    DEFAULT_WALL_TEXTURES,
    TEX_MASK,
    TEX_SIZE,
    build_atlas,
    register_wall_texture,
)


def test_size_is_a_power_of_two_so_masking_equals_modulo() -> None:
    assert TEX_SIZE & (TEX_SIZE - 1) == 0
    u = np.arange(-200, 200)
    assert np.array_equal(u & TEX_MASK, u % TEX_SIZE)


@pytest.mark.parametrize("wall_type", sorted(DEFAULT_WALL_TEXTURES))
def test_textures_are_64x64_uint8_deterministic_and_varied(wall_type: int) -> None:
    make = DEFAULT_WALL_TEXTURES[wall_type]
    tex = make()
    assert tex.shape == (TEX_SIZE, TEX_SIZE) and tex.dtype == np.uint8
    assert np.array_equal(tex, make())  # seeded, so identical every run
    assert len(np.unique(tex)) >= 4  # actually has detail
    assert not np.any((tex >> 4) == MAGENTA)  # debug ramp reserved for missing textures


def test_atlas_is_transposed_and_slot_zero_is_missing_texture() -> None:
    atlas, slot_table = build_atlas()
    assert atlas.shape[1:] == (TEX_SIZE, TEX_SIZE)
    tex = DEFAULT_WALL_TEXTURES[2]()
    # atlas is [slot, u, v]; generators draw [v, u].
    assert np.array_equal(atlas[slot_table[2, 0]], tex.T)
    assert (slot_table[2] == slot_table[2, 0]).all()  # one texture on every side
    assert np.all((atlas[0] >> 4) == MAGENTA)


def test_register_rejects_wrong_shape() -> None:
    with pytest.raises(ValueError):
        register_wall_texture(99, np.zeros((32, 32), dtype=np.uint8))

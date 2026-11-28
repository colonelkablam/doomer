"""The game's icon: a valid .ico at every size Windows uses, each a PNG of
the office worker; the one in the repository is what the generator makes."""

import io
import struct
from pathlib import Path

import pygame

from ok_doomer.client.icon import DEFAULT_OUT, SIZES, ico_bytes, portrait


def _entries(data: bytes) -> list[tuple[int, bytes]]:
    reserved, kind, count = struct.unpack_from("<HHH", data)
    assert (reserved, kind) == (0, 1)
    out = []
    for k in range(count):
        side, _, _, _, planes, bits, size, offset = struct.unpack_from(
            "<BBBBHHII", data, 6 + 16 * k
        )
        assert (planes, bits) == (1, 32)
        out.append((side or 256, data[offset : offset + size]))
    return out


def test_an_ico_at_every_size_each_a_png_of_that_size() -> None:
    pygame.init()
    try:
        entries = _entries(ico_bytes(portrait()))
        assert [side for side, _ in entries] == list(SIZES)
        for side, png in entries:
            assert png.startswith(b"\x89PNG")
            image = pygame.image.load(io.BytesIO(png), "icon.png")
            assert image.get_size() == (side, side)
    finally:
        pygame.quit()


def test_the_committed_icon_is_the_generators() -> None:
    """Remade if the sprite changes: python -m ok_doomer.client.icon."""
    pygame.init()
    try:
        made = ico_bytes(portrait())
    finally:
        pygame.quit()
    committed = Path(DEFAULT_OUT).read_bytes()
    assert [s for s, _ in _entries(committed)] == [s for s, _ in _entries(made)]

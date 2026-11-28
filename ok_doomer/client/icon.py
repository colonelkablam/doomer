"""The game's icon: the office worker's head and shoulders (the player
sprite, client/sprites.py) on the menu's navy, as a Windows .ico.

    python -m ok_doomer.client.icon [--out ok-doomer.ico]

An .ico holds the picture at several sizes and Windows picks the one it
needs (16 in lists, 32 and 48 on the desktop, 256 in big views). Each is
stored as a PNG inside the .ico (Windows Vista on reads them). Sizes below
the sprite's own are scaled down smoothly (nearest-neighbour would drop
whole rows of a 26-pixel face); the rest up, nearest-neighbour, keeping the
pixel art's edges. play.bat makes the desktop shortcut that shows it.
"""

import argparse
import io
import struct
import sys
from pathlib import Path

import numpy as np
import pygame

from ok_doomer.client.palette import BLUE, PALETTE, colour_index
from ok_doomer.client.sprites import PLAYER_SPRITE, SPRITES

SIZES = (16, 32, 48, 64, 128, 256)
SHOULDERS = 18  # rows of the sprite, from the top of his hair down
MARGIN = 2  # pixels of navy round him, at the sprite's own size
DEFAULT_OUT = Path(__file__).resolve().parents[2] / "ok-doomer.ico"


def portrait() -> pygame.Surface:
    """Head and shoulders, square, on navy, at the sprite's own size."""
    sprite = SPRITES[PLAYER_SPRITE]
    index, opaque = sprite.index[0], sprite.opaque[0]  # from the front: [u][v]
    cols = np.nonzero(opaque.any(axis=1))[0]
    rows = np.nonzero(opaque.any(axis=0))[0]
    top = rows[0]
    index = index[cols[0] : cols[-1] + 1, top : top + SHOULDERS]
    opaque = opaque[cols[0] : cols[-1] + 1, top : top + SHOULDERS]
    w, h = index.shape
    n = max(w, h) + 2 * MARGIN
    rgb = np.empty((n, n, 3), dtype=np.uint8)
    rgb[:] = PALETTE[colour_index(BLUE, 1)]
    left, upper = (n - w) // 2, n - h  # he stands on the bottom edge
    picture = PALETTE[index]
    region = rgb[left : left + w, upper : upper + h]
    region[opaque] = picture[opaque]
    return pygame.surfarray.make_surface(rgb)


def sized(picture: pygame.Surface, size: int) -> pygame.Surface:
    if size < picture.get_width():
        return pygame.transform.smoothscale(picture.convert(32), (size, size))
    return pygame.transform.scale(picture, (size, size))


def ico_bytes(picture: pygame.Surface, sizes: tuple[int, ...] = SIZES) -> bytes:
    """An .ico of `picture` at each of `sizes`, each a PNG."""
    pngs = []
    for size in sizes:
        buffer = io.BytesIO()
        pygame.image.save(sized(picture, size), buffer, "icon.png")
        pngs.append(buffer.getvalue())
    header = struct.pack("<HHH", 0, 1, len(sizes))  # reserved, type 1 (icon), count
    offset = len(header) + 16 * len(sizes)
    entries = b""
    for size, png in zip(sizes, pngs, strict=True):
        side = 0 if size >= 256 else size  # 0 means 256
        entries += struct.pack("<BBBBHHII", side, side, 0, 0, 1, 32, len(png), offset)
        offset += len(png)
    return header + entries + b"".join(pngs)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m ok_doomer.client.icon", description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args(argv)
    pygame.init()
    try:
        args.out.write_bytes(ico_bytes(portrait()))
    finally:
        pygame.quit()
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Draw a map's baked light map from above, for tuning the lighting.

python -m ok_doomer.client.lightmap_png maps/office.map --out out/light.png

Grey levels are light values (black 0, white 1 or more: where pools overlap
past 1, they show as white, then pale yellow); full walls are dark red, low
blocks are outlined, panels are marked in yellow.
"""

import argparse
import logging
from pathlib import Path

import numpy as np
import numpy.typing as npt

from ok_doomer.client.headless import save_png, use_dummy_video

use_dummy_video()

import pygame  # noqa: E402

from ok_doomer.client.lighting import LightMap, bake_light_map  # noqa: E402
from ok_doomer.shared.mapfile import GameMap, load_map  # noqa: E402

log = logging.getLogger(__name__)

WALL = (90, 20, 20)
PANEL = (255, 220, 40)
BLOCK_EDGE = (60, 110, 160)


def light_image(game_map: GameMap, light: LightMap, pixels_per_sample: int = 4) -> pygame.Surface:
    s = light.samples_per_cell
    v = light.values
    grey = np.clip(v, 0.0, 1.0) * 255
    over = np.clip(v - 1.0, 0.0, 1.0)  # above 1: tint towards yellow
    rgb = np.stack([grey, grey, grey * (1 - 0.6 * over)], axis=-1)  # [row][col][3]
    cells: npt.NDArray[np.int64] = np.indices((game_map.height, game_map.width)).transpose(1, 2, 0)
    for y, x in cells.reshape(-1, 2):
        rows, cols = slice(y * s, (y + 1) * s), slice(x * s, (x + 1) * s)
        if game_map.is_full_wall(x, y):
            rgb[rows, cols] = WALL
        elif game_map.is_low_block(x, y):
            rgb[rows, cols][0, :] = rgb[rows, cols][-1, :] = BLOCK_EDGE
            rgb[rows, cols][:, 0] = rgb[rows, cols][:, -1] = BLOCK_EDGE
    for patch in light.patches:
        r, c = int(patch.panel.y * s), int(patch.panel.x * s)
        rgb[r - 1 : r + 1, c - 1 : c + 1] = PANEL
    big = np.repeat(np.repeat(rgb, pixels_per_sample, axis=0), pixels_per_sample, axis=1)
    return pygame.surfarray.make_surface(big.transpose(1, 0, 2).astype(np.uint8))


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(prog="python -m ok_doomer.client.lightmap_png")
    parser.add_argument("map", type=Path)
    parser.add_argument("--out", type=Path, default=Path("out/light.png"))
    args = parser.parse_args(argv)
    game_map = load_map(args.map)
    light = bake_light_map(game_map)
    path = save_png(light_image(game_map, light), args.out)
    low, high = float(np.min(light.values)), float(np.max(light.values))
    log.info("%s (light %.2f..%.2f)", path, low, high)


if __name__ == "__main__":
    main()

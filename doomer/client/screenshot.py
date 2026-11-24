"""Render one frame to PNG without a display, for checking the renderer.

python -m doomer.client.screenshot maps/test.map --at 1.5,1.5,45 --out out/frame.png
"""

import argparse
import math
from pathlib import Path

from doomer.client.headless import save_png, use_dummy_video

use_dummy_video()

import pygame  # noqa: E402

from doomer.client.config import RenderConfig  # noqa: E402
from doomer.client.render import Camera, Renderer, make_frame_surface  # noqa: E402
from doomer.shared.mapfile import GameMap, load_map  # noqa: E402


def render_frame(
    game_map: GameMap, camera: Camera, config: RenderConfig | None = None
) -> pygame.Surface:
    config = config or RenderConfig()
    frame = make_frame_surface(config)
    Renderer(config).render(frame, game_map, camera)
    return frame


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="python -m doomer.client.screenshot")
    parser.add_argument("map", type=Path)
    parser.add_argument("--at", help="x,y,angle_degrees (default: first spawn point)")
    parser.add_argument("--out", type=Path, default=Path("out/frame.png"))
    parser.add_argument("--scale", type=int, default=2)
    args = parser.parse_args(argv)

    game_map = load_map(args.map)
    if args.at:
        x, y, deg = (float(v) for v in args.at.split(","))
        camera = Camera(x, y, math.radians(deg))
    else:
        t = game_map.things[0]
        camera = Camera(t.x, t.y, t.angle)
    print(save_png(render_frame(game_map, camera), args.out, scale=args.scale))


if __name__ == "__main__":
    main()

"""Render one frame to PNG without a display, for checking the renderer.

python -m doomer.client.screenshot maps/test.map --at 1.5,1.5,45 --out out/frame.png
"""

import argparse
import math
from pathlib import Path

from doomer.client.headless import save_png, use_dummy_video

use_dummy_video()

import pygame  # noqa: E402

from doomer.client.boxes import door_boxes  # noqa: E402
from doomer.client.config import LightingConfig, RenderConfig  # noqa: E402
from doomer.client.render import (  # noqa: E402
    Camera,
    Renderer,
    make_frame_surface,
    sprites_from_things,
)
from doomer.shared.doors import Door, doors_from_map  # noqa: E402
from doomer.shared.mapfile import GameMap, load_map  # noqa: E402


def render_frame(
    game_map: GameMap,
    camera: Camera,
    config: RenderConfig | None = None,
    things: bool = True,
    lighting: LightingConfig | None = None,
    doors: list[Door] | None = None,
    passes: frozenset[str] = frozenset(),
) -> pygame.Surface:
    """Render one frame; `things=False` draws walls, floor and ceiling only;
    `lighting` lights it from the map's ceiling panels; `doors` in a given
    state (default: the map's, shut)."""
    config = config or RenderConfig()
    frame = make_frame_surface(config)
    sprites = sprites_from_things(game_map) if things else []
    shown = doors if doors is not None else doors_from_map(game_map)
    boxes = door_boxes(shown, passes) if things else []
    Renderer(config, lighting).render(frame, game_map, camera, sprites, boxes)
    return frame


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="python -m doomer.client.screenshot")
    parser.add_argument("map", type=Path)
    parser.add_argument("--at", help="x,y,angle_degrees (default: first spawn point)")
    parser.add_argument("--out", type=Path, default=Path("out/frame.png"))
    parser.add_argument("--scale", type=int, default=2)
    parser.add_argument("--no-lighting", action="store_true", help="distance fade only")
    args = parser.parse_args(argv)

    game_map = load_map(args.map)
    if args.at:
        x, y, deg = (float(v) for v in args.at.split(","))
        camera = Camera(x, y, math.radians(deg))
    else:
        t = game_map.things[0]
        camera = Camera(t.x, t.y, t.angle)
    lighting = None if args.no_lighting else LightingConfig()
    print(save_png(render_frame(game_map, camera, lighting=lighting), args.out, scale=args.scale))


if __name__ == "__main__":
    main()

"""First-person view: one DDA ray per screen column, flat-shaded walls.

Projection: for screen column c, the ray angle is the view angle plus
atan(cam_x * tan(fov/2)), where cam_x runs from -1 (left edge) to +1 (right).
Using atan (not equal angle steps) keeps straight walls straight.

Fisheye: the ray's Euclidean distance is multiplied by cos(ray offset) to get
the perpendicular distance to the camera plane. Without that, a flat wall in
front of you bulges towards the middle of the screen.
"""

import math
from dataclasses import dataclass

import pygame

from doomer.client.config import RenderConfig
from doomer.client.palette import UNKNOWN_WALL, WALL_COLOURS
from doomer.shared.mapfile import GameMap
from doomer.shared.raycast import Side, cast_ray

MIN_SHADE = 0.25
Y_SIDE_SHADE = 0.72  # north/south faces darker than east/west, so corners read clearly


@dataclass(frozen=True, slots=True)
class Camera:
    x: float
    y: float
    angle: float


class Renderer:
    def __init__(self, config: RenderConfig) -> None:
        self.config = config
        w = config.width
        tan_half_fov = math.tan(math.radians(config.fov_degrees) / 2)
        # Distance from the eye to a virtual screen, in pixels.
        self.projection = (w / 2) / tan_half_fov
        self._offsets = [math.atan((2 * (c + 0.5) / w - 1) * tan_half_fov) for c in range(w)]
        self._cos = [math.cos(o) for o in self._offsets]
        # Perpendicular wall distance per column; sprites (milestone 2) clip against it.
        self.depth: list[float] = [math.inf] * w

    def render(self, target: pygame.Surface, game_map: GameMap, camera: Camera) -> None:
        cfg = self.config
        w, h = cfg.width, cfg.height
        horizon = h // 2
        target.fill(cfg.ceiling_colour, (0, 0, w, horizon))
        target.fill(cfg.floor_colour, (0, horizon, w, h - horizon))

        for col in range(w):
            hits = cast_ray(game_map, camera.x, camera.y, camera.angle + self._offsets[col])
            if not hits:
                self.depth[col] = math.inf
                continue
            hit = hits[-1]
            perp = max(hit.distance * self._cos[col], 1e-6)
            self.depth[col] = perp
            line = self.projection / perp
            top = max(0, round(horizon - line / 2))
            bottom = min(h, round(horizon + line / 2))
            if bottom > top:
                colour = _shade(WALL_COLOURS.get(hit.wall_type, UNKNOWN_WALL), perp, hit.side, cfg)
                target.fill(colour, (col, top, 1, bottom - top))


def _shade(
    colour: tuple[int, int, int], distance: float, side: Side, cfg: RenderConfig
) -> tuple[int, int, int]:
    f = max(MIN_SHADE, 1.0 - distance / cfg.shade_distance)
    if side is Side.Y:
        f *= Y_SIDE_SHADE
    r, g, b = colour
    return int(r * f), int(g * f), int(b * f)

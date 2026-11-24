"""First-person view: one DDA ray per screen column, then one numpy pass that
textures and shades the whole frame as 8-bit palette indices.

Projection: for screen column c, the ray angle is the view angle plus
atan(cam_x * tan(fov/2)), where cam_x runs from -1 (left edge) to +1 (right).
Using atan (not equal angle steps) keeps straight walls straight.

Fisheye: the ray's Euclidean distance is multiplied by cos(ray offset) to get
the perpendicular distance to the camera plane. Without that, a flat wall in
front of you bulges towards the middle of the screen.

Sprites (things) are drawn after the walls, farthest first (the painter's
algorithm), and only in columns where they are nearer than the wall: the
per-column wall depth from the ray pass doubles as a 1D depth buffer.

Why split it in two: the DDA is inherently a loop (each ray takes a different
number of steps), so it stays in Python, 320 iterations a frame. Everything
per-pixel (64,000 of them) is done as whole-array numpy operations, which run
in C. A per-pixel Python loop would be ~100x too slow.
"""

import math
from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import pygame

from doomer.client.config import RenderConfig
from doomer.client.palette import COLORMAP, PALETTE, U8Array, light_level, nearest_index
from doomer.client.sprites import SPRITES
from doomer.client.textures import TEX_MASK, TEX_SIZE, build_atlas
from doomer.shared.mapfile import GameMap
from doomer.shared.raycast import Side, cast_ray

MIN_SHADE = 0.25
Y_SIDE_SHADE = 0.72  # north/south faces darker than east/west, so corners read clearly
EYE_HEIGHT = 0.5  # in wall heights; used to work out how far away each floor row is
# Bayer matrix: each value 0-15 appears once, arranged so that any threshold
# switches on pixels that are spread as evenly as possible.
BAYER_4X4 = np.array([[0, 8, 2, 10], [12, 4, 14, 6], [3, 11, 1, 9], [15, 7, 13, 5]])

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.intp]


NEAR_PLANE = 0.05  # sprites closer than this (in cells) aren't drawn


@dataclass(frozen=True, slots=True)
class Camera:
    x: float
    y: float
    angle: float


@dataclass(frozen=True, slots=True)
class SpriteInstance:
    """Something to draw this frame: a sprite name (= thing type) at a position."""

    name: str
    x: float
    y: float


def make_frame_surface(config: RenderConfig) -> pygame.Surface:
    """An 8-bit surface carrying the game palette, for Renderer.render to fill."""
    surface = pygame.Surface((config.width, config.height), depth=8)
    surface.set_palette([tuple(int(c) for c in rgb) for rgb in PALETTE])
    return surface


class Renderer:
    def __init__(self, config: RenderConfig) -> None:
        self.config = config
        w, h = config.width, config.height
        self.horizon = h // 2
        tan_half_fov = math.tan(math.radians(config.fov_degrees) / 2)
        # Distance from the eye to a virtual screen, in pixels.
        self.projection = (w / 2) / tan_half_fov
        self._offsets = [math.atan((2 * (c + 0.5) / w - 1) * tan_half_fov) for c in range(w)]
        self._cos = np.cos(np.array(self._offsets))
        self._pixel_y = np.arange(h, dtype=np.float64) + 0.5  # pixel centres
        self.atlas, self.slot_for_type = build_atlas()
        self.background = self._build_background()
        # Perpendicular wall distance per column; sprites (milestone 2) clip against it.
        self.depth: FloatArray = np.full(w, np.inf)

    def _build_background(self) -> U8Array:
        """Ceiling and floor, darkened with distance. (w, h) palette indices.

        A floor row y pixels below the horizon is where a ray from eye height
        EYE_HEIGHT meets the floor: distance = EYE_HEIGHT * projection / y.
        Same for the ceiling above. Fixed until the horizon can move (look up/down).
        """
        cfg = self.config
        offset = np.abs(self._pixel_y - self.horizon)
        distance = EYE_HEIGHT * self.projection / offset
        ceiling = nearest_index(PALETTE, cfg.ceiling_colour)
        floor = nearest_index(PALETTE, cfg.floor_colour)
        base = np.where(self._pixel_y < self.horizon, ceiling, floor)  # (h,)

        # A dark base colour has only a few shades below it, so a smooth gradient
        # would show hard bands. Ordered dithering: compute the exact (fractional)
        # shade, then round up or down per pixel by comparing the fraction against
        # a tiled 4x4 Bayer threshold matrix. The eye averages the pattern.
        # Shade s has brightness (s + 1) / 16, so scaling by f gives (s + 1) f - 1.
        exact = ((base & 0x0F) + 1) * self._brightness(distance) - 1  # (h,)
        x, y = np.meshgrid(np.arange(cfg.width), np.arange(cfg.height), indexing="ij")
        threshold = (BAYER_4X4[x & 3, y & 3] + 0.5) / 16  # (w, h) in (0, 1)
        shade = np.floor(exact[None, :] + threshold).clip(0, 15).astype(np.intp)
        background: U8Array = ((base[None, :] & 0xF0) | shade).astype(np.uint8)
        return background

    def _brightness(self, distance: FloatArray) -> FloatArray:
        result: FloatArray = np.maximum(MIN_SHADE, 1.0 - distance / self.config.shade_distance)
        return result

    def render(
        self,
        target: pygame.Surface,
        game_map: GameMap,
        camera: Camera,
        sprites: Iterable[SpriteInstance] = (),
    ) -> None:
        """Draw into an 8-bit surface from make_frame_surface()."""
        pygame.surfarray.blit_array(target, self.render_indices(game_map, camera, sprites))

    def render_indices(
        self, game_map: GameMap, camera: Camera, sprites: Iterable[SpriteInstance] = ()
    ) -> U8Array:
        """The frame as a (width, height) array of palette indices."""
        w = self.config.width

        # 1. One ray per column (Python loop).
        dist = [math.inf] * w
        wall = [0] * w
        tex_u = [0.0] * w
        y_side = [False] * w
        for col in range(w):
            hits = cast_ray(game_map, camera.x, camera.y, camera.angle + self._offsets[col])
            if hits:
                hit = hits[-1]
                dist[col] = hit.distance
                wall[col] = hit.wall_type
                tex_u[col] = hit.u
                y_side[col] = hit.side is Side.Y

        # 2. Per-column values as arrays.
        perp = np.array(dist) * self._cos
        self.depth = perp
        has_wall = np.isfinite(perp)
        types = np.array(wall, dtype=np.intp)
        types[types >= len(self.slot_for_type)] = 0  # unknown type -> missing texture
        slot = self.slot_for_type[types]
        u = (np.array(tex_u) * TEX_SIZE).astype(np.intp) & TEX_MASK
        brightness = self._brightness(perp) * np.where(np.array(y_side), Y_SIDE_SHADE, 1.0)
        level = light_level(brightness)
        line = np.where(has_wall, self.projection / np.where(has_wall, perp, 1.0), 0.0)
        top = self.horizon - line / 2

        # 3. Every pixel at once (numpy). v = which texel row this pixel shows.
        scale = TEX_SIZE / np.maximum(line, 1e-9)
        v = (self._pixel_y[None, :] - top[:, None]) * scale[:, None]  # (w, h)
        on_wall = (v >= 0) & (v < TEX_SIZE) & has_wall[:, None]
        vi = v.astype(np.intp) & TEX_MASK
        texel = self.atlas[slot[:, None], u[:, None], vi]  # (w, h) palette indices
        lit = COLORMAP[level[:, None], texel]  # shading = one table lookup
        frame: U8Array = np.where(on_wall, lit, self.background)
        self._draw_sprites(frame, camera, sprites)
        return frame

    def _draw_sprites(
        self, frame: U8Array, camera: Camera, sprites: Iterable[SpriteInstance]
    ) -> None:
        w, h = self.config.width, self.config.height
        cos_a, sin_a = math.cos(camera.angle), math.sin(camera.angle)

        # Into camera space: depth along the view direction, lateral to the right.
        visible = []
        for s in sprites:
            art = SPRITES.get(s.name)
            if art is None:
                continue  # e.g. spawn points: no sprite
            dx, dy = s.x - camera.x, s.y - camera.y
            depth = dx * cos_a + dy * sin_a
            if depth > NEAR_PLANE:
                visible.append((depth, -dx * sin_a + dy * cos_a, art))
        visible.sort(key=lambda item: item[0], reverse=True)  # painter's: far to near

        for depth, lateral, art in visible:
            size = self.projection / depth  # one cell wide, one wall tall, in pixels
            left = w / 2 + lateral * self.projection / depth - size / 2
            bottom = self.horizon + EYE_HEIGHT * size  # standing on the floor
            top = bottom - size
            c0, c1 = max(0, math.ceil(left - 0.5)), min(w, math.ceil(left + size - 0.5))
            r0, r1 = max(0, math.ceil(top - 0.5)), min(h, math.ceil(bottom - 0.5))
            if c0 >= c1 or r0 >= r1:
                continue
            cols = np.arange(c0, c1)
            in_front = depth < self.depth[cols]  # the 1D depth buffer test
            if not in_front.any():
                continue
            scale = 64 / size
            u = ((cols + 0.5 - left) * scale).astype(np.intp).clip(0, 63)
            v = ((np.arange(r0, r1) + 0.5 - top) * scale).astype(np.intp).clip(0, 63)
            index = art.index[u[:, None], v[None, :]]
            draw = art.opaque[u[:, None], v[None, :]] & in_front[:, None]
            if not art.fullbright:
                level = light_level(self._brightness(np.array(depth)))
                index = COLORMAP[level, index]
            region = frame[c0:c1, r0:r1]
            region[draw] = index[draw]

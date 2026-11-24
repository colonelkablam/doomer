"""First-person view: one DDA ray per screen column, then one numpy pass that
textures and shades the whole frame as 8-bit palette indices.

Projection: for screen column c, the ray angle is the view angle plus
atan(cam_x * tan(fov/2)), where cam_x runs from -1 (left edge) to +1 (right).
Using atan (not equal angle steps) keeps straight walls straight.

Fisheye: the ray's Euclidean distance is multiplied by cos(ray offset) to get
the perpendicular distance to the camera plane. Without that, a flat wall in
front of you bulges towards the middle of the screen.

Depth: every pixel drawn records its perpendicular distance in a z-buffer
(w x h floats). Low blocks (desks) and sprites are only drawn where they are
nearer than what's already there, so a desk hides the bottom of the wall and of
any sprite behind it, whatever order things are drawn in. Doom couldn't afford
a per-pixel z-buffer in 1993 and used per-column clip ranges instead; with
numpy it's a handful of whole-array comparisons.

Low blocks: the ray carries on through them (see shared/raycast.py), and each
one is drawn as a front face (base..top at the distance the ray enters) plus a
top surface (between the entry and exit distances, only if below eye level).
A pixel on a horizontal surface at height h, row y, is at perpendicular
distance (eye - h) * projection / (y - horizon): the same similar-triangles
rule used for the floor.

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
from doomer.client.textures import TEX_MASK, TEX_SIZE, TOP_COLOURS, build_atlas
from doomer.shared.mapfile import GameMap
from doomer.shared.raycast import RayHit, Side, cast_ray

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
    z: float = 0.0  # height of the surface it stands on (a desk top), in walls


def sprites_from_things(game_map: GameMap) -> list[SpriteInstance]:
    """Map things as sprites, standing on whatever cell they're in (unless they
    hang from the ceiling)."""
    out = []
    for t in game_map.things:
        art = SPRITES.get(t.type)
        on_ceiling = art is not None and art.on_ceiling
        z = 0.0 if on_ceiling else game_map.top_height(math.floor(t.x), math.floor(t.y))
        out.append(SpriteInstance(t.type, t.x, t.y, z))
    return out


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
        self.top_colour = _top_colours(self.atlas, self.slot_for_type)
        # Per-pixel perpendicular distance of whatever was drawn last frame.
        self.zbuffer: FloatArray = np.full((w, h), np.inf)

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

        # 1. One ray per column (Python loop). The last hit is the full-height
        # wall; any before it are low blocks, sorted into layers: layer k holds
        # each column's k-th block.
        dist = [math.inf] * w
        wall = [0] * w
        tex_u = [0.0] * w
        y_side = [False] * w
        layers: list[list[RayHit | None]] = []
        for col in range(w):
            hits = cast_ray(game_map, camera.x, camera.y, camera.angle + self._offsets[col])
            if hits and hits[-1].top >= 1.0:
                hit = hits.pop()
                dist[col] = hit.distance
                wall[col] = hit.wall_type
                tex_u[col] = hit.u
                y_side[col] = hit.side is Side.Y
            for k, block in enumerate(hits):
                if k == len(layers):
                    layers.append([None] * w)
                layers[k][col] = block

        # 2. Full-height walls for every pixel at once (numpy).
        perp = np.array(dist) * self._cos
        frame, zbuffer = self._draw_walls(perp, wall, tex_u, y_side)

        # 3. Low blocks, one layer at a time, z-tested.
        for layer in layers:
            self._draw_block_layer(frame, zbuffer, layer)

        self._draw_sprites(frame, zbuffer, camera, sprites)
        self.zbuffer = zbuffer
        return frame

    def _wall_texels(self, types: IntArray, tex_u: FloatArray) -> tuple[IntArray, IntArray]:
        """Atlas slot and texture column for each wall type / hit position."""
        types = types.copy()
        types[types >= len(self.slot_for_type)] = 0  # unknown type -> missing texture
        slot = self.slot_for_type[types]
        u = (tex_u * TEX_SIZE).astype(np.intp) & TEX_MASK
        return slot, u

    def _draw_walls(
        self, perp: FloatArray, wall: list[int], tex_u: list[float], y_side: list[bool]
    ) -> tuple[U8Array, FloatArray]:
        has_wall = np.isfinite(perp)
        slot, u = self._wall_texels(np.array(wall, dtype=np.intp), np.array(tex_u))
        brightness = self._brightness(perp) * np.where(np.array(y_side), Y_SIDE_SHADE, 1.0)
        level = light_level(brightness)
        line = np.where(has_wall, self.projection / np.where(has_wall, perp, 1.0), 0.0)
        top = self.horizon - line / 2

        # v = which texel row each pixel shows.
        scale = TEX_SIZE / np.maximum(line, 1e-9)
        v = (self._pixel_y[None, :] - top[:, None]) * scale[:, None]  # (w, h)
        on_wall = (v >= 0) & (v < TEX_SIZE) & has_wall[:, None]
        vi = v.astype(np.intp) & TEX_MASK
        texel = self.atlas[slot[:, None], u[:, None], vi]  # (w, h) palette indices
        lit = COLORMAP[level[:, None], texel]  # shading = one table lookup
        frame: U8Array = np.where(on_wall, lit, self.background)
        # Floor and ceiling never hide anything, so they count as infinitely far.
        zbuffer: FloatArray = np.where(on_wall, perp[:, None], np.inf)
        return frame, zbuffer

    def _draw_block_layer(
        self, frame: U8Array, zbuffer: FloatArray, layer: list[RayHit | None]
    ) -> None:
        cols = np.array([c for c, hit in enumerate(layer) if hit is not None], dtype=np.intp)
        hits = [hit for hit in layer if hit is not None]
        cos = self._cos[cols]
        near = np.array([h.distance for h in hits]) * cos  # perpendicular distances
        far = np.array([h.exit_distance for h in hits]) * cos
        top = np.array([h.top for h in hits])
        base = np.array([h.base for h in hits])
        y_side = np.array([h.side is Side.Y for h in hits])
        slot, u = self._wall_texels(
            np.array([h.wall_type for h in hits], dtype=np.intp), np.array([h.u for h in hits])
        )
        # Screen rows of the front face (base..top at the entry distance) and of
        # the top surface (top height, from the exit distance to the entry).
        y_top = self.horizon + (EYE_HEIGHT - top) * self.projection / near
        y_base = self.horizon + (EYE_HEIGHT - base) * self.projection / near
        y_far = self.horizon + (EYE_HEIGHT - top) * self.projection / far
        # Only touch the band of rows this layer can occupy: a desk covers a
        # strip below the horizon, not the whole column. Big saving.
        r0 = max(0, math.floor(min(y_top.min(), y_far.min())))
        r1 = min(self.config.height, math.ceil(y_base.max()))
        if r0 >= r1:
            return
        py = self._pixel_y[None, r0:r1]
        sub_frame = frame[cols, r0:r1]
        sub_z = zbuffer[cols, r0:r1]

        # Front face, z-tested.
        face = (py >= y_top[:, None]) & (py < y_base[:, None]) & (near[:, None] < sub_z)
        height = EYE_HEIGHT - (py - self.horizon) * near[:, None] / self.projection
        vi = ((1.0 - height) * TEX_SIZE).astype(np.intp) & TEX_MASK  # textures floor-anchored
        texel = self.atlas[slot[:, None], u[:, None], vi]
        level = light_level(self._brightness(near) * np.where(y_side, Y_SIDE_SHADE, 1.0))
        sub_frame = np.where(face, COLORMAP[level[:, None], texel], sub_frame)
        sub_z = np.where(face, near[:, None], sub_z)

        # Top surface, if it's below eye level: rows whose distance lies in near..far.
        with np.errstate(divide="ignore", invalid="ignore"):
            row_depth = (EYE_HEIGHT - top)[:, None] * self.projection / (py - self.horizon)
        on_top = (
            (top < EYE_HEIGHT)[:, None]
            & (py > self.horizon)
            & (row_depth >= near[:, None])
            & (row_depth <= far[:, None])
            & (row_depth < sub_z)
        )
        colour = self.top_colour[slot]
        lit_top = COLORMAP[light_level(self._brightness(np.nan_to_num(row_depth))), colour[:, None]]
        frame[cols, r0:r1] = np.where(on_top, lit_top, sub_frame)
        zbuffer[cols, r0:r1] = np.where(on_top, row_depth, sub_z)

    def _draw_sprites(
        self,
        frame: U8Array,
        zbuffer: FloatArray,
        camera: Camera,
        sprites: Iterable[SpriteInstance],
    ) -> None:
        w, h = self.config.width, self.config.height
        cos_a, sin_a = math.cos(camera.angle), math.sin(camera.angle)

        for s in sprites:
            art = SPRITES.get(s.name)
            if art is None:
                continue  # e.g. spawn points: no sprite
            # Into camera space: depth along the view direction, lateral to the right.
            dx, dy = s.x - camera.x, s.y - camera.y
            depth = dx * cos_a + dy * sin_a
            if depth <= NEAR_PLANE:
                continue
            lateral = -dx * sin_a + dy * cos_a
            size = self.projection / depth  # one cell wide, one wall tall, in pixels
            left = w / 2 + lateral * self.projection / depth - size / 2
            bottom = self.horizon + (EYE_HEIGHT - s.z) * size  # standing on its surface
            top = bottom - size
            c0, c1 = max(0, math.ceil(left - 0.5)), min(w, math.ceil(left + size - 0.5))
            r0, r1 = max(0, math.ceil(top - 0.5)), min(h, math.ceil(bottom - 0.5))
            if c0 >= c1 or r0 >= r1:
                continue
            scale = 64 / size
            u = ((np.arange(c0, c1) + 0.5 - left) * scale).astype(np.intp).clip(0, 63)
            v = ((np.arange(r0, r1) + 0.5 - top) * scale).astype(np.intp).clip(0, 63)
            region_z = zbuffer[c0:c1, r0:r1]
            draw = art.opaque[u[:, None], v[None, :]] & (depth <= region_z)  # z-test
            if not draw.any():
                continue
            index = art.index[u[:, None], v[None, :]]
            if not art.fullbright:
                level = light_level(self._brightness(np.array(depth)))
                index = COLORMAP[level, index]
            region = frame[c0:c1, r0:r1]
            region[draw] = index[draw]
            region_z[draw] = depth


def _top_colours(atlas: U8Array, slot_for_type: IntArray) -> U8Array:
    """Palette index for the top surface of a low block, per atlas slot.

    From TOP_COLOURS where given, otherwise the commonest colour in the top
    quarter of the block's texture.
    """
    colours = np.zeros(len(atlas), dtype=np.uint8)
    for slot in range(len(atlas)):
        values, counts = np.unique(atlas[slot][:, :16], return_counts=True)
        colours[slot] = values[np.argmax(counts)]
    for wall_type, colour in TOP_COLOURS.items():
        if wall_type < len(slot_for_type):
            colours[slot_for_type[wall_type]] = colour
    return colours

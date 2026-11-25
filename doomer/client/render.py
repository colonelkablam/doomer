"""First-person view: one DDA ray per screen column, then numpy passes that
texture and shade the whole frame as 8-bit palette indices.

Projection: for screen column c, the ray angle is the view angle plus
atan(cam_x * tan(fov/2)), where cam_x runs from -1 (left edge) to +1 (right).
Using atan (not equal angle steps) keeps straight walls straight.

Fisheye: the ray's Euclidean distance is multiplied by cos(ray offset) to get
the perpendicular distance to the camera plane. Without that, a flat wall in
front of you bulges towards the middle of the screen.

Heights: a point at height h and perpendicular distance d appears on screen at
row  horizon + (eye - h) * projection / d,  where `eye` is the camera's height
(it changes when you jump, crouch or stand on a desk). Turned round, a pixel at
row y on a horizontal surface at height h is at distance
(eye - h) * projection / (y - horizon). Every wall, desk face, floor and
ceiling below uses one of those two formulas.

Looking up and down is y-shearing: the horizon moves up or down the screen
(horizon = middle + tan(pitch) * projection) and everything else follows from
the formulas above. Verticals stay vertical, so it isn't true 3D pitch (Doom
ports, Heretic and Duke Nukem 3D did the same); it looks right up to ~25 deg.

Spans: the raycaster reports where the space along the ray changes (desks,
raised floors, a taller or lower ceiling) and stops at a full wall. The wall
is drawn from the floor to the ceiling of the space in front of it. Each change
can show a riser face (floor goes up), a bulkhead face (ceiling comes down),
and a stretch of floor and ceiling surface until the next change. The default
floor (height 0) and ceiling (height 1) are a precomputed background; only
surfaces at other heights are drawn explicitly.

Depth: every pixel drawn records its perpendicular distance in a z-buffer, and
everything after the walls is z-tested, so draw order doesn't matter. Doom
couldn't afford a per-pixel z-buffer in 1993 and used per-column clip ranges;
with numpy it's a handful of whole-array comparisons.

Why split work this way: the DDA is inherently a loop (each ray takes a
different number of steps), so it stays in Python, 320 iterations a frame.
Everything per-pixel is whole-array numpy, which runs in C.
"""

import functools
import math
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt
import pygame

from doomer.client import fastcast as fc
from doomer.client.areas import (
    AREA_FACES,
    AREA_WALL_TYPE,
    CEILING,
    REVEAL_WALL_TYPE,
    SOFFIT,
    TOP_KIND,
    WINDOW_FACES,
    Surfaces,
    build_surfaces,
    cell_tops,
    window_type,
)
from doomer.client.boxes import (
    BOX_MODELS,
    SPINNING,
    TOP,
    Box,
    PlacedBox,
    box_textures,
    desk_screens,
    lit_now,
    poster_plans,
    stair_boxes,
)
from doomer.client.config import LightingConfig, RenderConfig
from doomer.client.lighting import LightMap, bake_light_map
from doomer.client.palette import (
    COLORMAP,
    CREAM,
    GREY,
    PALETTE,
    U8Array,
    light_level,
    nearest_index,
)
from doomer.client.sprites import SPRITES
from doomer.client.textures import (
    GLASS,
    PANORAMA_ROWS,
    PANORAMA_ROWS_PER_TAN,
    PANORAMA_WIDTH,
    TEX_MASK,
    TEX_SIZE,
    TOP_COLOURS,
    TOP_SEAM_WALL_TYPES,
    UPPER,
    build_atlas,
    build_opacity,
    panorama,
    railing,
)
from doomer.shared.doors import DOOR_THING_KINDS
from doomer.shared.furniture import resting_height
from doomer.shared.mapfile import (
    AREA_CHARS,
    EAST,
    NORTH,
    RAIL_HEIGHT,
    RAILING_WALL_TYPE,
    SOUTH,
    STAIRS_WALL_TYPE,
    WEST,
    GameMap,
)

MIN_SHADE = 0.25
Y_SIDE_SHADE = 0.72  # north/south faces darker than east/west, so corners read clearly
# Default camera height in walls: a standing eye at ~1.6 m under a 2.7 m ceiling.
EYE_HEIGHT = 0.6
MAX_PITCH = math.radians(25)  # y-shearing looks wrong much beyond this
# Bayer matrix: each value 0-15 appears once, arranged so that any threshold
# switches on pixels that are spread as evenly as possible.
BAYER_4X4 = np.array([[0, 8, 2, 10], [12, 4, 14, 6], [3, 11, 1, 9], [15, 7, 13, 5]])
NEAR_PLANE = 0.05  # sprites closer than this (in cells) aren't drawn
# A face is lit by the light map this far back along the ray from where the
# ray meets it: in the open space it faces (the first light sample in front
# of a wall is 1/8 cell out), not inside the wall, which no light reaches.
FACE_LIGHT_BACKOFF = 0.13
# Floor and ceiling mipmaps: + makes them blurrier sooner, - sharper but
# shimmering. 0 picks the level whose texels are about a pixel.
MIP_BIAS = 0.0
MAX_BOX_DISTANCE = 16.0  # box models further than this (cells) aren't drawn: a few pixels
UNDER_DESK_SHADOW = 12  # colormap levels darker (of 32) for the floor under a desk
DESK_SEAM_DARKEN = 9  # colormap levels darker for the join between two desks' tops
DESK_JOIN_DARKEN = 5  # ... and for the joins between neighbouring pairs in a row
DESK_SEAM_WIDTH = 0.01  # half-width of those lines, in cells (~3 cm wide)
STEP_HEIGHT = 0.125  # a rise up to this is a step, not a drop that needs a railing
# Ceiling light panels: half-size in cells, and colours (diffuser is full-bright).
# The middle 2 x 2 ceiling tiles of a cell (areas.ceiling_tiles: 4 x 4 per
# cell, lined up with the cell's edges), the grid's bars carried across it.
LIGHT_HALF = (0.25, 0.25)
LIGHT_BAR = 0.012  # half-width of the bars across a panel, cells (as the T-bars)
LIGHT_DIFFUSER = (CREAM << 4) | 15
DEAD_DIFFUSER = (GREY << 4) | 8  # a panel with no power
LIGHT_FRAME = (GREY << 4) | 9
_BACKGROUND_CACHE_SIZE = 64

# The neighbouring cell across each compass face.
_FACE_STEP = {NORTH: (0, -1), EAST: (1, 0), SOUTH: (0, 1), WEST: (-1, 0)}
_PANORAMA = panorama()  # the view through windows (textures.panorama)

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.intp]
Int64Array = npt.NDArray[np.int64]


@dataclass(frozen=True, slots=True)
class Camera:
    x: float
    y: float
    angle: float
    z: float = EYE_HEIGHT  # eye height above the base floor, in walls
    pitch: float = 0.0  # radians, positive = looking up; clamped to MAX_PITCH


@dataclass(frozen=True, slots=True)
class SpriteInstance:
    """Something to draw this frame: a sprite name (= thing type) at a position."""

    name: str
    x: float
    y: float
    z: float = 0.0  # height of its base (a desk top; for hanging things, ceiling - 1)
    angle: float = 0.0  # which way it faces (radians), for sprites with views


def sprites_from_things(game_map: GameMap) -> list[SpriteInstance]:
    """Map things as sprites, standing on whatever cell they're in, or hanging
    from its ceiling."""
    out = []
    for t in game_map.things:
        art = SPRITES.get(t.type)
        cx, cy = math.floor(t.x), math.floor(t.y)
        if art is not None and art.on_ceiling:
            z = game_map.ceiling_height(cx, cy) - 1.0  # the image's top at the ceiling
        else:
            z = resting_height(game_map, t)  # a block's top, furniture's, or the floor
        # A thing with a kind (a poster) draws as "type:kind" (BOX_MODELS).
        kind = t.prop("kind")
        name = f"{t.type}:{kind}" if kind else t.type
        out.append(SpriteInstance(name, t.x, t.y, z, t.angle))
    return out


def _placed(s: SpriteInstance, box: Box, now: float) -> PlacedBox:
    """A model's box where its thing is; a SPINNING one turned (and bobbed)
    by the clock."""
    spin = SPINNING.get(s.name)
    if spin is None:
        return PlacedBox(s.x, s.y, s.z, s.angle, box)
    rate, bob = spin
    return PlacedBox(s.x, s.y, s.z + bob * math.sin(now * rate), s.angle + now * rate, box)


def make_frame_surface(config: RenderConfig) -> pygame.Surface:
    """An 8-bit surface carrying the game palette, for Renderer.render to fill."""
    surface = pygame.Surface((config.width, config.height), depth=8)
    surface.set_palette([tuple(int(c) for c in rgb) for rgb in PALETTE])
    return surface


class Renderer:
    def __init__(self, config: RenderConfig, lighting: LightingConfig | None = None) -> None:
        """`lighting`: light surfaces by a light map baked from the map's
        ceiling panels (doomer/client/lighting.py). None: distance fade only."""
        self.config = config
        self.lighting = lighting
        self._light_map_map: GameMap | None = None
        self._light_map: LightMap | None = None
        self._light: LightMap | None = None  # this frame's
        self._power = True  # this frame's
        self._frame_background: U8Array = np.zeros((0, 0), dtype=np.uint8)
        w, h = config.width, config.height
        tan_half_fov = math.tan(math.radians(config.fov_degrees) / 2)
        # Distance from the eye to a virtual screen, in pixels.
        self.projection = (w / 2) / tan_half_fov
        self._offsets = [math.atan((2 * (c + 0.5) / w - 1) * tan_half_fov) for c in range(w)]
        self._offsets_array = np.array(self._offsets)
        self._cos = np.cos(self._offsets_array)
        self._pixel_y = np.arange(h, dtype=np.float64) + 0.5  # pixel centres
        # slot_table[wall_type, side] -> atlas slot; side = FRONT/RIGHT/BACK/LEFT/UPPER
        self.atlas, self.slot_table = build_atlas()
        self.opacity = build_opacity()  # (slot, u, v): False = see-through (open desks)
        rail_tex, rail_opaque = railing()
        self.rail_tex = np.ascontiguousarray(rail_tex.T)  # [u][v]
        self.rail_opaque = np.ascontiguousarray(rail_opaque.T)
        self.top_colour = _top_colours(self.atlas, self.slot_table)
        # Floors and ceilings: textured by area, unless the config gives flat
        # colours. The flat floor colour is also used for the odd raised
        # surface that isn't a block (none in practice): the office carpet's.
        self.surfaces: Surfaces = build_surfaces()
        # Per wall type: the surface kind of its top if it's textured like a
        # floor (a balcony's carpet), else -1 (a flat colour: desk tops).
        self._top_kind = np.full(max(TOP_KIND, default=0) + 2, -1, dtype=np.int64)
        for wall_type, kind in TOP_KIND.items():
            self._top_kind[wall_type] = kind
        self.textured = config.floor_colour is None or config.ceiling_colour is None
        average = self.surfaces.mips[:, -1, 0, 0]  # each texture's single-colour level
        self.ceiling_colour = (
            int(average[self.surfaces.ceiling[0]])
            if config.ceiling_colour is None
            else nearest_index(PALETTE, config.ceiling_colour)
        )
        self.floor_colour = (
            int(average[self.surfaces.floor[0]])
            if config.floor_colour is None
            else nearest_index(PALETTE, config.floor_colour)
        )
        x, y = np.meshgrid(np.arange(w), np.arange(h), indexing="ij")
        self._dither = (BAYER_4X4[x & 3, y & 3] + 0.5) / 16  # (w, h) thresholds in (0, 1)
        # For choosing between mip levels: the same pattern, out of step with
        # the shading's (else both round up at the same pixels).
        self._mip_dither = (self._dither + 0.5) % 1.0
        self._backgrounds: dict[tuple[int, int], U8Array] = {}
        # Per-map caches remember the map object itself (and compare with `is`),
        # not id(map): Python reuses the id of a freed object, so a new map
        # could otherwise be served an old map's table.
        self._lights_map: GameMap | None = None
        self._slots_map: GameMap | None = None
        self._map_boxes_map: GameMap | None = None
        self._seam_map: GameMap | None = None
        self._seam_grid: IntArray = np.zeros((0, 0), dtype=np.intp)
        self._map_boxes: list[PlacedBox] = []
        self._map_arrays: fc.MapArrays | None = None
        self._cell_tops_map: GameMap | None = None
        self._door_cells_map: GameMap | None = None
        self._door_cells: frozenset[tuple[int, int]] = frozenset()
        self._cell_tops = np.zeros((0, 0), dtype=np.int64)
        self._light_list: list[tuple[float, float, float, bool]] = []
        # Last frame's per-pixel perpendicular distances, and its horizon row.
        self.zbuffer: FloatArray = np.full((w, h), np.inf)
        self.horizon: float = h / 2

    # --- background (floor at 0, ceiling at 1) ---------------------------------------

    def background(self, horizon: float, eye: float) -> U8Array:
        """Default floor and ceiling, darkened with distance and dithered.

        Depends on the horizon (looking up/down) and eye height (jumping,
        crouching), so it's cached by both, rounded to a pixel and 1/64 wall.
        """
        key = (round(horizon), round(eye * 64))
        cached = self._backgrounds.get(key)
        if cached is not None:
            return cached
        horizon, eye = float(key[0]), key[1] / 64
        rows = self._pixel_y - horizon
        with np.errstate(divide="ignore", invalid="ignore"):
            floor_depth = eye * self.projection / rows
            ceiling_depth = (1.0 - eye) * self.projection / -rows
        below = rows > 0
        depth = np.where(below, floor_depth, ceiling_depth)
        depth = np.where(np.isfinite(depth) & (depth > 0), depth, np.inf)
        base = np.where(below, self.floor_colour, self.ceiling_colour)  # (h,)
        # A dark base colour has only a few shades below it, so a smooth gradient
        # would show hard bands. Ordered dithering: compute the exact (fractional)
        # shade, then round up or down per pixel by comparing the fraction against
        # a tiled 4x4 Bayer threshold matrix. The eye averages the pattern.
        # Shade s has brightness (s + 1) / 16, so scaling by f gives (s + 1) f - 1.
        exact = ((base & 0x0F) + 1) * self._brightness(depth) - 1  # (h,)
        shade = np.floor(exact[None, :] + self._dither).clip(0, 15).astype(np.intp)
        background: U8Array = ((base[None, :] & 0xF0) | shade).astype(np.uint8)
        if len(self._backgrounds) >= _BACKGROUND_CACHE_SIZE:
            self._backgrounds.pop(next(iter(self._backgrounds)))
        self._backgrounds[key] = background
        return background

    def _background_depths(
        self, horizon: float, eye: float
    ) -> tuple[FloatArray, npt.NDArray[np.bool_]]:
        """Per screen row: the distance to the default floor (below the
        horizon) or ceiling (above), inf at the horizon; and which are floor."""
        rows = self._pixel_y - horizon
        with np.errstate(divide="ignore", invalid="ignore"):
            floor_depth = eye * self.projection / rows
            ceiling_depth = (1.0 - eye) * self.projection / -rows
        below = rows > 0
        depth = np.where(below, floor_depth, ceiling_depth)
        return np.where(np.isfinite(depth) & (depth > 0), depth, np.inf), below

    def _lit_background(
        self, light: LightMap, camera: Camera, horizon: float, eye: float
    ) -> U8Array:
        """The default floor and ceiling, lit: each pixel's distance (the
        usual floor-casting formula) puts it at a point in the world (camera +
        ray x distance), where the light map is sampled. Ceilings take the
        square root of the light: lit mostly by light bounced off the floor,
        they vary less than floors. Built every frame (it depends on where
        you are), unlike the unlit background, which is cached."""
        depth, below = self._background_depths(horizon, eye)  # where each pixel is
        # Shaded exactly as the cached background (horizon to a pixel, eye to
        # 1/64), so with light 1 everywhere the frame is unchanged.
        shade_depth, _ = self._background_depths(round(horizon), round(eye * 64) / 64)
        # Light changes slowly across the screen (the map has 4 samples per
        # cell), so it's looked up once per 2x2 pixels and spread; the shading
        # below is still per pixel, and the dither hides the steps. A quarter
        # of the lookups: this is the costliest part of lighting.
        w, h = self.config.width, self.config.height
        cs, rs = slice(0, w, 2), slice(0, h, 2)
        along = np.minimum(depth[None, rs] / self._cos[cs, None], 64.0)
        x = camera.x + self._ray_dx[cs, None] * along
        y = camera.y + self._ray_dy[cs, None] * along
        coarse = self._floor_and_ceiling_light(light, x, y, below[rs])
        # Ceilings: the square root (light bounced off a lit floor), but with
        # the power off there's little to bounce: the plain light.
        ceiling = np.sqrt(coarse) if self._power else coarse
        coarse = np.where(below[None, rs], coarse, ceiling)
        lit = np.repeat(np.repeat(coarse, 2, axis=0), 2, axis=1)[:w, :h]
        base = np.where(below, self.floor_colour, self.ceiling_colour)  # (h,)
        exact = ((base & 0x0F) + 1)[None, :] * self._brightness(shade_depth)[None, :] * lit - 1
        shade = np.floor(exact + self._dither).clip(0, 15).astype(np.intp)
        result: U8Array = ((base[None, :] & 0xF0) | shade).astype(np.uint8)
        return result

    def _textured_background(self, camera: Camera, horizon: float, eye: float) -> U8Array:
        """The default floor (height 0) and ceiling (height 1), textured by
        area: each pixel's distance (the floor-casting formula) puts it at a
        point in the world, camera + ray x distance, whose texel it shows.
        Lit and shaded as _lit_background (or by distance only, without a
        light map). Built every frame: it depends on where you are."""
        depth, below = self._background_depths(horizon, eye)
        shade_depth, _ = self._background_depths(round(horizon), round(eye * 64) / 64)
        w, h = self.config.width, self.config.height
        light = np.ones(((w + 1) // 2, (h + 1) // 2))
        if self._light is not None:
            # Light changes slowly across the screen (4 samples per cell), so
            # it's looked up once per 2x2 pixels; the dither hides the steps.
            # Ceilings take the square root (bounce light) with the power on,
            # as in _lit_background.
            cs, rs = slice(0, w, 2), slice(0, h, 2)
            along = np.minimum(depth[None, rs] / self._cos[cs, None], 64.0)
            x = camera.x + self._ray_dx[cs, None] * along
            y = camera.y + self._ray_dy[cs, None] * along
            light = self._floor_and_ceiling_light(self._light, x, y, below[rs])
            ceiling = np.sqrt(light) if self._power else light
            light = np.where(below[None, rs], light, ceiling)
        arrays = self._map_arrays_for(self._frame_map)
        return fc.textured_background(
            camera.x,
            camera.y,
            self._ray_dx,
            self._ray_dy,
            self._cos,
            depth,
            np.where(below, eye, 1.0 - eye),  # the eye's height above / below the surface
            ~below,
            self._brightness(shade_depth),
            np.ascontiguousarray(light),
            self._dither,
            self._mip_dither,
            self.projection,
            MIP_BIAS,
            arrays.areas,
            self.surfaces.floor,
            self.surfaces.ceiling,
            self.surfaces.mips,
        )

    def _cell_tops_for(self, game_map: GameMap) -> npt.NDArray[np.int64]:
        """Per cell, its own top texture (desk pairs' paperwork) or -1."""
        if self._cell_tops_map is not game_map:
            self._cell_tops = cell_tops(game_map, self.surfaces)
            self._cell_tops_map = game_map
        return self._cell_tops

    def _surface_texels(
        self,
        x: FloatArray,
        y: FloatArray,
        depth: FloatArray,
        drop: FloatArray | float,
        cos: FloatArray,
        kind: npt.NDArray[np.int64] | int,
        dither: FloatArray,
    ) -> U8Array:
        """Texels of horizontal surface pixels (all arguments broadcast to
        x's shape): world points (x, y), perpendicular `depth`, the eye's
        `drop` to the surface (its height difference), the ray's `cos`, the
        kind of surface (areas.FLOOR, CEILING, SOFFIT, a raised top) and a
        dither threshold for choosing the mip level. The level is picked so a
        texel is about a pixel: a pixel covers depth / projection across the
        ray and, seen at a slant, depth^2 / (projection * drop) along it; the
        larger decides."""
        footprint = depth / self.projection / cos * np.maximum(1.0, depth / np.maximum(drop, 1e-3))
        with np.errstate(divide="ignore", invalid="ignore"):
            level = np.log2(footprint * TEX_SIZE) + MIP_BIAS
        level = np.nan_to_num(level, nan=99.0, posinf=99.0, neginf=0.0)
        shape = x.shape

        def flat(a: npt.ArrayLike, dtype: type) -> npt.NDArray[Any]:
            return np.ascontiguousarray(np.broadcast_to(a, shape), dtype=dtype)

        return fc.surface_texels(
            flat(x, np.float64),
            flat(y, np.float64),
            flat(level, np.float64),
            flat(kind, np.int64),
            flat(dither, np.float64),
            self._map_arrays_for(self._frame_map).areas,
            self.surfaces.table,
            self._cell_tops_for(self._frame_map),
            self.surfaces.mips,
        )

    def _dithered(
        self,
        colour: npt.NDArray[np.integer],
        depth: FloatArray,
        cols: IntArray,
        r0: int,
        r1: int,
        light: FloatArray | float = 1.0,
    ) -> U8Array:
        """Distance-shaded, dithered palette indices for a horizontal surface:
        `colour` per column, `depth` per pixel of the (cols, r0:r1) block. Same
        ordered dithering as the background, so large floors and ceilings at
        other heights don't show hard bands. `light`: per pixel, or 1."""
        brightness = self._brightness(np.nan_to_num(depth, nan=np.inf, posinf=1e9)) * light
        base = colour if colour.ndim == 2 else colour[:, None]  # per pixel, or per column
        exact = ((base & 0x0F) + 1) * brightness - 1
        shade = np.floor(exact + self._dither[cols, r0:r1]).clip(0, 15).astype(np.intp)
        result: U8Array = ((base & 0xF0) | shade).astype(np.uint8)
        return result

    def _ceiling_light(
        self, cols: IntArray, depth: FloatArray, height: FloatArray
    ) -> FloatArray | float:
        """Light on a ceiling at per-pixel `depth` in columns `cols` (a block
        of rows), each column's ceiling at `height`: the light map below it,
        square-rooted (bounce light, as the background's ceiling), or 1
        without lighting. A ceiling lower than a storey (the soffit of a
        lintel over a door) sits in shadow: the plain light, a little less."""
        if self._light is None:
            return 1.0
        along = np.minimum(
            np.nan_to_num(depth, nan=64.0, posinf=64.0) / self._cos[cols][:, None], 64.0
        )
        camera = self._frame_camera
        x = camera.x + self._ray_dx[cols][:, None] * along
        y = camera.y + self._ray_dy[cols][:, None] * along
        # Ceilings take the walls' light (no light through a door reaches
        # them); soffits, low over a doorway, the floor's.
        low = (height < 1.0)[:, None]
        soffit = low[:, 0]
        light = np.empty(x.shape)
        light[soffit] = self._light.at(x[soffit], y[soffit])
        light[~soffit] = self._light.wall_faces().at(x[~soffit], y[~soffit])
        light = np.minimum(light, 1.0)
        shade = self.lighting.soffit_shade if self.lighting else 1.0
        bounce = np.sqrt(light) if self._power else light  # (as the background's ceiling)
        result: FloatArray = np.where(low, light * shade, bounce)
        return result

    def _light_map_for(self, game_map: GameMap) -> LightMap | None:
        """The map's light map, baked once per map (None without lighting)."""
        if self.lighting is None:
            return None
        if self._light_map_map is not game_map:
            self._light_map = bake_light_map(game_map, self.lighting)
            self._light_map_map = game_map
        return self._light_map

    @staticmethod
    def _floor_and_ceiling_light(
        light: LightMap, x: FloatArray, y: FloatArray, below: npt.NDArray[np.bool_]
    ) -> FloatArray:
        """Light at points [col][row] on the floor (rows `below` the
        horizon) or the ceiling: floors take the floor map, ceilings the
        walls' (light through a door lights the floor beyond, not the
        ceiling). Each point looked up once."""
        result = np.empty(x.shape)
        result[:, below] = light.at(x[:, below], y[:, below])
        result[:, ~below] = light.wall_faces().at(x[:, ~below], y[:, ~below])
        return np.minimum(result, 1.0)

    def _ray_light(
        self,
        cols: IntArray,
        along: FloatArray,
        backoff: float = FACE_LIGHT_BACKOFF,
        high: bool = False,
        wall: bool = False,
    ) -> FloatArray:
        """Light (0..1) at a distance `along` each column's ray (moved back
        `backoff` towards the camera), from this frame's light map; 1 without
        one. Light over 1 (overlapping pools) is clamped: the colormap can
        darken, not brighten. `high`: for a face above head height (over a
        doorway), lit only by light that doesn't pass under a lintel. `wall`:
        for a full wall's face, lit only by light that passes no door."""
        if self._light is None:
            return np.ones(np.shape(cols))
        t = np.clip(np.nan_to_num(along - backoff, posinf=64.0), 0.0, 64.0)
        camera = self._frame_camera
        x = camera.x + self._ray_dx[cols] * t
        y = camera.y + self._ray_dy[cols] * t
        light = self._light.high() if high else self._light.wall_faces() if wall else self._light
        result: FloatArray = np.minimum(light.at(x, y), 1.0)
        return result

    def _brightness(self, distance: FloatArray) -> FloatArray:
        fade = self.lighting.shade_distance if self.lighting else self.config.shade_distance
        result: FloatArray = np.maximum(MIN_SHADE, 1.0 - distance / fade)
        return result

    # --- frame ------------------------------------------------------------------------

    def render(
        self,
        target: pygame.Surface,
        game_map: GameMap,
        camera: Camera,
        sprites: Iterable[SpriteInstance] = (),
        boxes: Iterable[PlacedBox] = (),
        power: bool = True,
        now: float = 0.0,
    ) -> None:
        """Draw into an 8-bit surface from make_frame_surface()."""
        frame = self.render_indices(game_map, camera, sprites, boxes, power, now)
        pygame.surfarray.blit_array(target, frame)

    def render_indices(
        self,
        game_map: GameMap,
        camera: Camera,
        sprites: Iterable[SpriteInstance] = (),
        boxes: Iterable[PlacedBox] = (),
        power: bool = True,
        now: float = 0.0,
    ) -> U8Array:
        """The frame as a (width, height) array of palette indices. `boxes`:
        moving geometry this frame (doors), drawn like box models. `power`:
        False lights the floor by its emergency panels only (and draws the
        rest dark). `now`: seconds, for models that move by the clock (an ID
        pass turning, boxes.SPINNING)."""
        self._power = power
        w = self.config.width
        pitch = max(-MAX_PITCH, min(MAX_PITCH, camera.pitch))
        horizon = self.config.height / 2 + math.tan(pitch) * self.projection
        eye = camera.z
        self.horizon = horizon
        # For finding where a pixel of a horizontal surface is in the world.
        self._frame_map, self._frame_camera = game_map, camera
        ray = camera.angle + self._offsets_array
        self._ray_dx, self._ray_dy = np.cos(ray), np.sin(ray)
        self._light = self._light_map_for(game_map)
        if self._light is not None and not power:
            self._light = self._light.without_power()
        if self.textured:
            self._frame_background = self._textured_background(camera, horizon, eye)
        elif self._light is None:
            self._frame_background = self.background(horizon, eye)
        else:
            self._frame_background = self._lit_background(self._light, camera, horizon, eye)

        # 1. One ray per column, compiled (doomer/client/fastcast.py): per
        # column, the full wall it ends at and the changes of space before it,
        # nearest first (the camera's own space first of all), sorted into
        # layers: layer k holds each column's k-th change.
        columns = fc.cast_columns(
            self._map_arrays_for(game_map), camera.x, camera.y, camera.angle + self._offsets_array
        )
        dist = columns.wall_floats[fc.W_DIST]
        present = np.arange(columns.layers)[:, None] < columns.counts[None, :]
        ceiling_changes = bool((present & (columns.floats[fc.CEILING] != 1.0)).any())

        # 2. Walls, floor to ceiling of the space in front of them (numpy).
        perp = dist * self._cos
        frame, zbuffer = self._draw_walls(
            perp,
            self._ray_light(np.arange(w), dist, wall=True),
            columns.wall_ints[fc.W_SLOT].astype(np.intp),
            columns.wall_ints[fc.W_UPPER].astype(np.intp),
            columns.wall_floats[fc.W_U],
            columns.wall_ints[fc.W_Y_SIDE] != 0,
            columns.wall_floats[fc.W_BASE],
            columns.wall_floats[fc.W_TOP],
            horizon,
            eye,
        )

        # The default ceiling is a background with no geometry of its own; give
        # its pixels their real depth where it really is the ceiling, so a low
        # ceiling hides a higher one beyond it (looking from a normal room into
        # the atrium). Inside a stretch with a different ceiling it doesn't
        # exist, so those pixels stay infinitely far and the real ceiling wins.
        # Where it exists and is nearer than a wall already drawn, it covers
        # the wall: a wall in a tall room is drawn up to that room's ceiling,
        # but from a low room you only see it below your own ceiling.
        if ceiling_changes:
            self._ceiling_depths(frame, zbuffer, columns, present, horizon, eye)
        else:
            self._default_ceiling_depths(zbuffer, horizon, eye)

        # 3. Changes of space, one layer at a time, z-tested.
        for k in range(columns.layers):
            self._draw_span_layer(frame, zbuffer, columns, present[k], k, horizon, eye)

        self._draw_ceiling_lights(frame, zbuffer, game_map, camera, horizon, eye)
        sprites = list(sprites)
        moving = list(boxes)
        boxes = (
            moving
            + self._map_boxes_for(game_map)
            + [
                _placed(s, box, now)
                for s in sprites
                for box in BOX_MODELS.get(s.name, ())
                if (s.x - camera.x) ** 2 + (s.y - camera.y) ** 2 < MAX_BOX_DISTANCE**2
            ]
        )
        # Indicator lamps: only while lit (a blink's off half, no mains power).
        boxes = [p for p in boxes if not p.box.indicator or lit_now(p.box, p.x, p.y, now, power)]
        self._draw_boxes(frame, zbuffer, camera, boxes, horizon, eye)
        self._draw_sprites(frame, zbuffer, camera, sprites, horizon, eye)
        self.zbuffer = zbuffer
        return frame

    def _map_arrays_for(self, game_map: GameMap) -> fc.MapArrays:
        """The map's grids as arrays for the compiled caster, with each cell's
        atlas slots: per compass face, its texture and its texture above one
        storey (a cell's textures never change during a map). Outside the
        map counts as wall type 1."""
        if self._map_arrays is None or self._slots_map is not game_map:
            slots = np.array(
                [
                    [
                        [self._slot_for(game_map, x, y, f) for f in range(4)]
                        + [self._upper_slot(self._face_type(game_map, x, y, f)) for f in range(4)]
                        for x in range(game_map.width)
                    ]
                    for y in range(game_map.height)
                ],
                dtype=np.int64,
            ).reshape(game_map.height, game_map.width, 8)
            outside = np.array(
                [int(self.slot_table[1, side]) for side in range(4)] + [self._upper_slot(1)] * 4,
                dtype=np.int64,
            )
            bulkhead = np.array(
                [self._upper_slot(AREA_WALL_TYPE + area) for area in range(len(AREA_CHARS))],
                dtype=np.int64,
            )
            self._map_arrays = fc.map_arrays(game_map, slots, outside, bulkhead)
            self._slots_map = game_map
        return self._map_arrays

    def _face_type(self, game_map: GameMap, x: int, y: int, face: int) -> int:
        """The wall type whose textures a face of a cell shows: its own, but
        a plain wall looks like the area it faces (areas.AREA_FACES), and a
        window is cut into that area's wall (areas.WINDOW_FACES)."""
        wall_type = game_map.walls[y][x]
        side = (face - game_map.facing_at(x, y)) % 4
        dx, dy = _FACE_STEP[face]
        if side in WINDOW_FACES.get(wall_type, ()):  # a window: cut into the area's wall
            return window_type(game_map.area_at(x + dx, y + dy))
        if side not in AREA_FACES.get(wall_type, ()):
            return wall_type
        if (x + dx, y + dy) in self._door_cells_in(game_map):
            return REVEAL_WALL_TYPE  # a doorway's side wall: its lining
        return AREA_WALL_TYPE + game_map.area_at(x + dx, y + dy)

    def _door_cells_in(self, game_map: GameMap) -> frozenset[tuple[int, int]]:
        """The cells with a door in them (doorways)."""
        if self._door_cells_map is not game_map:
            self._door_cells = frozenset(
                (math.floor(t.x), math.floor(t.y))
                for t in game_map.things
                if t.type in DOOR_THING_KINDS
            )
            self._door_cells_map = game_map
        return self._door_cells

    def _slot_for(self, game_map: GameMap, x: int, y: int, face: int) -> int:
        wall_type = self._face_type(game_map, x, y, face)
        if wall_type >= len(self.slot_table):
            return 0  # unknown wall type -> missing texture
        side = (face - game_map.facing_at(x, y)) % 4
        return int(self.slot_table[wall_type, side])

    def _upper_slot(self, wall_type: int) -> int:
        if wall_type >= len(self.slot_table):
            return 0
        return int(self.slot_table[wall_type, UPPER])

    def _rows(
        self, height: FloatArray, depth: FloatArray, horizon: float, eye: float
    ) -> FloatArray:
        """Screen row of points at `height` and perpendicular `depth` (per column)."""
        rows: FloatArray = horizon + (eye - height) * self.projection / depth
        return rows

    def _face_texels(
        self,
        py: FloatArray,
        depth: FloatArray,
        slot: IntArray,
        upper: IntArray,
        horizon: float,
        eye: float,
    ) -> tuple[IntArray, IntArray]:
        """(slot, v) per pixel of a vertical face at `depth`. Textures are anchored
        to the floor (height 0), one texture per storey; above one storey the
        wall type's upper texture is used."""
        height = eye - (py - horizon) * depth[:, None] / self.projection
        vi = ((1.0 - np.mod(height, 1.0)) * TEX_SIZE).astype(np.intp) & TEX_MASK
        which = np.where(height >= 1.0, upper[:, None], slot[:, None])
        return which, vi

    def _ceiling_depths(
        self,
        frame: U8Array,
        zbuffer: FloatArray,
        columns: fc.Columns,
        present: npt.NDArray[np.bool_],
        horizon: float,
        eye: float,
    ) -> None:
        rows = min(self.config.height, math.floor(horizon))
        if rows <= 0 or eye >= 1.0:
            return  # no default ceiling above the eye on screen
        # Per column, the [near, far] perpendicular intervals where the ceiling
        # isn't the default: open stretches, the camera's own space, and blocks
        # (a step or desk under the atrium's high ceiling is under it too).
        changed = present & (columns.floats[fc.CEILING] != 1.0)
        lo = np.where(changed, columns.floats[fc.DISTANCE] * self._cos, np.nan)
        hi = np.where(changed, columns.floats[fc.EXIT_DISTANCE] * self._cos, np.nan)
        depth = (1.0 - eye) * self.projection / (horizon - self._pixel_y[:rows])  # (rows,)
        inside = (lo[:, :, None] <= depth[None, None, :]) & (depth[None, None, :] <= hi[:, :, None])
        real = ~inside.any(axis=0)  # (w, rows)
        region = zbuffer[:, :rows]
        nearer = real & (depth[None, :] < region)
        zbuffer[:, :rows] = np.where(nearer, depth[None, :], region)
        # Uncover the ceiling where a wall beyond was drawn over it.
        covered = nearer & ~np.isinf(region)
        if covered.any():
            sky = self._frame_background[:, :rows]
            frame[:, :rows] = np.where(covered, sky, frame[:, :rows])

    def _default_ceiling_depths(self, zbuffer: FloatArray, horizon: float, eye: float) -> None:
        """No ceiling changes in view: the default ceiling is real everywhere."""
        rows = min(self.config.height, math.floor(horizon))
        if rows <= 0 or eye >= 1.0:
            return
        depth = (1.0 - eye) * self.projection / (horizon - self._pixel_y[:rows])
        region = zbuffer[:, :rows]
        zbuffer[:, :rows] = np.where(np.isinf(region), depth[None, :], region)

    def _draw_walls(
        self,
        perp: FloatArray,
        light: FloatArray,
        slot: IntArray,
        upper: IntArray,
        tex_u: FloatArray,
        y_side: npt.NDArray[np.bool_],
        base: FloatArray,
        top: FloatArray,
        horizon: float,
        eye: float,
    ) -> tuple[U8Array, FloatArray]:
        has_wall = np.isfinite(perp)
        depth = np.where(has_wall, perp, 1.0)
        u = _texel_u(tex_u)
        brightness = self._brightness(perp) * np.where(y_side, Y_SIDE_SHADE, 1.0) * light
        level = light_level(brightness)
        y_top = self._rows(top, depth, horizon, eye)
        y_bottom = self._rows(base, depth, horizon, eye)
        py = self._pixel_y[None, :]
        on_wall = has_wall[:, None] & (py >= y_top[:, None]) & (py < y_bottom[:, None])
        which, vi = self._face_texels(py, depth, slot, upper, horizon, eye)
        texel = self.atlas[which, u[:, None], vi]  # (w, h) palette indices
        lit = COLORMAP[level[:, None], texel]  # shading = one table lookup
        frame: U8Array = np.where(on_wall, lit, self._frame_background)
        glass = on_wall & (texel == GLASS)
        if glass.any():  # through windows: the city outside
            frame = np.where(glass, self._outside(horizon), frame)
        # Default floor and ceiling never hide anything: they count as infinitely far.
        zbuffer: FloatArray = np.where(on_wall, perp[:, None], np.inf)
        return frame, zbuffer

    def _outside(self, horizon: float) -> U8Array:
        """The view through glass, (w, h): the panorama, by each column's
        ray angle and each row's height over the horizon (as a tangent), so
        it's infinitely far: it turns with you but never moves as you walk.
        Unshaded: the city lights itself."""
        angle = self._frame_camera.angle + self._offsets_array
        col = (np.floor(angle * (PANORAMA_WIDTH / (2 * math.pi))).astype(np.intp)) % PANORAMA_WIDTH
        tan = (horizon - self._pixel_y) / self.projection
        row = np.floor(tan * PANORAMA_ROWS_PER_TAN).astype(np.intp) + 1  # 0: the ground
        row = np.clip(row, 0, PANORAMA_ROWS + 1)
        view: U8Array = _PANORAMA[col[:, None], row[None, :]]
        return view

    def _draw_span_layer(
        self,
        frame: U8Array,
        zbuffer: FloatArray,
        columns: fc.Columns,
        present: npt.NDArray[np.bool_],
        k: int,
        horizon: float,
        eye: float,
    ) -> None:
        """Draw one layer of changes of space, z-tested, in two passes: a lower
        band (risers, back faces, contact shadows, floor surfaces) and an upper
        band (bulkheads, ceiling surfaces). Each pass is cropped to the rows it
        can touch and skipped if the layer has nothing of its kind, which keeps
        a desk layer from paying for the whole screen."""
        cols = np.nonzero(present)[0]
        f = columns.floats[:, k, cols]
        i = columns.ints[:, k, cols]
        cos = self._cos[cols]
        distance = f[fc.DISTANCE]
        wall_type = i[fc.WALL_TYPE]
        # Perpendicular distances. The camera's own space starts at 0: nudge it
        # so the formulas don't divide by zero (it has no faces to draw anyway).
        near = np.maximum(distance, 1e-6) * cos
        far = f[fc.EXIT_DISTANCE] * cos
        span = _Span(
            cols=cols,
            near=near,
            far=far,
            # Never closer than `near` (a camera exactly on a cell boundary can
            # give a zero-length span), never infinite (open-ended stretches).
            finite_far=np.maximum(np.where(np.isfinite(far), far, 1e9), near),
            top=f[fc.TOP],
            exit_top=f[fc.EXIT_TOP],
            base=f[fc.BASE],
            stairs=wall_type == STAIRS_WALL_TYPE,
            rail=wall_type == RAILING_WALL_TYPE,
            ceiling=f[fc.CEILING],
            prev_ceiling=f[fc.PREV_CEILING],
            block=wall_type != 0,
            slot=i[fc.SLOT].astype(np.intp),
            upper=i[fc.UPPER_SLOT].astype(np.intp),
            u=_texel_u(f[fc.U]),
            level_near=light_level(
                self._brightness(near)
                * np.where(i[fc.Y_SIDE] != 0, Y_SIDE_SHADE, 1.0)
                * self._ray_light(cols, distance)
            ),
            level_high=light_level(
                self._brightness(near)
                * np.where(i[fc.Y_SIDE] != 0, Y_SIDE_SHADE, 1.0)
                * self._ray_light(cols, distance, high=True)
            ),
            wall_type=wall_type,
            exit_distance=f[fc.EXIT_DISTANCE],
            exit_u=f[fc.EXIT_U],
            exit_y_side=i[fc.EXIT_Y_SIDE] != 0,
            exits_to_floor=i[fc.EXITS_TO_FLOOR] != 0,
            back_slot=i[fc.BACK_SLOT].astype(np.intp),
        )
        if span.block.any() or (span.top > 0).any() or (span.exit_top > 0).any():
            self._draw_lower(frame, zbuffer, span, horizon, eye)
        if (span.ceiling != 1.0).any() or (span.prev_ceiling > span.ceiling).any():
            self._draw_upper(frame, zbuffer, span, horizon, eye)

    def _crop(self, lows: list[FloatArray], highs: list[FloatArray]) -> tuple[int, int]:
        """The band of screen rows [r0, r1) that rows in `lows`..`highs` can reach.
        Off-screen and infinite rows (spans starting at the camera) clamp."""
        h = self.config.height
        low = min(float(np.min(np.clip(a, -1.0, h + 1.0))) for a in lows)
        high = max(float(np.max(np.clip(a, -1.0, h + 1.0))) for a in highs)
        return max(0, math.floor(low)), min(h, math.ceil(high))

    def _draw_lower(
        self,
        frame: U8Array,
        zbuffer: FloatArray,
        s: "_Span",
        horizon: float,
        eye: float,
    ) -> None:
        """Risers, back faces, contact shadows, floor surfaces and railings.

        Each part builds its mask with cheap comparisons over the band of rows,
        then does the expensive work (texel lookup, transparency, shading,
        dithering) only at the pixels that pass: `ci, ri = np.nonzero(mask)`.
        Faces and surfaces usually cover a small part of the band, so this is
        several times faster than evaluating textures for every pixel.
        """
        y_top = self._rows(s.top, s.near, horizon, eye)  # riser top / floor near edge
        y_base = self._rows(s.base, s.near, horizon, eye)  # riser bottom
        exit_top = np.where(np.isfinite(s.exit_top), s.exit_top, s.top)
        y_far = self._rows(exit_top, s.finite_far, horizon, eye)  # floor surface far edge
        y_far_floor = self._rows(np.zeros_like(s.top), s.finite_far, horizon, eye)
        lows = [y_top, y_far]
        if s.rail.any():  # railings stand above the balcony's edges
            lows += [
                self._rows(s.top + RAIL_HEIGHT, s.near, horizon, eye),
                self._rows(exit_top + RAIL_HEIGHT, s.finite_far, horizon, eye),
            ]
        r0, r1 = self._crop(lows, [y_base, y_far_floor, y_top])
        if r0 >= r1:
            return
        cols = s.cols
        py = self._pixel_y[None, r0:r1]
        rows_y = self._pixel_y[r0:r1]
        sub_frame = frame[cols, r0:r1]
        sub_z = zbuffer[cols, r0:r1]
        near, far = s.near, s.far

        # Riser face (a desk front, a step): floor before .. this floor, at `near`.
        # Transparent texels (the space under a desk) keep what's behind.
        # (Not for stairs: their steps are drawn as boxes, see _draw_boxes.)
        face = (
            ~s.stairs[:, None]
            & (py >= y_top[:, None])
            & (py < y_base[:, None])
            & (near[:, None] < sub_z)
        )
        ci, ri = np.nonzero(face)
        if ci.size:
            which, vi = self._texel_at(rows_y[ri], near[ci], s.slot[ci], s.upper[ci], horizon, eye)
            uu = s.u[ci]
            keep = self.opacity[which, uu, vi]
            ci, ri = ci[keep], ri[keep]
            texel = self.atlas[which[keep], uu[keep], vi[keep]]
            sub_frame[ci, ri] = COLORMAP[s.level_near[ci], texel]
            sub_z[ci, ri] = near[ci]

        # Back face of see-through blocks (an open desk's back legs), from the
        # floor up to its top at the exit distance: only shows through the gaps.
        exits_to_floor = s.exits_to_floor
        back_cols = s.block & exits_to_floor & ~self.opacity[s.slot].all(axis=(1, 2))
        if back_cols.any():
            back = (
                back_cols[:, None]
                & (py >= y_far[:, None])
                & (py < y_far_floor[:, None])
                & (far[:, None] < sub_z)
            )
            ci, ri = np.nonzero(back)
            if ci.size:
                back_slot = s.back_slot
                # Seen from inside the cell, so mirror u to match the outside view.
                back_u = _texel_u(1.0 - s.exit_u)
                back_y_side = s.exit_y_side
                depth = s.finite_far[ci]
                which, vi = self._texel_at(
                    rows_y[ri], depth, back_slot[ci], s.upper[ci], horizon, eye
                )
                uu = back_u[ci]
                keep = self.opacity[which, uu, vi]
                ci, ri, which, uu, vi = ci[keep], ri[keep], which[keep], uu[keep], vi[keep]
                shade = self._brightness(far[ci]) * np.where(back_y_side[ci], Y_SIDE_SHADE, 1.0)
                exits = s.exit_distance[ci]
                shade = shade * self._ray_light(cols[ci], exits)
                sub_frame[ci, ri] = COLORMAP[light_level(shade), self.atlas[which, uu, vi]]
                sub_z[ci, ri] = far[ci]

        with np.errstate(divide="ignore", invalid="ignore"):
            # Contact shadow: floor seen through the gap under an open desk lies
            # inside its footprint, so darken it. Grounds the furniture.
            if s.block.any():
                floor_depth = eye * self.projection / (py - horizon)
                under = (
                    (s.block & ~s.stairs & (s.base == 0.0))[:, None]
                    & (py > horizon)
                    & (floor_depth >= near[:, None])
                    & (floor_depth <= far[:, None])
                    & np.isinf(sub_z)  # still showing the default floor
                )
                sub_frame[under] = COLORMAP[UNDER_DESK_SHADOW, sub_frame[under]]

            # Floor surface of this span (a desk top, a raised floor, stairs), if
            # it isn't the default floor. Along a ray its height goes linearly
            # from `top` at `near` to `exit_top` at `far` (equal unless it's a
            # sloping stair), slope m per unit of perpendicular distance. The
            # pixel at row y sees height eye - (y - horizon) d / projection at
            # distance d; setting that equal to top + m (d - near) gives
            #   d = (eye - top + m near) / ((y - horizon) / projection + m),
            # which for a flat surface (m = 0) is the usual floor formula.
            raised = ((s.top > 0.0) | (exit_top > 0.0)) & ~s.stairs
            if raised.any():
                span_len = np.maximum(s.finite_far - near, 1e-9)
                m = np.where(np.isfinite(s.far), (exit_top - s.top) / span_len, 0.0)
                denom = (py - horizon) / self.projection + m[:, None]
                row_depth = (eye - s.top + m * near)[:, None] / denom
                # No "below the horizon" test: a stair rising ahead of you climbs
                # above eye level, so its far part is above the horizon. A flat
                # surface above eye level gives a negative distance, excluded by
                # the range test anyway.
                on_floor = (
                    raised[:, None]
                    & (denom > 0)
                    & (row_depth >= near[:, None])
                    & (row_depth <= far[:, None])
                    & (row_depth < sub_z)
                )
                ci, ri = np.nonzero(on_floor)
                if ci.size:
                    depth = row_depth[ci, ri]
                    colour = np.where(s.block, self.top_colour[s.slot], self.floor_colour)[ci]
                    kind = self._top_kind[np.minimum(s.wall_type, len(self._top_kind) - 1)][ci]
                    carpeted = kind >= 0  # textured like a floor (areas.TOP_TEXTURES)
                    if self.textured and carpeted.any():
                        c, k, d = cols[ci][carpeted], kind[carpeted], depth[carpeted]
                        along = d / self._cos[c]
                        colour[carpeted] = self._surface_texels(
                            self._frame_camera.x + self._ray_dx[c] * along,
                            self._frame_camera.y + self._ray_dy[c] * along,
                            d,
                            eye - s.top[ci][carpeted],
                            self._cos[c],
                            k,
                            self._mip_dither[c, (r0 + ri)[carpeted]],
                        )
                    on_light = self._ray_light(cols[ci], depth / self._cos[cols[ci]], backoff=0.0)
                    lit = self._dithered_at(colour, depth, cols[ci], r0 + ri, on_light)
                    lit = self._desk_seams(lit, s.wall_type, ci, cols[ci], depth, eye - s.top[ci])
                    sub_frame[ci, ri] = lit
                    sub_z[ci, ri] = depth

        if s.rail.any():
            # Railings along balcony edges that drop to a lower floor: above
            # the front face where the ray climbs onto the balcony (seen from
            # below), and where it leaves over the drop (seen from on it).
            drop_in = s.rail & (s.top - s.base > STEP_HEIGHT)
            drop_out = s.rail & exits_to_floor
            # Leaving, the ray sees that edge from inside: mirror u (as for
            # desks' back faces) so the balusters line up with the outside view.
            exit_u = _texel_u(1.0 - s.exit_u)
            for mask, edge_top, depth_cols, rail_u in (
                (drop_in, s.top, near, s.u),
                (drop_out, exit_top, s.finite_far, exit_u),
            ):
                if not mask.any():
                    continue
                y_rail = self._rows(edge_top + RAIL_HEIGHT, depth_cols, horizon, eye)
                y_edge = self._rows(edge_top, depth_cols, horizon, eye)
                rail = (
                    mask[:, None]
                    & (py >= y_rail[:, None])
                    & (py < y_edge[:, None])
                    & (depth_cols[:, None] < sub_z)
                )
                ci, ri = np.nonzero(rail)
                if not ci.size:
                    continue
                depth = depth_cols[ci]
                height = eye - (rows_y[ri] - horizon) * depth / self.projection
                above = (height - edge_top[ci]) / RAIL_HEIGHT  # 0 at the floor, 1 at the top
                rv = np.clip(((1.0 - above) * TEX_SIZE).astype(np.intp), 0, TEX_MASK)
                ru = rail_u[ci]
                keep = self.rail_opaque[ru, rv]
                ci, ri, ru, rv, depth = ci[keep], ri[keep], ru[keep], rv[keep], depth[keep]
                rail_light = self._ray_light(cols[ci], depth / self._cos[cols[ci]], backoff=0.0)
                level = light_level(self._brightness(depth) * rail_light)
                sub_frame[ci, ri] = COLORMAP[level, self.rail_tex[ru, rv]]
                sub_z[ci, ri] = depth

        frame[cols, r0:r1] = sub_frame
        zbuffer[cols, r0:r1] = sub_z

    def _texel_at(
        self,
        y: FloatArray,
        depth: FloatArray,
        slot: IntArray,
        upper: IntArray,
        horizon: float,
        eye: float,
    ) -> tuple[IntArray, IntArray]:
        """(slot, v) for individual pixels of vertical faces: row y, perpendicular
        depth. Same rule as _face_texels: floor-anchored, upper texture above a storey."""
        height = eye - (y - horizon) * depth / self.projection
        vi: IntArray = ((1.0 - np.mod(height, 1.0)) * TEX_SIZE).astype(np.intp) & TEX_MASK
        which: IntArray = np.where(height >= 1.0, upper, slot)
        return which, vi

    def _desk_seams(
        self,
        lit: U8Array,
        wall_type: Int64Array,
        ci: IntArray,
        cols: IntArray,
        depth: FloatArray,
        drop: FloatArray,
    ) -> U8Array:
        """Joins on the tops of desk pairs: a dark seam across the middle of
        each cell, perpendicular to its facing, where the two desks meet; and
        a fainter join at the cell edges between neighbouring pairs in a row.
        Pixels are put back into the world (camera + ray * distance) to find
        which are on a line. `drop` is the eye's height above the surface."""
        seam = np.isin(wall_type, list(TOP_SEAM_WALL_TYPES))[ci]
        if not seam.any():
            return lit
        camera = self._frame_camera
        grid = self._seam_grid_for(self._frame_map)
        t = depth / self._cos[cols]
        wx = camera.x + self._ray_dx[cols] * t
        wy = camera.y + self._ray_dy[cols] * t
        # A pixel's footprint on the surface: depth / projection across the ray
        # and, seen at a slant, depth^2 / (projection * drop) along it. Lines are
        # at least half of it wide, or they fall between pixel centres.
        across = depth / self.projection / self._cos[cols]
        along = across * depth / np.maximum(drop, 1e-6)
        rx, ry = np.abs(self._ray_dx[cols]), np.abs(self._ray_dy[cols])
        reach_x, reach_y = 0.5 * (along * rx + across * ry), 0.5 * (along * ry + across * rx)

        gx = np.clip(np.floor(wx).astype(np.intp), 0, grid.shape[1] - 1)
        gy = np.clip(np.floor(wy).astype(np.intp), 0, grid.shape[0] - 1)
        facing = grid[gy, gx]  # -1 unless this pixel is on a desk pair
        ns = facing % 2 == 0  # facing north/south: the pair meets along y
        # The seam: the middle of the cell across the facing.
        middle = np.where(ns, np.abs(wy - gy - 0.5), np.abs(wx - gx - 0.5))
        on_seam = (
            seam
            & (facing >= 0)
            & (middle < np.maximum(DESK_SEAM_WIDTH, np.where(ns, reach_y, reach_x)))
        )
        # Joins: the nearest cell edge along the row, if the pair beyond it is
        # part of the same row (a desk pair facing the same way).
        edge = np.where(ns, np.round(wx), np.round(wy)).astype(np.intp)
        own = np.where(ns, gx, gy)
        other = np.where(own == edge, edge - 1, edge)  # the cell across that edge
        h, w = grid.shape
        neighbour = np.where(
            ns,
            grid[gy, np.clip(other, 0, w - 1)],
            grid[np.clip(other, 0, h - 1), gx],
        )
        gap = np.where(ns, np.abs(wx - edge), np.abs(wy - edge))
        width = np.maximum(DESK_SEAM_WIDTH, np.where(ns, reach_x, reach_y))
        on_join = seam & (facing >= 0) & (neighbour == facing) & (gap < width)
        result: U8Array = np.where(
            on_seam,
            COLORMAP[DESK_SEAM_DARKEN, lit],
            np.where(on_join, COLORMAP[DESK_JOIN_DARKEN, lit], lit),
        )
        return result

    def _seam_grid_for(self, game_map: GameMap) -> IntArray:
        """Per cell: its facing if it's a desk pair (has a top seam), else -1.
        Built once per map."""
        if self._seam_map is not game_map:
            self._seam_grid = np.array(
                [
                    [
                        game_map.facing_at(x, y)
                        if wall in TOP_SEAM_WALL_TYPES and game_map.is_low_block(x, y)
                        else -1
                        for x, wall in enumerate(row)
                    ]
                    for y, row in enumerate(game_map.walls)
                ],
                dtype=np.intp,
            )
            self._seam_map = game_map
        return self._seam_grid

    def _dithered_at(
        self,
        colour: npt.NDArray[np.integer],
        depth: FloatArray,
        cols: IntArray,
        rows: IntArray,
        light: FloatArray | float = 1.0,
    ) -> U8Array:
        """Distance-shaded, dithered palette indices for individual pixels of a
        horizontal surface (same rule as _dithered)."""
        brightness = self._brightness(np.nan_to_num(depth, nan=np.inf, posinf=1e9)) * light
        exact = ((colour & 0x0F) + 1) * brightness - 1
        shade = np.floor(exact + self._dither[cols, rows]).clip(0, 15).astype(np.intp)
        result: U8Array = ((colour & 0xF0) | shade).astype(np.uint8)
        return result

    def _draw_upper(
        self, frame: U8Array, zbuffer: FloatArray, s: "_Span", horizon: float, eye: float
    ) -> None:
        y_ceiling = self._rows(s.ceiling, s.near, horizon, eye)  # bulkhead bottom
        y_prev_ceiling = self._rows(s.prev_ceiling, s.near, horizon, eye)  # bulkhead top
        y_ceiling_far = self._rows(s.ceiling, s.finite_far, horizon, eye)
        # Ceiling surfaces are above eye level (you only see a ceiling from
        # below), but a bulkhead is a wall face: from a balcony, with your eye
        # above its bottom edge, it reaches below the horizon.
        _, ceilings_end = self._crop([y_ceiling], [y_ceiling_far])
        r0, bulkheads_end = self._crop([y_prev_ceiling, y_ceiling], [y_ceiling])
        r1 = max(min(ceilings_end, math.ceil(horizon)), bulkheads_end)
        if r0 >= r1:
            return
        cols = s.cols
        py = self._pixel_y[None, r0:r1]
        sub_frame = frame[cols, r0:r1]
        sub_z = zbuffer[cols, r0:r1]
        near, far = s.near, s.far

        # Bulkhead face: where the ceiling comes down, prev ceiling .. this ceiling.
        bulk = (py >= y_prev_ceiling[:, None]) & (py < y_ceiling[:, None]) & (near[:, None] < sub_z)
        if bulk.any():
            which, vi = self._face_texels(py, near, s.upper, s.upper, horizon, eye)
            texel = self.atlas[which, s.u[:, None], vi]
            sub_frame = np.where(bulk, COLORMAP[s.level_high[:, None], texel], sub_frame)
            sub_z = np.where(bulk, near[:, None], sub_z)

        # Ceiling surface at this span's height, if it isn't the default.
        lifted = (s.ceiling != 1.0) & (s.ceiling > eye)
        if lifted.any():
            with np.errstate(divide="ignore", invalid="ignore"):
                row_depth = (s.ceiling - eye)[:, None] * self.projection / (horizon - py)
            on_ceiling = (
                lifted[:, None]
                & (py < horizon)
                & (row_depth >= near[:, None])
                & (row_depth <= far[:, None])
                & (row_depth < sub_z)
            )
            if on_ceiling.any():
                colour: npt.NDArray[np.integer] = np.full(len(cols), self.ceiling_colour)
                if self.textured:
                    seen = np.where(on_ceiling, row_depth, near[:, None])  # finite everywhere
                    cos = self._cos[cols][:, None]
                    along = np.minimum(seen / cos, 64.0)
                    # Lower than a storey: a soffit (over a doorway), lined, not tiled.
                    kind = np.where(s.ceiling < 1.0, SOFFIT, CEILING)[:, None]
                    colour = self._surface_texels(
                        self._frame_camera.x + self._ray_dx[cols][:, None] * along,
                        self._frame_camera.y + self._ray_dy[cols][:, None] * along,
                        seen,
                        (s.ceiling - eye)[:, None],
                        cos,
                        kind,
                        self._mip_dither[cols, r0:r1],
                    )
                lit = self._dithered(
                    colour, row_depth, cols, r0, r1, self._ceiling_light(cols, row_depth, s.ceiling)
                )
                sub_frame = np.where(on_ceiling, lit, sub_frame)
                sub_z = np.where(on_ceiling, row_depth, sub_z)

        frame[cols, r0:r1] = sub_frame
        zbuffer[cols, r0:r1] = sub_z

    def _lights_in(self, game_map: GameMap) -> list[tuple[float, float, float, bool]]:
        """(x, y, ceiling height, on the emergency circuit) of each light
        panel: one per `ceiling_light` thing, centred in its cell."""
        if self._lights_map is not game_map:
            self._light_list = []
            for t in game_map.things:
                if t.type == "ceiling_light":
                    cx, cy = math.floor(t.x), math.floor(t.y)
                    emergency = t.prop("circuit") == "emergency"
                    centre = (cx + 0.5, cy + 0.5, game_map.ceiling_height(cx, cy), emergency)
                    self._light_list.append(centre)
            self._lights_map = game_map
        return self._light_list

    def _draw_ceiling_lights(
        self,
        frame: U8Array,
        zbuffer: FloatArray,
        game_map: GameMap,
        camera: Camera,
        horizon: float,
        eye: float,
    ) -> None:
        """Fluorescent light panels set into the ceiling, by ceiling casting.

        Each panel's corners are projected to find its screen bounding box (as
        with sprites), so only those pixels are examined. For each, the column's
        ray direction and the row give the world point on the panel's ceiling
        plane; it's drawn if that point is inside the panel and nearer than
        whatever is there already (the z-buffer, or for the default ceiling,
        that ceiling's own depth).
        """
        w, h = self.config.width, self.config.height
        cos_a, sin_a = math.cos(camera.angle), math.sin(camera.angle)
        half_x, half_y = LIGHT_HALF
        for lx, ly, ceiling, emergency in self._lights_in(game_map):
            if ceiling <= eye:
                continue
            corners = [(lx + sx * half_x, ly + sy * half_y) for sx in (-1, 1) for sy in (-1, 1)]
            cols, rows = [], []
            for x, y in corners:
                dx, dy = x - camera.x, y - camera.y
                depth = dx * cos_a + dy * sin_a
                if depth <= NEAR_PLANE:
                    break  # partly behind you: skip (it's overhead and nearly edge-on)
                lateral = -dx * sin_a + dy * cos_a
                cols.append(w / 2 + lateral * self.projection / depth)
                rows.append(horizon + (eye - ceiling) * self.projection / depth)
            else:
                c0, c1 = max(0, math.floor(min(cols))), min(w, math.ceil(max(cols)) + 1)
                r0, r1 = max(0, math.floor(min(rows))), min(h, math.ceil(max(rows)) + 1)
                r1 = min(r1, math.floor(horizon))
                if c0 < c1 and r0 < r1:
                    lit = self._power or emergency
                    self._draw_light_panel(
                        frame, zbuffer, camera, lx, ly, ceiling, c0, c1, r0, r1, horizon, eye, lit
                    )

    def _draw_light_panel(
        self,
        frame: U8Array,
        zbuffer: FloatArray,
        camera: Camera,
        lx: float,
        ly: float,
        ceiling: float,
        c0: int,
        c1: int,
        r0: int,
        r1: int,
        horizon: float,
        eye: float,
        lit: bool = True,
    ) -> None:
        cols = np.arange(c0, c1)
        up = horizon - self._pixel_y[None, r0:r1]  # > 0 above the horizon
        depth = (ceiling - eye) * self.projection / up  # (1, rows) perpendicular
        angles = camera.angle + self._offsets_array[cols]
        along = depth / self._cos[cols][:, None]  # (cols, rows) along each ray
        wx = camera.x + np.cos(angles)[:, None] * along
        wy = camera.y + np.sin(angles)[:, None] * along
        ox, oy = np.abs(wx - lx), np.abs(wy - ly)
        half_x, half_y = LIGHT_HALF
        panel = (ox < half_x) & (oy < half_y)
        # Visible only if nothing nearer is drawn there. Where the z-buffer is
        # empty, the default ceiling (height 1) is showing at its own depth.
        there = zbuffer[c0:c1, r0:r1]
        default_ceiling = (1.0 - eye) * self.projection / up if eye < 1.0 else np.inf
        there = np.where(np.isinf(there), default_ceiling, there)
        panel &= depth <= there * (1 + 1e-6)
        if not panel.any():
            return
        diffuser = panel & (ox < half_x - 0.02) & (oy < half_y - 0.02)
        diffuser &= (ox > LIGHT_BAR) & (oy > LIGHT_BAR)  # four lit tiles, the grid across them
        frame_colour = COLORMAP[
            light_level(self._brightness(np.broadcast_to(depth, panel.shape))), LIGHT_FRAME
        ]
        region = frame[c0:c1, r0:r1]
        region[panel] = frame_colour[panel]
        if lit:
            region[diffuser] = LIGHT_DIFFUSER
        else:  # no power: a dead tube behind grey plastic, shaded like the frame
            dead = COLORMAP[
                light_level(self._brightness(np.broadcast_to(depth, panel.shape))), DEAD_DIFFUSER
            ]
            region[diffuser] = dead[diffuser]

    def _map_boxes_for(self, game_map: GameMap) -> list[PlacedBox]:
        """Boxes that are part of the map (stairs' steps, desk screens, fire
        plans), built once per map."""
        if self._map_boxes_map is not game_map:
            self._map_boxes = (
                stair_boxes(game_map) + desk_screens(game_map) + poster_plans(game_map)
            )
            self._map_boxes_map = game_map
        return self._map_boxes

    def _draw_boxes(
        self,
        frame: U8Array,
        zbuffer: FloatArray,
        camera: Camera,
        boxes: list[PlacedBox],
        horizon: float,
        eye: float,
    ) -> None:
        """Box models (boxes.py), drawn as solid geometry. Each box is tested
        against the rays of the columns it can cover, all at once (numpy, one
        row per box-column pair): the slab test, in each box's own frame.
        Where a ray enters a box by a side, that face is a vertical run of
        pixels at the entry depth (like a wall); the top (or bottom) is a
        stretch of rows between the entry and exit
        depths, cast back onto its plane (like a floor). Pixels from all boxes
        are then z-tested together, nearest wins. Used for box-model things
        and for the steps of stairs."""
        box_tex, box_opaque = box_textures()
        boxes, c0, c1 = self._in_view(boxes, camera)
        if not boxes:
            return
        h = self.config.height
        proj = self.projection

        def per_box(values: list[float]) -> FloatArray:
            return np.array(values)  # (B,)

        turned_b = per_box([p.angle + p.box.turn for p in boxes])
        ca, sa = np.cos(turned_b), np.sin(turned_b)
        rel_x = per_box([camera.x - p.x for p in boxes])
        rel_y = per_box([camera.y - p.y for p in boxes])
        # Camera into each box's frame (rotate by -angle).
        ox_b, oy_b = rel_x * ca + rel_y * sa, -rel_x * sa + rel_y * ca

        # The (box, column) pairs to test: each box only against the columns
        # its bounding circle covers on screen.
        counts = c1 - c0
        pair_box = np.repeat(np.arange(len(boxes)), counts)
        pair_col = (
            c0[pair_box] + np.arange(pair_box.size) - np.repeat(np.cumsum(counts) - counts, counts)
        )
        ray = camera.angle + self._offsets_array[pair_col]
        wx, wy = np.cos(ray), np.sin(ray)
        cb, sb = ca[pair_box], sa[pair_box]
        dx, dy = wx * cb + wy * sb, -wx * sb + wy * cb
        dx = np.where(np.abs(dx) < 1e-12, 1e-12, dx)
        dy = np.where(np.abs(dy) < 1e-12, 1e-12, dy)
        ox, oy = ox_b[pair_box], oy_b[pair_box]
        x0 = per_box([p.box.x[0] for p in boxes])[pair_box]
        x1 = per_box([p.box.x[1] for p in boxes])[pair_box]
        y0 = per_box([p.box.y[0] for p in boxes])[pair_box]
        y1 = per_box([p.box.y[1] for p in boxes])[pair_box]
        tx0, tx1 = (x0 - ox) / dx, (x1 - ox) / dx
        ty0, ty1 = (y0 - oy) / dy, (y1 - oy) / dy
        in_x, in_y = np.minimum(tx0, tx1), np.minimum(ty0, ty1)
        t_in = np.maximum(in_x, in_y)
        t_out = np.minimum(np.maximum(tx0, tx1), np.maximum(ty0, ty1))
        near_t = np.maximum(t_in, NEAR_PLANE)
        hit = np.flatnonzero(t_out > near_t)  # pairs whose ray crosses the box
        if not hit.size:
            return
        bi, ci = pair_box[hit], pair_col[hit]
        dx, dy, ox, oy = dx[hit], dy[hit], ox[hit], oy[hit]
        x0, x1, y0, y1 = x0[hit], x1[hit], y0[hit], y1[hit]
        in_x, in_y, t_in, t_out, near_t = in_x[hit], in_y[hit], t_in[hit], t_out[hit], near_t[hit]
        faces = np.array([p.box.faces for p in boxes])  # (B, 5)
        tiled = np.array([p.box.tiled_sides for p in boxes])
        riser = np.array([p.box.riser for p in boxes])
        cos_c = self._cos[ci]
        d_near = near_t * cos_c  # perpendicular depths, as in the z-buffer
        d_far = t_out * cos_c
        zt = per_box([p.z + p.box.z[1] for p in boxes])[bi]
        zb = per_box([p.z + p.box.z[0] for p in boxes])[bi]

        # Which face the ray enters by, where along it (u), and its light.
        from_x = in_x > in_y
        rdx, rdy = dx, dy
        face = np.where(from_x, np.where(rdx < 0, 0, 2), np.where(rdy < 0, 1, 3))
        hx, hy = ox + rdx * t_in, oy + rdy * t_in
        bx0, bx1, by0, by1 = x0, x1, y0, y1
        across = np.choose(
            face,
            [
                (by1 - hy) / (by1 - by0),  # front: the viewer's right is -y
                (hx - bx0) / (bx1 - bx0),  # right side: towards the front
                (hy - by0) / (by1 - by0),  # back
                (bx1 - hx) / (bx1 - bx0),  # left side
            ],
        )
        # World direction of the face's normal: north/south faces darker, as walls.
        normal = turned_b[bi] + face * (math.pi / 2)
        side_light = 1.0 - (1.0 - Y_SIDE_SHADE) * np.abs(np.sin(normal))

        cols: list[IntArray] = []
        rows: list[IntArray] = []
        depths: list[FloatArray] = []
        texels: list[npt.NDArray[np.uint8]] = []
        solid: list[npt.NDArray[np.bool_]] = []  # False: see-through (leaves)
        glow_b = np.array([p.box.glow and (self._power or not p.box.mains) for p in boxes])
        glows: list[npt.NDArray[np.bool_]] = []  # lit signs: full brightness
        lights: list[FloatArray] = []

        # Sides: where the ray enters by one (not from inside the footprint).
        # A riser (front/back of a step) is only its top slice.
        side = t_in > NEAR_PLANE
        short = (riser[bi] > 0) & (face % 2 == 0)
        face_bottom = np.where(short, zt - riser[bi], zb)
        p, r = self._runs(
            horizon + (eye - zt[side]) * proj / d_near[side],
            horizon + (eye - face_bottom[side]) * proj / d_near[side],
        )
        if p.size:
            k = np.flatnonzero(side)[p]
            depth = d_near[k]
            height = eye - (r + 0.5 - horizon) * depth / proj
            v = (zt[k] - height) / (zt[k] - face_bottom[k])
            tex = faces[bi[k], face[k]]
            tu, tv = _texel_clamped(across[k]), _texel_clamped(v)
            # Tiled left/right faces: texels from position in the world, like walls.
            world = tiled[bi[k]] & (face[k] % 2 == 1)
            if world.any():
                along = np.where(face[k] == 1, hx[k], -hx[k])
                tu = np.where(world, np.floor(along * TEX_SIZE).astype(np.intp) & TEX_MASK, tu)
                tile_v = np.floor((1.0 - height) * TEX_SIZE).astype(np.intp) & TEX_MASK
                tv = np.where(world, tile_v, tv)
            cols.append(ci[k])
            rows.append(r)
            depths.append(depth)
            texels.append(box_tex[tex, tu, tv])
            solid.append(box_opaque[tex, tu, tv])
            lights.append(side_light[k])
            glows.append(glow_b[bi[k]])

        # Top (seen from above) or bottom (from below): between entry and exit.
        for above in (True, False):
            plane = zt if above else zb
            sees = (eye > plane) if above else (eye < plane)
            first = np.where(above, d_far, d_near)
            last = np.where(above, d_near, d_far)
            p, r = self._runs(
                horizon + (eye - plane[sees]) * proj / first[sees],
                horizon + (eye - plane[sees]) * proj / last[sees],
            )
            if not p.size:
                continue
            k = np.flatnonzero(sees)[p]
            depth = (eye - plane[k]) * proj / (r + 0.5 - horizon)
            depth = np.clip(depth, d_near[k], d_far[k])
            t = depth / cos_c[k]
            px, py = ox[k] + dx[k] * t, oy[k] + dy[k] * t
            u = (y1[k] - py) / (y1[k] - y0[k])
            v = (px - x0[k]) / (x1[k] - x0[k])
            cols.append(ci[k])
            rows.append(r)
            depths.append(depth)
            top_tex, tu, tv = faces[bi[k], TOP], _texel_clamped(u), _texel_clamped(v)
            texels.append(box_tex[top_tex, tu, tv])
            solid.append(box_opaque[top_tex, tu, tv])
            lights.append(np.full(k.size, 1.0 if above else Y_SIDE_SHADE))
            glows.append(glow_b[bi[k]])

        if not cols:
            return
        keep = np.concatenate(solid)
        col, row = np.concatenate(cols)[keep], np.concatenate(rows)[keep]
        depth, texel = np.concatenate(depths)[keep], np.concatenate(texels)[keep]
        light = np.concatenate(lights)[keep]
        glow = np.concatenate(glows)[keep]
        # Where boxes overlap on screen, keep the nearest pixel (sort by pixel
        # then depth, take the first of each pixel), then z-test it.
        key = col * h + row
        order = np.lexsort((depth, key))
        first = np.ones(order.size, dtype=np.bool_)
        first[1:] = key[order][1:] != key[order][:-1]
        pick = order[first]
        pick = pick[depth[pick] < zbuffer[col[pick], row[pick]]]
        col, row, depth = col[pick], row[pick], depth[pick]
        lit = self._ray_light(col, depth / self._cos[col], backoff=0.0)
        shade = np.where(glow[pick], 1.0, self._brightness(depth) * light[pick] * lit)
        level = light_level(shade)
        frame[col, row] = COLORMAP[level, texel[pick]]
        zbuffer[col, row] = depth

    def _in_view(
        self, boxes: list[PlacedBox], camera: Camera
    ) -> tuple[list[PlacedBox], IntArray, IntArray]:
        """Boxes that may be on screen, and the screen columns [c0, c1) each
        can cover: from its bounding circle (round the placement, so turns
        don't matter), the angles it spans either side of its centre.
        Conservative, so nothing visible is dropped; most boxes are behind or
        beside you, or far away and a few columns wide."""
        w = self.config.width
        cos_a, sin_a = math.cos(camera.angle), math.sin(camera.angle)
        half_fov = math.atan((w / 2) / self.projection)
        kept, c0s, c1s = [], [], []
        for p in boxes:
            reach = _box_reach(p.box)
            dx, dy = p.x - camera.x, p.y - camera.y
            depth = dx * cos_a + dy * sin_a
            lateral = -dx * sin_a + dy * cos_a
            dist = math.hypot(depth, lateral)
            if dist <= reach + NEAR_PLANE:
                lo, hi = -half_fov, half_fov  # around the camera: anywhere
            else:
                centre, spread = math.atan2(lateral, depth), math.asin(reach / dist)
                lo, hi = max(centre - spread, -half_fov), min(centre + spread, half_fov)
                if lo >= hi:
                    continue  # outside the field of view (or behind)
            c0 = max(0, math.floor(w / 2 + math.tan(lo) * self.projection - 1))
            c1 = min(w, math.ceil(w / 2 + math.tan(hi) * self.projection + 1))
            if c0 < c1:
                kept.append(p)
                c0s.append(c0)
                c1s.append(c1)
        return kept, np.array(c0s, dtype=np.intp), np.array(c1s, dtype=np.intp)

    def _runs(self, start: FloatArray, end: FloatArray) -> tuple[IntArray, IntArray]:
        """Screen rows covered from row `start` to `end` (pixel centres in
        [start, end)), clipped to the screen, for many runs at once. Returns
        (run index, row) for every covered pixel."""
        h = self.config.height
        r0 = np.clip(np.ceil(start - 0.5), 0, h).astype(np.intp)
        r1 = np.clip(np.ceil(end - 0.5), 0, h).astype(np.intp)
        count = np.maximum(r1 - r0, 0)
        run = np.repeat(np.arange(count.size), count)
        within = np.arange(run.size) - np.repeat(np.cumsum(count) - count, count)
        return run, r0[run] + within

    def _draw_sprites(
        self,
        frame: U8Array,
        zbuffer: FloatArray,
        camera: Camera,
        sprites: Iterable[SpriteInstance],
        horizon: float,
        eye: float,
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
            bottom = horizon + (eye - s.z) * size + art.sink * size / 64  # on its surface
            top = bottom - size
            c0, c1 = max(0, math.ceil(left - 0.5)), min(w, math.ceil(left + size - 0.5))
            r0, r1 = max(0, math.ceil(top - 0.5)), min(h, math.ceil(bottom - 0.5))
            if c0 >= c1 or r0 >= r1:
                continue
            scale = 64 / size
            su = ((np.arange(c0, c1) + 0.5 - left) * scale).astype(np.intp).clip(0, 63)
            sv = ((np.arange(r0, r1) + 0.5 - top) * scale).astype(np.intp).clip(0, 63)
            region_z = zbuffer[c0:c1, r0:r1]
            frame_no = art.frame(math.atan2(-dy, -dx), s.angle)
            opaque, texels = art.opaque[frame_no], art.index[frame_no]
            near = max(depth - art.reach, NEAR_PLANE)  # its nearest point, for the z-test
            draw = opaque[su[:, None], sv[None, :]] & (near <= region_z)
            if not draw.any():
                continue
            index = texels[su[:, None], sv[None, :]]
            if not art.fullbright:
                lit = self._light.at(np.array([s.x]), np.array([s.y])) if self._light else 1.0
                level = light_level(self._brightness(np.array(depth)) * np.minimum(lit, 1.0))
                index = COLORMAP[level, index]
            region = frame[c0:c1, r0:r1]
            region[draw] = index[draw]
            region_z[draw] = depth


@dataclass(slots=True)
class _Span:
    """One layer's per-column values, shared by its lower and upper passes."""

    cols: IntArray
    near: FloatArray
    far: FloatArray
    finite_far: FloatArray
    top: FloatArray
    exit_top: FloatArray
    base: FloatArray
    stairs: npt.NDArray[np.bool_]
    rail: npt.NDArray[np.bool_]
    ceiling: FloatArray
    prev_ceiling: FloatArray
    block: npt.NDArray[np.bool_]
    slot: IntArray
    upper: IntArray
    u: IntArray
    level_near: IntArray
    level_high: IntArray  # for faces above head height (bulkheads): lintels stop light
    wall_type: Int64Array
    exit_distance: FloatArray
    exit_u: FloatArray
    exit_y_side: npt.NDArray[np.bool_]
    exits_to_floor: npt.NDArray[np.bool_]
    back_slot: IntArray


@functools.cache
def _box_reach(box: Box) -> float:
    """How far a box reaches from its model's origin, in any direction."""
    return max(math.hypot(x, y) for x in box.x for y in box.y)


def _texel_clamped(v: FloatArray) -> IntArray:
    """Texel (0-63) for positions 0..1 across a face, clamped (not wrapped):
    box faces don't tile."""
    result: IntArray = np.clip((v * TEX_SIZE).astype(np.intp), 0, TEX_MASK)
    return result


def _texel_u(u: FloatArray) -> IntArray:
    """Texture column (0-63) for positions 0..1 along a face."""
    result: IntArray = (u * TEX_SIZE).astype(np.intp) & TEX_MASK
    return result


def _top_colours(atlas: U8Array, slot_table: IntArray) -> U8Array:
    """Palette index for the top surface of a low block, per atlas slot.

    From TOP_COLOURS where given (for every slot of that wall type), otherwise
    the commonest colour in the top quarter of the slot's texture.
    """
    colours = np.zeros(len(atlas), dtype=np.uint8)
    for slot in range(len(atlas)):
        values, counts = np.unique(atlas[slot][:, :16], return_counts=True)
        colours[slot] = values[np.argmax(counts)]
    for wall_type, colour in TOP_COLOURS.items():
        if wall_type < len(slot_table):
            colours[slot_table[wall_type]] = colour
    return colours

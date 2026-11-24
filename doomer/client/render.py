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

import math
from collections.abc import Iterable
from dataclasses import dataclass
from typing import NamedTuple

import numpy as np
import numpy.typing as npt
import pygame

from doomer.client.config import RenderConfig
from doomer.client.palette import COLORMAP, PALETTE, U8Array, light_level, nearest_index
from doomer.client.sprites import SPRITES
from doomer.client.textures import (
    TEX_MASK,
    TEX_SIZE,
    TOP_COLOURS,
    UPPER,
    build_atlas,
    build_opacity,
)
from doomer.shared.mapfile import WALL_UNITS, GameMap
from doomer.shared.raycast import RayHit, Side, cast_ray, cell_exit_distance, start_heights

MIN_SHADE = 0.25
Y_SIDE_SHADE = 0.72  # north/south faces darker than east/west, so corners read clearly
# Default camera height in walls: a standing eye at ~1.6 m under a 2.7 m ceiling.
EYE_HEIGHT = 0.6
MAX_PITCH = math.radians(25)  # y-shearing looks wrong much beyond this
# Bayer matrix: each value 0-15 appears once, arranged so that any threshold
# switches on pixels that are spread as evenly as possible.
BAYER_4X4 = np.array([[0, 8, 2, 10], [12, 4, 14, 6], [3, 11, 1, 9], [15, 7, 13, 5]])
BULKHEAD_TYPE = 1  # wall type whose upper texture faces a dropped ceiling
NEAR_PLANE = 0.05  # sprites closer than this (in cells) aren't drawn
UNDER_DESK_SHADOW = 12  # colormap levels darker (of 32) for the floor under a desk
_BACKGROUND_CACHE_SIZE = 64

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.intp]


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
            z = game_map.top_height(cx, cy)
        out.append(SpriteInstance(t.type, t.x, t.y, z))
    return out


class SpanHit(NamedTuple):
    """A change of space on one column's ray, with what the renderer needs."""

    hit: RayHit
    exits_to_floor: bool  # a low block left into open floor: its back face may show
    slot: int  # atlas slot of the face the ray enters by
    back_slot: int  # ... and of the face it leaves by
    upper_slot: int  # texture for a bulkhead face above it


def make_frame_surface(config: RenderConfig) -> pygame.Surface:
    """An 8-bit surface carrying the game palette, for Renderer.render to fill."""
    surface = pygame.Surface((config.width, config.height), depth=8)
    surface.set_palette([tuple(int(c) for c in rgb) for rgb in PALETTE])
    return surface


class Renderer:
    def __init__(self, config: RenderConfig) -> None:
        self.config = config
        w, h = config.width, config.height
        tan_half_fov = math.tan(math.radians(config.fov_degrees) / 2)
        # Distance from the eye to a virtual screen, in pixels.
        self.projection = (w / 2) / tan_half_fov
        self._offsets = [math.atan((2 * (c + 0.5) / w - 1) * tan_half_fov) for c in range(w)]
        self._cos = np.cos(np.array(self._offsets))
        self._pixel_y = np.arange(h, dtype=np.float64) + 0.5  # pixel centres
        # slot_table[wall_type, side] -> atlas slot; side = FRONT/RIGHT/BACK/LEFT/UPPER
        self.atlas, self.slot_table = build_atlas()
        self.opacity = build_opacity()  # (slot, u, v): False = see-through (open desks)
        self.top_colour = _top_colours(self.atlas, self.slot_table)
        self.ceiling_colour = nearest_index(PALETTE, config.ceiling_colour)
        self.floor_colour = nearest_index(PALETTE, config.floor_colour)
        x, y = np.meshgrid(np.arange(w), np.arange(h), indexing="ij")
        self._dither = (BAYER_4X4[x & 3, y & 3] + 0.5) / 16  # (w, h) thresholds in (0, 1)
        self._backgrounds: dict[tuple[int, int], U8Array] = {}
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

    def _dithered(
        self, colour: npt.NDArray[np.integer], depth: FloatArray, cols: IntArray, r0: int, r1: int
    ) -> U8Array:
        """Distance-shaded, dithered palette indices for a horizontal surface:
        `colour` per column, `depth` per pixel of the (cols, r0:r1) block. Same
        ordered dithering as the background, so large floors and ceilings at
        other heights don't show hard bands."""
        brightness = self._brightness(np.nan_to_num(depth, nan=np.inf, posinf=1e9))
        exact = ((colour[:, None] & 0x0F) + 1) * brightness - 1
        shade = np.floor(exact + self._dither[cols, r0:r1]).clip(0, 15).astype(np.intp)
        result: U8Array = ((colour[:, None] & 0xF0) | shade).astype(np.uint8)
        return result

    def _brightness(self, distance: FloatArray) -> FloatArray:
        result: FloatArray = np.maximum(MIN_SHADE, 1.0 - distance / self.config.shade_distance)
        return result

    # --- frame ------------------------------------------------------------------------

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
        pitch = max(-MAX_PITCH, min(MAX_PITCH, camera.pitch))
        horizon = self.config.height / 2 + math.tan(pitch) * self.projection
        eye = camera.z
        self.horizon = horizon

        # The space the camera is in. If it isn't the default (standing on a
        # desk, under the atrium's high ceiling), its own floor and ceiling are
        # drawn too, as a span that starts at distance 0.
        cx, cy = math.floor(camera.x), math.floor(camera.y)
        floor0, ceiling0 = start_heights(game_map, cx, cy)
        own_space = (floor0, ceiling0) != (0, WALL_UNITS)
        on_block = game_map.is_low_block(cx, cy)

        # 1. One ray per column (Python loop). The last hit is the full wall;
        # the others are changes of space, sorted into layers: layer k holds
        # each column's k-th change.
        dist = [math.inf] * w
        wall_slot = [0] * w
        upper_slot = [0] * w
        tex_u = [0.0] * w
        y_side = [False] * w
        wall_base = [0.0] * w
        wall_top = [1.0] * w
        layers: list[list[SpanHit | None]] = []
        bulkhead = int(self.slot_table[BULKHEAD_TYPE, UPPER])
        for col in range(w):
            hits = cast_ray(game_map, camera.x, camera.y, camera.angle + self._offsets[col])
            if hits and hits[-1].full:
                hit = hits.pop()
                dist[col] = hit.distance
                wall_slot[col] = self._slot(game_map, hit, hit.face)
                upper_slot[col] = self._upper_slot(hit.wall_type)
                tex_u[col] = hit.u
                y_side[col] = hit.side is Side.Y
                wall_base[col] = hit.base
                wall_top[col] = hit.prev_ceiling
            if own_space:
                # The camera's own span ends at the next change, or (standing on
                # a desk) where the ray leaves the desk's cell.
                ray_angle = camera.angle + self._offsets[col]
                end = hits[0].distance if hits else dist[col]
                if on_block:
                    end = min(end, cell_exit_distance(camera.x, camera.y, ray_angle))
                hits.insert(0, _own_space(floor0, ceiling0, end))
            for k, span in enumerate(hits):
                if k == len(layers):
                    layers.append([None] * w)
                block = span.wall_type != 0
                after = hits[k + 1].distance if k + 1 < len(hits) else dist[col]
                layers[k][col] = SpanHit(
                    span,
                    block and after > span.exit_distance + 1e-9,
                    self._slot(game_map, span, span.face) if block else 0,
                    self._slot(game_map, span, span.exit_face) if block else 0,
                    self._upper_slot(span.wall_type) if block else bulkhead,
                )

        # 2. Walls, floor to ceiling of the space in front of them (numpy).
        perp = np.array(dist) * self._cos
        frame, zbuffer = self._draw_walls(
            perp,
            np.array(wall_slot, dtype=np.intp),
            np.array(upper_slot, dtype=np.intp),
            np.array(tex_u),
            np.array(y_side),
            np.array(wall_base),
            np.array(wall_top),
            horizon,
            eye,
        )

        # 3. Changes of space, one layer at a time, z-tested.
        for layer in layers:
            self._draw_span_layer(frame, zbuffer, layer, horizon, eye)

        self._draw_sprites(frame, zbuffer, camera, sprites, horizon, eye)
        self.zbuffer = zbuffer
        return frame

    def _slot(self, game_map: GameMap, hit: RayHit, face: int) -> int:
        """Atlas slot for one face of a wall cell: which side of the cell that
        face is (from the cell's own point of view) picks the texture."""
        if hit.wall_type >= len(self.slot_table):
            return 0  # unknown wall type -> missing texture
        side = (face - game_map.facing_at(hit.cell_x, hit.cell_y)) % 4
        return int(self.slot_table[hit.wall_type, side])

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

    def _draw_walls(
        self,
        perp: FloatArray,
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
        brightness = self._brightness(perp) * np.where(y_side, Y_SIDE_SHADE, 1.0)
        level = light_level(brightness)
        y_top = self._rows(top, depth, horizon, eye)
        y_bottom = self._rows(base, depth, horizon, eye)
        py = self._pixel_y[None, :]
        on_wall = has_wall[:, None] & (py >= y_top[:, None]) & (py < y_bottom[:, None])
        which, vi = self._face_texels(py, depth, slot, upper, horizon, eye)
        texel = self.atlas[which, u[:, None], vi]  # (w, h) palette indices
        lit = COLORMAP[level[:, None], texel]  # shading = one table lookup
        frame: U8Array = np.where(on_wall, lit, self.background(horizon, eye))
        # Default floor and ceiling never hide anything: they count as infinitely far.
        zbuffer: FloatArray = np.where(on_wall, perp[:, None], np.inf)
        return frame, zbuffer

    def _draw_span_layer(
        self,
        frame: U8Array,
        zbuffer: FloatArray,
        layer: list[SpanHit | None],
        horizon: float,
        eye: float,
    ) -> None:
        """Draw one layer of changes of space, z-tested, in two passes: a lower
        band (risers, back faces, contact shadows, floor surfaces) and an upper
        band (bulkheads, ceiling surfaces). Each pass is cropped to the rows it
        can touch and skipped if the layer has nothing of its kind, which keeps
        a desk layer from paying for the whole screen."""
        cols = np.array([c for c, entry in enumerate(layer) if entry is not None], dtype=np.intp)
        entries = [entry for entry in layer if entry is not None]
        hits = [e.hit for e in entries]
        cos = self._cos[cols]
        # Perpendicular distances. The camera's own space starts at 0: nudge it
        # so the formulas don't divide by zero (it has no faces to draw anyway).
        near = np.maximum(np.array([h.distance for h in hits]), 1e-6) * cos
        far = np.array([h.exit_distance for h in hits]) * cos
        span = _Span(
            cols=cols,
            near=near,
            far=far,
            finite_far=np.where(np.isfinite(far), far, 1e9),
            top=np.array([h.top for h in hits]),
            base=np.array([h.base for h in hits]),
            ceiling=np.array([h.ceiling for h in hits]),
            prev_ceiling=np.array([h.prev_ceiling for h in hits]),
            block=np.array([h.wall_type != 0 for h in hits]),
            slot=np.array([e.slot for e in entries], dtype=np.intp),
            upper=np.array([e.upper_slot for e in entries], dtype=np.intp),
            u=_texel_u(np.array([h.u for h in hits])),
            level_near=light_level(
                self._brightness(near)
                * np.where(np.array([h.side is Side.Y for h in hits]), Y_SIDE_SHADE, 1.0)
            ),
        )
        if span.block.any() or (span.top > 0).any():
            self._draw_lower(frame, zbuffer, span, entries, horizon, eye)
        if (span.ceiling != 1.0).any() or (span.prev_ceiling > span.ceiling).any():
            self._draw_upper(frame, zbuffer, span, horizon, eye)

    def _crop(self, lows: list[FloatArray], highs: list[FloatArray]) -> tuple[int, int]:
        r0 = max(0, math.floor(min(float(np.min(a)) for a in lows)))
        r1 = min(self.config.height, math.ceil(max(float(np.max(a)) for a in highs)))
        return r0, r1

    def _draw_lower(
        self,
        frame: U8Array,
        zbuffer: FloatArray,
        s: "_Span",
        entries: list[SpanHit],
        horizon: float,
        eye: float,
    ) -> None:
        hits = [e.hit for e in entries]
        y_top = self._rows(s.top, s.near, horizon, eye)  # riser top / floor near edge
        y_base = self._rows(s.base, s.near, horizon, eye)  # riser bottom
        y_far = self._rows(s.top, s.finite_far, horizon, eye)  # floor surface far edge
        y_far_floor = self._rows(np.zeros_like(s.top), s.finite_far, horizon, eye)
        r0, r1 = self._crop([y_top, y_far], [y_base, y_far_floor, y_top])
        if r0 >= r1:
            return
        cols = s.cols
        py = self._pixel_y[None, r0:r1]
        sub_frame = frame[cols, r0:r1]
        sub_z = zbuffer[cols, r0:r1]
        near, far = s.near, s.far

        # Riser face (a desk front, a step): floor before .. this floor, at `near`.
        # Transparent texels (the space under a desk) keep what's behind.
        face = (py >= y_top[:, None]) & (py < y_base[:, None]) & (near[:, None] < sub_z)
        if face.any():
            which, vi = self._face_texels(py, near, s.slot, s.upper, horizon, eye)
            face &= self.opacity[which, s.u[:, None], vi]
            texel = self.atlas[which, s.u[:, None], vi]
            sub_frame = np.where(face, COLORMAP[s.level_near[:, None], texel], sub_frame)
            sub_z = np.where(face, near[:, None], sub_z)

        # Back face of see-through blocks (an open desk's back legs), from the
        # floor up to its top at the exit distance: only shows through the gaps.
        exits_to_floor = np.array([e.exits_to_floor for e in entries])
        back_cols = s.block & exits_to_floor & ~self.opacity[s.slot].all(axis=(1, 2))
        if back_cols.any():
            back_slot = np.array([e.back_slot for e in entries], dtype=np.intp)
            # Seen from inside the cell, so mirror u to match the outside view.
            back_u = _texel_u(1.0 - np.array([h.exit_u for h in hits]))
            back_y_side = np.array([h.exit_side is Side.Y for h in hits])
            back = (
                back_cols[:, None]
                & (py >= y_far[:, None])
                & (py < y_far_floor[:, None])
                & (far[:, None] < sub_z)
            )
            which_f, vi_f = self._face_texels(py, s.finite_far, back_slot, s.upper, horizon, eye)
            back &= self.opacity[which_f, back_u[:, None], vi_f]
            texel_f = self.atlas[which_f, back_u[:, None], vi_f]
            level_far = light_level(
                self._brightness(far) * np.where(back_y_side, Y_SIDE_SHADE, 1.0)
            )
            sub_frame = np.where(back, COLORMAP[level_far[:, None], texel_f], sub_frame)
            sub_z = np.where(back, far[:, None], sub_z)

        with np.errstate(divide="ignore", invalid="ignore"):
            below = py > horizon
            # Contact shadow: floor seen through the gap under an open desk lies
            # inside its footprint, so darken it. Grounds the furniture.
            if s.block.any():
                floor_depth = eye * self.projection / (py - horizon)
                under = (
                    (s.block & (s.base == 0.0))[:, None]
                    & below
                    & (floor_depth >= near[:, None])
                    & (floor_depth <= far[:, None])
                    & np.isinf(sub_z)  # still showing the default floor
                )
                sub_frame = np.where(under, COLORMAP[UNDER_DESK_SHADOW, sub_frame], sub_frame)

            # Floor surface at this span's height (a desk top, a raised floor),
            # if it isn't the default floor and is below eye level.
            raised = (s.top > 0.0) & (s.top < eye)
            if raised.any():
                row_depth = (eye - s.top)[:, None] * self.projection / (py - horizon)
                on_floor = (
                    raised[:, None]
                    & below
                    & (row_depth >= near[:, None])
                    & (row_depth <= far[:, None])
                    & (row_depth < sub_z)
                )
                if on_floor.any():
                    colour = np.where(s.block, self.top_colour[s.slot], self.floor_colour)
                    lit = self._dithered(colour, row_depth, cols, r0, r1)
                    sub_frame = np.where(on_floor, lit, sub_frame)
                    sub_z = np.where(on_floor, row_depth, sub_z)

        frame[cols, r0:r1] = sub_frame
        zbuffer[cols, r0:r1] = sub_z

    def _draw_upper(
        self, frame: U8Array, zbuffer: FloatArray, s: "_Span", horizon: float, eye: float
    ) -> None:
        y_ceiling = self._rows(s.ceiling, s.near, horizon, eye)  # bulkhead bottom
        y_prev_ceiling = self._rows(s.prev_ceiling, s.near, horizon, eye)  # bulkhead top
        y_ceiling_far = self._rows(s.ceiling, s.finite_far, horizon, eye)
        r0, r1 = self._crop([y_prev_ceiling, y_ceiling], [y_ceiling, y_ceiling_far])
        r1 = min(r1, math.ceil(horizon))  # ceilings and bulkheads are above eye level
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
            sub_frame = np.where(bulk, COLORMAP[s.level_near[:, None], texel], sub_frame)
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
                colour = np.full(len(cols), self.ceiling_colour)
                lit = self._dithered(colour, row_depth, cols, r0, r1)
                sub_frame = np.where(on_ceiling, lit, sub_frame)
                sub_z = np.where(on_ceiling, row_depth, sub_z)

        frame[cols, r0:r1] = sub_frame
        zbuffer[cols, r0:r1] = sub_z

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
            bottom = horizon + (eye - s.z) * size  # standing on its surface
            top = bottom - size
            c0, c1 = max(0, math.ceil(left - 0.5)), min(w, math.ceil(left + size - 0.5))
            r0, r1 = max(0, math.ceil(top - 0.5)), min(h, math.ceil(bottom - 0.5))
            if c0 >= c1 or r0 >= r1:
                continue
            scale = 64 / size
            su = ((np.arange(c0, c1) + 0.5 - left) * scale).astype(np.intp).clip(0, 63)
            sv = ((np.arange(r0, r1) + 0.5 - top) * scale).astype(np.intp).clip(0, 63)
            region_z = zbuffer[c0:c1, r0:r1]
            draw = art.opaque[su[:, None], sv[None, :]] & (depth <= region_z)  # z-test
            if not draw.any():
                continue
            index = art.index[su[:, None], sv[None, :]]
            if not art.fullbright:
                level = light_level(self._brightness(np.array(depth)))
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
    base: FloatArray
    ceiling: FloatArray
    prev_ceiling: FloatArray
    block: npt.NDArray[np.bool_]
    slot: IntArray
    upper: IntArray
    u: IntArray
    level_near: IntArray


def _own_space(floor_units: int, ceiling_units: int, exit_distance: float) -> RayHit:
    """A pseudo-hit at distance 0 for the space the camera is in, so its own
    floor and ceiling (if not the defaults) get drawn like any other span."""
    floor = floor_units / WALL_UNITS
    ceiling = ceiling_units / WALL_UNITS
    return RayHit(
        0.0,
        0,
        0,
        0,
        Side.X,
        0.0,
        floor,
        floor,
        exit_distance,
        ceiling=ceiling,
        prev_ceiling=ceiling,
    )


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

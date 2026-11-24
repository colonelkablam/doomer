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

from doomer.client.boxes import BOX_MODELS, BOX_TEXTURES, TOP, PlacedBox, stair_boxes
from doomer.client.config import RenderConfig
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
    TEX_MASK,
    TEX_SIZE,
    TOP_COLOURS,
    TOP_SEAM_WALL_TYPES,
    UPPER,
    build_atlas,
    build_opacity,
    railing,
)
from doomer.shared.mapfile import (
    RAIL_HEIGHT,
    RAILING_WALL_TYPE,
    STAIRS_WALL_TYPE,
    WALL_UNITS,
    GameMap,
)
from doomer.shared.raycast import (
    RayHit,
    Side,
    cast_ray,
    cell_exit_distance,
    cell_exit_u,
    start_heights,
)

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
MAX_BOX_DISTANCE = 16.0  # box models further than this (cells) aren't drawn: a few pixels
UNDER_DESK_SHADOW = 12  # colormap levels darker (of 32) for the floor under a desk
DESK_SEAM_DARKEN = 9  # colormap levels darker for the join between two desks' tops
DESK_SEAM_WIDTH = 0.01  # half-width of that line, in cells (~3 cm wide)
STEP_HEIGHT = 0.125  # a rise up to this is a step, not a drop that needs a railing
# Ceiling light panels: half-size in cells, and colours (diffuser is full-bright).
LIGHT_HALF = (0.22, 0.11)
LIGHT_DIFFUSER = (CREAM << 4) | 15
LIGHT_FRAME = (GREY << 4) | 9
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
            z = game_map.floor_at(t.x, t.y) if game_map.is_low_block(cx, cy) else 0.0
        out.append(SpriteInstance(t.type, t.x, t.y, z, t.angle))
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
        rail_tex, rail_opaque = railing()
        self.rail_tex = np.ascontiguousarray(rail_tex.T)  # [u][v]
        self.rail_opaque = np.ascontiguousarray(rail_opaque.T)
        self.top_colour = _top_colours(self.atlas, self.slot_table)
        self.ceiling_colour = nearest_index(PALETTE, config.ceiling_colour)
        self.floor_colour = nearest_index(PALETTE, config.floor_colour)
        x, y = np.meshgrid(np.arange(w), np.arange(h), indexing="ij")
        self._dither = (BAYER_4X4[x & 3, y & 3] + 0.5) / 16  # (w, h) thresholds in (0, 1)
        self._backgrounds: dict[tuple[int, int], U8Array] = {}
        # Per-map caches remember the map object itself (and compare with `is`),
        # not id(map): Python reuses the id of a freed object, so a new map
        # could otherwise be served an old map's table.
        self._lights_map: GameMap | None = None
        self._slots_map: GameMap | None = None
        self._stairs_map: GameMap | None = None
        self._stairs: list[PlacedBox] = []
        self._cell_slots: list[list[list[int]]] = []
        self._light_list: list[tuple[float, float, float]] = []
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
        # For finding where a pixel of a horizontal surface is in the world.
        self._frame_map, self._frame_camera = game_map, camera
        ray = camera.angle + np.array(self._offsets)
        self._ray_dx, self._ray_dy = np.cos(ray), np.sin(ray)

        # The space the camera is in. If it isn't the default (standing on a
        # desk, under the atrium's high ceiling), its own floor and ceiling are
        # drawn too, as a span that starts at distance 0.
        cx, cy = math.floor(camera.x), math.floor(camera.y)
        floor0, ceiling0 = start_heights(game_map, camera.x, camera.y)
        own_space = (floor0, ceiling0) != (0.0, WALL_UNITS)
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
        ceiling_changes = False  # any open stretch with a non-default ceiling?
        bulkhead = int(self.slot_table[BULKHEAD_TYPE, UPPER])
        for col in range(w):
            hits = cast_ray(game_map, camera.x, camera.y, camera.angle + self._offsets[col])
            if hits and hits[-1].full:
                hit = hits.pop()
                dist[col] = hit.distance
                cell = self._cell_slots_for(game_map, hit)
                wall_slot[col] = cell[hit.face]
                upper_slot[col] = cell[4]
                tex_u[col] = hit.u
                y_side[col] = hit.side is Side.Y
                wall_base[col] = hit.base
                wall_top[col] = hit.prev_ceiling
            if own_space:
                # The camera's own space, as up to two pseudo-spans from distance
                # 0: its ceiling (if not the default) runs to the next change of
                # ceiling or a wall, carrying on over any desks in between; the
                # floor it's standing on (a desk, a gallery) ends where the ray
                # leaves that cell.
                own: list[RayHit] = []
                if ceiling0 != WALL_UNITS:
                    open_change = next((h.distance for h in hits if h.wall_type == 0), dist[col])
                    own.append(_own_space(0.0, ceiling0, open_change))
                if on_block:
                    ray_angle = camera.angle + self._offsets[col]
                    edge = cell_exit_distance(camera.x, camera.y, ray_angle)
                    first = hits[0].distance if hits else dist[col]
                    end = min(first, edge)
                    # The surface you're standing on: your cell's wall type (a
                    # desk top's colour, not the carpet's) and, on stairs, its
                    # height where this column's ray leaves the cell.
                    leave = _floor_at_ray(game_map, camera, ray_angle, end, cx, cy)
                    own.append(
                        _own_space(
                            floor0,
                            WALL_UNITS,
                            end,
                            exit_top=leave,
                            wall_type=game_map.wall_at(cx, cy),
                            cell=(cx, cy),
                            # Where it leaves, for railings and back faces on
                            # that edge (u = 0 put every column between balusters).
                            exit_u=cell_exit_u(camera.x, camera.y, ray_angle),
                        )
                    )
                hits[0:0] = own
            for k, span in enumerate(hits):
                if k == len(layers):
                    layers.append([None] * w)
                block = span.wall_type != 0
                if span.ceiling != 1.0:
                    ceiling_changes = True
                after = hits[k + 1].distance if k + 1 < len(hits) else dist[col]
                layers[k][col] = SpanHit(
                    span,
                    block and after > span.exit_distance + 1e-9,
                    *self._span_slots(game_map, span, block, bulkhead),
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

        # The default ceiling is a background with no geometry of its own; give
        # its pixels their real depth where it really is the ceiling, so a low
        # ceiling hides a higher one beyond it (looking from a normal room into
        # the atrium). Inside a stretch with a different ceiling it doesn't
        # exist, so those pixels stay infinitely far and the real ceiling wins.
        if ceiling_changes:
            self._ceiling_depths(zbuffer, layers, horizon, eye)
        else:
            self._default_ceiling_depths(zbuffer, horizon, eye)

        # 3. Changes of space, one layer at a time, z-tested.
        for layer in layers:
            self._draw_span_layer(frame, zbuffer, layer, horizon, eye)

        self._draw_ceiling_lights(frame, zbuffer, game_map, camera, horizon, eye)
        sprites = list(sprites)
        boxes = self._stair_boxes_for(game_map) + [
            PlacedBox(s.x, s.y, s.z, s.angle, box)
            for s in sprites
            for box in BOX_MODELS.get(s.name, ())
            if (s.x - camera.x) ** 2 + (s.y - camera.y) ** 2 < MAX_BOX_DISTANCE**2
        ]
        self._draw_boxes(frame, zbuffer, camera, boxes, horizon, eye)
        self._draw_sprites(frame, zbuffer, camera, sprites, horizon, eye)
        self.zbuffer = zbuffer
        return frame

    def _cell_slots_for(self, game_map: GameMap, hit: RayHit) -> list[int]:
        """[slot for each compass face 0-3, upper slot] of the hit's cell, from a
        per-map table (a cell's textures never change during a map). Plain
        nested lists: single-element lookups are much faster than numpy's."""
        if self._slots_map is not game_map:
            self._cell_slots = [
                [
                    [self._slot_for(game_map, x, y, f) for f in range(4)]
                    + [self._upper_slot(game_map.walls[y][x])]
                    for x in range(game_map.width)
                ]
                for y in range(game_map.height)
            ]
            self._slots_map = game_map
        if 0 <= hit.cell_y < game_map.height and 0 <= hit.cell_x < game_map.width:
            return self._cell_slots[hit.cell_y][hit.cell_x]
        # Outside the map (solid): wall type 1.
        return [int(self.slot_table[1, side]) for side in range(4)] + [self._upper_slot(1)]

    def _span_slots(
        self, game_map: GameMap, span: RayHit, block: bool, bulkhead: int
    ) -> tuple[int, int, int]:
        """(entry slot, exit slot, upper slot) for a change of space."""
        if not block:
            return 0, 0, bulkhead
        cell = self._cell_slots_for(game_map, span)
        return cell[span.face], cell[span.exit_face], cell[4]

    def _slot_for(self, game_map: GameMap, x: int, y: int, face: int) -> int:
        wall_type = game_map.walls[y][x]
        if wall_type >= len(self.slot_table):
            return 0  # unknown wall type -> missing texture
        side = (face - game_map.facing_at(x, y)) % 4
        return int(self.slot_table[wall_type, side])

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

    def _ceiling_depths(
        self, zbuffer: FloatArray, layers: list[list[SpanHit | None]], horizon: float, eye: float
    ) -> None:
        rows = min(self.config.height, math.floor(horizon))
        if rows <= 0 or eye >= 1.0:
            return  # no default ceiling above the eye on screen
        w = self.config.width
        # Per column, the [near, far] perpendicular intervals where the ceiling
        # isn't the default: open stretches, the camera's own space, and blocks
        # (a step or desk under the atrium's high ceiling is under it too).
        lo = np.full((len(layers), w), np.nan)
        hi = np.full((len(layers), w), np.nan)
        for k, layer in enumerate(layers):
            for col, entry in enumerate(layer):
                if entry is not None and entry.hit.ceiling != 1.0:
                    lo[k, col] = entry.hit.distance * self._cos[col]
                    hi[k, col] = entry.hit.exit_distance * self._cos[col]
        depth = (1.0 - eye) * self.projection / (horizon - self._pixel_y[:rows])  # (rows,)
        inside = (lo[:, :, None] <= depth[None, None, :]) & (depth[None, None, :] <= hi[:, :, None])
        real = ~inside.any(axis=0)  # (w, rows)
        region = zbuffer[:, :rows]
        zbuffer[:, :rows] = np.where(np.isinf(region) & real, depth[None, :], region)

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
            # Never closer than `near` (a camera exactly on a cell boundary can
            # give a zero-length span), never infinite (open-ended stretches).
            finite_far=np.maximum(np.where(np.isfinite(far), far, 1e9), near),
            top=np.array([h.top for h in hits]),
            exit_top=np.array([h.exit_top for h in hits]),
            base=np.array([h.base for h in hits]),
            stairs=np.array([h.wall_type == STAIRS_WALL_TYPE for h in hits]),
            rail=np.array([h.wall_type == RAILING_WALL_TYPE for h in hits]),
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
        if span.block.any() or (span.top > 0).any() or (span.exit_top > 0).any():
            self._draw_lower(frame, zbuffer, span, entries, horizon, eye)
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
        entries: list[SpanHit],
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
        hits = [e.hit for e in entries]
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
        exits_to_floor = np.array([e.exits_to_floor for e in entries])
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
                back_slot = np.array([e.back_slot for e in entries], dtype=np.intp)
                # Seen from inside the cell, so mirror u to match the outside view.
                back_u = _texel_u(1.0 - np.array([h.exit_u for h in hits]))
                back_y_side = np.array([h.exit_side is Side.Y for h in hits])
                depth = s.finite_far[ci]
                which, vi = self._texel_at(
                    rows_y[ri], depth, back_slot[ci], s.upper[ci], horizon, eye
                )
                uu = back_u[ci]
                keep = self.opacity[which, uu, vi]
                ci, ri, which, uu, vi = ci[keep], ri[keep], which[keep], uu[keep], vi[keep]
                shade = self._brightness(far[ci]) * np.where(back_y_side[ci], Y_SIDE_SHADE, 1.0)
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
                    lit = self._dithered_at(colour, depth, cols[ci], r0 + ri)
                    lit = self._desk_seams(lit, hits, ci, cols[ci], depth, eye - s.top[ci])
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
            exit_u = _texel_u(1.0 - np.array([h.exit_u for h in hits]))
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
                level = light_level(self._brightness(depth))
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
        hits: list[RayHit],
        ci: IntArray,
        cols: IntArray,
        depth: FloatArray,
        drop: FloatArray,
    ) -> U8Array:
        """Darken a line across the top of desk pairs, where the two desks meet:
        the middle of the cell, perpendicular to its facing. Pixels are put
        back into the world (camera + ray * distance) to find which are on it.
        `drop` is the eye's height above the surface."""
        seam = np.array([h.wall_type in TOP_SEAM_WALL_TYPES for h in hits])
        if not seam[ci].any():
            return lit
        game_map, camera = self._frame_map, self._frame_camera
        # Facing north or south: the desks meet along y = middle of the cell.
        along_y = np.array([game_map.facing_at(h.cell_x, h.cell_y) % 2 == 0 for h in hits])
        middle = np.array(
            [(h.cell_y if y else h.cell_x) + 0.5 for h, y in zip(hits, along_y, strict=True)]
        )
        t = depth / self._cos[cols]
        coord = np.where(
            along_y[ci], camera.y + self._ray_dy[cols] * t, camera.x + self._ray_dx[cols] * t
        )
        # At least half a pixel's footprint on the surface wide, or the line
        # falls between pixel centres. A pixel covers depth / projection across
        # the ray and, seen at a slant, depth^2 / (projection * drop) along it.
        across = depth / self.projection / self._cos[cols]
        along = across * depth / np.maximum(drop, 1e-6)
        dx, dy = np.abs(self._ray_dx[cols]), np.abs(self._ray_dy[cols])
        footprint = np.where(along_y[ci], along * dy + across * dx, along * dx + across * dy)
        width = np.maximum(DESK_SEAM_WIDTH, 0.5 * footprint)
        on = seam[ci] & (np.abs(coord - middle[ci]) < width)
        result: U8Array = np.where(on, COLORMAP[DESK_SEAM_DARKEN, lit], lit)
        return result

    def _dithered_at(
        self, colour: npt.NDArray[np.integer], depth: FloatArray, cols: IntArray, rows: IntArray
    ) -> U8Array:
        """Distance-shaded, dithered palette indices for individual pixels of a
        horizontal surface (same rule as _dithered)."""
        brightness = self._brightness(np.nan_to_num(depth, nan=np.inf, posinf=1e9))
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

    def _lights_in(self, game_map: GameMap) -> list[tuple[float, float, float]]:
        """(x, y, ceiling height) of each light panel: one per `ceiling_light`
        thing, centred in its cell."""
        if self._lights_map is not game_map:
            self._light_list = []
            for t in game_map.things:
                if t.type == "ceiling_light":
                    cx, cy = math.floor(t.x), math.floor(t.y)
                    self._light_list.append((cx + 0.5, cy + 0.5, game_map.ceiling_height(cx, cy)))
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
        for lx, ly, ceiling in self._lights_in(game_map):
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
                    self._draw_light_panel(
                        frame, zbuffer, camera, lx, ly, ceiling, c0, c1, r0, r1, horizon, eye
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
    ) -> None:
        cols = np.arange(c0, c1)
        up = horizon - self._pixel_y[None, r0:r1]  # > 0 above the horizon
        depth = (ceiling - eye) * self.projection / up  # (1, rows) perpendicular
        angles = camera.angle + np.array(self._offsets)[cols]
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
        frame_colour = COLORMAP[
            light_level(self._brightness(np.broadcast_to(depth, panel.shape))), LIGHT_FRAME
        ]
        region = frame[c0:c1, r0:r1]
        region[panel] = frame_colour[panel]
        region[diffuser] = LIGHT_DIFFUSER

    def _stair_boxes_for(self, game_map: GameMap) -> list[PlacedBox]:
        """The map's stairs as step boxes, built once per map."""
        if self._stairs_map is not game_map:
            self._stairs = stair_boxes(game_map)
            self._stairs_map = game_map
        return self._stairs

    def _draw_boxes(
        self,
        frame: U8Array,
        zbuffer: FloatArray,
        camera: Camera,
        boxes: list[PlacedBox],
        horizon: float,
        eye: float,
    ) -> None:
        """Box models (boxes.py), drawn as solid geometry. Every box is tested
        against every column's ray at once (B boxes x W columns): the slab
        test, in each box's own frame. Where a ray enters a box by a side, that
        face is a vertical run of pixels at the entry depth (like a wall); the
        top (or bottom) is a stretch of rows between the entry and exit
        depths, cast back onto its plane (like a floor). Pixels from all boxes
        are then z-tested together, nearest wins. Used for box-model things
        and for the steps of stairs."""
        if not boxes:
            return
        h = self.config.height
        proj = self.projection

        def per_box(values: list[float]) -> FloatArray:
            return np.array(values)[:, None]  # (B, 1), broadcasts over columns

        ca = per_box([math.cos(p.angle) for p in boxes])
        sa = per_box([math.sin(p.angle) for p in boxes])
        rel_x = per_box([camera.x - p.x for p in boxes])
        rel_y = per_box([camera.y - p.y for p in boxes])
        x0, x1 = per_box([p.box.x[0] for p in boxes]), per_box([p.box.x[1] for p in boxes])
        y0, y1 = per_box([p.box.y[0] for p in boxes]), per_box([p.box.y[1] for p in boxes])
        z0 = per_box([p.z + p.box.z[0] for p in boxes])
        z1 = per_box([p.z + p.box.z[1] for p in boxes])
        faces = np.array([p.box.faces for p in boxes])  # (B, 5)
        tiled = np.array([p.box.tiled_sides for p in boxes])
        riser = np.array([p.box.riser for p in boxes])

        # Camera and rays into each box's frame (rotate by -angle).
        ox, oy = rel_x * ca + rel_y * sa, -rel_x * sa + rel_y * ca
        ray = camera.angle + np.array(self._offsets)
        wx, wy = np.cos(ray)[None, :], np.sin(ray)[None, :]
        dx, dy = wx * ca + wy * sa, -wx * sa + wy * ca
        dx = np.where(np.abs(dx) < 1e-12, 1e-12, dx)
        dy = np.where(np.abs(dy) < 1e-12, 1e-12, dy)
        tx0, tx1 = (x0 - ox) / dx, (x1 - ox) / dx
        ty0, ty1 = (y0 - oy) / dy, (y1 - oy) / dy
        in_x, in_y = np.minimum(tx0, tx1), np.minimum(ty0, ty1)
        t_in = np.maximum(in_x, in_y)
        t_out = np.minimum(np.maximum(tx0, tx1), np.maximum(ty0, ty1))
        near_t = np.maximum(t_in, NEAR_PLANE)
        hit = t_out > near_t
        bi, ci = np.nonzero(hit)  # the (box, column) pairs whose ray crosses the box
        if not bi.size:
            return
        cos_c = self._cos[ci]
        d_near = near_t[bi, ci] * cos_c  # perpendicular depths, as in the z-buffer
        d_far = t_out[bi, ci] * cos_c
        zt, zb = z1[bi, 0], z0[bi, 0]

        # Which face the ray enters by, where along it (u), and its light.
        from_x = in_x[bi, ci] > in_y[bi, ci]
        rdx, rdy = dx[bi, ci], dy[bi, ci]
        face = np.where(from_x, np.where(rdx < 0, 0, 2), np.where(rdy < 0, 1, 3))
        ti = t_in[bi, ci]
        hx, hy = ox[bi, 0] + rdx * ti, oy[bi, 0] + rdy * ti
        bx0, bx1, by0, by1 = x0[bi, 0], x1[bi, 0], y0[bi, 0], y1[bi, 0]
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
        normal = np.array([p.angle for p in boxes])[bi] + face * (math.pi / 2)
        side_light = 1.0 - (1.0 - Y_SIDE_SHADE) * np.abs(np.sin(normal))

        cols: list[IntArray] = []
        rows: list[IntArray] = []
        depths: list[FloatArray] = []
        texels: list[npt.NDArray[np.uint8]] = []
        lights: list[FloatArray] = []

        # Sides: where the ray enters by one (not from inside the footprint).
        # A riser (front/back of a step) is only its top slice.
        side = t_in[bi, ci] > NEAR_PLANE
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
            texels.append(BOX_TEXTURES[tex, tu, tv])
            lights.append(side_light[k])

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
            px, py = ox[bi[k], 0] + dx[bi[k], ci[k]] * t, oy[bi[k], 0] + dy[bi[k], ci[k]] * t
            u = (y1[bi[k], 0] - py) / (y1[bi[k], 0] - y0[bi[k], 0])
            v = (px - x0[bi[k], 0]) / (x1[bi[k], 0] - x0[bi[k], 0])
            cols.append(ci[k])
            rows.append(r)
            depths.append(depth)
            texels.append(BOX_TEXTURES[faces[bi[k], TOP], _texel_clamped(u), _texel_clamped(v)])
            lights.append(np.full(k.size, 1.0 if above else Y_SIDE_SHADE))

        if not cols:
            return
        col, row, depth = np.concatenate(cols), np.concatenate(rows), np.concatenate(depths)
        texel, light = np.concatenate(texels), np.concatenate(lights)
        # Where boxes overlap on screen, keep the nearest pixel (sort by pixel
        # then depth, take the first of each pixel), then z-test it.
        key = col * h + row
        order = np.lexsort((depth, key))
        first = np.ones(order.size, dtype=np.bool_)
        first[1:] = key[order][1:] != key[order][:-1]
        pick = order[first]
        pick = pick[depth[pick] < zbuffer[col[pick], row[pick]]]
        col, row, depth = col[pick], row[pick], depth[pick]
        level = light_level(self._brightness(depth) * light[pick])
        frame[col, row] = COLORMAP[level, texel[pick]]
        zbuffer[col, row] = depth

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


def _own_space(
    floor: float,
    ceiling_units: int,
    exit_distance: float,
    exit_top: float | None = None,
    wall_type: int = 0,
    cell: tuple[int, int] = (0, 0),
    exit_u: float = 0.0,
) -> RayHit:
    """A pseudo-hit at distance 0 for the space the camera is in, so its own
    floor and ceiling (if not the defaults) get drawn like any other span.
    Standing on something, it carries that cell's wall type and position."""
    ceiling = ceiling_units / WALL_UNITS
    return RayHit(
        0.0,
        cell[0],
        cell[1],
        wall_type,
        Side.X,
        0.0,
        floor,
        floor,
        exit_distance,
        ceiling=ceiling,
        prev_ceiling=ceiling,
        exit_top=floor if exit_top is None else exit_top,
        exit_u=exit_u,
    )


def _floor_at_ray(
    game_map: GameMap, camera: Camera, angle: float, distance: float, cx: int, cy: int
) -> float:
    """Floor height of cell (cx, cy) where a ray from the camera leaves it."""
    x = camera.x + math.cos(angle) * distance
    y = camera.y + math.sin(angle) * distance
    eps = 1e-9
    return game_map.floor_at(
        min(max(x, cx + eps), cx + 1 - eps), min(max(y, cy + eps), cy + 1 - eps)
    )


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

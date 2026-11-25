"""The light map: how brightly the ceiling panels light each part of the floor.

Baked once per map, client-side (lighting is cosmetic; the server never needs
it). A grid of samples over the floor plan, `samples_per_cell` per cell along
each axis, sample (i, j) at ((j + 0.5) / S, (i + 0.5) / S) in cells. Each one:

    ambient + sum over panels of intensity * falloff * fade * visible

- falloff = 1 / (1 + (d / r)^2), d the straight-line distance from the panel
  (at its ceiling) to the floor at the sample, r = `falloff_radius`: the
  light halves at r. Because d includes the height, a panel in a two-storey
  ceiling gives a dimmer, wider pool with no special case.
- fade takes the light smoothly to zero at `cutoff` cells horizontally, so
  there's no ring where a panel's reach ends.
- visible: the straight line from the panel to the sample, looked at from
  above, crosses no full wall and no opaque door's plane (hinged, double,
  lift: they stand shut, and the map is baked once; glass doors let light
  through). Only the plane: the doorway either side of it is lit.
  Low blocks (desks) don't stop light.

Each panel's contribution is kept as its own patch (its reach only), so a
light can later be switched on or off (flicker, power cuts) by adding or
subtracting one patch, without baking again. The values aren't clamped:
where pools overlap they can exceed 1, and whoever uses them clamps.

Phase 1 of the lighting plan (docs/DESIGN.md, "Lighting"): the map and its
tests. The renderer doesn't use it yet.
"""

import math
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from doomer.client.config import LightingConfig
from doomer.shared.doors import SLIDING, doors_from_map
from doomer.shared.mapfile import GameMap

FloatArray = npt.NDArray[np.float64]

_VISIBILITY_STEP = 0.125  # cells between checks along a panel-to-sample line


@dataclass(frozen=True, slots=True)
class Panel:
    """A ceiling light: the centre of its cell, at its ceiling's height."""

    x: float
    y: float
    height: float  # walls
    emergency: bool = False  # on the emergency circuit: lit without power


@dataclass(frozen=True, slots=True)
class LightPatch:
    """One panel's light: `values` covers samples [row0, row0 + rows) x
    [col0, col0 + cols) of the map."""

    panel: Panel
    row0: int
    col0: int
    values: FloatArray


@dataclass(frozen=True, slots=True)
class LightMap:
    samples_per_cell: int
    ambient: float
    values: FloatArray  # [row][col]: ambient plus every patch
    patches: tuple[LightPatch, ...]
    # The same with the power off: a lower ambient, emergency panels only.
    unpowered: FloatArray | None = None

    def without_power(self) -> "LightMap":
        """This map as it is with the power off (itself, if it has no
        unpowered version)."""
        if self.unpowered is None:
            return self
        return LightMap(self.samples_per_cell, self.ambient, self.unpowered, self.patches)

    def at(self, x: FloatArray, y: FloatArray) -> FloatArray:
        """Light at points (cells), interpolated bilinearly between samples;
        points off the map take the nearest edge sample."""
        s = self.samples_per_cell
        rows, cols = self.values.shape
        fx = np.clip(np.asarray(x, dtype=np.float64) * s - 0.5, 0.0, cols - 1.0)
        fy = np.clip(np.asarray(y, dtype=np.float64) * s - 0.5, 0.0, rows - 1.0)
        c0, r0 = np.floor(fx).astype(np.intp), np.floor(fy).astype(np.intp)
        c1, r1 = np.minimum(c0 + 1, cols - 1), np.minimum(r0 + 1, rows - 1)
        tx, ty = fx - c0, fy - r0
        v = self.values
        top = v[r0, c0] * (1 - tx) + v[r0, c1] * tx
        bottom = v[r1, c0] * (1 - tx) + v[r1, c1] * tx
        result: FloatArray = top * (1 - ty) + bottom * ty
        return result


def panels_in(game_map: GameMap) -> list[Panel]:
    """One panel per `ceiling_light` thing, as the renderer draws them."""
    out = []
    for t in game_map.things:
        if t.type == "ceiling_light":
            cx, cy = math.floor(t.x), math.floor(t.y)
            emergency = t.prop("circuit") == "emergency"
            out.append(Panel(cx + 0.5, cy + 0.5, game_map.ceiling_height(cx, cy), emergency))
    return out


def bake_light_map(game_map: GameMap, config: LightingConfig | None = None) -> LightMap:
    config = config or LightingConfig()
    s = config.samples_per_cell
    rows, cols = game_map.height * s, game_map.width * s
    xs, ys = range(game_map.width), range(game_map.height)
    walls = np.array([[game_map.is_full_wall(x, y) for x in xs] for y in ys], dtype=np.bool_)
    floors = np.array(
        [[0.0 if walls[y, x] else game_map.floor_height(x, y) for x in xs] for y in ys]
    )
    # Shut opaque doors: (axis, at, lo, hi), a plane x = at (axis 0) or
    # y = at (1), from lo to hi along it.
    doors = [
        (0, d.x, d.y - 0.5, d.y + 0.5) if d.across_x else (1, d.y, d.x - 0.5, d.x + 0.5)
        for d in doors_from_map(game_map)
        if d.kind != SLIDING
    ]
    values = np.full((rows, cols), config.ambient)
    unpowered = np.full((rows, cols), config.unpowered_ambient)
    patches = []
    for panel in panels_in(game_map):
        patch = _patch(panel, config, walls, floors, rows, cols, doors)
        if patch is not None:
            r, c = patch.row0, patch.col0
            h, w = patch.values.shape
            values[r : r + h, c : c + w] += patch.values
            if panel.emergency:
                unpowered[r : r + h, c : c + w] += patch.values
            patches.append(patch)
    return LightMap(s, config.ambient, values, tuple(patches), unpowered)


def _patch(
    panel: Panel,
    config: LightingConfig,
    walls: npt.NDArray[np.bool_],
    floors: FloatArray,
    rows: int,
    cols: int,
    doors: list[tuple[int, float, float, float]] = [],  # noqa: B006 (never mutated)
) -> LightPatch | None:
    s = config.samples_per_cell
    reach = config.cutoff
    # The samples within the panel's reach (a square round it).
    r0 = max(0, math.floor((panel.y - reach) * s))
    r1 = min(rows, math.ceil((panel.y + reach) * s))
    c0 = max(0, math.floor((panel.x - reach) * s))
    c1 = min(cols, math.ceil((panel.x + reach) * s))
    if r0 >= r1 or c0 >= c1:
        return None
    sy = (np.arange(r0, r1) + 0.5) / s
    sx = (np.arange(c0, c1) + 0.5) / s
    gx, gy = np.meshgrid(sx, sy)  # [row][col], in cells
    cell_x = np.clip(gx.astype(np.intp), 0, walls.shape[1] - 1)
    cell_y = np.clip(gy.astype(np.intp), 0, walls.shape[0] - 1)
    horizontal = np.hypot(gx - panel.x, gy - panel.y)
    drop = np.maximum(panel.height - floors[cell_y, cell_x], 0.0)
    distance = np.hypot(horizontal, drop)
    falloff = 1.0 / (1.0 + (distance / config.falloff_radius) ** 2)
    fade = np.clip(1.0 - (horizontal / reach) ** 2, 0.0, 1.0) ** 2
    visible = _visible(panel, gx, gy, walls) & ~_behind_doors(panel, gx, gy, doors)
    light = config.panel_intensity * falloff * fade * visible
    return LightPatch(panel, r0, c0, light)


def _behind_doors(
    panel: Panel, gx: FloatArray, gy: FloatArray, doors: list[tuple[int, float, float, float]]
) -> npt.NDArray[np.bool_]:
    """For each sample point, whether the line from the panel to it crosses
    a shut door: it passes from one side of the door's plane to the other,
    within the doorway."""
    hidden = np.zeros(gx.shape, dtype=np.bool_)
    for axis, at, lo, hi in doors:
        p_across, p_along = (panel.x, panel.y) if axis == 0 else (panel.y, panel.x)
        s_across, s_along = (gx, gy) if axis == 0 else (gy, gx)
        crosses = (p_across - at) * (s_across - at) < 0
        with np.errstate(divide="ignore", invalid="ignore"):
            t = (at - p_across) / (s_across - p_across)
        where = p_along + (s_along - p_along) * t
        hidden |= crosses & (where >= lo) & (where <= hi)
    return hidden


def _visible(
    panel: Panel, gx: FloatArray, gy: FloatArray, walls: npt.NDArray[np.bool_]
) -> npt.NDArray[np.bool_]:
    """For each sample point, whether the line from the panel to it (seen
    from above) stays out of full walls. Checked at points every ~1/8 cell
    along the line, all lines at once."""
    longest = float(np.max(np.hypot(gx - panel.x, gy - panel.y)))
    steps = max(1, math.ceil(longest / _VISIBILITY_STEP))
    clear = np.ones(gx.shape, dtype=np.bool_)
    h, w = walls.shape
    for k in range(steps):
        t = (k + 0.5) / steps  # strictly between the ends
        px = panel.x + (gx - panel.x) * t
        py = panel.y + (gy - panel.y) * t
        ix = np.clip(np.floor(px).astype(np.intp), 0, w - 1)
        iy = np.clip(np.floor(py).astype(np.intp), 0, h - 1)
        clear &= ~walls[iy, ix]
    return clear

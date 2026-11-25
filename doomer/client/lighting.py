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
from doomer.shared.mapfile import WALL_UNITS, GameMap

FloatArray = npt.NDArray[np.float64]

_VISIBILITY_STEP = 0.125  # cells between checks along a panel-to-sample line


@dataclass(frozen=True, slots=True)
class Panel:
    """A ceiling light: the centre of its cell, at its ceiling's height."""

    x: float
    y: float
    height: float  # walls
    emergency: bool = False  # on the emergency circuit: lit without power
    intensity: float = 1.0  # times LightingConfig.panel_intensity


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
    # Light for faces above head height (the face over a doorway): baked with
    # lowered ceilings (lintels) and door panels stopping light, as light
    # through a doorway passes under its lintel and lights the floor beyond,
    # not the wall above it on that side.
    upper: "LightMap | None" = None

    def without_power(self) -> "LightMap":
        """This map as it is with the power off (itself, if it has no
        unpowered version)."""
        if self.unpowered is None:
            return self
        upper = self.upper.without_power() if self.upper is not None else None
        return LightMap(
            self.samples_per_cell, self.ambient, self.unpowered, self.patches, None, upper
        )

    def high(self) -> "LightMap":
        """The light for faces above head height (itself, if it has none)."""
        return self.upper if self.upper is not None else self

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


def panels_in(game_map: GameMap, config: LightingConfig | None = None) -> list[Panel]:
    """The lights: one panel per `ceiling_light` thing, as the renderer draws
    them; and a dim emergency light at each exit sign, over its door."""
    config = config or LightingConfig()
    out = []
    for t in game_map.things:
        if t.type == "ceiling_light":
            cx, cy = math.floor(t.x), math.floor(t.y)
            emergency = t.prop("circuit") == "emergency"
            out.append(Panel(cx + 0.5, cy + 0.5, game_map.ceiling_height(cx, cy), emergency))
        elif t.type == "exit_sign":
            # A little out from the wall it hangs on, so walls don't hide it.
            x, y = t.x + 0.1 * math.cos(t.angle), t.y + 0.1 * math.sin(t.angle)
            out.append(Panel(x, y, 0.83, emergency=True, intensity=config.exit_sign_intensity))
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
    # Doors: (axis, at, lo, hi), a plane x = at (axis 0) or y = at (1), from
    # lo to hi along it. Shut opaque ones stop light; for faces above head
    # height, glass ones too (above the door is the lintel, not glass).
    planes = [
        (
            (0, d.plane, d.y - 0.5, d.y + 0.5)
            if d.across_x
            else (1, d.plane, d.x - 0.5, d.x + 0.5),
            d,
        )
        for d in doors_from_map(game_map)
    ]
    doors = [p for p, d in planes if d.kind != SLIDING]
    # Lowered ceilings (a doorway's lintel) stop light for faces above them.
    low = np.array(
        [[not walls[y, x] and game_map.ceiling[y][x] < WALL_UNITS for x in xs] for y in ys],
        dtype=np.bool_,
    )
    lights = panels_in(game_map, config)
    floor_light = _compose(lights, config, walls, floors, doors, rows, cols, keep=True)
    face_light = _compose(lights, config, walls | low, floors, [p for p, _ in planes], rows, cols)
    values, unpowered, patches = floor_light
    upper = LightMap(s, config.ambient, face_light[0], (), face_light[1])
    return LightMap(s, config.ambient, values, patches, unpowered, upper)


def _compose(
    lights: list[Panel],
    config: LightingConfig,
    walls: npt.NDArray[np.bool_],
    floors: FloatArray,
    doors: list[tuple[int, float, float, float]],
    rows: int,
    cols: int,
    keep: bool = False,
) -> tuple[FloatArray, FloatArray, tuple[LightPatch, ...]]:
    """(lit, unpowered, patches if `keep`): the ambient plus every light's
    patch, with the power on and off, lifted (LightingConfig.lift). The
    lift bends the finished map, not the patches: they stay plain sums, to
    add or take away."""
    values = np.full((rows, cols), config.ambient)
    unpowered = np.full((rows, cols), config.unpowered_ambient)
    patches = []
    for panel in lights:
        patch = _patch(panel, config, walls, floors, rows, cols, doors)
        if patch is not None:
            r, c = patch.row0, patch.col0
            h, w = patch.values.shape
            values[r : r + h, c : c + w] += patch.values
            if panel.emergency:
                unpowered[r : r + h, c : c + w] += patch.values
            if keep:
                patches.append(patch)
    return np.power(values, config.lift), np.power(unpowered, config.lift), tuple(patches)


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
    light = config.panel_intensity * panel.intensity * falloff * fade * visible
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

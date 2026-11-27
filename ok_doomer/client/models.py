"""Voxel models, baked into rotation frames for sprites.

A sprite that should look different from different sides (a PC: screen in
front, vents behind) needs one picture per viewing direction, as Doom's
sprites had 8 (frames A1..A8). Drawing 8 views by hand is a lot of pixel art
and the views never quite agree. Instead the thing is built here as a few
boxes of voxels, and each view is rendered from the model once, at startup.
The renderer then just picks the frame for the direction you see it from.

Units: one voxel = one sprite pixel (1/64 of a cell wide, 1/64 of a wall
tall), so a frame lines up with the sprites drawn by hand in sprites.py.

Model axes: x points out of the thing's front (the way its map angle faces),
y to its right, z up from its base. The origin is the thing's position.

Each view is an orthographic projection (the model is small, so perspective
within it is negligible) from ELEVATION above level, because you mostly look
down on furniture: a view from level would show no tops and read as a flat
cut-out. Seen from above, the parts nearest you drop below the thing's base
line, so frames carry a `sink`: rows of picture below where the base stands.
Views are shaded per voxel from its surface normal with a
light fixed to the model, so its sides keep the same light as it turns.
"""

import math
from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt

from ok_doomer.client.palette import SHADES, U8Array
from ok_doomer.client.pixelart import ART_SIZE, BoolArray, indices

VIEWS = 16  # frames per model: one per 22.5 degrees (Doom had 8; baked, more is free)
ELEVATION = math.radians(10)  # views look down on the model this much
GRID = ART_SIZE  # model space is GRID voxels across, deep and tall
_HALF = GRID // 2

# Light direction in model space (towards the light): above, in front, a
# little to the thing's right. Normalised.
_LIGHT = np.array([0.5, 0.35, 0.8]) / math.sqrt(0.5**2 + 0.35**2 + 0.8**2)
_SHADE_RANGE = 4.0  # shades between the darkest and brightest lit face


@dataclass(slots=True)
class VoxelModel:
    """A 64^3 grid of voxels: ramp and shade per voxel; -1 ramp = empty.
    `glow` voxels (a screen) ignore the model's light."""

    ramp: npt.NDArray[np.int8] = field(
        default_factory=lambda: np.full((GRID, GRID, GRID), -1, dtype=np.int8)
    )
    shade: npt.NDArray[np.float64] = field(
        default_factory=lambda: np.zeros((GRID, GRID, GRID), dtype=np.float64)
    )
    glow: BoolArray = field(default_factory=lambda: np.zeros((GRID, GRID, GRID), dtype=np.bool_))
    # Shade from normals smoothed over this many voxels (0: each voxel's own
    # faces). Slopes built of voxels (a leaning body, a swung arm) are
    # staircases, and faceted normals light their steps alternately: stripes.
    smooth: int = 0

    def box(
        self,
        x: tuple[int, int],
        y: tuple[int, int],
        z: tuple[int, int],
        ramp: int,
        shade: float,
        glow: bool = False,
    ) -> None:
        """Fill the half-open box [x0, x1) x [y0, y1) x [z0, z1) (model
        coordinates, origin at the thing) with one colour. Later boxes
        overwrite earlier ones, so details are painted over the bodies."""
        where = _slices(x, y, z)
        self.ramp[where] = ramp
        self.shade[where] = shade
        self.glow[where] = glow

    def clear(self, x: tuple[int, int], y: tuple[int, int], z: tuple[int, int]) -> None:
        """Cut a box out (a recess)."""
        self.ramp[_slices(x, y, z)] = -1


def _slices(
    x: tuple[int, int], y: tuple[int, int], z: tuple[int, int]
) -> tuple[slice, slice, slice]:
    return slice(x[0] + _HALF, x[1] + _HALF), slice(y[0] + _HALF, y[1] + _HALF), slice(*z)


@dataclass(frozen=True, slots=True)
class Baked:
    index: U8Array  # [view][u][v]
    opaque: BoolArray  # [view][u][v]
    sink: int  # image rows below the thing's base line (nearest parts, seen from above)
    reach: float  # how far the model extends from its centre, in cells


def bake(model: VoxelModel, views: int = VIEWS) -> Baked:
    """Render the model from `views` directions (VIEWS for sprites in the
    world, which pick theirs with `view_for`), each [view][u][v] like a sprite.
    View k sees the model from k * 360 / VIEWS degrees round from its front, in the
    map's angle direction (view 0 = from the front; with y pointing down,
    view VIEWS / 4 = from the thing's right)."""
    solid = model.ramp >= 0
    normals = _normals(solid)
    if model.smooth:
        normals = np.where(
            normals.any(axis=-1, keepdims=True), _smooth_normals(solid, model.smooth), 0.0
        )
    surface = solid & normals.any(axis=-1)
    px, py, pz = (a.astype(np.float64) for a in np.nonzero(surface))
    px, py = px - _HALF + 0.5, py - _HALF + 0.5  # voxel centres, origin at the thing
    normal = normals[surface]
    normal = normal / np.linalg.norm(normal, axis=1, keepdims=True)
    lit = np.clip(normal @ _LIGHT, 0.0, 1.0)
    shade = model.shade[surface] + np.where(model.glow[surface], 0.0, (lit - 0.5) * _SHADE_RANGE)
    colour = indices(model.ramp[surface].astype(np.int64), np.clip(shade, 0, SHADES - 1))
    reach = float(np.hypot(np.abs(px) + 0.5, np.abs(py) + 0.5).max())
    sin_e, cos_e = math.sin(ELEVATION), math.cos(ELEVATION)
    sink = math.ceil(reach * sin_e)

    index = np.zeros((views, ART_SIZE, ART_SIZE), dtype=np.uint8)
    opaque = np.zeros((views, ART_SIZE, ART_SIZE), dtype=np.bool_)
    for k in range(views):
        phi = k * math.tau / views
        # The viewer stands at direction phi from the thing, looking back at
        # it; their right is (sin phi, -cos phi) (y points down).
        u = np.floor(_HALF + px * math.sin(phi) - py * math.cos(phi)).astype(np.intp)
        ahead = px * math.cos(phi) + py * math.sin(phi)  # towards the viewer
        # Image rows: the base line is row GRID - 1 - sink; up is z, tipped back.
        v = np.floor(GRID - sink - (pz + 0.5) * cos_e + ahead * sin_e).astype(np.intp)
        nearness = ahead * cos_e + pz * sin_e
        keep = (u >= 0) & (u < ART_SIZE) & (v >= 0) & (v < ART_SIZE)
        # Painter's order: draw far to near, so the nearest voxel in each
        # pixel is written last and wins.
        order = np.argsort(nearness[keep], kind="stable")
        uu, vv, cc = u[keep][order], v[keep][order], colour[keep][order]
        index[k, uu, vv] = cc
        opaque[k, uu, vv] = True
        _fill_pinholes(index[k], opaque[k])
    return Baked(index, opaque, sink, reach / GRID)


def _normals(solid: BoolArray) -> npt.NDArray[np.float64]:
    """Per voxel, the sum of directions to its empty neighbours (zero inside)."""
    padded = np.pad(solid, 1)
    out = np.zeros((*solid.shape, 3), dtype=np.float64)
    for axis in range(3):
        for sign in (-1, 1):
            neighbour = np.roll(padded, -sign, axis=axis)[1:-1, 1:-1, 1:-1]
            out[..., axis] += np.where(neighbour, 0.0, float(sign))
    return out


def _smooth_normals(solid: BoolArray, radius: int) -> npt.NDArray[np.float64]:
    """Outward normals of the surface as a whole: down the gradient of the
    solid's occupancy, box-blurred over `radius` voxels each way."""
    blurred = solid.astype(np.float64)
    for axis in range(3):
        blurred = _box_blur(blurred, radius, axis)
    out = -np.stack([np.asarray(g) for g in np.gradient(blurred)], axis=-1)
    # A thin part (a voxel or two) has no gradient across it: fall back on
    # its own faces there.
    flat = np.linalg.norm(out, axis=-1) < 1e-6
    out[flat] = _normals(solid)[flat]
    return out


def _box_blur(a: npt.NDArray[np.float64], radius: int, axis: int) -> npt.NDArray[np.float64]:
    """The mean over 2 * radius + 1 cells along `axis` (zero beyond the edge)."""
    pad = [(0, 0)] * a.ndim
    pad[axis] = (radius + 1, radius)
    c = np.cumsum(np.pad(a, pad), axis=axis)
    n = a.shape[axis]
    hi = np.take(c, np.arange(2 * radius + 1, 2 * radius + 1 + n), axis=axis)
    lo = np.take(c, np.arange(0, n), axis=axis)
    return (hi - lo) / (2 * radius + 1)


def _fill_pinholes(index: U8Array, opaque: BoolArray) -> None:
    """Rotated voxels can miss a pixel inside a surface; fill a pixel whose
    left and right (or upper and lower) neighbours are both drawn."""
    hole = ~opaque
    for axis in (0, 1):
        before = np.roll(opaque, 1, axis=axis)
        after = np.roll(opaque, -1, axis=axis)
        fill = hole & before & after
        if axis == 0:
            fill[0, :] = fill[-1, :] = False
        else:
            fill[:, 0] = fill[:, -1] = False
        index[fill] = np.roll(index, 1, axis=axis)[fill]
        opaque |= fill
        hole = ~opaque


def view_for(angle_to_viewer: float) -> int:
    """Which frame to show: the view nearest the direction the viewer is in,
    measured from the thing's facing (radians)."""
    return round(angle_to_viewer / (math.tau / VIEWS)) % VIEWS

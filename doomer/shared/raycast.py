"""Grid raycasting with DDA (digital differential analyser).

Shared because both the renderer (one ray per screen column) and the server
(hitscan weapons, later) need it.

How DDA works: a ray crosses vertical grid lines at regular distances
(`delta_x` apart) and horizontal grid lines at regular distances (`delta_y`
apart). Keep a running "distance to the next vertical line" and "distance to
the next horizontal line", always step to whichever is nearer, and check the
cell you've entered. Each step is a couple of additions, and you never test a
cell the ray doesn't actually pass through.
"""

import math
from dataclasses import dataclass
from enum import Enum

from doomer.shared.mapfile import GameMap


class Side(Enum):
    X = "x"  # hit a wall face perpendicular to the x axis (an east or west face)
    Y = "y"  # hit a north or south face


@dataclass(frozen=True, slots=True)
class RayHit:
    distance: float  # Euclidean, along the ray. Renderers correct for fisheye themselves.
    cell_x: int
    cell_y: int
    wall_type: int
    side: Side
    # Where along the wall face the ray hit, 0..1, for texture lookup. Oriented so
    # textures read left-to-right when you face the wall.
    u: float


def cast_ray(
    game_map: GameMap, x: float, y: float, angle: float, max_distance: float = 64.0
) -> list[RayHit]:
    """Cast a ray and return the hits along it, nearest first.

    Currently every wall is opaque and full-height, so the list ends at the first
    wall (or is empty if nothing is hit within max_distance). With floor and
    ceiling heights (milestone 4), the ray will continue past steps and ledges and
    report each one.
    """
    dx, dy = math.cos(angle), math.sin(angle)
    cell_x, cell_y = math.floor(x), math.floor(y)

    # Distance along the ray between successive vertical (x) / horizontal (y) grid lines.
    delta_x = abs(1.0 / dx) if dx != 0.0 else math.inf
    delta_y = abs(1.0 / dy) if dy != 0.0 else math.inf

    if dx < 0:
        step_x, next_x = -1, (x - cell_x) * delta_x
    else:
        step_x, next_x = 1, (cell_x + 1 - x) * delta_x if dx != 0.0 else math.inf
    if dy < 0:
        step_y, next_y = -1, (y - cell_y) * delta_y
    else:
        step_y, next_y = 1, (cell_y + 1 - y) * delta_y if dy != 0.0 else math.inf

    while True:
        if next_x < next_y:
            distance, side = next_x, Side.X
            next_x += delta_x
            cell_x += step_x
        else:
            distance, side = next_y, Side.Y
            next_y += delta_y
            cell_y += step_y
        if distance > max_distance:
            return []
        wall = game_map.wall_at(cell_x, cell_y)
        if wall:
            return [
                RayHit(distance, cell_x, cell_y, wall, side, _wall_u(x, y, dx, dy, distance, side))
            ]


def _wall_u(x: float, y: float, dx: float, dy: float, distance: float, side: Side) -> float:
    # u increases towards the viewer's right. Facing east, right is +y (south);
    # facing west it's -y; facing south it's -x; facing north it's +x.
    if side is Side.X:
        u = (y + dy * distance) % 1.0
        return u if dx > 0 else 1.0 - u
    u = (x + dx * distance) % 1.0
    return 1.0 - u if dy > 0 else u

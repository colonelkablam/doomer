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
from enum import Enum
from typing import NamedTuple

from doomer.shared.mapfile import DEFAULT_WALL, GameMap


class Side(Enum):
    X = "x"  # hit a wall face perpendicular to the x axis (an east or west face)
    Y = "y"  # hit a north or south face


class RayHit(NamedTuple):
    # A NamedTuple rather than a frozen dataclass: the renderer creates ~1,000 of
    # these per frame, and tuple construction is several times cheaper.
    distance: float  # Euclidean, along the ray. Renderers correct for fisheye themselves.
    cell_x: int
    cell_y: int
    wall_type: int
    side: Side
    # Where along the wall face the ray hit, 0..1, for texture lookup. Oriented so
    # textures read left-to-right when you face the wall.
    u: float
    # Heights as fractions of a wall. A full wall has top 1. A low block (desk)
    # has its own top, and `base` is the top of the cell the ray came from, so
    # the visible front face spans base..top (nothing if base >= top).
    top: float = 1.0
    base: float = 0.0
    # Where the ray leaves the cell: the far edge of a low block's top surface,
    # and the back face it leaves through (seen through an open desk).
    exit_distance: float = math.inf
    exit_side: Side = Side.X
    exit_u: float = 0.0


def cast_ray(
    game_map: GameMap, x: float, y: float, angle: float, max_distance: float = 64.0
) -> list[RayHit]:
    """Cast a ray and return the hits along it, nearest first.

    Low blocks don't stop the ray: each one is reported and the ray carries on,
    so the renderer can draw the desk and the wall behind it. The list ends at
    the first full-height wall, or at max_distance.
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

    # Hot loop (320+ rays a frame): local variables and an inlined grid lookup
    # instead of method calls.
    walls, width, height = game_map.walls, game_map.width, game_map.height
    hits: list[RayHit] = []
    current_top = 0.0  # top of the cell the ray is in (0 = open floor)
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
            return hits
        if 0 <= cell_x < width and 0 <= cell_y < height:
            wall = walls[cell_y][cell_x]
        else:
            wall = DEFAULT_WALL  # outside the map is solid
        if not wall:
            current_top = 0.0
            continue
        top = game_map.top_height(cell_x, cell_y)
        u = _wall_u(x, y, dx, dy, distance, side)
        # The ray leaves this cell at the next grid line it crosses.
        if next_x < next_y:
            exit_distance, exit_side = next_x, Side.X
        else:
            exit_distance, exit_side = next_y, Side.Y
        exit_u = _wall_u(x, y, dx, dy, exit_distance, exit_side)
        hits.append(
            RayHit(
                distance,
                cell_x,
                cell_y,
                wall,
                side,
                u,
                top,
                current_top,
                exit_distance,
                exit_side,
                exit_u,
            )
        )
        if top >= 1.0:
            return hits
        current_top = top


def _wall_u(x: float, y: float, dx: float, dy: float, distance: float, side: Side) -> float:
    # u increases towards the viewer's right. Facing east, right is +y (south);
    # facing west it's -y; facing south it's -x; facing north it's +x.
    if side is Side.X:
        u = (y + dy * distance) % 1.0
        return u if dx > 0 else 1.0 - u
    u = (x + dx * distance) % 1.0
    return 1.0 - u if dy > 0 else u

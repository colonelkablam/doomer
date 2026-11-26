"""The automap: a small green map in the top corner, Tab to show or hide it.

North up (it doesn't turn with you: "the plant room's at the bottom" stays
true, and the fire plans on the walls agree), you in the middle with a line
showing which way you face. See-through where you can walk: walls are
solid dark green blocks with bright green faces, desks and counters dim,
doors as a line (a locked one in its pass's colour), lift cars in amber;
each has a dark shadow below and to its right, so it reads over any
background. A faint green border marks the
display. Only what you've seen is drawn: each frame the cells along a spread
of rays, out to what the renderer hit, are marked seen.

Three zooms (pixels a cell); a zoom's image of the whole floor is drawn once,
and each frame a square is cut out of it round you, so drawing costs little.
Client-side only: what you've seen is yours, not the server's.
"""

import math

import numpy as np
import numpy.typing as npt

from doomer.client.palette import BLUE, BRICK, CREAM, GREEN, OCHRE, ORANGE, U8Array
from doomer.shared.doors import LIFT, doors_from_map
from doomer.shared.mapfile import GameMap

ZOOMS = (2, 4, 8)  # pixels a cell
SIZE = 88  # pixels across the display (of the 320 x 200 frame)
MARGIN = 4  # from the frame's top and right edges
SEE_STEP = 0.33  # cells between the points sampled along each ray
SEE_FAR = 24.0  # cells: the furthest a glance reveals
RAY_STRIDE = 4  # every 4th screen column's ray

WALL = (GREEN << 4) | 14
WALL_FILL = (GREEN << 4) | 4  # inside a wall: solid, darker than its faces
LOW = (GREEN << 4) | 8  # desks, counters, raised floors
DOOR = (GREEN << 4) | 11
LIFT_CAR = (ORANGE << 4) | 13
YOU = (CREAM << 4) | 15
FACING = (CREAM << 4) | 12
BORDER = (GREEN << 4) | 5
SHADOW = 0  # black
PASS_RAMPS = {"blue": BLUE, "red": BRICK, "gold": OCHRE}

BoolArray = npt.NDArray[np.bool_]


def _grow(mask: BoolArray) -> BoolArray:
    """`mask` and its 4-neighbours."""
    out = mask.copy()
    out[1:, :] |= mask[:-1, :]
    out[:-1, :] |= mask[1:, :]
    out[:, 1:] |= mask[:, :-1]
    out[:, :-1] |= mask[:, 1:]
    return out


def _grow8(mask: BoolArray) -> BoolArray:
    """`mask` and all 8 neighbours: an outline drawn with it has no gaps at
    its corners (a corner pixel touches the floor only diagonally)."""
    out = _grow(mask)
    out[1:, 1:] |= mask[:-1, :-1]
    out[1:, :-1] |= mask[:-1, 1:]
    out[:-1, 1:] |= mask[1:, :-1]
    out[:-1, :-1] |= mask[1:, 1:]
    return out


def _cells_to_pixels(cells: BoolArray, zoom: int) -> BoolArray:
    result: BoolArray = np.kron(cells, np.ones((zoom, zoom), dtype=np.bool_))
    return result


def floor_image(game_map: GameMap, zoom: int) -> U8Array:
    """The whole floor at `zoom` pixels a cell, [row (y)][column (x)]: the
    palette index to draw at each pixel, 0 where nothing is drawn."""
    w, h = game_map.width, game_map.height
    full = np.array([[game_map.is_full_wall(x, y) for x in range(w)] for y in range(h)])
    low = np.array([[game_map.is_low_block(x, y) for x in range(w)] for y in range(h)])
    open_ = ~full & ~low
    full_px, low_px = _cells_to_pixels(full, zoom), _cells_to_pixels(low, zoom)
    open_px = _cells_to_pixels(open_, zoom)
    image = np.zeros((h * zoom, w * zoom), dtype=np.uint8)
    image[low_px & _grow8(open_px)] = LOW  # a desk's or counter's outline
    image[full_px] = WALL_FILL  # walls solid, so rooms read as the spaces between
    image[full_px & _grow8(open_px | low_px)] = WALL  # their faces, bright, unbroken
    for door in doors_from_map(game_map):
        if door.kind == LIFT:  # the car: outlined in amber
            cx, cy = door.car
            car = np.zeros((h, w), dtype=np.bool_)
            car[cy, cx] = True
            car_px = _cells_to_pixels(car, zoom)
            image[car_px & ~_grow(~car_px)] = 0
            image[car_px & _grow(~car_px)] = LIFT_CAR
            continue
        colour = DOOR if door.lock is None else (PASS_RAMPS[door.lock] << 4) | 13
        at = round(door.plane * zoom)
        if door.across_x:  # a line x = plane, down the doorway
            y0, y1 = round((door.y - 0.5) * zoom), round((door.y + 0.5) * zoom)
            image[y0:y1, min(at, w * zoom - 1)] = colour
        else:
            x0, x1 = round((door.x - 0.5) * zoom), round((door.x + 0.5) * zoom)
            image[min(at, h * zoom - 1), x0:x1] = colour
    return image


class Automap:
    """What a player has seen of a floor, and the corner map of it."""

    def __init__(self, game_map: GameMap, zoom_index: int = 1) -> None:
        self.map = game_map
        self.seen: BoolArray = np.zeros((game_map.height, game_map.width), dtype=np.bool_)
        self.zoom_index = zoom_index
        self._images: dict[int, U8Array] = {}

    @property
    def zoom(self) -> int:
        return ZOOMS[self.zoom_index]

    def change_zoom(self, step: int) -> None:
        """Zoom in (+1) or out (-1), within the zooms there are."""
        self.zoom_index = max(0, min(len(ZOOMS) - 1, self.zoom_index + step))

    def observe(
        self,
        x: float,
        y: float,
        angles: npt.NDArray[np.float64],
        distances: npt.NDArray[np.float64],
    ) -> None:
        """Mark seen the cells along rays from (x, y) at `angles`, each out to
        its `distance` (what it hit) and just past it (the wall itself)."""
        reach = np.minimum(np.nan_to_num(distances, posinf=SEE_FAR), SEE_FAR) + 0.3
        t = np.arange(0.0, SEE_FAR + 0.3, SEE_STEP)
        inside = t[None, :] <= reach[:, None]
        px = x + np.cos(angles)[:, None] * t[None, :]
        py = y + np.sin(angles)[:, None] * t[None, :]
        cx, cy = np.floor(px[inside]).astype(np.intp), np.floor(py[inside]).astype(np.intp)
        ok = (cx >= 0) & (cy >= 0) & (cx < self.map.width) & (cy < self.map.height)
        self.seen[cy[ok], cx[ok]] = True

    def image(self, zoom: int) -> U8Array:
        if zoom not in self._images:
            self._images[zoom] = floor_image(self.map, zoom)
        return self._images[zoom]

    def draw(self, frame: U8Array, x: float, y: float, angle: float) -> None:
        """Draw the corner map into `frame` ([x][y] palette indices), you at
        (x, y) facing `angle`, in its middle."""
        z = self.zoom
        image = self.image(z)
        rows, cols = image.shape
        # The display's pixels, and the floor image's pixels under them.
        j, i = np.mgrid[0:SIZE, 0:SIZE]
        r = math.floor(y * z) + j - SIZE // 2
        c = math.floor(x * z) + i - SIZE // 2
        inside = (r >= 0) & (c >= 0) & (r < rows) & (c < cols)
        rc, cc = np.clip(r, 0, rows - 1), np.clip(c, 0, cols - 1)
        drawn = image[rc, cc] * (inside & self.seen[rc // z, cc // z])  # [j][i]
        lit = drawn != 0
        shadow = np.zeros_like(lit)
        shadow[1:, 1:] = lit[:-1, :-1]
        shadow &= ~lit
        border = np.zeros_like(lit)
        border[[0, -1], :] = True
        border[:, [0, -1]] = True
        width = frame.shape[0]
        x0, y0 = width - SIZE - MARGIN, MARGIN
        view = frame[x0 : x0 + SIZE, y0 : y0 + SIZE]  # [i][j]
        view[shadow.T] = SHADOW
        view[lit.T] = drawn.T[lit.T]
        view[border.T & ~lit.T] = BORDER
        # You: a dot in the middle, a line the way you face; shadowed too.
        mid = SIZE // 2
        you = np.zeros((SIZE, SIZE), dtype=np.bool_)  # [i][j], as `view`
        facing = np.zeros_like(you)
        you[mid - 1 : mid + 2, mid - 1 : mid + 2] = True
        for k in range(2, 9):
            facing[mid + round(math.cos(angle) * k), mid + round(math.sin(angle) * k)] = True
        # Only the dot is shadowed: a shadow beside a slanting line would
        # interleave with its steps and make it look dashed.
        under = np.zeros_like(you)
        under[1:, 1:] = you[:-1, :-1]
        view[under & ~you & ~facing] = SHADOW
        view[facing] = FACING
        view[you] = YOU

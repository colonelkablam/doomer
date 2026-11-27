"""Behind the start screen: an 80s office block at night, turning slowly,
people working late (windows going on and off).

The block, from a photo of one: a slab tower of brown brick, rows of
punched windows, window-less brick end towers standing proud, a dark
louvred plant floor near the top, a lit sign; on a wide low podium with
two ribbons of glass and a dark entrance. A voxel model (models.py), baked
to TURN_VIEWS views (not the 16 sprites have: it turns slowly, and steps of
22.5 degrees would show) and drawn TOWER_SCALE times its 64 pixels.

Windows going on and off: a baked view is a picture, so a window can't be
switched in it. Instead there's a chain of STATES lightings, each one
window different from the last, baked one by one on a thread (~0.1 s
each) while the menu shows; it steps along the chain every SWITCH_SECONDS,
there and back (never jumping several windows at once), using those baked
so far. A navy night sky with stars behind (dithered lighter over the
city: three shades of navy in the palette would band).
"""

import math
import random
import threading

import numpy as np

from ok_doomer.client.models import Baked, VoxelModel, bake
from ok_doomer.client.palette import BLUE, CREAM, GREY, OCHRE, STEEL, WOOD, U8Array, colour_index

TURN_VIEWS = 64  # baked views round: a step of 5.6 degrees
TURN_SECONDS = 50.0  # once round
STATES = 24  # lightings in the chain
SWITCH_SECONDS = 1.2  # a window on or off this often
LIT_SHARE = 0.3  # of windows lit, to start
TOWER_SCALE = 2.3
WINDOW_FLOORS = range(14, 46, 3)  # each floor's windows' lowest voxel
WINDOW_COLUMNS = range(-19, 19, 3)
FACES = (7, -8)  # the slab's front and back (model x)
WINDOWS = [(x, y, z) for x in FACES for z in WINDOW_FLOORS for y in WINDOW_COLUMNS]


def office_block(lit: frozenset[int], warmth: tuple[float, ...] = (13.0,)) -> VoxelModel:
    """The block, with the windows numbered in `lit` (indices into WINDOWS)
    lit; each lit window's shade from `warmth`, by its number."""
    m = VoxelModel()
    brick, shaft = (WOOD, 7.0), (WOOD, 5.5)
    # The podium: wide and low, two ribbons of glass, a dark entrance.
    m.box((-14, 14), (-30, 30), (0, 12), *brick)
    for z0 in (4, 8):
        m.box((13, 14), (-29, 29), (z0, z0 + 2), STEEL, 5.0)
        m.box((-14, -13), (-29, 29), (z0, z0 + 2), STEEL, 4.0)
        for side in (-30, 29):
            m.box((-13, 13), (side, side + 1), (z0, z0 + 2), STEEL, 4.5)
        for y0 in range(-28, 29, 5):  # mullions
            m.box((13, 14), (y0, y0 + 1), (z0, z0 + 2), *brick)
    m.clear((11, 14), (-20, 20), (0, 4))  # the entrance, set back
    m.box((10, 11), (-20, 20), (0, 4), GREY, 1.0)
    # The slab, its brick end towers standing proud.
    m.box((-8, 8), (-21, 21), (12, 54), *brick)
    for ends in ((-25, -20), (20, 25)):
        m.box((-9, 9), ends, (12, 55), *shaft)
    for i, (x, y, z) in enumerate(WINDOWS):
        if i in lit:
            m.box((x, x + 1), (y, y + 2), (z, z + 2), OCHRE, warmth[i % len(warmth)], glow=True)
        else:
            m.box((x, x + 1), (y, y + 2), (z, z + 2), BLUE, 2.0)
    for z0 in (47, 49, 51):  # the plant floor's louvres
        m.box((7, 8), (-20, 20), (z0, z0 + 1), GREY, 3.0)
        m.box((-8, -7), (-20, 20), (z0, z0 + 1), GREY, 3.0)
    m.box((7, 8), (-19, -16), (52, 54), CREAM, 14.0, glow=True)  # the company's sign
    return m


def lighting_chain(seed: int = 3) -> list[frozenset[int]]:
    """STATES sets of lit windows, each one window different from the last."""
    rng = random.Random(seed)
    lit = {i for i in range(len(WINDOWS)) if rng.random() < LIT_SHARE}
    chain = [frozenset(lit)]
    while len(chain) < STATES:
        lit ^= {rng.randrange(len(WINDOWS))}  # one on, or off
        if frozenset(lit) != chain[-1]:
            chain.append(frozenset(lit))
    return chain


def chain_index(now: float, ready: int) -> int:
    """Where along the chain to be at `now`, there and back, over the first
    `ready` states (those baked so far)."""
    if ready <= 1:
        return 0
    period = 2 * (ready - 1)
    step = int(now / SWITCH_SECONDS) % period
    return step if step < ready else period - step


def night_sky(width: int, height: int, seed: int = 1) -> U8Array:
    """Navy, a little lighter (dithered) over the city, stars, dark ground."""
    rng = np.random.default_rng(seed)
    sky = np.full((width, height), colour_index(BLUE, 1), dtype=np.uint8)
    us, vs = np.meshgrid(np.arange(width), np.arange(height), indexing="ij")
    sky[(vs > height * 0.55) & ((us + vs) % 2 == 0)] = colour_index(BLUE, 2)
    sky[(vs > height * 0.75) & ((us + vs) % 2 == 1)] = colour_index(BLUE, 2)
    for _ in range(width * height // 1000):
        star = colour_index(GREY, int(rng.integers(8, 15)))
        sky[rng.integers(0, width), rng.integers(0, height // 2)] = star
    sky[:, height - 8 :] = colour_index(GREY, 1)
    return sky


class Skyline:
    """The scene: `draw` it each frame. Its lightings bake on a thread from
    the moment it's made; until they have, the first shows."""

    def __init__(self, width: int, height: int, seed: int = 3) -> None:
        self.sky = night_sky(width, height)
        rng = random.Random(seed)
        self.warmth = tuple(rng.choice((12.0, 13.0, 14.0)) for _ in WINDOWS)
        self.chain = lighting_chain(seed)
        self.baked: list[Baked] = [bake(office_block(self.chain[0], self.warmth), TURN_VIEWS)]
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._bake_rest, name="skyline", daemon=True)
        self._thread.start()

    def _bake_rest(self) -> None:
        for lit in self.chain[1:]:
            if self._stop.is_set():
                return
            self.baked.append(bake(office_block(lit, self.warmth), TURN_VIEWS))

    def close(self) -> None:
        self._stop.set()

    def draw(self, pixels: U8Array, now: float) -> None:
        """The sky, and the block as it is at `now` (turned, lit)."""
        pixels[:] = self.sky
        baked = self.baked[chain_index(now, len(self.baked))]
        view = int(now / TURN_SECONDS * TURN_VIEWS) % TURN_VIEWS
        w, h = pixels.shape
        _place(pixels, baked, view, TOWER_SCALE, w / 2, h - 6)


def _place(
    pixels: U8Array, baked: Baked, view: int, scale: float, centre: float, bottom: int
) -> None:
    """A baked view, scaled up (nearest), standing on `bottom`."""
    index, opaque = baked.index[view], baked.opaque[view]
    size = math.floor(index.shape[0] * scale)
    src = (np.arange(size) / scale).astype(np.intp)
    big_index, big_opaque = index[src][:, src], opaque[src][:, src]
    left = round(centre - size / 2)
    top = bottom - size + round(baked.sink * scale)
    w, h = pixels.shape
    u0, v0 = max(0, -left), max(0, -top)
    u1, v1 = min(size, w - left), min(size, h - top)
    if u0 >= u1 or v0 >= v1:
        return
    region = pixels[left + u0 : left + u1, top + v0 : top + v1]
    show = big_opaque[u0:u1, v0:v1]
    region[show] = big_index[u0:u1, v0:v1][show]

"""Beside the start screen's menu: one of the game's own objects, picked at
random each time, turning slowly. The Mac SE/30, the boss's computer, a
cup of coffee, the water cooler, a desk phone.

They're box models (client/boxes.py), drawn by the renderer itself, so
they're shown as they are in the game: rendered in an empty room from
VIEWS points round them, the camera placed by the model's size to fill
SIZE pixels. Each view is rendered twice, with the object and without,
and the pixels that differ are the object: a cut-out, the room left
behind. Baked on a thread (two renders a view) while the menu shows; the
first view is ready at once, and the turn uses the views baked so far.
"""

import math
import random
import threading

import numpy as np

from ok_doomer.client.boxes import BOX_MODELS
from ok_doomer.client.config import RenderConfig
from ok_doomer.client.palette import U8Array
from ok_doomer.client.render import Camera, Renderer, SpriteInstance
from ok_doomer.shared.mapfile import parse_map

SHOWPIECES = ("mac", "pc", "coffee_cup", "water_cooler", "phone")
VIEWS = 48  # round it: a step of 7.5 degrees
TURN_SECONDS = 12.0  # once round
SIZE = 120  # pixels: the square it's drawn in
FILL = 0.85  # of SIZE its widest view fills
# Looking down on it at this angle: its top shows (a phone's buttons, a
# keyboard), whatever its height. Within the renderer's MAX_PITCH (25).
LOOK_DOWN = math.radians(20)
ROOM = 21  # cells across the empty room it's shown in


def _extent(where: U8Array) -> int:
    """Pixels across or down (the larger) that an object covers."""
    cols, rows = np.nonzero(where.any(axis=1))[0], np.nonzero(where.any(axis=0))[0]
    if not len(cols):
        return 1
    return int(max(cols[-1] - cols[0], rows[-1] - rows[0])) + 1


def _room() -> str:
    wall = "#" * ROOM
    inside = "#" + "." * (ROOM - 2) + "#"
    return "[walls]\n" + "\n".join([wall, *[inside] * (ROOM - 2), wall]) + "\n"


class Showcase:
    """One object, turning: `draw` it each frame."""

    def __init__(self, name: str | None = None, seed: int | None = None) -> None:
        self.name = name or random.Random(seed).choice(SHOWPIECES)
        self.config = RenderConfig()
        self.renderer = Renderer(self.config)
        self.map = parse_map(_room())
        self.centre = ROOM / 2
        boxes = BOX_MODELS[self.name]
        xs = [v for b in boxes for v in b.x]
        ys = [v for b in boxes for v in b.y]
        zs = [v for b in boxes for v in b.z]
        # Turned about its middle, it sweeps a circle this wide.
        self.mid = ((min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2)
        reach = max(math.hypot(x - self.mid[0], y - self.mid[1]) for x in xs for y in ys)
        self.height = max(zs)
        # A first guess at how far off to fill SIZE, then measured and put
        # right: a model's widest turn (a keyboard out in front) isn't its
        # bulk, so a guess from its size leaves most things small.
        projection = self.renderer.projection
        self.distance = max(self.height, 2 * reach) * projection / SIZE + reach
        extent = max(_extent(self._view(k)[1]) for k in range(0, VIEWS, VIEWS // 8))
        self.distance *= extent / (FILL * SIZE)
        self.views: list[tuple[U8Array, U8Array]] = [self._view(0)]
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._bake_rest, name="showcase", daemon=True)
        self._thread.start()

    def _view(self, k: int) -> tuple[U8Array, U8Array]:
        """View k round it: (palette indices, where it is), each
        [column][row], SIZE x SIZE about its middle."""
        turn = k * math.tau / VIEWS
        # The object turns; the camera stays put, looking at its middle.
        cos_t, sin_t = math.cos(turn), math.sin(turn)
        mx, my = self.mid
        ox = self.centre - (mx * cos_t - my * sin_t)  # so its middle is at the centre
        oy = self.centre - (mx * sin_t + my * cos_t)
        thing = SpriteInstance(self.name, ox, oy, 0.0, turn)
        eye = self.height / 2 + math.tan(LOOK_DOWN) * self.distance
        pitch = -LOOK_DOWN
        camera = Camera(self.centre - self.distance, self.centre, 0.0, z=eye, pitch=pitch)
        with_it = self.renderer.render_indices(self.map, camera, [thing]).copy()
        without = self.renderer.render_indices(self.map, camera, [])
        w, h = self.config.width, self.config.height
        u0, v0 = w // 2 - SIZE // 2, h // 2 - SIZE // 2
        crop = (slice(u0, u0 + SIZE), slice(v0, v0 + SIZE))
        index = with_it[crop]
        return index, index != without[crop]

    def _bake_rest(self) -> None:
        for k in range(1, VIEWS):
            if self._stop.is_set():
                return
            self.views.append(self._view(k))

    def close(self) -> None:
        self._stop.set()

    def draw(self, pixels: U8Array, now: float, centre: tuple[float, float]) -> None:
        """As it is at `now`, its middle at `centre` (frame pixels)."""
        k = int(now / TURN_SECONDS * VIEWS) % VIEWS
        index, where = self.views[min(k, len(self.views) - 1)]
        w, h = pixels.shape
        left, top = round(centre[0] - SIZE / 2), round(centre[1] - SIZE / 2)
        u0, v0 = max(0, -left), max(0, -top)
        u1, v1 = min(SIZE, w - left), min(SIZE, h - top)
        if u0 >= u1 or v0 >= v1:
            return
        region = pixels[left + u0 : left + u1, top + v0 : top + v1]
        show = where[u0:u1, v0:v1]
        region[show] = index[u0:u1, v0:v1][show]

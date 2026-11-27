"""The start screen's office block: lit windows as asked, a chain of
lightings a window apart, stepped there and back, baked on a thread; the
scene turning and changing as time passes."""

import time

import numpy as np

from ok_doomer.client.palette import OCHRE
from ok_doomer.client.skyline import (
    STATES,
    SWITCH_SECONDS,
    TURN_SECONDS,
    TURN_VIEWS,
    WINDOWS,
    Skyline,
    chain_index,
    lighting_chain,
    office_block,
)


def test_the_lit_windows_are_the_ones_asked_for() -> None:
    lit = frozenset({0, 5, len(WINDOWS) - 1})
    model = office_block(lit)
    glowing_windows = (model.ramp == OCHRE) & model.glow
    assert glowing_windows.sum() == 4 * len(lit)  # each window 2 x 2
    assert (office_block(frozenset()).ramp == OCHRE).sum() == 0


def test_each_lighting_is_one_window_from_the_last() -> None:
    chain = lighting_chain()
    assert len(chain) == STATES
    for a, b in zip(chain, chain[1:], strict=False):
        assert len(a ^ b) == 1
    assert 0.15 < len(chain[0]) / len(WINDOWS) < 0.45  # about a third lit


def test_the_chain_is_walked_there_and_back() -> None:
    steps = [chain_index(k * SWITCH_SECONDS + 0.01, 4) for k in range(9)]
    assert steps == [0, 1, 2, 3, 2, 1, 0, 1, 2]  # never more than a window at a step
    assert chain_index(123.0, 1) == 0  # only the first baked yet: that one


def test_the_scene_bakes_turns_and_changes() -> None:
    skyline = Skyline(320, 200)
    try:
        assert skyline.baked[0].index.shape == (TURN_VIEWS, 64, 64)
        end = time.monotonic() + 30
        while len(skyline.baked) < STATES and time.monotonic() < end:
            time.sleep(0.05)
        assert len(skyline.baked) == STATES  # all baked, on the thread

        def frame(now: float) -> np.ndarray:
            pixels = np.zeros((320, 200), dtype=np.uint8)
            skyline.draw(pixels, now)
            return pixels

        first = frame(0.0)
        assert ((first >> 4) == OCHRE).any()  # lit windows show
        turned = frame(TURN_SECONDS / 4)
        assert (turned != first).any()
        # The next lighting, seen the same way: a window's worth of pixels
        # differ (at most 2 x 2 voxels' picture), in some views, nothing more.
        a, b = skyline.baked[0].index, skyline.baked[1].index
        per_view = (a != b).reshape(TURN_VIEWS, -1).sum(axis=1)
        assert per_view.max() <= 12 and per_view.sum() > 0
    finally:
        skyline.close()


def test_closing_stops_the_baking() -> None:
    skyline = Skyline(320, 200)
    skyline.close()
    skyline._thread.join(5)
    assert not skyline._thread.is_alive() and len(skyline.baked) < STATES

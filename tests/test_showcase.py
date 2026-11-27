"""The object turning beside the start screen: cut out of its room clean,
filling its square, turning, one of the three picked at random."""

import time

import numpy as np

from ok_doomer.client.showcase import FILL, SHOWPIECES, SIZE, TURN_SECONDS, VIEWS, Showcase, _extent


def _baked(name: str) -> Showcase:
    showcase = Showcase(name)
    end = time.monotonic() + 60
    while len(showcase.views) < VIEWS and time.monotonic() < end:
        time.sleep(0.05)
    return showcase


def test_each_is_cut_out_filling_its_square_and_turning() -> None:
    for name in SHOWPIECES:
        showcase = _baked(name)
        try:
            assert len(showcase.views) == VIEWS
            extents = [_extent(where) for _, where in showcase.views]
            assert FILL * SIZE * 0.9 <= max(extents) <= SIZE  # fills it, doesn't spill
            index, where = showcase.views[0]
            assert where.any() and not where[0, :].any() and not where[:, 0].any()  # room gone
            first = np.full((320, 200), 7, dtype=np.uint8)
            showcase.draw(first, 0.0, (160, 100))
            later = np.full((320, 200), 7, dtype=np.uint8)
            showcase.draw(later, TURN_SECONDS / 4, (160, 100))
            assert (first != later).any()  # turned
            assert (first[:, :30] == 7).all()  # only its square drawn on
        finally:
            showcase.close()


def test_one_is_picked_at_random() -> None:
    picks = {Showcase(seed=seed).name for seed in range(12)}
    assert picks == set(SHOWPIECES)

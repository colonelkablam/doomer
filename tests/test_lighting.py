"""The baked light map (doomer/client/lighting.py)."""

import numpy as np
import pytest

from doomer.client.config import LightingConfig
from doomer.client.lighting import LightMap, bake_light_map
from doomer.shared.mapfile import parse_map

CONFIG = LightingConfig()

# A long room, 14 x 3 open cells, one-storey ceiling; panels at x = 3.5 and 7.5.
ROOM = """\
[walls]
################
#..............#
#..............#
#..............#
################

[things]
ceiling_light 3.5 2.5 0
ceiling_light 7.5 2.5 0
"""


def bake(text: str) -> LightMap:
    return bake_light_map(parse_map(text), CONFIG)


def light(m: LightMap, x: float, y: float) -> float:
    return float(m.at(np.array([x]), np.array([y]))[0])


def test_brightest_under_a_panel_dimmer_between_ambient_out_of_reach() -> None:
    m = bake(ROOM)
    under, between, far = light(m, 3.5, 2.5), light(m, 5.5, 2.5), light(m, 14.5, 2.5)
    assert under > between > far
    assert far == pytest.approx(CONFIG.ambient, abs=1e-3)  # 7 cells from the nearest panel
    # Directly under a panel: its own light at distance = its height (one storey).
    own = CONFIG.panel_intensity / (1 + (1.0 / CONFIG.falloff_radius) ** 2)
    assert under > CONFIG.ambient + own


def test_a_wall_stops_light() -> None:
    # The same room split by a wall at x = 6: the panel at 3.5 can't light x 7..
    split = ROOM.replace("#..............#\n", "#.....#........#\n").replace(
        "ceiling_light 7.5 2.5 0\n", ""
    )
    m = bake(split)
    assert light(m, 7.5, 2.5) == pytest.approx(CONFIG.ambient)  # 4 away, behind the wall
    assert light(m, 1.5, 2.5) > CONFIG.ambient + 0.1  # 2 away, same side


def test_light_goes_through_a_doorway() -> None:
    door = ROOM.replace(
        "#..............#\n#..............#\n#..............#\n",
        "#.....#........#\n#..............#\n#.....#........#\n",
    )
    door = door.replace("ceiling_light 7.5 2.5 0\n", "")
    m = bake(door)
    through = light(m, 7.5, 2.5)  # in line with the doorway, 4 cells away: faint
    assert through > CONFIG.ambient + 0.01
    assert light(m, 7.5, 1.2) == pytest.approx(CONFIG.ambient, abs=0.005)  # round the corner


def test_a_panel_in_a_two_storey_ceiling_lights_its_floor_less() -> None:
    tall = ROOM.replace("[things]", "[ceiling]\n" + "gggggggggggggggg\n" * 5 + "\n[things]")
    assert light(bake(tall), 3.5, 2.5) < light(bake(ROOM), 3.5, 2.5)


def test_the_map_is_ambient_plus_its_patches() -> None:
    m = bake(ROOM)
    total = np.full_like(m.values, m.ambient)
    for p in m.patches:
        h, w = p.values.shape
        total[p.row0 : p.row0 + h, p.col0 : p.col0 + w] += p.values
    assert np.allclose(total, m.values)
    assert len(m.patches) == 2


def test_sampling_at_sample_centres_returns_the_samples() -> None:
    m = bake(ROOM)
    s = m.samples_per_cell
    rows, cols = np.mgrid[0 : m.values.shape[0], 0 : m.values.shape[1]]
    assert np.allclose(m.at((cols + 0.5) / s, (rows + 0.5) / s), m.values)


def test_office_bakes_quickly_enough_to_do_at_load() -> None:
    import time

    from doomer.shared.mapfile import load_map

    office = load_map("maps/office.map")
    start = time.perf_counter()
    m = bake_light_map(office, CONFIG)
    assert time.perf_counter() - start < 2.0  # ~0.2 s here; generous for CI
    assert len(m.patches) == sum(t.type == "ceiling_light" for t in office.things)

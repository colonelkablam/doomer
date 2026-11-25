"""The baked light map (doomer/client/lighting.py)."""

import math

import numpy as np
import pytest

from doomer.client.config import LightingConfig
from doomer.client.lighting import LightMap, bake_light_map
from doomer.shared.mapfile import parse_map

CONFIG = LightingConfig(lift=1.0)  # the bake's sums, unbent (lift: its own test)

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
    lights = sum(t.type in ("ceiling_light", "exit_sign") for t in office.things)
    assert len(m.patches) == lights


# --- applied by the renderer (phase 2: walls, faces, box models, sprites) ---

# A long room, panel only at the east end; desks and a railing-free low block
# so block faces are drawn too.
LIT_ROOM = """\
[walls]
################
#..............#
#..2........2..#
#..............#
################

[floor]
0000000000000000
0000000000000000
0003000000003000

[things]
ceiling_light 13.5 2.5 0
"""


def _shade(frame: np.ndarray, mask: np.ndarray | None = None) -> float:
    """Mean shade (low nibble: brightness within the ramp), over a mask."""
    return float(np.mean((frame if mask is None else frame[mask]) & 0x0F))


def test_full_light_everywhere_renders_exactly_as_unlit() -> None:
    from doomer.client.config import RenderConfig
    from doomer.client.render import Camera, Renderer, SpriteInstance

    game_map = parse_map(LIT_ROOM.replace("ceiling_light 13.5 2.5 0\n", ""))
    camera = Camera(1.5, 1.5, 0.3)
    things = [SpriteInstance("mac", 6.5, 2.5, 0.0, 3.0), SpriteInstance("desk", 9.5, 1.5)]
    unlit = Renderer(RenderConfig()).render_indices(game_map, camera, things).copy()
    fade = RenderConfig().shade_distance  # the same distance fade as unlit
    full = Renderer(RenderConfig(), LightingConfig(ambient=1.0, shade_distance=fade))
    assert (full.render_indices(game_map, camera, things) == unlit).all()


def test_walls_and_block_faces_near_a_panel_are_brighter() -> None:
    from doomer.client.config import RenderConfig
    from doomer.client.render import Camera, Renderer

    game_map = parse_map(LIT_ROOM)
    lit = Renderer(RenderConfig(), LightingConfig())
    unlit = Renderer(RenderConfig())
    # The same view of each end of the room: the counter and the end wall
    # behind it, 2.5 cells away. Only the east end has a panel.
    east = Camera(12.5, 1.5, math.pi / 2)
    west = Camera(3.5, 1.5, math.pi / 2)
    rows = slice(90, 130)  # around eye level: counter front and wall
    shades = {}
    for name, camera in (("east", east), ("west", west)):
        frame = lit.render_indices(game_map, camera)[:, rows].copy()
        plain = unlit.render_indices(game_map, camera)[:, rows]
        shades[name] = _shade(frame, np.ones_like(frame, dtype=bool))
        assert _shade(frame, np.ones_like(frame, dtype=bool)) <= _shade(
            plain, np.ones_like(plain, dtype=bool)
        )
    assert shades["east"] > shades["west"] + 1.0


def test_things_are_lit_where_they_stand() -> None:
    from doomer.client.config import RenderConfig
    from doomer.client.render import Camera, Renderer, SpriteInstance

    game_map = parse_map(LIT_ROOM)
    lit = Renderer(RenderConfig(), LightingConfig())
    for name in ("filing_cabinet", "desk"):  # a box model and a sprite
        shades = []
        for x in (13.5, 4.5):  # under the panel; far from it
            camera = Camera(x - 2.0, 2.5, 0.0)
            empty = lit.render_indices(game_map, camera).copy()
            frame = lit.render_indices(
                game_map, camera, [SpriteInstance(name, x, 2.5, 0.0, math.pi)]
            )
            shades.append(_shade(frame, frame != empty))
        assert shades[0] > shades[1] + 1.0, name


# --- phase 3: floors, ceilings, desk tops ---


def test_full_light_renders_every_surface_of_the_office_exactly_as_unlit() -> None:
    # The office has every kind of surface: high ceilings, desk tops, a
    # balcony, stairs, box models, sprites.
    from doomer.client.config import RenderConfig
    from doomer.client.render import Camera, Renderer, sprites_from_things
    from doomer.shared.mapfile import load_map

    office = load_map("maps/office.map")
    things = sprites_from_things(office)
    unlit = Renderer(RenderConfig())
    fade = RenderConfig().shade_distance
    no_panels = LightingConfig(
        ambient=1.0, panel_intensity=0.0, shade_distance=fade, soffit_shade=1.0
    )
    full = Renderer(RenderConfig(), no_panels)
    for camera in (
        Camera(3.0, 5.0, math.radians(-20)),
        Camera(11.5, 4.3, math.radians(-90), pitch=-0.3),
        Camera(6.3, 1.5, math.radians(60), z=1.6, pitch=-0.4),
        Camera(4.5, 5.0, 0.0, pitch=0.4),  # looking up into the lobby
    ):
        expected = unlit.render_indices(office, camera, things).copy()
        assert (full.render_indices(office, camera, things) == expected).all()


def test_floor_is_brighter_near_a_panel_and_ceilings_vary_less() -> None:
    from doomer.client.config import RenderConfig
    from doomer.client.render import Camera, Renderer

    game_map = parse_map(LIT_ROOM)
    lit = Renderer(RenderConfig(), LightingConfig())
    floor_rows, ceiling_rows = slice(150, 200), slice(0, 50)
    shades = {}
    for name, x in (("east", 12.5), ("west", 3.5)):  # the panel is at the east end
        frame = lit.render_indices(game_map, Camera(x, 2.5, 0.0)).copy()
        shades[name] = (_shade(frame[:, floor_rows]), _shade(frame[:, ceiling_rows]))
    # Shade s is brightness (s + 1) / 16, so compare ratios: the ceiling is a
    # much brighter colour than the carpet, so equal light moves it more steps.
    floor_ratio = (shades["east"][0] + 1) / (shades["west"][0] + 1)
    ceiling_ratio = (shades["east"][1] + 1) / (shades["west"][1] + 1)
    assert floor_ratio > 1.3
    assert 1.0 < ceiling_ratio < floor_ratio


@pytest.mark.parametrize(
    "door, blocks", [("door_hinged", True), ("door_lift", True), ("door_sliding", False)]
)
def test_shut_opaque_doors_stop_light_glass_ones_dont(door: str, blocks: bool) -> None:
    # Two rooms joined by a doorway at x = 4, a panel in the east room only.
    two_rooms = parse_map(
        "[walls]\n#########\n#...#...#\n#.......#\n#...#...#\n#########\n"
        f"\n[things]\nceiling_light 6.5 2.5 0\n{door} 4.5 2.5 0\n"
    )
    light = bake_light_map(two_rooms, CONFIG)
    west = float(light.at(np.array([3.0]), np.array([2.5]))[0])
    # On the panel's side of the door (it stands at the east end of its doorway).
    beside = float(light.at(np.array([5.3]), np.array([2.5]))[0])
    if blocks:
        assert west == pytest.approx(CONFIG.ambient)
    else:  # 3.5 cells from the panel: dim, but lit
        assert west > CONFIG.ambient + 0.02
    assert beside > CONFIG.ambient + 0.1  # the doorway on the lit side is lit


def test_lift_raises_dim_light_most() -> None:
    m = parse_map("[walls]\n#######\n#.....#\n#######\n\n[things]\nceiling_light 1.5 1.5 0\n")
    plain = bake_light_map(m, CONFIG)
    lifted = bake_light_map(m, LightingConfig(lift=0.65))
    xs, ys = np.array([1.5, 5.5]), np.array([1.5, 1.5])  # under the panel, far from it
    assert np.allclose(lifted.values, np.power(plain.values, 0.65))  # sample by sample
    before, after = plain.at(xs, ys), lifted.at(xs, ys)
    assert after[1] / before[1] > after[0] / before[0]  # the dim end gains most


def test_exit_signs_are_dim_emergency_lights() -> None:
    m = parse_map(
        "[walls]\n#########\n#.......#\n#.......#\n#########\n"
        "\n[things]\nexit_sign 1.02 1.5 0\nceiling_light 6.5 1.5 0\n"
    )
    light = bake_light_map(m, CONFIG)
    dark = light.without_power()

    def at(lm: LightMap, x: float) -> float:
        return float(lm.at(np.array([x]), np.array([1.5]))[0])

    assert at(dark, 1.5) > CONFIG.unpowered_ambient + 0.2  # the sign's pool, power off
    assert at(dark, 6.5) < CONFIG.unpowered_ambient + 0.05  # the panel's dead
    assert at(light, 6.5) > at(light, 1.5)  # dimmer than a panel


def test_light_through_a_doorway_lights_the_floor_not_the_wall_over_it() -> None:
    # A glass door under a lintel (ceiling 6) between two rooms; a panel in
    # the east one. The west room's floor by the door gets some; the face
    # over the doorway, on the west side, doesn't (the light came under it).
    two_rooms = parse_map(
        "[walls]\n#########\n#...#...#\n#.......#\n#...#...#\n#########\n"
        "\n[ceiling]\n888888888\n888888888\n888868888\n888888888\n888888888\n"
        "\n[things]\nceiling_light 6.5 2.5 0\ndoor_sliding 4.5 2.5 0\n"
    )
    light = bake_light_map(two_rooms, CONFIG)
    west = (np.array([3.7]), np.array([2.5]))
    assert float(light.at(*west)[0]) > CONFIG.ambient + 0.05
    assert float(light.high().at(*west)[0]) == pytest.approx(CONFIG.ambient)

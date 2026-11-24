import math

import numpy as np
import numpy.typing as npt

from doomer.client.config import RenderConfig
from doomer.client.palette import GREEN, GREY
from doomer.client.render import EYE_HEIGHT, Camera, Renderer, SpriteInstance
from doomer.client.sprites import SPRITES
from doomer.shared.mapfile import GameMap, parse_map

OPEN = parse_map("[walls]\n###########\n#.........#\n#.........#\n#.........#\n###########\n")
PILLAR = parse_map("[walls]\n###########\n#.........#\n#..#......#\n#.........#\n###########\n")
CFG = RenderConfig()
IntArray = npt.NDArray[np.intp]
CAM = Camera(1.5, 2.5, 0.0)  # facing east along the middle row
R = Renderer(CFG)


def changed_columns(sprites: list[SpriteInstance], game_map: GameMap = OPEN) -> IntArray:
    """Screen columns that differ once the sprites are drawn."""
    base = R.render_indices(game_map, CAM)
    with_sprites = R.render_indices(game_map, CAM, sprites)
    return np.flatnonzero((base != with_sprites).any(axis=1))


def screen_rect(depth: float) -> tuple[float, float, float]:
    """(left, top, size) in pixels of a sprite straight ahead at `depth`."""
    size = R.projection / depth
    return CFG.width / 2 - size / 2, CFG.height / 2 + EYE_HEIGHT * size - size, size


def test_sprite_straight_ahead_is_centred() -> None:
    cols = changed_columns([SpriteInstance("filing_cabinet", 4.5, 2.5)])
    assert len(cols) > 0
    assert abs((int(cols[0]) + int(cols[-1])) / 2 - CFG.width / 2) <= 1.5  # sorted


def test_sprite_width_is_inversely_proportional_to_depth() -> None:
    near = len(changed_columns([SpriteInstance("filing_cabinet", 3.5, 2.5)]))  # depth 2
    far = len(changed_columns([SpriteInstance("filing_cabinet", 5.5, 2.5)]))  # depth 4
    assert abs(near - 2 * far) <= 2


def test_sprite_behind_wall_is_hidden_and_behind_camera_is_skipped() -> None:
    assert len(changed_columns([SpriteInstance("filing_cabinet", 5.5, 2.5)], PILLAR)) == 0
    assert len(changed_columns([SpriteInstance("filing_cabinet", 0.9, 2.5)])) == 0


def test_partly_hidden_sprite_is_clipped_per_column() -> None:
    # The pillar (cell 3,2) hides lateral offsets |tan| < 1/3 as seen from the camera,
    # i.e. +/-1.33 at depth 4. A sprite at lateral 1.1 spans 0.6..1.6: partly hidden.
    cols = changed_columns([SpriteInstance("desk", 5.5, 3.6)], PILLAR)
    full = changed_columns([SpriteInstance("desk", 5.5, 3.6)])
    assert 0 < len(cols) < len(full)


def test_transparent_texels_show_what_is_behind() -> None:
    sprite = SPRITES["desk"]
    assert not sprite.opaque[0, 2, 2]  # top-left corner of the desk image is empty
    depth = 3.0
    left, top, size = screen_rect(depth)
    x, y = int(left + 2.5 * size / 64), int(top + 2.5 * size / 64)
    base = R.render_indices(OPEN, CAM)
    with_sprite = R.render_indices(OPEN, CAM, [SpriteInstance("desk", 1.5 + depth, 2.5)])
    assert with_sprite[x, y] == base[x, y]


def test_nearer_sprite_is_drawn_over_farther_one() -> None:
    # Filing cabinet (grey) in front of a water cooler (cream/blue); listed far-last
    # to check the renderer sorts rather than trusting input order.
    sprites = [SpriteInstance("filing_cabinet", 3.5, 2.5), SpriteInstance("water_cooler", 5.5, 2.5)]
    frame = R.render_indices(OPEN, CAM, sprites)
    left, top, size = screen_rect(2.0)
    y = int(top + 50.5 * size / 64)  # well inside the cabinet's drawers
    assert frame[CFG.width // 2, y] >> 4 == GREY


def test_fullbright_sprites_are_not_darkened() -> None:
    # The exit sign is full-bright: 4 cells away its green body still appears at
    # its unshaded shade (9). A shaded sprite would be darker at that distance.
    depth = 4.0
    frame = R.render_indices(OPEN, CAM, [SpriteInstance("exit", 1.5 + depth, 2.5)])
    left, top, size = screen_rect(depth)
    region = frame[int(left) : int(left + size), max(0, int(top)) : int(top + size / 2)]
    assert ((GREEN << 4) | 9) in set(region.ravel().tolist())


def test_unknown_sprite_names_are_ignored() -> None:
    assert len(changed_columns([SpriteInstance("coop_start", 4.5, 2.5)])) == 0


def test_every_sprite_is_64x64_with_something_drawn() -> None:
    for name, sprite in SPRITES.items():
        assert sprite.index.shape == sprite.opaque.shape, name
        assert sprite.index.shape[1:] == (64, 64) and len(sprite.index) in (1, 8), name
        assert sprite.opaque.any() and not sprite.opaque.all(), name


def _green(index: npt.NDArray[np.uint8]) -> int:
    return int(((index >> 4) == GREEN).sum())


def test_terminal_has_eight_views_screen_in_front_only() -> None:
    sprite = SPRITES["terminal"]
    assert len(sprite.index) == 8
    front, back = sprite.index[0][sprite.opaque[0]], sprite.index[4][sprite.opaque[4]]
    assert _green(front) > 50 and _green(back) == 0


def test_view_is_picked_from_where_you_see_the_thing() -> None:
    from doomer.client.models import view_for

    assert view_for(0.0) == 0  # from the front
    assert view_for(math.pi) == 4  # from behind
    assert view_for(math.radians(80)) == 2  # from its right (y points down)
    assert view_for(math.radians(-44)) == 7


def test_rotated_terminal_shows_its_screen_only_when_facing_you() -> None:
    # The camera is west of the thing, looking east.
    facing_camera = R.render_indices(
        OPEN, CAM, [SpriteInstance("terminal", 4.5, 2.5, 0.0, math.pi)]
    )
    facing_away = R.render_indices(OPEN, CAM, [SpriteInstance("terminal", 4.5, 2.5, 0.0, 0.0)])
    assert _green(facing_camera) > 20 and _green(facing_away) == 0

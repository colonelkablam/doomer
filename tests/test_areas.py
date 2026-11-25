"""Areas: the [areas] grid, and how it sets floors, ceilings and plain walls."""

import math

import numpy as np
import pytest

from doomer.client import fastcast as fc
from doomer.client.areas import (
    AREA_WALL_TYPE,
    CEILING,
    FLOOR,
    MIP_LEVELS,
    SOFFIT,
    build_surfaces,
    mipmaps,
)
from doomer.client.config import LightingConfig, RenderConfig
from doomer.client.palette import BLUE, STEEL, WARM_GREY, WOOD
from doomer.client.render import Camera, Renderer
from doomer.shared.mapfile import (
    EAST,
    EXECUTIVE,
    LOBBY,
    NORTH,
    OFFICE,
    SERVICE,
    SOUTH,
    WEST,
    MapError,
    format_map,
    parse_map,
)

# Two rooms either side of a plain wall at x = 4: the lobby west, service east.
ROOMS = """\
[walls]
#########
#...#...#
#.......#
#...#...#
#########

[areas]
lllll
lllll
lllll
lllll
lllll
"""


def test_areas_parse_default_to_office_and_round_trip() -> None:
    m = parse_map(ROOMS)
    assert m.area_at(2, 2) == LOBBY and m.area_at(6, 2) == OFFICE  # past the rows given: '.'
    assert m.area_at(-1, 0) == OFFICE  # outside the map
    assert parse_map(format_map(m)) == m
    plain = parse_map(ROOMS.split("[areas]")[0])
    assert plain.area_at(2, 2) == OFFICE and "[areas]" not in format_map(plain)


def test_a_bad_area_letter_is_an_error() -> None:
    with pytest.raises(MapError, match="bad character"):
        parse_map(
            ROOMS.replace("lllll\n[", "llzll\n[", 1).replace("[areas]\nlllll", "[areas]\nllzll")
        )


def test_each_side_of_a_plain_wall_looks_like_the_area_it_faces() -> None:
    m = parse_map(ROOMS.replace("[areas]\n" + "lllll\n" * 5, "[areas]\n" + "lllllssss\n" * 5))
    r = Renderer(RenderConfig())
    assert r._face_type(m, 4, 1, WEST) == AREA_WALL_TYPE + LOBBY
    assert r._face_type(m, 4, 1, EAST) == AREA_WALL_TYPE + SERVICE
    assert r._face_type(m, 0, 2, EAST) == AREA_WALL_TYPE + LOBBY  # the outer wall too


def test_other_wall_types_keep_their_own_textures() -> None:
    m = parse_map(ROOMS.replace("#...#...#\n#....", "#...4...#\n#....", 1))
    assert Renderer(RenderConfig())._face_type(m, 4, 1, WEST) == 4  # a window stays a window


def test_mipmaps_average_down_to_one_colour() -> None:
    surfaces = build_surfaces()
    assert surfaces.mips.shape[1:] == (MIP_LEVELS, 64, 64)
    for level in surfaces.mips[:, -1]:
        assert (level == level[0, 0]).all()  # the last level is a single colour
    flat = np.full((64, 64), 0x37, dtype=np.uint8)
    assert (mipmaps(flat) == 0x37).all()  # averaging one colour changes nothing


def test_each_surface_shows_its_areas_texture() -> None:
    surfaces = build_surfaces()
    areas = np.array([[OFFICE, LOBBY, EXECUTIVE]], dtype=np.int64)
    xs = np.array([0.3, 1.3, 2.3, 1.3, 0.3])
    ys = np.full(5, 0.4)
    kind = np.array([FLOOR, FLOOR, FLOOR, CEILING, SOFFIT])
    texels = fc.surface_texels(
        xs,
        ys,
        np.zeros(5),
        kind,
        np.zeros(5),
        areas,
        surfaces.table,
        np.full((1, 3), -1),
        surfaces.mips,
    )
    u, v = math.floor(0.3 * 64), math.floor(0.4 * 64)
    expected = [
        surfaces.mips[surfaces.table[OFFICE, FLOOR], 0, v, u],
        surfaces.mips[surfaces.table[LOBBY, FLOOR], 0, v, u],
        surfaces.mips[surfaces.table[EXECUTIVE, FLOOR], 0, v, u],
        surfaces.mips[surfaces.table[LOBBY, CEILING], 0, v, u],
        surfaces.mips[surfaces.table[OFFICE, SOFFIT], 0, v, u],
    ]
    assert list(texels) == expected
    # Soffits aren't the room's ceiling tiles.
    assert surfaces.table[OFFICE, SOFFIT] != surfaces.table[OFFICE, CEILING]


def test_far_floor_uses_the_smallest_mip_level() -> None:
    surfaces = build_surfaces()
    areas = np.array([[OFFICE]], dtype=np.int64)
    texel = fc.surface_texels(
        np.array([0.5]),
        np.array([0.5]),
        np.array([99.0]),
        np.array([FLOOR]),
        np.zeros(1),
        areas,
        surfaces.table,
        np.full((1, 1), -1),
        surfaces.mips,
    )
    assert texel[0] == surfaces.mips[surfaces.table[OFFICE, FLOOR], -1, 0, 0]


def test_a_balcony_top_is_carpeted_like_a_floor() -> None:
    # A raised floor (e, height 4) across the room, seen from the floor below.
    m = parse_map(
        "[walls]\n########\n#..eee.#\n#..eee.#\n#..eee.#\n########\n"
        "\n[floor]\n00000000\n00044400\n00044400\n00044400\n00000000\n"
    )
    r = Renderer(RenderConfig())  # unlit: shaded by distance only
    frame = r.render_indices(m, Camera(1.5, 2.5, 0.0, z=0.9, pitch=-0.4))  # eye above it
    # Along one screen row (one distance), a flat colour dithers between at
    # most two shades; textured carpet has more. The nearest row of it: further
    # off, the mipmaps rightly average its fine speckle away.
    rows = [frame[:, y] for y in range(int(r.horizon) + 1, r.config.height)]
    row = [r_ for r_ in rows if ((r_ >> 4) == BLUE).sum() > 100][-1]
    assert len(set((row[(row >> 4) == BLUE] & 0x0F).tolist())) >= 3


def test_rendered_floors_follow_the_areas() -> None:
    hall = parse_map(
        "[walls]\n"
        + "#" * 12
        + "\n"
        + "#..........#\n" * 3
        + "#" * 12
        + "\n\n[areas]\n"
        + "llllll\n" * 5
    )
    r = Renderer(RenderConfig(), LightingConfig(enabled=False))
    frame = r.render_indices(hall, Camera(1.5, 2.5, 0.0, pitch=-0.3))  # facing east, looking down
    # Down the middle column: terrazzo (warm grey) near, in the lobby (x < 6),
    # and carpet tiles (steel) beyond, in the office.
    ramps = set((frame[160, int(r.horizon) + 1 :] >> 4).tolist())
    assert WARM_GREY in ramps and STEEL in ramps


def test_a_bulkhead_looks_like_the_area_it_faces() -> None:
    # A lobby (two storeys) whose ceiling steps down into an office doorway:
    # the face above the doorway is seen from the lobby, so it's dressed as
    # the lobby's walls are.
    m = parse_map(
        "[walls]\n#######\n#.....#\n#######\n\n[ceiling]\n8888888\n8gg8888\n8888888\n"
        "\n[areas]\nlll\nlll\nlll\n"
    )
    r = Renderer(RenderConfig())
    columns = fc.cast_columns(r._map_arrays_for(m), 1.5, 1.5, np.array([0.0]))
    # Span 0: the camera's own (high) ceiling; span 1: into x = 3, the office.
    assert columns.ints[fc.UPPER_SLOT, 1, 0] == r._upper_slot(AREA_WALL_TYPE + LOBBY)
    back = fc.cast_columns(r._map_arrays_for(m), 4.5, 1.5, np.array([math.pi]))
    # From the office, looking back into the lobby, the ceiling steps up:
    # no bulkhead faces you, but the span's look is the office's.
    assert back.ints[fc.UPPER_SLOT, 0, 0] == r._upper_slot(AREA_WALL_TYPE + OFFICE)


def test_a_balcony_carries_on_its_rooms_walls_and_floor() -> None:
    from doomer.client.areas import TOP_KIND

    m = parse_map(
        "[walls]\n#####\n#iii#\n#...#\n#####\n\n[floor]\n00000\n08880\n\n[areas]\n" + "lllll\n" * 4
    )
    r = Renderer(RenderConfig())
    assert r._face_type(m, 2, 1, SOUTH) == AREA_WALL_TYPE + LOBBY  # its front: stone
    assert TOP_KIND[18] == FLOOR  # its top: the area's floor (terrazzo)


def test_desk_pairs_get_paperwork_tops_turned_to_their_facing() -> None:
    from doomer.client.areas import DESK_VARIANTS, cell_tops

    surfaces = build_surfaces()
    m = parse_map("[walls]\n######\n#dddd#\n#jj..#\n######\n\n[facing]\n......\n.nnee.\n.ee...\n")
    tops = cell_tops(m, surfaces)
    assert tops[1, 1] in surfaces.desk_tops[:, 0] and tops[1, 3] in surfaces.desk_tops[:, 1]
    assert tops[2, 1] in surfaces.desk_tops[:, 1]  # j (with screens) too
    assert tops[2, 3] == -1 and tops[0, 0] == -1  # floor, walls: none
    assert tops[1, 1] != tops[1, 2]  # neighbours differ
    assert surfaces.desk_tops.shape == (DESK_VARIANTS, 2)


def test_paperwork_keeps_clear_of_the_macs() -> None:
    from doomer.client.areas import desk_top

    # Each sitter's Mac and keyboard: north half round (0.68, 0.25) of the
    # cell, south half round (0.32, 0.75), in texels. Bare laminate there.
    for variant in range(4):
        top = desk_top(variant)
        bare = (top >> 4) == WOOD
        assert bare[2:30, 37:56].all() and bare[34:62, 8:27].all()


def test_one_desk_variant_has_a_pencil() -> None:
    from doomer.client.areas import DESK_VARIANTS, PENCIL_VARIANT, desk_top
    from doomer.client.palette import PINK

    # The pencil's eraser is a pink no message slip uses.
    eraser = (PINK << 4) | 12
    for variant in range(DESK_VARIANTS):
        has = bool((desk_top(variant) == eraser).any())
        assert has == (variant == PENCIL_VARIANT)


def test_one_desk_variant_has_two_square_notes() -> None:
    from doomer.client.areas import DESK_VARIANTS, NOTE_VARIANT, desk_top
    from doomer.client.palette import OCHRE, PINK

    for variant in range(DESK_VARIANTS):
        top = desk_top(variant)
        pink = ((top >> 4) == PINK) & ((top & 15) != 12)  # not the pencil's eraser
        yellow = ((top >> 4) == OCHRE) & ((top & 15) >= 11)
        if variant != NOTE_VARIANT:
            assert not pink.any()
            continue
        # A pink and a yellow note, each a clean 5 x 5 square on the texel grid.
        rows, cols = np.nonzero(pink)
        assert np.ptp(rows) == 4 and np.ptp(cols) == 4 and pink.sum() == 25
        note = yellow & ((top & 15) >= 11)
        assert note.sum() >= 25


def test_a_doorways_side_walls_are_lined() -> None:
    from doomer.client.areas import REVEAL_WALL_TYPE

    m = parse_map(ROOMS + "\n[things]\ndoor_hinged 4.5 2.5 0\n")
    r = Renderer(RenderConfig())
    assert r._face_type(m, 4, 1, SOUTH) == REVEAL_WALL_TYPE  # the wall north of the doorway
    assert r._face_type(m, 4, 3, NORTH) == REVEAL_WALL_TYPE  # ... and south of it
    assert r._face_type(m, 4, 1, WEST) == AREA_WALL_TYPE + LOBBY  # its room faces: as before

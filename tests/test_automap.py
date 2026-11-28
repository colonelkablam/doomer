"""The automap: a see-through green corner map of what you've seen."""

import math

import numpy as np

from ok_doomer.client.automap import (
    BORDER,
    DOOR,
    LOW,
    MARGIN,
    SIZE,
    WALL,
    WALL_FILL,
    YOU,
    ZOOMS,
    Automap,
    floor_image,
)
from ok_doomer.client.palette import BRICK
from ok_doomer.shared.mapfile import parse_map

# Two rooms joined by a red-locked doorway at (3, 2); a desk in the north room.
ROOMS = parse_map(
    "[walls]\n#######\n#..d..#\n###.###\n#.....#\n#######\n\n"
    "[floor]\n0000000\n0002000\n0000000\n0000000\n0000000\n\n"
    "[things]\ncoop_start 1.5 1.5 0\nexit 5.5 3.5 0\ndoor_hinged 3.5 2.5 90 lock=red\n"
)


def test_the_floor_image_draws_walls_desks_and_locked_doors() -> None:
    image = floor_image(ROOMS, 4)
    assert image.shape == (5 * 4, 7 * 4)
    assert (image == WALL).any() and (image == LOW).any()
    # The wall between the rooms is solid: its middle filled, its faces bright.
    middle = image[2 * 4 + 1 : 3 * 4 - 1, 1 * 4 + 1 : 2 * 4 - 1]
    assert (middle == WALL_FILL).all()
    assert (image == (BRICK << 4) | 13).any()  # the red-locked door, in red
    assert not (image == DOOR).any()  # it's the only door, and it's locked
    assert (image[1 * 4 + 1 : 2 * 4 - 1, 1 * 4 + 1 : 2 * 4 - 1] == 0).all()  # open floor: nothing


def test_seeing_stops_just_past_what_you_hit() -> None:
    m = Automap(ROOMS)
    # Looking east along the north room from (1.5, 1.5): a desk 1.5 away.
    m.observe(1.5, 1.5, np.array([0.0]), np.array([1.5]))
    assert m.seen[1, 1] and m.seen[1, 2] and m.seen[1, 3]  # up to and including it
    assert not m.seen[1, 5]  # not beyond
    assert not m.seen[3].any()  # nor the other room


def test_only_what_you_have_seen_is_drawn_and_the_rest_shows_through() -> None:
    m = Automap(ROOMS)
    frame = np.full((320, 200), 0x55, dtype=np.uint8)
    m.draw(frame, 1.5, 1.5, 0.0)
    x0 = 320 - SIZE - MARGIN
    view = frame[x0 : x0 + SIZE, MARGIN : MARGIN + SIZE]
    assert not (view == WALL).any()  # nothing seen yet: no walls
    assert (view == YOU).any() and (view == BORDER).any()
    assert (frame[:x0] == 0x55).all()  # the rest of the frame untouched
    m.seen[:] = True
    m.draw(frame, 1.5, 1.5, 0.0)
    assert (view == WALL).any()
    assert (view == 0x55).mean() > 0.2  # see-through where you can walk


def test_you_are_in_the_middle_facing_the_way_you_look() -> None:
    m = Automap(ROOMS)
    frame = np.zeros((320, 200), dtype=np.uint8)
    m.draw(frame, 3.0, 2.0, math.pi / 2)  # facing south: y down
    x0 = 320 - SIZE - MARGIN
    mid = SIZE // 2
    assert frame[x0 + mid, MARGIN + mid] == YOU
    column = frame[x0 + mid, MARGIN + mid + 3 : MARGIN + mid + 8]
    assert (column != 0).all()  # the facing line runs down from you


def test_zoom_stays_within_its_steps() -> None:
    m = Automap(ROOMS, zoom_index=0)
    m.change_zoom(-1)
    assert m.zoom == ZOOMS[0]
    for _ in range(5):
        m.change_zoom(1)
    assert m.zoom == ZOOMS[-1]


def test_wall_faces_join_round_corners_without_a_gap() -> None:
    # A pillar in a room: its outline must be a closed ring, corners and all.
    walls = "[walls]\n#####\n#...#\n#.#.#\n#...#\n#####\n\n"
    room = parse_map(walls + "[things]\ncoop_start 1.5 1.5 0\nexit 3.5 3.5 0\n")
    for z in ZOOMS:
        image = floor_image(room, z)
        ring = image[2 * z : 3 * z, 2 * z : 3 * z]  # the pillar's cell
        edge = np.zeros_like(ring, dtype=bool)
        edge[[0, -1], :] = True
        edge[:, [0, -1]] = True
        assert (ring[edge] == WALL).all(), z  # its whole outline, corners included

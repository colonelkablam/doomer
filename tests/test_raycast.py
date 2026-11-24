import math

import pytest

from doomer.shared.mapfile import EAST, NORTH, SOUTH, WEST, parse_map
from doomer.shared.raycast import Side, cast_ray

# 7x5 room; interior x in [1, 6), y in [1, 4). Wall type 2 on the east side.
ROOM = parse_map(
    """\
[walls]
#######
#.....2
#.....2
#.....2
#######
"""
)


def test_east_wall_distance_side_and_type() -> None:
    (hit,) = cast_ray(ROOM, 1.5, 2.5, 0.0)
    assert hit.distance == pytest.approx(4.5)
    assert (hit.cell_x, hit.cell_y, hit.wall_type, hit.side) == (6, 2, 2, Side.X)


def test_north_and_south_walls() -> None:
    (north,) = cast_ray(ROOM, 2.5, 2.25, -math.pi / 2)
    (south,) = cast_ray(ROOM, 2.5, 2.25, math.pi / 2)
    assert north.distance == pytest.approx(1.25)
    assert south.distance == pytest.approx(1.75)
    assert north.side is south.side is Side.Y
    assert (north.cell_y, south.cell_y) == (0, 4)


def test_diagonal_distance() -> None:
    # From (1.5, 1.5) heading 45 deg south-east, the ray crosses (3, 3)... until the
    # south wall at y = 4: distance = 2.5 * sqrt(2).
    (hit,) = cast_ray(ROOM, 1.5, 1.5, math.pi / 4)
    assert hit.distance == pytest.approx(2.5 * math.sqrt(2))
    assert hit.side is Side.Y


def test_exact_axis_angles_have_no_nan() -> None:
    for angle in (0.0, math.pi / 2, math.pi, 3 * math.pi / 2):
        hits = cast_ray(ROOM, 3.0, 2.0, angle)  # start exactly on grid lines
        assert len(hits) == 1
        assert math.isfinite(hits[0].distance)


def test_max_distance_returns_no_hits() -> None:
    assert cast_ray(ROOM, 1.5, 2.5, 0.0, max_distance=2.0) == []


def test_map_edge_is_solid() -> None:
    open_map = parse_map("[walls]\n...\n...\n")
    (hit,) = cast_ray(open_map, 0.5, 0.5, 0.0)
    assert hit.cell_x == 3
    assert hit.distance == pytest.approx(2.5)


@pytest.mark.parametrize(
    "angle",
    [0.0, math.pi, math.pi / 2, -math.pi / 2],
    ids=["east", "west", "south", "north"],
)
def test_u_increases_towards_viewers_right(angle: float) -> None:
    # From a cell centre, rays slightly left and right of straight ahead hit the
    # same wall cell; +angle is clockwise, i.e. to the viewer's right.
    (left,) = cast_ray(ROOM, 3.5, 2.5, angle - 0.1)
    (right,) = cast_ray(ROOM, 3.5, 2.5, angle + 0.1)
    assert (left.cell_x, left.cell_y) == (right.cell_x, right.cell_y)
    assert 0.0 <= left.u < 0.5 < right.u <= 1.0


DESKS = parse_map(
    """\
[walls]
##########
#........#
#..22.3..#
#........#
##########

[floor]
0000000000
0000000000
0002305000
0000000000
"""
)


def test_ray_continues_through_low_blocks_to_the_wall() -> None:
    hits = cast_ray(DESKS, 1.5, 2.5, 0.0)
    assert [(h.cell_x, h.top) for h in hits] == [(3, 0.25), (4, 0.375), (6, 0.625), (9, 1.0)]
    first, second, third, wall = hits
    assert (first.distance, first.exit_distance) == pytest.approx((1.5, 2.5))
    assert (second.distance, second.exit_distance) == pytest.approx((2.5, 3.5))
    assert first.base == 0.0  # came from open floor
    assert second.base == 0.25  # came straight off the first desk
    assert third.base == 0.0  # open floor in between
    assert wall.distance == pytest.approx(7.5)


def test_low_blocks_within_max_distance_are_still_reported() -> None:
    hits = cast_ray(DESKS, 1.5, 2.5, 0.0, max_distance=3.0)
    assert [h.cell_x for h in hits] == [3, 4]


@pytest.mark.parametrize(
    ("angle", "face", "exit_face"),
    [
        (0.0, WEST, EAST),
        (math.pi, EAST, WEST),
        (math.pi / 2, NORTH, SOUTH),
        (-math.pi / 2, SOUTH, NORTH),
    ],
)
def test_entry_and_exit_faces(angle: float, face: int, exit_face: int) -> None:
    block = parse_map(
        "[walls]\n#####\n#...#\n#.2.#\n#...#\n#####\n\n[floor]\n00000\n00000\n00200\n"
    )
    # Start beside the block on the side opposite the direction of travel.
    starts = {WEST: (1.5, 2.5), EAST: (3.5, 2.5), NORTH: (2.5, 1.5), SOUTH: (2.5, 3.5)}
    x, y = starts[face]
    hit = cast_ray(block, x, y, angle)[0]
    assert (hit.cell_x, hit.cell_y) == (2, 2)
    assert (hit.face, hit.exit_face) == (face, exit_face)

import math

import pytest

from doomer.shared.mapfile import parse_map
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

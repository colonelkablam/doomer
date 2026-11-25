"""Furniture with a real floor plan (doomer/shared/furniture.py): footprints,
collision against them, and things resting on furniture."""

import math

import pytest

from doomer.shared.collision import Obstacle, move_box
from doomer.shared.furniture import placed_footprints, resting_height
from doomer.shared.mapfile import Thing, parse_map
from doomer.shared.things import THING_TYPES, Footprint, ThingType, register_thing_type

if "test_bench" not in THING_TYPES:
    register_thing_type(
        ThingType(
            "test_bench",
            "A bench for tests: a long, shallow footprint ahead of its centre",
            blocking=True,
            radius=0.1,
            footprints=(Footprint((0.1, 0.3), (-0.5, 0.5), 0.2),),
        )
    )

ROOM = "[walls]\n##########\n#........#\n#........#\n#........#\n#........#\n##########\n"


@pytest.mark.parametrize(
    "degrees, expected",
    [
        (0, (5.1, 5.3, 4.5, 5.5)),  # facing east: in front is +x
        (90, (4.5, 5.5, 5.1, 5.3)),  # facing south: +y
        (180, (4.7, 4.9, 4.5, 5.5)),  # facing west: -x
        (270, (4.5, 5.5, 4.7, 4.9)),  # facing north: -y
    ],
)
def test_footprints_turn_with_the_thing(degrees: int, expected: tuple[float, ...]) -> None:
    (p,) = placed_footprints(Thing("test_bench", 5.0, 5.0, math.radians(degrees)))
    assert (p.x0, p.x1, p.y0, p.y1) == pytest.approx(expected)
    assert p.top == 0.2


def test_a_rectangle_blocks_on_each_axis_by_its_own_half_width() -> None:
    room = parse_map(ROOM)
    ob = Obstacle(4.0, 2.5, 0.1, 0.8)  # thin across x, long across y
    r = 0.25
    assert move_box(room, 2.0, 2.5, 3.0, 0.0, r, [ob])[0] == pytest.approx(4.0 - 0.1 - r)
    assert move_box(room, 4.0, 1.0, 0.0, 3.0, r, [ob])[1] == pytest.approx(2.5 - 0.8 - r)
    # Beside its end (y beyond 2.5 + 0.8 + r): no effect.
    assert move_box(room, 2.0, 3.6, 3.0, 0.0, r, [ob])[0] == pytest.approx(5.0)


def test_things_rest_on_the_furniture_they_are_placed_on() -> None:
    room = parse_map(
        ROOM + "[things]\nreception_desk 4.5 2.5 180\npass_blue 4.25 2.4 0\n"
        "mac 4.7 2.0 0\nficus 6.5 2.5 0\n"
    )
    by_type = {t.type: t for t in room.things}
    assert resting_height(room, by_type["reception_desk"]) == 0.0  # not on itself
    assert resting_height(room, by_type["pass_blue"]) == pytest.approx(0.425)  # the ledge
    assert resting_height(room, by_type["mac"]) == pytest.approx(0.27)  # the work surface
    assert resting_height(room, by_type["ficus"]) == 0.0  # beside it, on the floor


def test_the_reception_desk_blocks_where_it_is_drawn() -> None:
    from doomer.client.boxes import BOX_MODELS

    footprints = THING_TYPES["reception_desk"].footprints
    xs = [x for f in footprints for x in f.x]
    ys = [y for f in footprints for y in f.y]
    for box in BOX_MODELS["reception_desk"]:  # everything drawn is within what blocks
        assert min(xs) - 1e-9 <= box.x[0] and box.x[1] <= max(xs) + 1e-9, box
        assert min(ys) - 1e-9 <= box.y[0] and box.y[1] <= max(ys) + 1e-9, box
    tops = sorted(f.top for f in footprints)
    drawn_tops = {round(b.z[1], 3) for b in BOX_MODELS["reception_desk"]}
    assert {round(t, 3) for t in tops} <= drawn_tops  # the surfaces things rest on exist


def test_walking_up_to_the_desk_you_stop_at_its_ledge() -> None:
    from doomer.shared.player import PlayerInput
    from doomer.shared.sim import step
    from doomer.shared.world import new_world, spawn_player

    room = parse_map(
        ROOM + "[things]\ncoop_start 1.5 2.5 0\nexit 8.5 4.5 0\nreception_desk 4.5 2.5 180\n"
    )
    world = new_world(room, "coop")
    p = spawn_player(world, 1, "p")
    for _ in range(90):
        step(world, {1: PlayerInput(forward=1.0, angle=0.0)})
    # The ledge's front edge (the desk faces west): x = 4.5 - 0.31.
    assert p.x == pytest.approx(4.5 - 0.31 - world.config.movement.radius)


def test_the_boss_desk_blocks_where_it_is_drawn() -> None:
    from doomer.client.boxes import BOX_MODELS

    footprints = THING_TYPES["executive_desk"].footprints
    for box in BOX_MODELS["executive_desk"]:  # every box is inside some footprint
        assert any(
            f.x[0] - 1e-9 <= box.x[0]
            and box.x[1] <= f.x[1] + 1e-9
            and f.y[0] - 1e-9 <= box.y[0]
            and box.y[1] <= f.y[1] + 1e-9
            for f in footprints
        ), box


def test_the_red_pass_on_the_boss_desk_can_be_reached_from_the_front() -> None:
    from doomer.shared.mapfile import load_map
    from doomer.shared.player import PlayerInput
    from doomer.shared.sim import step
    from doomer.shared.world import new_world, spawn_player

    world = new_world(load_map("maps/office.map"), "coop")
    p = spawn_player(world, 1, "p")
    p.x, p.y = 5.5, 12.0
    red = next(k for k in world.pickups if k.type == "pass_red")
    for _ in range(120):  # walk west, into the desk's front
        step(world, {1: PlayerInput(forward=1.0, angle=math.pi)})
    assert p.x == pytest.approx(3.0 + 0.46 + world.config.movement.radius)  # stopped at it
    assert red.taken

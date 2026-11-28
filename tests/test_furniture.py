"""Furniture with a real floor plan (ok_doomer/shared/furniture.py): footprints,
collision against them, and things resting on furniture."""

import math

import pytest

from ok_doomer.shared.collision import Obstacle, move_box
from ok_doomer.shared.furniture import placed_footprints, resting_height
from ok_doomer.shared.mapfile import Thing, parse_map
from ok_doomer.shared.things import THING_TYPES, Footprint, ThingType, register_thing_type

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
    from ok_doomer.client.boxes import BOX_MODELS

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
    from ok_doomer.shared.player import PlayerInput
    from ok_doomer.shared.sim import step
    from ok_doomer.shared.spawn import spawn_player
    from ok_doomer.shared.world import new_world

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
    from ok_doomer.client.boxes import BOX_MODELS

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
    from ok_doomer.shared.mapfile import load_map
    from ok_doomer.shared.player import PlayerInput
    from ok_doomer.shared.sim import step
    from ok_doomer.shared.spawn import spawn_player
    from ok_doomer.shared.world import new_world

    world = new_world(load_map("maps/office.map"), "coop")
    p = spawn_player(world, 1, "p")
    p.x, p.y = 5.5, 18.0
    red = next(k for k in world.pickups if k.type == "pass_red")
    for _ in range(120):  # walk west, into the desk's front
        step(world, {1: PlayerInput(forward=1.0, angle=math.pi)})
    assert p.x == pytest.approx(3.0 + 0.46 + world.config.movement.radius)  # stopped at it
    assert red.taken


def test_phones_sit_on_the_boss_and_reception_desks_handsets_front_to_back() -> None:
    from ok_doomer.client.boxes import BOX_MODELS
    from ok_doomer.shared.mapfile import load_map

    office = load_map("maps/office.map")
    phones = office.things_of_type("phone")
    (desk,) = office.things_of_type("executive_desk")
    (counter,) = office.things_of_type("reception_desk")
    boss = min(phones, key=lambda t: math.hypot(t.x - desk.x, t.y - desk.y))
    reception = min(phones, key=lambda t: math.hypot(t.x - counter.x, t.y - counter.y))
    assert all(resting_height(office, t) > 0.2 for t in phones)  # all on desks
    assert resting_height(office, boss) > 0.3  # on the boss's desk top
    assert resting_height(office, reception) == pytest.approx(0.27)  # reception's work surface
    keypad, cradle, handset = BOX_MODELS["phone"]  # three boxes
    assert cradle.z[1] > keypad.z[1]  # the cradle raised beside the keypad
    assert cradle.y[0] >= keypad.y[1] - 1e-9  # on the user's left (+y), the keypad right
    assert handset.z[0] >= cradle.z[1] - 1e-9  # lying on it
    # Front to back: longer along x than across.
    assert handset.x[1] - handset.x[0] > 2 * (handset.y[1] - handset.y[0])


def test_the_boss_has_a_green_screen_pc_on_the_desk() -> None:
    from ok_doomer.client.boxes import BOX_MODELS
    from ok_doomer.shared.mapfile import load_map

    office = load_map("maps/office.map")
    (pc,) = office.things_of_type("pc")
    assert resting_height(office, pc) > 0.3  # on the boss's desk top
    unit, monitor, tube, screen, keyboard = BOX_MODELS["pc"]
    assert monitor.z[0] == unit.z[1] and tube.x[1] <= monitor.x[0]  # standing on it, tube behind
    assert screen.glow and screen.mains  # lit only while there's power
    assert screen.x[0] == monitor.x[1] and keyboard.x[0] > unit.x[1]  # on its face; keys in front


def test_a_swivel_chair_is_four_boxes_its_base_a_painted_star() -> None:
    from ok_doomer.client.boxes import BOX_MODELS, BOX_OPAQUE

    base, column, seat, back = BOX_MODELS["chair"]
    assert base.z[1] <= column.z[0] and column.z[1] <= seat.z[0] + 1e-9  # stacked up
    assert back.x[1] <= seat.x[0] + 0.01 and back.z[1] > seat.z[1]  # behind the sitter, higher
    star = BOX_OPAQUE[base.faces[4]]
    assert 0.1 < star.mean() < 0.5  # arms and casters; the rest see-through
    assert not any(BOX_OPAQUE[f].any() for f in base.faces[:4])  # no sides: just the star


def test_every_sitter_has_a_chair_at_their_mac_and_the_aisles_stay_passable() -> None:
    # (Chairs without a Mac too: round the lounge and meeting room tables.)
    from ok_doomer.shared.config import MovementConfig
    from ok_doomer.shared.mapfile import load_map

    office = load_map("maps/office.map")
    macs, chairs = office.things_of_type("mac"), office.things_of_type("chair")
    radius = THING_TYPES["chair"].radius
    for mac in macs:
        chair = min(chairs, key=lambda c: math.hypot(c.x - mac.x, c.y - mac.y))
        assert math.hypot(chair.x - mac.x, chair.y - mac.y) < 0.8  # pulled up to it
        assert not office.is_solid(math.floor(chair.x), math.floor(chair.y))  # on the floor
    # Between the north wall (y = 1) and the first pod row (y = 2): room
    # for a player past the chairs.
    north = [c for c in chairs if 1.0 < c.y < 2.0]
    assert north and min(c.y - radius for c in north) - 1.0 >= 2 * MovementConfig().radius


@pytest.mark.parametrize("name", ["sofa_two", "sofa_three", "sofa_corner", "coffee_table"])
def test_lobby_seating_footprints_match_their_models(name: str) -> None:
    from ok_doomer.client.boxes import BOX_MODELS

    boxes = BOX_MODELS[name]
    (fp,) = THING_TYPES[name].footprints
    assert min(b.x[0] for b in boxes) == pytest.approx(fp.x[0])
    assert max(b.x[1] for b in boxes) == pytest.approx(fp.x[1])
    assert min(b.y[0] for b in boxes) == pytest.approx(fp.y[0])
    assert max(b.y[1] for b in boxes) == pytest.approx(fp.y[1])
    assert max(b.z[1] for b in boxes) == pytest.approx(fp.top)


def test_the_nooks_ferns_stand_on_their_coffee_tables() -> None:
    from ok_doomer.shared.mapfile import load_map

    office = load_map("maps/office.map")
    ferns = office.things_of_type("fern")
    assert len(ferns) == len(office.things_of_type("coffee_table")) >= 2  # lobby, lounge
    for fern in ferns:
        assert resting_height(office, fern) == pytest.approx(0.15)  # the table's top


@pytest.mark.parametrize("size", ["bin", "bin_tall"])
@pytest.mark.parametrize("kind", ["steel", "wood", "orange", "chrome"])
def test_a_bins_footprint_matches_its_model(size: str, kind: str) -> None:
    from ok_doomer.client.boxes import BOX_MODELS

    (body,) = BOX_MODELS[f"{size}:{kind}"]
    (fp,) = THING_TYPES[size].footprints
    assert (body.x, body.y, body.z[1]) == (fp.x, fp.y, fp.top)


def test_there_are_bins_round_the_floor_and_in_both_washrooms() -> None:
    from ok_doomer.shared.mapfile import WASHROOM, load_map

    office = load_map("maps/office.map")
    bins = office.things_of_type("bin") + office.things_of_type("bin_tall")
    assert len(bins) >= 10 and office.things_of_type("bin_tall")
    in_washrooms = [b for b in bins if office.area_at(math.floor(b.x), math.floor(b.y)) == WASHROOM]
    assert len(in_washrooms) == 2 and all(b.prop("kind") == "chrome" for b in in_washrooms)
    for b in bins:  # on the floor, not in a desk or a wall
        assert not office.is_solid(math.floor(b.x), math.floor(b.y))

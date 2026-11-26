"""ID passes and locked doors: picking passes up, locks, the HUD's messages."""

import math

import pytest

from ok_doomer.shared.doors import HINGED, OPEN_SECONDS, SLIDING
from ok_doomer.shared.events import DoorLocked, DoorToggled, PassPickedUp
from ok_doomer.shared.mapfile import parse_map
from ok_doomer.shared.player import PlayerInput, PlayerState
from ok_doomer.shared.sim import step
from ok_doomer.shared.spawn import spawn_player
from ok_doomer.shared.world import World, drain_events, new_world

NORTH, SOUTH, EAST, WEST = -math.pi / 2, math.pi / 2, 0.0, math.pi

# Two rooms divided by a wall along y = 4: a blue-locked office door at x = 3
# and red-locked glass doors at x = 8. The blue pass lies in the south room at
# (6.5, 6.5); a counter (height 3/8) at (10, 6) has the red pass on top, near
# its south edge.
LOCKED = """\
[walls]
############
#..........#
#..........#
#..........#
###.####.###
#..........#
#.........2#
#..........#
############

[floor]
000000000000
000000000000
000000000000
000000000000
000000000000
000000000000
000000000030

[ceiling]
888888888888
888888888888
888888888888
888888888888
888688886888

[things]
coop_start 3.5 6.5 270
exit 10.5 1.5 0
dm_spawn 1.5 1.5 0
dm_spawn 10.5 1.5 0
dm_spawn 1.5 7.5 0
dm_spawn 10.5 7.5 0
door_hinged 3.5 4.5 270 lock=blue
door_sliding 8.5 4.5 90 lock=red
pass_blue 6.5 6.5 0
pass_red 10.5 6.7 0
"""


def world_at(x: float, y: float, mode: str = "coop") -> tuple[World, PlayerState]:
    world = new_world(parse_map(LOCKED), mode)
    player = spawn_player(world, 1, "p")
    player.x, player.y = x, y
    return world, player


def run(world: World, ticks: int, angle: float, **held: float | bool) -> list[object]:
    events: list[object] = []
    for _ in range(ticks):
        step(world, {1: PlayerInput(angle=angle, **held)})  # type: ignore[arg-type]
        events += drain_events(world)
    return events


def settle(kind: str) -> int:
    return math.ceil(OPEN_SECONDS[kind] * 60) + 2


def test_walking_over_a_pass_gives_it_to_the_team() -> None:
    world, p = world_at(4.5, 6.5)
    events = run(world, 60, EAST, forward=1.0)
    assert "blue" in world.passes
    assert [e.colour for e in events if isinstance(e, PassPickedUp)] == ["blue"]
    assert world.pickups[0].taken


def test_you_can_take_a_pass_off_a_counter_from_the_floor() -> None:
    world, p = world_at(10.5, 7.5)  # in front of the counter (it's too high to step onto)
    run(world, 30, NORTH, forward=1.0)
    assert "red" in world.passes and p.z == 0.0


def test_a_locked_door_stays_shut_and_says_which_pass_it_needs() -> None:
    world, p = world_at(3.5, 5.6)  # south of the door, 0.65 from it
    events = run(world, 1, NORTH, use=True)
    run(world, settle(HINGED), NORTH)
    assert world.doors[0].open == 0.0
    assert [(e.lock, e.door_id) for e in events if isinstance(e, DoorLocked)] == [("blue", 0)]
    assert not any(isinstance(e, DoorToggled) for e in events)


def test_with_the_pass_a_locked_door_opens() -> None:
    world, p = world_at(3.5, 5.6)  # south of the door, 0.65 from it
    world.passes.add("blue")
    events = run(world, 1, NORTH, use=True)
    run(world, settle(HINGED), NORTH)
    assert world.doors[0].open == 1.0
    assert any(isinstance(e, DoorToggled) for e in events)


def test_locked_glass_doors_dont_open_until_the_team_has_the_pass() -> None:
    world, p = world_at(8.5, 5.6)  # right in front of them
    run(world, settle(SLIDING) + 10, NORTH)
    assert world.doors[1].open == 0.0
    events = run(world, 1, NORTH, use=True)  # pressing use says why
    assert any(isinstance(e, DoorLocked) and e.lock == "red" for e in events)
    world.passes.add("red")
    run(world, settle(SLIDING) + 10, NORTH)
    assert world.doors[1].open == 1.0


def test_the_whole_loop_blue_then_red() -> None:
    # Fetch the blue pass, open the blue door, then the red pass, the red door.
    world, p = world_at(5.5, 6.5)
    run(world, 30, EAST, forward=1.0)  # over the blue pass
    p.x, p.y = 3.5, 5.6
    run(world, 1, NORTH, use=True)
    run(world, settle(HINGED), NORTH)
    run(world, 90, NORTH, forward=1.0)
    assert p.y < 3.5  # through the blue door
    world.pickups[1].taken, p.x, p.y = False, 10.5, 7.5  # and the red pass on the counter
    run(world, 30, NORTH, forward=1.0)
    assert world.passes == {"blue", "red"}


def test_deathmatch_starts_with_every_pass() -> None:
    world, _ = world_at(1.5, 1.5, mode="deathmatch")
    assert world.passes == {"blue", "red", "gold"}


def test_card_readers_by_locked_doors_light_once_the_team_can_open_them() -> None:
    from ok_doomer.client.boxes import door_boxes
    from ok_doomer.shared.doors import doors_from_map

    doors = doors_from_map(parse_map(LOCKED))
    readers = [b for b in door_boxes(doors) if b.box.z[0] > 0.3]
    assert len(readers) == 4  # two per locked door, one each side of the wall
    lit = [b for b in door_boxes(doors, {"blue"}) if b.box.z[0] > 0.3]
    changed = {b.box.faces for b in lit} - {b.box.faces for b in readers}
    assert len(changed) == 1  # the blue door's readers turned green; the red ones didn't


def test_taken_passes_are_no_longer_drawn() -> None:
    from ok_doomer.client.session import LocalSession

    session = LocalSession(new_world(parse_map(LOCKED), "coop"), "p")
    assert sum(s.name.startswith("pass_") for s in session.sprites()) == 2
    session.world.pickups[0].taken = True
    assert [s.name for s in session.sprites() if s.name.startswith("pass_")] == ["pass_red"]


@pytest.mark.parametrize(
    "event, text",
    [
        (PassPickedUp(0, 1, "blue"), "Picked up the blue pass"),
        (DoorLocked(0, 0, 1, "red"), "Locked: you need the red pass"),
        (DoorToggled(0, 0, 1, True), None),
    ],
)
def test_hud_messages(event: object, text: str | None) -> None:
    from ok_doomer.client.hud import message_for

    assert message_for(event) == text  # type: ignore[arg-type]


def test_a_pass_deep_in_a_counter_is_out_of_reach_from_the_floor() -> None:
    deep = LOCKED.replace("pass_red 10.5 6.7 0", "pass_red 10.5 6.3 0")
    world = new_world(parse_map(deep), "coop")
    p = spawn_player(world, 1, "p")
    p.x, p.y = 10.5, 7.5
    run(world, 30, NORTH, forward=1.0)
    assert "red" not in world.passes  # stopped at the edge, 0.95 from it


def test_the_office_passes_can_be_reached() -> None:
    from ok_doomer.shared.mapfile import load_map
    from ok_doomer.shared.pickups import REACH

    office = load_map("maps/office.map")
    for t in office.things:
        if t.type.startswith("pass_"):
            cx, cy = math.floor(t.x), math.floor(t.y)
            if office.is_low_block(cx, cy):  # within reach of the block's nearest edge
                edge = min(t.x - cx, cx + 1 - t.x, t.y - cy, cy + 1 - t.y)
                assert edge + 0.25 <= REACH, t


def test_a_pass_is_a_card_that_turns_and_bobs_by_the_clock() -> None:
    from ok_doomer.client.boxes import BOX_MODELS, SPINNING
    from ok_doomer.client.render import SpriteInstance, _placed

    (card,) = BOX_MODELS["pass_red"]
    assert card.glow and card.z[0] > 0  # lit by itself, hovering
    assert card.x[1] - card.x[0] < 0.2 * (card.y[1] - card.y[0])  # thin: a card
    lying = SpriteInstance("pass_red", 1.5, 1.5, 0.27, 0.0)
    a, b = _placed(lying, card, 0.0), _placed(lying, card, 0.5)
    rate, _ = SPINNING["pass_red"]
    assert b.angle - a.angle == pytest.approx(0.5 * rate) and b.z != a.z
    desk = SpriteInstance("mac", 1.5, 1.5, 0.27, 1.0)
    assert _placed(desk, BOX_MODELS["mac"][0], 9.0).angle == 1.0  # others stay put


def test_the_hud_shows_passes_held_as_the_same_cards() -> None:
    import pygame

    from ok_doomer.client.hud import CARD_SIZE, Hud

    pygame.font.init()
    window = pygame.Surface((320, 200))
    Hud(pygame.font.Font(None, 16)).draw(window, {"red"}, 0.0)
    corner = pygame.surfarray.array3d(window)[:60, 200 - CARD_SIZE[1] - 12 :]
    assert (corner[..., 0] > 150).sum() > 200  # mostly red, like the card itself

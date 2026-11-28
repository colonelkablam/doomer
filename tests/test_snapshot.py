"""Snapshots: every field classified, and a snapshot applied to a fresh world
of the same map makes it the same world, one that then plays on the same."""

import dataclasses
import math
import random
from collections.abc import Callable

import pytest

from ok_doomer.shared.building import floor_map_path
from ok_doomer.shared.doors import LIFT
from ok_doomer.shared.mapfile import load_map, parse_map
from ok_doomer.shared.player import PlayerInput
from ok_doomer.shared.sim import step
from ok_doomer.shared.snapshot import (
    FIXED,
    PLAYER_FIELDS,
    STATE,
    WORLD_NOT_SENT,
    WORLD_SENT,
    Snapshot,
    SnapshotMismatch,
    apply,
    capture,
    player_from_row,
)
from ok_doomer.shared.spawn import spawn_player
from ok_doomer.shared.switches import POWER
from ok_doomer.shared.world import World, new_world

OFFICE = floor_map_path("G").read_text()


@pytest.mark.parametrize("kind", list(STATE))
def test_every_entity_field_is_sent_or_fixed(kind: type) -> None:
    fields = {f.name for f in dataclasses.fields(kind)}
    sent, fixed = set(STATE[kind]), set(FIXED[kind])
    assert not sent & fixed
    assert sent | fixed == fields, f"classify {sorted(fields - sent - fixed)} in snapshot.py"


def test_every_world_field_is_sent_or_not() -> None:
    fields = {f.name for f in dataclasses.fields(World)}
    assert set(WORLD_SENT) | set(WORLD_NOT_SENT) == fields
    assert not set(WORLD_SENT) & set(WORLD_NOT_SENT)
    assert set(WORLD_SENT) <= {f.name for f in dataclasses.fields(Snapshot)}


def _inputs(rng: random.Random, ids: list[int], t: int) -> dict[int, PlayerInput]:
    """Wandering about: walking, turning, jumping, crouching, pressing use."""
    return {
        i: PlayerInput(
            forward=rng.choice((-1.0, 0.0, 1.0, 1.0)),
            strafe=rng.choice((-1.0, 0.0, 1.0)),
            angle=(t * 0.01 * i + i) % math.tau,
            pitch=rng.uniform(-0.3, 0.3),
            jump=rng.random() < 0.05,
            crouch=rng.random() < 0.1,
            use=rng.random() < 0.05,
        )
        for i in ids
    }


def _played_world() -> World:
    """The ground floor some way into a game, with something non-default in
    every kind of state: power on, a door moving, a cup gone, the cooler
    used, a pass held."""
    world = new_world(parse_map(OFFICE), "coop", seed=3)
    for i in (1, 2, 3):
        spawn_player(world, i, f"p{i}")
    world.flags.add(POWER)
    world.passes.add("blue")
    world.doors[0].open, world.doors[0].opening = 0.4, True
    world.pickups[-1].taken = True
    world.dispensers[0].used[2] = 25
    world.players[3].coffee, world.players[2].health = 30, 55
    # The lift called from three floors down: on its way, counting up.
    lift = next(d for d in world.doors if d.kind == LIFT)
    lift.called, lift.car_floor = True, lift.here - 3.0
    rng = random.Random(9)
    for t in range(120):
        step(world, _inputs(rng, [1, 2, 3], t))
    world.events.clear()
    return world


def _use_at(world: World, x: float, y: float, after: Callable[[], None]) -> None:
    """Player 1 presses use on whatever's at (x, y), from each side in turn,
    calling `after` each tick."""
    p = world.players[1]
    for k in range(8):
        a = k * math.tau / 8
        p.x, p.y = x + 0.6 * math.cos(a), y + 0.6 * math.sin(a)
        facing = (a + math.pi) % math.tau
        step(world, {1: PlayerInput(angle=facing, use=True)})
        after()
        step(world, {1: PlayerInput(angle=facing)})
        after()


def _fields_seen_changing(world: World) -> dict[type, set[str]]:
    """Play on (wandering, the lift arriving, every switch, dispenser and
    door used) and note which entity fields ever change from tick to tick."""
    lists = [getattr(world, n) for n in WORLD_SENT if isinstance(getattr(world, n), list)]

    def values() -> list[tuple[type, str, object]]:
        return [
            (type(e), f.name, repr(getattr(e, f.name)))
            for lst in lists
            for e in lst
            for f in dataclasses.fields(e)
        ]

    seen: dict[type, set[str]] = {kind: set() for kind in STATE}
    before = values()

    def note() -> None:
        nonlocal before
        now = values()
        for (kind, name, old), (_, _, new) in zip(before, now, strict=True):
            if old != new:
                seen[kind].add(name)
        before = now

    lift = next(d for d in world.doors if d.kind == LIFT)
    lift.arrive_in = 1.0  # as if the team had just arrived by it
    rng = random.Random(4)
    for t in range(300):
        step(world, _inputs(rng, [1, 2, 3], t))
        note()
    targets = [(s.x, s.y) for s in world.switches] + [(d.x, d.y) for d in world.dispensers]
    targets += [d.centre() for d in world.cubicle_doors] + [(d.x, d.y) for d in world.doors]
    world.players[1].health = 50  # so the cooler has something to give
    for x, y in targets:
        _use_at(world, x, y, note)
    for pickup in world.pickups:  # walk over each one
        world.players[1].x, world.players[1].y = pickup.x, pickup.y
        step(world, {})
        note()
    for _ in range(120):
        step(world, {})
        note()
    return seen


@pytest.mark.parametrize("kind", list(STATE))
def test_what_changes_in_play_is_what_is_sent(kind: type) -> None:
    """Every field that play changes must be sent (one marked fixed by
    mistake would leave clients stale). And every sent field is seen
    changing, so the run really covers it."""
    seen = _fields_seen_changing(_played_world())[kind]
    assert seen <= set(STATE[kind]), f"send {sorted(seen - set(STATE[kind]))}"
    assert set(STATE[kind]) <= seen, f"never saw {sorted(set(STATE[kind]) - seen)} change"


def _state(world: World) -> dict[str, object]:
    return {name: getattr(world, name) for name in WORLD_SENT}


def test_a_snapshot_makes_a_fresh_world_the_same() -> None:
    world = _played_world()
    fresh = new_world(parse_map(OFFICE), "coop")
    apply(fresh, capture(world))
    assert _state(fresh) == _state(world)


def test_the_lift_arrives_during_the_run() -> None:
    """So the run below covers a lift travelling, arriving and opening."""
    world = _played_world()
    lift = next(d for d in world.doors if d.kind == LIFT)
    assert lift.called and lift.car_floor < lift.here
    for _ in range(300):
        step(world, {})
    assert not lift.called and lift.car_floor == lift.here and lift.open > 0.0


def test_and_it_plays_on_the_same() -> None:
    """The snapshot holds everything `step` depends on: both worlds, given
    the same inputs, stay identical (what client prediction relies on)."""
    world = _played_world()
    fresh = new_world(parse_map(OFFICE), "coop")
    apply(fresh, capture(world))
    rng_a, rng_b = random.Random(5), random.Random(5)
    for t in range(300):
        step(world, _inputs(rng_a, [1, 2, 3], t))
        step(fresh, _inputs(rng_b, [1, 2, 3], t))
        assert capture(fresh) == capture(world), f"diverged at tick {t}"
    assert fresh.events == world.events


def _changed(value: object) -> object:
    """A value the field wouldn't otherwise have."""
    if isinstance(value, bool):
        return not value
    if isinstance(value, int | float):
        return value + 0.25 if isinstance(value, float) else value + 1
    if isinstance(value, dict):
        return {9: 9}
    raise AssertionError(f"teach _changed about {value!r}")


@pytest.mark.parametrize("kind", list(STATE))
def test_every_sent_field_arrives(kind: type) -> None:
    """Change each sent field of the first entity of each kind: the change
    must reach a fresh world. (A played game can leave a field at its
    default by chance; this can't.)"""
    world = _played_world()
    lists = [getattr(world, n) for n in WORLD_SENT if isinstance(getattr(world, n), list)]
    (first,) = [lst[0] for lst in lists if lst and type(lst[0]) is kind]
    for name in STATE[kind]:
        setattr(first, name, _changed(getattr(first, name)))
    fresh = new_world(parse_map(OFFICE), "coop")
    apply(fresh, capture(world))
    assert _state(fresh) == _state(world)


def test_every_player_field_arrives() -> None:
    world = _played_world()
    p = world.players[2]
    for name in PLAYER_FIELDS:
        if name not in ("id", "name"):
            setattr(p, name, _changed(getattr(p, name)))
    p.name = "renamed"
    fresh = new_world(parse_map(OFFICE), "coop")
    apply(fresh, capture(world))
    assert fresh.players[2] == p


def test_players_join_and_leave_and_update_in_place() -> None:
    world = new_world(parse_map(OFFICE), "coop")
    a, _ = spawn_player(world, 1, "ann"), spawn_player(world, 2, "bob")
    client = new_world(parse_map(OFFICE), "coop")
    mine = spawn_player(client, 1, "ann")
    spawn_player(client, 7, "gone")
    a.x += 0.5
    apply(client, capture(world))
    assert sorted(client.players) == [1, 2]  # bob joined, 7 left
    assert client.players[1] is mine and mine.x == a.x  # the same object, moved
    assert client.players[2] == world.players[2]


def test_a_snapshot_of_another_map_is_refused() -> None:
    snap = capture(_played_world())
    other = new_world(load_map(floor_map_path("1")), "coop")
    with pytest.raises(SnapshotMismatch):
        apply(other, snap)


def test_rows_follow_the_field_lists() -> None:
    world = _played_world()
    snap = capture(world)
    assert len(snap.players) == 3 and all(len(r) == len(PLAYER_FIELDS) for r in snap.players)
    assert snap.players[0][0] == 1  # id first
    assert snap.doors[0][:2] == (world.doors[0].open, world.doors[0].opening)
    assert dict(snap.dispensers[0][0]) == {2: 25}  # type: ignore[arg-type]
    assert player_from_row(snap.players[1]) == world.players[2]

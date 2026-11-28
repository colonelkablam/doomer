"""The proximity filter in sim.move_player (only obstacles and doors within
reach of a move are checked) changes nothing: thousands of random moves on
the busiest floor, with doors part open and players up on desks, land
exactly where the unfiltered version puts them."""

import random

import pytest

from ok_doomer.shared import sim
from ok_doomer.shared.building import floor_map_text
from ok_doomer.shared.mapfile import parse_map
from ok_doomer.shared.player import PlayerState
from ok_doomer.shared.spawn import spawn_player
from ok_doomer.shared.world import World, new_world


def _office() -> World:
    world = new_world(parse_map(floor_map_text("G")), "coop", seed=2)
    spawn_player(world, 1, "p")
    rng = random.Random(5)
    for door in world.doors:  # every door somewhere between shut and open
        door.open = rng.choice((0.0, 0.3, 0.6, 1.0))
    for cubicle in world.cubicle_doors:
        cubicle.open = rng.choice((0.0, 0.4, 0.7, 1.0))
    return world


def _moves(world: World, n: int) -> list[tuple[float, float, float, float, float]]:
    """(x, y, z, dx, dy): starts near things that stop you (furniture,
    doors, cubicles), moves up to a few ticks' worth, at floor and desk height."""
    rng = random.Random(11)
    spots = [(ob.x, ob.y) for ob in world.obstacles]
    spots += [(d.x, d.y) for d in world.doors] + [d.centre() for d in world.cubicle_doors]
    out = []
    for _ in range(n):
        x, y = rng.choice(spots)
        x, y = x + rng.uniform(-1.0, 1.0), y + rng.uniform(-1.0, 1.0)
        z = rng.choice((0.0, 0.25))  # on the floor, up on a desk
        out.append((x, y, z, rng.uniform(-0.2, 0.2), rng.uniform(-0.2, 0.2)))
    # And at each cubicle door (shut, or its leaf out in the room): from
    # round about it, heading for its middle.
    for door in world.cubicle_doors:
        cx, cy = door.centre()
        for _ in range(60):
            x, y = cx + rng.uniform(-0.9, 0.9), cy + rng.uniform(-0.9, 0.9)
            out.append((x, y, 0.0, (cx - x) * 0.4, (cy - y) * 0.4))
    return out


def test_the_filter_changes_nothing(monkeypatch: pytest.MonkeyPatch) -> None:
    world = _office()
    moves = _moves(world, 3000)

    def run() -> list[tuple[float, float]]:
        out = []
        for x, y, z, dx, dy in moves:
            p = PlayerState(1, "p", x, y, 0.0, z=z)
            sim.move_player(world, p, dx, dy)
            out.append((p.x, p.y))
        return out

    filtered = run()
    monkeypatch.setattr(sim, "_NEAR_MARGIN", 1000.0)  # "near" = everything: the old way
    assert run() == filtered
    blocked = sum(
        (x, y) != (x0 + dx, y0 + dy)
        for (x, y), (x0, y0, _, dx, dy) in zip(filtered, moves, strict=True)
    )
    assert blocked > 300  # the moves really do run into things

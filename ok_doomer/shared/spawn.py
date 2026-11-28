"""Putting players into a world: spawning them, and moving the team on to the
next floor when the lift arrives.

Single-player and the server both do this the same way, so it lives here
rather than in either of them.
"""

import math

from ok_doomer.shared.events import PlayerSpawned
from ok_doomer.shared.mapfile import Thing, parse_map
from ok_doomer.shared.modes import GAME_MODES
from ok_doomer.shared.player import PlayerState
from ok_doomer.shared.sim import all_obstacles, move_player, standing_height
from ok_doomer.shared.world import World, new_world

# Where to try putting a player when someone's already on the spawn point:
# a body width to the right, left, behind, the back corners, ahead, the front
# corners (as (forward, right) in body widths, turned to the spawn's facing).
_AROUND = ((0, 1), (0, -1), (-1, 0), (-1, 1), (-1, -1), (1, 0), (1, 1), (1, -1))


def spawn_player(world: World, player_id: int, name: str) -> PlayerState:
    """A new player, placed by `place_player`."""
    player = PlayerState(player_id, name, 0.0, 0.0, 0.0)
    place_player(world, player)
    world.players[player_id] = player
    return player


def place_player(world: World, player: PlayerState) -> None:
    """Put a player (new, or back from the dead) at a spawn point for the
    world's mode: the campaign's start; in deathmatch the spawn farthest
    from everyone living (ties: the world's rng). If another player's in
    the way there, beside it (as far as walls and furniture allow); if
    there's no room anywhere near, on it anyway."""
    spawns = world.map.things_of_type(GAME_MODES[world.mode].spawn_thing)
    if world.mode == "coop":
        spot = spawns[0]
    else:
        others = [p for p in world.players.values() if p is not player and not p.dead]

        def room(t: Thing) -> float:
            return min((math.hypot(p.x - t.x, p.y - t.y) for p in others), default=0.0)

        best = max(room(t) for t in spawns)
        spot = world.rng.choice([t for t in spawns if room(t) == best])
    player.x, player.y, player.angle = spot.x, spot.y, spot.angle
    player.z = world.map.floor_height(math.floor(spot.x), math.floor(spot.y))
    player.z_velocity, player.on_ground, player.crouching = 0.0, True, False
    _make_room(world, player)
    world.events.append(PlayerSpawned(world.tick, player.id, player.x, player.y))


def return_player(world: World, player_id: int, before: PlayerState) -> PlayerState:
    """A player who left a moment ago, back where they were (as `before`,
    their state when they left), under a new id: health, coffee and all;
    only motion (a jump, a crouch, a button held) starts afresh. If someone's
    now standing there, beside it."""
    player = PlayerState(
        player_id,
        before.name,
        before.x,
        before.y,
        before.angle,
        z=standing_height(
            world.map,
            all_obstacles(world),
            before.x,
            before.y,
            world.config.movement.radius,
            before.z,
            world.config.movement.step_height,
        ),  # fmt: skip
        pitch=before.pitch,
        health=before.health,
        coffee=before.coffee,
    )
    _make_room(world, player)
    world.players[player_id] = player
    world.events.append(PlayerSpawned(world.tick, player_id, player.x, player.y))
    return player


def _make_room(world: World, player: PlayerState) -> None:
    """Step the player off the spawn point to the first place beside it
    where they don't overlap anyone, moving as they would walk there."""
    if not _crowded(world, player.x, player.y, player):
        return
    move = world.config.movement
    sx, sy, sz = player.x, player.y, player.z
    width = 2 * move.radius
    cos_a, sin_a = math.cos(player.angle), math.sin(player.angle)
    for forward, right in _AROUND:
        player.x, player.y = sx, sy
        # Right of facing (cos, sin) is (-sin, cos): y points down.
        dx = (forward * cos_a - right * sin_a) * width
        dy = (forward * sin_a + right * cos_a) * width
        move_player(world, player, dx, dy)
        if not _crowded(world, player.x, player.y, player):
            player.z = standing_height(
                world.map, all_obstacles(world), player.x, player.y, move.radius, sz,
                move.step_height,
            )  # fmt: skip
            return
    player.x, player.y = sx, sy  # nowhere: share the spot


def _crowded(world: World, x: float, y: float, player: PlayerState | None = None) -> bool:
    """Would a player's box at (x, y) overlap another (living) player's?"""
    width = 2 * world.config.movement.radius - 1e-9
    return any(
        abs(p.x - x) < width and abs(p.y - y) < width
        for p in world.players.values()
        if p is not player and not p.dead
    )


def arrive_on_floor(world: World, map_text: str) -> World:
    """The lift's arrived: a new world for the next floor (its map file's
    text: `building.floor_map_text`, as the server sends it), with
    every player in it (at its start, in the lift car they arrive in), each
    keeping their health and coffee. Passes are a floor's own: the new one
    starts without."""
    seed = world.rng.randrange(2**32)  # deterministic: the next world's from this one
    arrived = new_world(parse_map(map_text), world.mode, world.config, seed)
    for before in sorted(world.players.values(), key=lambda p: p.id):
        player = spawn_player(arrived, before.id, before.name)
        player.health, player.coffee = before.health, before.coffee
    return arrived

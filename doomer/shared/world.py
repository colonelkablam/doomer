"""The World: everything that changes during play, plus the static map."""

import random
from dataclasses import dataclass, field

from doomer.shared.collision import Obstacle
from doomer.shared.config import SimConfig
from doomer.shared.events import Event, PlayerSpawned
from doomer.shared.mapfile import GameMap
from doomer.shared.modes import GAME_MODES, missing_requirements
from doomer.shared.player import PlayerState
from doomer.shared.things import THING_TYPES


@dataclass(slots=True)
class World:
    map: GameMap
    mode: str
    config: SimConfig
    rng: random.Random
    players: dict[int, PlayerState] = field(default_factory=dict)
    tick: int = 0
    # Events emitted since the last drain_events(). Consumers drain each frame/tick.
    events: list[Event] = field(default_factory=list)
    # Footprints of blocking things placed in the map. Static for now; moving
    # entities (enemies) will be collided with separately.
    obstacles: tuple[Obstacle, ...] = ()


class ModeNotSupported(ValueError):
    pass


def new_world(
    game_map: GameMap, mode: str, config: SimConfig | None = None, seed: int = 0
) -> World:
    missing = missing_requirements(game_map, mode)
    if missing:
        raise ModeNotSupported(f"{game_map.name!r} can't be played as {mode}: {', '.join(missing)}")
    obstacles = tuple(
        Obstacle(t.x, t.y, THING_TYPES[t.type].radius)
        for t in game_map.things
        if THING_TYPES[t.type].blocking
    )
    return World(game_map, mode, config or SimConfig(), random.Random(seed), obstacles=obstacles)


def spawn_player(world: World, player_id: int, name: str) -> PlayerState:
    """Place a player at a spawn point for the world's mode."""
    spawns = world.map.things_of_type(GAME_MODES[world.mode].spawn_thing)
    # TODO(milestone 3b): deathmatch should pick the spawn farthest from other players.
    spot = spawns[0] if world.mode == "coop" else world.rng.choice(spawns)
    player = PlayerState(player_id, name, spot.x, spot.y, spot.angle)
    world.players[player_id] = player
    world.events.append(PlayerSpawned(world.tick, player_id, spot.x, spot.y))
    return player


def drain_events(world: World) -> list[Event]:
    events, world.events = world.events, []
    return events

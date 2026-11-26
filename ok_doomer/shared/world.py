"""The World: everything that changes during play, plus the static map."""

import math
import random
from dataclasses import dataclass, field

from ok_doomer.shared.collision import Obstacle
from ok_doomer.shared.config import SimConfig
from ok_doomer.shared.cubicles import CubicleDoor, cubicle_doors_from_map
from ok_doomer.shared.dispensers import Dispenser, dispensers_from_map
from ok_doomer.shared.doors import Door, doors_from_map
from ok_doomer.shared.events import Event, PlayerSpawned
from ok_doomer.shared.furniture import footprint_obstacles
from ok_doomer.shared.mapfile import GameMap
from ok_doomer.shared.modes import GAME_MODES, missing_requirements
from ok_doomer.shared.pickups import Pickup, pickups_from_map
from ok_doomer.shared.player import PlayerState
from ok_doomer.shared.screens import Screen, desk_screens
from ok_doomer.shared.switches import Switch, initial_flags, switches_from_map
from ok_doomer.shared.things import PASS_COLOURS, THING_TYPES


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
    # Cubicle screens on desk tops (static, part of the map).
    screens: tuple[Screen, ...] = ()
    # Doors and how far open each is (ok_doomer/shared/doors.py).
    doors: list[Door] = field(default_factory=list)
    # Pickups (ID passes) and the passes the team holds (ok_doomer/shared/pickups.py).
    pickups: list[Pickup] = field(default_factory=list)
    passes: set[str] = field(default_factory=set)
    # Switches and the level flags they set (ok_doomer/shared/switches.py): power.
    switches: list[Switch] = field(default_factory=list)
    # Dispensers (the water cooler) and what each has given whom
    # (ok_doomer/shared/dispensers.py).
    dispensers: list[Dispenser] = field(default_factory=list)
    # Toilet cubicle doors, open or shut (ok_doomer/shared/cubicles.py).
    cubicle_doors: list[CubicleDoor] = field(default_factory=list)
    flags: set[str] = field(default_factory=set)
    # The lift ride up: seconds into it, or -1 (not riding). It starts when
    # everyone's in a lift car with its doors shut (ok_doomer/shared/sim.py).
    ride: float = -1.0
    complete: bool = False  # everyone left by lift (LevelComplete)


class ModeNotSupported(ValueError):
    pass


def new_world(
    game_map: GameMap, mode: str, config: SimConfig | None = None, seed: int = 0
) -> World:
    missing = missing_requirements(game_map, mode)
    if missing:
        raise ModeNotSupported(f"{game_map.name!r} can't be played as {mode}: {', '.join(missing)}")
    obstacles = tuple(
        ob
        for t in game_map.things
        if THING_TYPES[t.type].blocking
        for ob in (
            footprint_obstacles(t)
            if THING_TYPES[t.type].footprints
            else [Obstacle.square(t.x, t.y, THING_TYPES[t.type].radius)]
        )
    )
    return World(
        game_map,
        mode,
        config or SimConfig(),
        random.Random(seed),
        obstacles=obstacles,
        screens=tuple(desk_screens(game_map)),
        doors=doors_from_map(game_map),
        pickups=pickups_from_map(game_map),
        switches=switches_from_map(game_map),
        dispensers=dispensers_from_map(game_map),
        cubicle_doors=cubicle_doors_from_map(game_map),
        flags=initial_flags(game_map),
        passes=set(PASS_COLOURS) if GAME_MODES[mode].all_passes else set(),
    )


def spawn_player(world: World, player_id: int, name: str) -> PlayerState:
    """Place a player at a spawn point for the world's mode."""
    spawns = world.map.things_of_type(GAME_MODES[world.mode].spawn_thing)
    # TODO(milestone 3b): deathmatch should pick the spawn farthest from other players.
    spot = spawns[0] if world.mode == "coop" else world.rng.choice(spawns)
    floor = world.map.floor_height(math.floor(spot.x), math.floor(spot.y))
    player = PlayerState(player_id, name, spot.x, spot.y, spot.angle, z=floor)
    world.players[player_id] = player
    world.events.append(PlayerSpawned(world.tick, player_id, spot.x, spot.y))
    return player


def drain_events(world: World) -> list[Event]:
    events, world.events = world.events, []
    return events

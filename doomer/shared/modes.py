"""Registry of game modes and what a map needs to support each one.

Modes are derived from a map's things, never declared in the file. The editor
uses `missing_requirements` to show readiness ("needs 2 more DM spawns").
"""

from collections.abc import Callable
from dataclasses import dataclass

from doomer.shared.mapfile import GameMap
from doomer.shared.things import COOP_START, DM_SPAWN, EXIT

MIN_DM_SPAWNS = 4


@dataclass(frozen=True, slots=True)
class GameMode:
    name: str
    spawn_thing: str
    # Returns human-readable unmet requirements; empty means the map supports the mode.
    requirements: Callable[[GameMap], list[str]]


GAME_MODES: dict[str, GameMode] = {}


def register_game_mode(mode: GameMode) -> GameMode:
    if mode.name in GAME_MODES:
        raise ValueError(f"game mode already registered: {mode.name}")
    GAME_MODES[mode.name] = mode
    return mode


def missing_requirements(game_map: GameMap, mode_name: str) -> list[str]:
    return GAME_MODES[mode_name].requirements(game_map)


def supported_modes(game_map: GameMap) -> list[str]:
    return [name for name, mode in GAME_MODES.items() if not mode.requirements(game_map)]


def _coop_requirements(game_map: GameMap) -> list[str]:
    missing = []
    if not game_map.things_of_type(COOP_START.name):
        missing.append("needs a coop_start")
    if not game_map.things_of_type(EXIT.name):
        missing.append("needs an exit")
    return missing


def _dm_requirements(game_map: GameMap) -> list[str]:
    short = MIN_DM_SPAWNS - len(game_map.things_of_type(DM_SPAWN.name))
    if short <= 0:
        return []
    return [f"needs {short} more DM spawn{'s' if short > 1 else ''}"]


COOP = register_game_mode(GameMode("coop", COOP_START.name, _coop_requirements))
DEATHMATCH = register_game_mode(GameMode("deathmatch", DM_SPAWN.name, _dm_requirements))

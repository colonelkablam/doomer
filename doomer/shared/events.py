"""Events emitted by the simulation.

The simulation never calls out to sound, HUD or networking. It appends events
to a list, and those systems consume them. Add new event types here.
"""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Event:
    tick: int


@dataclass(frozen=True, slots=True)
class DoorToggled(Event):
    """A player pressed use on a door: it's now opening (or closing)."""

    door_id: int
    player_id: int
    opening: bool


@dataclass(frozen=True, slots=True)
class PlayerSpawned(Event):
    player_id: int
    x: float
    y: float

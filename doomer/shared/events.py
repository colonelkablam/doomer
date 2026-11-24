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
class DoorLocked(Event):
    """A player tried a locked door without its pass."""

    door_id: int
    player_id: int
    lock: str  # the pass colour it needs


@dataclass(frozen=True, slots=True)
class PassPickedUp(Event):
    """A player picked up an ID pass; the whole team now holds it."""

    player_id: int
    colour: str


@dataclass(frozen=True, slots=True)
class SwitchThrown(Event):
    """A player threw a switch, setting a level flag (the breaker: power)."""

    switch_id: int
    player_id: int
    flag: str


@dataclass(frozen=True, slots=True)
class LiftNoPower(Event):
    """A player pressed a lift's call button with the power off."""

    door_id: int
    player_id: int


@dataclass(frozen=True, slots=True)
class LevelComplete(Event):
    """Everyone's in a lift and its doors have shut: the floor is done."""


@dataclass(frozen=True, slots=True)
class PlayerSpawned(Event):
    player_id: int
    x: float
    y: float

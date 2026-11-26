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
class CubicleDoorUsed(Event):
    """A player used a toilet cubicle's door: it's now `opening` (or
    shutting); or it's `engaged` (locked: someone's in) and didn't move."""

    door_id: int
    player_id: int
    opening: bool
    engaged: bool = False


@dataclass(frozen=True, slots=True)
class DoorLocked(Event):
    """A player tried a locked door without its pass."""

    door_id: int
    player_id: int
    lock: str  # the pass colour it needs


@dataclass(frozen=True, slots=True)
class Dispensed(Event):
    """A player used a dispenser (the water cooler): `amount` added to their
    `meter`; if nothing, why: "full" (their meter), "empty" (nothing's
    `left` for them) or "no power"."""

    player_id: int
    dispenser: str  # its thing type
    meter: str
    amount: int
    left: int  # for this player, after this use
    reason: str = ""


@dataclass(frozen=True, slots=True)
class Refreshed(Event):
    """A player took a pickup that tops up a meter (a coffee cup)."""

    player_id: int
    pickup: str  # its thing type
    meter: str
    amount: int  # what it added (less if the meter filled up)


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
    """Everyone rode a lift up: the floor is done. `next_floor`: the floor
    the lift arrives at (its label), if it has a map; None at the top of
    what's built."""

    next_floor: str | None = None


@dataclass(frozen=True, slots=True)
class LiftCalled(Event):
    """A player called a lift whose car is on another floor: it's coming."""

    door_id: int
    player_id: int


@dataclass(frozen=True, slots=True)
class LiftDeparting(Event):
    """Everyone's aboard with the doors shut: the lift sets off up."""


@dataclass(frozen=True, slots=True)
class PlayerSpawned(Event):
    player_id: int
    x: float
    y: float

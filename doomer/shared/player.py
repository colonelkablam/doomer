from dataclasses import dataclass


@dataclass(slots=True)
class PlayerState:
    id: int
    name: str
    x: float
    y: float
    angle: float  # radians, 0 = east, clockwise on the top-down view
    z: float = 0.0  # unused until floor heights (milestone 4)
    z_velocity: float = 0.0
    health: int = 100


@dataclass(frozen=True, slots=True)
class PlayerInput:
    """One tick's worth of intent from a player.

    The client owns its view angle, so `angle` is absolute rather than a turn
    delta. forward/strafe are in [-1, 1]; positive strafe is to the right.
    """

    forward: float = 0.0
    strafe: float = 0.0
    angle: float = 0.0

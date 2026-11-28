from dataclasses import dataclass


@dataclass(slots=True)
class PlayerState:
    id: int
    name: str
    x: float
    y: float
    angle: float  # radians, 0 = east, clockwise on the top-down view
    z: float = 0.0  # feet height, in walls
    z_velocity: float = 0.0  # walls per second, up is positive
    on_ground: bool = True
    crouching: bool = False
    pitch: float = 0.0  # radians, up is positive (for aiming)
    health: int = 100
    coffee: int = 0  # the armour meter, 0..MAX_COFFEE (GAME.md, Pickups)
    use_held: bool = False  # last tick's use button, so one press is one use
    frags: int = 0  # deathmatch score: kills, less suicides
    respawn_at: int = -1  # dead: the world tick they come back; -1 alive

    @property
    def dead(self) -> bool:
        return self.respawn_at >= 0


@dataclass(frozen=True, slots=True)
class PlayerInput:
    """One tick's worth of intent from a player.

    The client owns its view direction, so `angle` and `pitch` are absolute
    rather than deltas. forward/strafe are in [-1, 1]; positive strafe is to the
    right. `jump`, `crouch` and `use` are held buttons.
    """

    forward: float = 0.0
    strafe: float = 0.0
    angle: float = 0.0
    pitch: float = 0.0
    jump: bool = False
    crouch: bool = False
    use: bool = False  # open/close a door in front of you

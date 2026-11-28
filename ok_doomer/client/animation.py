"""Which pose to draw another player in: standing, a step of the walk,
crouching, in the air.

The walk advances by distance, not time: STRIDE_CELLS travelled is one
full cycle (WALK's poses in turn), so the feet keep pace with the ground
at any speed, a crouch-walk steps slowly, and someone walking into a wall
stops mid-stride. Walking backwards runs the cycle backwards. The game
moves at ~7x a real walk (3.5 cells/s, ~9.5 m/s), so a stride matched to
real legs would spin them: STRIDE_CELLS is set by eye (a cycle a cell:
~14 poses a second at walking pace, near Doom's rhythm).

Positions come from the client's interpolated view of the player (every
frame), so nothing new crosses the network.
"""

import math
from dataclasses import dataclass

from ok_doomer.client.sprites import PLAYER_CROUCH, PLAYER_JUMP, PLAYER_SPRITE, PLAYER_WALK

STRIDE_CELLS = 1.0  # travel for one whole walk cycle (two steps)
STILL_FRAMES = 4  # frames without moving before standing (one quiet frame isn't a stop)
_MOVING = 1e-4  # cells in a frame: less is standing still


@dataclass(slots=True)
class Stride:
    """One player's walk, frame to frame."""

    phase: float = 0.0  # through the cycle, 0..1
    last: tuple[float, float] | None = None
    still: int = STILL_FRAMES
    walking: bool = False

    def pose(self, x: float, y: float, facing: float, crouching: bool, airborne: bool) -> str:
        """The sprite for this frame, the player now at (x, y)."""
        moved = 0.0 if self.last is None else math.dist(self.last, (x, y))
        if moved > _MOVING:
            assert self.last is not None
            dx, dy = x - self.last[0], y - self.last[1]
            ahead = dx * math.cos(facing) + dy * math.sin(facing)
            backwards = ahead < -0.3 * moved  # mostly backwards (strafing walks on)
            self.phase = (self.phase + (-moved if backwards else moved) / STRIDE_CELLS) % 1.0
            self.still = 0
        else:
            self.still += 1
        self.last = (x, y)
        if airborne:
            return PLAYER_JUMP
        if crouching:
            return PLAYER_CROUCH
        if self.still >= STILL_FRAMES:
            return PLAYER_SPRITE
        return PLAYER_WALK[int(self.phase * len(PLAYER_WALK)) % len(PLAYER_WALK)]

"""What a bot does each tick: wander. Walk the way it's facing; when it
stops getting anywhere (a wall, a desk, a shut door), press use (it may be
a door) and turn a random new way. Deterministic for a given seed."""

import math
import random
from dataclasses import dataclass, field

from ok_doomer.shared.player import PlayerInput, PlayerState

STUCK_TICKS = 10  # this many ticks of barely moving: it's stuck
STUCK_DISTANCE = 0.01  # cells a tick that counts as barely moving (walking is ~0.06)


@dataclass(slots=True)
class Wanderer:
    rng: random.Random
    heading: float | None = None
    stuck: int = 0
    last: tuple[float, float] | None = None
    _press: bool = field(default=False)

    def think(self, me: PlayerState) -> PlayerInput:
        if self.heading is None:
            self.heading = me.angle
        if self.last is not None and math.dist(self.last, (me.x, me.y)) < STUCK_DISTANCE:
            self.stuck += 1
        else:
            self.stuck = 0
        self.last = (me.x, me.y)
        press = False
        if self.stuck >= STUCK_TICKS:
            self.heading = self.rng.uniform(0.0, math.tau)
            self.stuck, press = 0, True
        elif self.rng.random() < 0.02:  # now and then, a small change of mind
            self.heading = (self.heading + self.rng.uniform(-0.5, 0.5)) % math.tau
        return PlayerInput(forward=1.0, angle=self.heading, use=press)

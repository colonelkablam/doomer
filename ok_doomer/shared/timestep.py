"""Fixed-timestep accumulator ("Fix Your Timestep", Glenn Fiedler).

Real frame time goes in, a whole number of fixed simulation steps comes out.
The leftover fraction (`alpha`) lets the renderer interpolate between the
previous and current simulation state, so motion is smooth at any frame rate.
"""

from dataclasses import dataclass


@dataclass(slots=True)
class FixedTimestep:
    dt: float
    # Cap on catch-up after a stall (window drag, breakpoint), to avoid a
    # spiral where each frame has to simulate more than the last.
    max_steps: int = 8
    accumulator: float = 0.0

    def advance(self, real_dt: float) -> int:
        self.accumulator += max(0.0, real_dt)
        steps = int(self.accumulator / self.dt)
        if steps > self.max_steps:
            steps = self.max_steps
            self.accumulator = 0.0
        else:
            self.accumulator -= steps * self.dt
        return steps

    @property
    def alpha(self) -> float:
        return self.accumulator / self.dt

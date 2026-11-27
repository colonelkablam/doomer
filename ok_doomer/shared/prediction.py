"""Client-side prediction of your own movement.

Without it, what you see of yourself waits for the server: your input
goes there, waits for a tick, comes back in an Update. With it, each input
moves a private copy of your player at once (`sim.predict_move`: look,
walk, jump, fall; never use or pickups, which change the world and are the
server's to decide), and the server's word is used to correct it:

- Each tick: number the input, keep it as pending, predict its move.
- Each Update: drop the pending inputs the server says it's applied
  (`ack`); start again from where the server has you, and replay the
  rest. That's where you are now, by the server's rules plus what it
  hasn't seen yet.

Usually the two agree to the last digit (the same `step` code, the same
doors). When they don't (a door that opened a moment sooner there, an
input the server merged), the difference goes into a visual `offset` that
fades over CORRECT_SECONDS, so a correction is a glide, not a snap; more
than SNAP_DISTANCE (a teleport, a new floor) snaps.
"""

import dataclasses
import math
from collections import deque
from dataclasses import dataclass, field

from ok_doomer.shared.player import PlayerInput, PlayerState
from ok_doomer.shared.sim import predict_move
from ok_doomer.shared.world import World

CORRECT_SECONDS = 0.1  # a correction fades to 1/e in this long
SNAP_DISTANCE = 0.5  # cells: further out than this, jump to the server's answer
MAX_PENDING = 120  # inputs unconfirmed (2 s): beyond that, the server's not answering

Vec = tuple[float, float, float]


@dataclass(slots=True)
class Predictor:
    pending: deque[tuple[int, PlayerInput]] = field(default_factory=deque)
    player: PlayerState | None = None  # where you are, predicted
    previous: PlayerState | None = None  # a tick before, for drawing between
    offset: Vec = (0.0, 0.0, 0.0)  # drawn displacement, fading (corrections)
    corrections: int = 0  # times the server disagreed (for tests and tuning)

    def reset(self, authoritative: PlayerState | None) -> None:
        """Start again from the server's word (joining, a new floor)."""
        self.pending.clear()
        self.player = dataclasses.replace(authoritative) if authoritative else None
        self.previous = dataclasses.replace(authoritative) if authoritative else None
        self.offset = (0.0, 0.0, 0.0)

    def input(self, world: World, seq: int, player_input: PlayerInput) -> None:
        """A tick's input, sent as `seq`: predict its move now."""
        if self.player is None:
            return
        self.pending.append((seq, player_input))
        while len(self.pending) > MAX_PENDING:
            self.pending.popleft()
        self.previous = dataclasses.replace(self.player)
        predict_move(world, self.player, player_input)

    def reconcile(self, world: World, authoritative: PlayerState, ack: int) -> None:
        """An Update: where the server has you, having applied inputs up to
        `ack`. Replay the rest from there."""
        if self.player is None:
            self.reset(authoritative)
            return
        while self.pending and self.pending[0][0] <= ack:
            self.pending.popleft()
        redone = dataclasses.replace(authoritative)
        for _, player_input in self.pending:
            predict_move(world, redone, player_input)
        error = (
            self.player.x - redone.x,
            self.player.y - redone.y,
            self.player.z - redone.z,
        )
        if any(abs(e) > 1e-9 for e in error):
            self.corrections += 1
            if math.hypot(*error) > SNAP_DISTANCE:
                self.offset = (0.0, 0.0, 0.0)
                self.previous = dataclasses.replace(redone)
            else:
                # Keep drawing where you were, then glide: the error goes into
                # the offset, and the tick before shifts with the correction.
                self.offset = _add(self.offset, error)
                assert self.previous is not None
                self.previous.x -= error[0]
                self.previous.y -= error[1]
                self.previous.z -= error[2]
        self.player = redone

    def fade(self, real_dt: float) -> None:
        """Let a correction's offset fade, a frame at a time."""
        k = math.exp(-real_dt / CORRECT_SECONDS)
        self.offset = (self.offset[0] * k, self.offset[1] * k, self.offset[2] * k)

    def position(self, alpha: float) -> Vec | None:
        """Where to draw you: between the tick before and now (`alpha` of
        the way, as the fixed timestep has it), plus the fading offset."""
        if self.player is None or self.previous is None:
            return None
        a, b = self.previous, self.player
        return _add(
            (a.x + (b.x - a.x) * alpha, a.y + (b.y - a.y) * alpha, a.z + (b.z - a.z) * alpha),
            self.offset,
        )


def _add(a: Vec, b: Vec) -> Vec:
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])

"""Dispensers: fixed things you use (E) to top up a meter, a little each
use, up to an allowance for each player (GAME.md, Pickups).

The water cooler gives health a paper cup of water at a time; the vending
machine a snack at a time, but only with the power on. Each player
has their own allowance at each dispenser, so one player draining it
doesn't empty it for another. What each type gives is data (DISPENSERS).
The state (what's left for whom) is in the World, so the server will own it;
a player who joins later starts with the full allowance.
"""

import math
from dataclasses import dataclass, field

from ok_doomer.shared.mapfile import GameMap

USE_REACH = 1.0  # cells from the dispenser's middle
USE_CONE = math.cos(math.radians(50))  # must be roughly facing it


@dataclass(frozen=True, slots=True)
class DispenserType:
    meter: str  # the PlayerState field it tops up
    per_use: int
    allowance: int  # for each player, for the floor
    name: str  # in messages: "the water cooler"
    needs_power: bool = False  # on the main circuit: works only with the power on


DISPENSERS: dict[str, DispenserType] = {
    "water_cooler": DispenserType("health", 5, 25, "the water cooler"),
    # A snack a use; lit and working only once the power's back.
    "vending_machine": DispenserType("health", 10, 40, "the vending machine", needs_power=True),
}


@dataclass(slots=True)
class Dispenser:
    id: int
    type: str
    x: float
    y: float
    used: dict[int, int] = field(default_factory=dict)  # player id -> given so far

    @property
    def kind(self) -> DispenserType:
        return DISPENSERS[self.type]

    def left_for(self, player_id: int) -> int:
        return self.kind.allowance - self.used.get(player_id, 0)


def dispensers_from_map(game_map: GameMap) -> list[Dispenser]:
    out: list[Dispenser] = []
    for t in game_map.things:
        if t.type in DISPENSERS:
            out.append(Dispenser(len(out), t.type, t.x, t.y))
    return out


def dispenser_in_reach(
    dispensers: list[Dispenser], x: float, y: float, angle: float
) -> Dispenser | None:
    """The dispenser a player at (x, y) facing `angle` is using: the nearest
    within reach that they're roughly facing."""
    best: Dispenser | None = None
    best_d = USE_REACH
    for d in dispensers:
        dx, dy = d.x - x, d.y - y
        dist = math.hypot(dx, dy)
        if dist > best_d or dist < 1e-9:
            continue
        if (dx * math.cos(angle) + dy * math.sin(angle)) / dist < USE_CONE:
            continue
        best, best_d = d, dist
    return best


def dispense(dispenser: Dispenser, player_id: int, current: int, cap: int) -> int:
    """Use it: how much it gives a player whose meter is at `current` (0 if
    their meter's full or their allowance is spent), recorded against them."""
    amount = min(dispenser.kind.per_use, dispenser.left_for(player_id), cap - current)
    amount = max(amount, 0)
    if amount:
        dispenser.used[player_id] = dispenser.used.get(player_id, 0) + amount
    return amount

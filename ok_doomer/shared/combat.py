"""Damage, death and respawning (GAME.md, Pickups and Player characters).

A hit (`hurt`) comes off coffee first, by its share (`CombatConfig.
coffee_soaks`: a third), the rest off health. At no health a player dies:
the killer scores a frag (a suicide loses one), and the dead come back
`respawn_seconds` later (`respawn_due`, each tick, by the server: at the
spawn farthest from the living in deathmatch, the floor's start in the
campaign) with full health and no coffee. The dead don't act: the
simulation ignores their inputs (shared/sim.py).
"""

from ok_doomer.shared.events import Died, Hurt
from ok_doomer.shared.pickups import METER_MAX
from ok_doomer.shared.player import PlayerState
from ok_doomer.shared.spawn import place_player
from ok_doomer.shared.world import World

NOBODY = 0  # `by` for harm that's nobody's doing (no player has id 0)


def hurt(world: World, victim: PlayerState, amount: int, by: int = NOBODY) -> None:
    """`victim` takes a hit of `amount` from player `by`."""
    if victim.dead or amount <= 0:
        return
    soaked = min(victim.coffee, round(amount * world.config.combat.coffee_soaks))
    victim.coffee -= soaked
    victim.health = max(0, victim.health - (amount - soaked))
    world.events.append(Hurt(world.tick, victim.id, amount, soaked, by, victim.health))
    if victim.health == 0:
        _die(world, victim, by)


def _die(world: World, victim: PlayerState, by: int) -> None:
    killer = world.players.get(by)
    if killer is victim:
        victim.frags -= 1  # a suicide
    elif killer is not None:
        killer.frags += 1
    delay = round(world.config.combat.respawn_seconds * world.config.tick_rate)
    victim.respawn_at = world.tick + max(1, delay)
    victim.z_velocity, victim.crouching, victim.use_held = 0.0, False, False
    by_name = killer.name if killer is not None else ""
    world.events.append(Died(world.tick, victim.id, victim.name, by, by_name))


def respawn_due(world: World) -> None:
    """Bring back the dead whose time's up (the server, each tick)."""
    for player in sorted(world.players.values(), key=lambda p: p.id):
        if player.dead and world.tick >= player.respawn_at:
            player.respawn_at = -1
            player.health, player.coffee = METER_MAX["health"], 0
            place_player(world, player)

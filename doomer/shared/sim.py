"""The simulation step. Deterministic: no clocks, sockets, files or global random.

Single-player, the server, and (later) client prediction all call `step`.
"""

import math
from collections.abc import Mapping

from doomer.shared.collision import move_box
from doomer.shared.player import PlayerInput, PlayerState
from doomer.shared.world import World

TAU = math.tau


def step(world: World, inputs: Mapping[int, PlayerInput]) -> None:
    """Advance the world by one fixed tick. Players without input stand still."""
    for player_id, player in world.players.items():
        player_input = inputs.get(player_id)
        if player_input is not None:
            apply_input(world, player, player_input)
    world.tick += 1


def apply_input(world: World, player: PlayerState, player_input: PlayerInput) -> None:
    move = world.config.movement
    player.angle = player_input.angle % TAU

    forward = _clamp(player_input.forward)
    strafe = _clamp(player_input.strafe)
    length = math.hypot(forward, strafe)
    if length > 1.0:  # diagonal movement isn't faster
        forward, strafe = forward / length, strafe / length
    if length == 0.0:
        return

    cos_a, sin_a = math.cos(player.angle), math.sin(player.angle)
    # Facing (cos, sin); right of that is (-sin, cos) because y points down.
    vx = forward * cos_a - strafe * sin_a
    vy = forward * sin_a + strafe * cos_a
    distance = move.walk_speed * world.config.dt
    player.x, player.y = move_box(
        world.map,
        player.x,
        player.y,
        vx * distance,
        vy * distance,
        move.radius,
        world.obstacles,
    )


def _clamp(v: float) -> float:
    return max(-1.0, min(1.0, v))

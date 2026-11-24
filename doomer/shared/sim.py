"""The simulation step. Deterministic: no clocks, sockets, files or global random.

Single-player, the server, and (later) client prediction all call `step`.

Vertical movement: the player has a feet height `z`. Each tick, horizontal
movement goes first, with a height-aware collision test (a cell blocks you if
its floor is above your feet plus the step height, or its ceiling leaves no
room for your body). Then gravity: you stand on the highest floor under your
collision box, climb small rises automatically (stairs), fall off edges,
and bump your head on ceilings.
"""

import math
from collections.abc import Mapping

from doomer.shared.collision import move_box
from doomer.shared.config import MovementConfig
from doomer.shared.mapfile import GameMap
from doomer.shared.player import PlayerInput, PlayerState
from doomer.shared.world import World

TAU = math.tau
_EPS = 1e-9


def step(world: World, inputs: Mapping[int, PlayerInput]) -> None:
    """Advance the world by one fixed tick. Players without input stand still
    (but still fall)."""
    for player_id, player in world.players.items():
        apply_input(world, player, inputs.get(player_id, PlayerInput(angle=player.angle)))
    world.tick += 1


def apply_input(world: World, player: PlayerState, player_input: PlayerInput) -> None:
    move = world.config.movement
    dt = world.config.dt
    player.angle = player_input.angle % TAU
    player.pitch = player_input.pitch
    _update_crouch(world.map, player, player_input.crouch, move)

    forward = _clamp(player_input.forward)
    strafe = _clamp(player_input.strafe)
    length = math.hypot(forward, strafe)
    if length > 1.0:  # diagonal movement isn't faster
        forward, strafe = forward / length, strafe / length
    if length > 0.0:
        cos_a, sin_a = math.cos(player.angle), math.sin(player.angle)
        # Facing (cos, sin); right of that is (-sin, cos) because y points down.
        vx = forward * cos_a - strafe * sin_a
        vy = forward * sin_a + strafe * cos_a
        speed = move.walk_speed * (move.crouch_speed_factor if player.crouching else 1.0)
        distance = speed * dt
        body = body_height(player, move)
        player.x, player.y = move_box(
            world.map,
            player.x,
            player.y,
            vx * distance,
            vy * distance,
            move.radius,
            world.obstacles,
            lambda cx, cy: cell_blocks(world.map, cx, cy, player.z, body, move.step_height),
        )

    if player_input.jump and player.on_ground:
        player.z_velocity = move.jump_speed
        player.on_ground = False
    _apply_gravity(world.map, player, move, dt)


def body_height(player: PlayerState, move: MovementConfig) -> float:
    return move.crouch_height if player.crouching else move.height


def eye_height(player: PlayerState, move: MovementConfig) -> float:
    """Eye height above the feet, for the camera (and later, for aiming)."""
    return move.crouch_eye_height if player.crouching else move.eye_height


def cell_blocks(game_map: GameMap, cx: int, cy: int, feet: float, body: float, step: float) -> bool:
    """Can't move into a cell whose floor is too high to step up onto, or whose
    ceiling is too low for your body once you're on its floor."""
    floor = game_map.floor_height(cx, cy)  # inf for full walls
    if floor > feet + step + _EPS:
        return True
    return game_map.ceiling_height(cx, cy) - max(floor, feet) < body - _EPS


def support_height(game_map: GameMap, x: float, y: float, radius: float) -> float:
    """The highest floor under a collision box: what you're standing on."""
    return max(game_map.floor_height(cx, cy) for cx, cy in _cells_under(x, y, radius))


def headroom(game_map: GameMap, x: float, y: float, radius: float) -> float:
    """The lowest ceiling over a collision box."""
    return min(game_map.ceiling_height(cx, cy) for cx, cy in _cells_under(x, y, radius))


def _cells_under(x: float, y: float, r: float) -> list[tuple[int, int]]:
    xs = range(math.floor(x - r + _EPS), math.ceil(x + r - _EPS))
    ys = range(math.floor(y - r + _EPS), math.ceil(y + r - _EPS))
    return [(cx, cy) for cx in xs for cy in ys]


def _update_crouch(
    game_map: GameMap, player: PlayerState, want: bool, move: MovementConfig
) -> None:
    if want:
        player.crouching = True
    elif player.crouching:
        # Only stand up if there's room above you.
        room = headroom(game_map, player.x, player.y, move.radius) - player.z
        player.crouching = room < move.height - _EPS


def _apply_gravity(game_map: GameMap, player: PlayerState, move: MovementConfig, dt: float) -> None:
    support = support_height(game_map, player.x, player.y, move.radius)
    if player.on_ground and player.z < support + _EPS:
        # Climbing a step (or still standing): snap to the floor.
        player.z = support
        player.z_velocity = 0.0
        return
    player.z_velocity -= move.gravity * dt
    player.z += player.z_velocity * dt
    if player.z <= support:  # landed
        player.z, player.z_velocity, player.on_ground = support, 0.0, True
    else:
        player.on_ground = False
    ceiling = headroom(game_map, player.x, player.y, move.radius) - body_height(player, move)
    if player.z > ceiling:  # bumped your head
        player.z, player.z_velocity = max(support, ceiling), min(player.z_velocity, 0.0)


def _clamp(v: float) -> float:
    return max(-1.0, min(1.0, v))

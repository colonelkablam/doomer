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

from ok_doomer.shared.building import next_floor
from ok_doomer.shared.collision import (
    Obstacle,
    ThinWall,
    move_box,
    obstacle_support,
    obstacles_in_the_way,
)
from ok_doomer.shared.config import MovementConfig
from ok_doomer.shared.cubicles import (
    cubicle_door_in_reach,
    cubicle_door_leaves,
    cubicle_door_walls,
    update_cubicle_doors,
    use_cubicle_door,
)
from ok_doomer.shared.dispensers import dispense, dispenser_in_reach
from ok_doomer.shared.doors import LIFT, door_walls, try_use, update_doors
from ok_doomer.shared.events import (
    CubicleDoorUsed,
    Dispensed,
    DoorLocked,
    DoorToggled,
    LevelComplete,
    LiftCalled,
    LiftDeparting,
    LiftNoPower,
    PassPickedUp,
    Refreshed,
    SwitchThrown,
)
from ok_doomer.shared.mapfile import RAIL_HEIGHT, RAILING_WALL_TYPE, GameMap
from ok_doomer.shared.pickups import METER_MAX, Pickup, take_pickups
from ok_doomer.shared.player import PlayerInput, PlayerState
from ok_doomer.shared.screens import SCREEN_HALF, screens_at_height
from ok_doomer.shared.switches import POWER, try_switch
from ok_doomer.shared.world import World

TAU = math.tau
RIDE_SECONDS = 2.5  # the lift ride up a floor, doors shut
_EPS = 1e-9


def step(world: World, inputs: Mapping[int, PlayerInput]) -> None:
    """Advance the world by one fixed tick. Players without input stand still
    (but still fall)."""
    for player_id, player in world.players.items():
        apply_input(world, player, inputs.get(player_id, PlayerInput(angle=player.angle)))
    positions = [(p.x, p.y) for p in world.players.values()]
    power = POWER in world.flags
    if world.ride < 0.0 and power and positions and _ready_to_ride(world, positions):
        world.ride = 0.0
        world.events.append(LiftDeparting(world.tick))
    update_doors(
        world.doors,
        positions,
        world.config.movement.radius,
        world.config.dt,
        world.passes,
        power,
        departing=world.ride >= 0.0,
    )
    update_cubicle_doors(
        world.cubicle_doors, world.config.dt, positions, world.config.movement.radius
    )
    if world.ride >= 0.0 and not world.complete:
        _ride(world, positions)
    world.tick += 1


def _ride(world: World, positions: list[tuple[float, float]]) -> None:
    """The ride up: the occupied cars climb a floor over RIDE_SECONDS (their
    indicators count), then the floor's done."""
    world.ride += world.config.dt
    occupied = {(math.floor(x), math.floor(y)) for x, y in positions}
    for door in world.doors:
        if door.kind == LIFT and door.car in occupied:
            door.car_floor = door.here + min(1.0, world.ride / RIDE_SECONDS)
    if world.ride >= RIDE_SECONDS:
        world.complete = True
        world.events.append(LevelComplete(world.tick, next_floor(world.map.storey)))


def apply_input(world: World, player: PlayerState, player_input: PlayerInput) -> None:
    move = world.config.movement
    dt = world.config.dt
    player.angle = player_input.angle % TAU
    player.pitch = player_input.pitch
    _update_crouch(world.map, player, player_input.crouch, move)
    if player_input.use and not player.use_held:  # a press, not a hold
        thrown = try_switch(world.switches, player.x, player.y, player.angle)
        power = POWER in world.flags
        tap = (
            None
            if thrown
            else dispenser_in_reach(world.dispensers, player.x, player.y, player.angle)
        )
        if tap is not None:
            meter = tap.kind.meter
            before = _meter(player, meter)
            dead = tap.kind.needs_power and not power
            amount = 0 if dead else dispense(tap, player.id, before, METER_MAX[meter])
            setattr(player, meter, before + amount)
            left = tap.left_for(player.id)
            reason = "no power" if dead else "" if amount else "empty" if left == 0 else "full"
            world.events.append(
                Dispensed(world.tick, player.id, tap.type, meter, amount, left, reason)
            )
        cubicle = (
            None
            if thrown or tap
            else cubicle_door_in_reach(world.cubicle_doors, player.x, player.y, player.angle)
        )
        if cubicle is not None:
            moved = use_cubicle_door(cubicle)
            world.events.append(
                CubicleDoorUsed(world.tick, cubicle.id, player.id, cubicle.opening, not moved)
            )
        used = (
            None
            if thrown or tap or cubicle
            else try_use(world.doors, player.x, player.y, player.angle, world.passes, power)
        )
        if thrown is not None:
            world.flags.add(thrown.flag)
            world.events.append(SwitchThrown(world.tick, thrown.id, player.id, thrown.flag))
        if used is not None:
            door, allowed = used
            if allowed and door.kind == LIFT and door.called:
                world.events.append(LiftCalled(world.tick, door.id, player.id))
            elif allowed:
                world.events.append(DoorToggled(world.tick, door.id, player.id, door.opening))
            elif door.lock is not None and door.lock not in world.passes:
                world.events.append(DoorLocked(world.tick, door.id, player.id, door.lock))
            elif door.kind == LIFT:
                world.events.append(LiftNoPower(world.tick, door.id, player.id))
    player.use_held = player_input.use

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
        railed = on_railed_floor(world.map, player, move.radius)
        # Cubicle screens block you only while you're up on the desks.
        walls = [
            ThinWall(sc.axis, sc.at, sc.lo, sc.hi, SCREEN_HALF)
            for sc in screens_at_height(world.screens, player.z)
        ]
        walls += [wall for door in world.doors for wall in door_walls(door)]
        walls += cubicle_door_walls(world.cubicle_doors)
        player.x, player.y = move_box(
            world.map,
            player.x,
            player.y,
            vx * distance,
            vy * distance,
            move.radius,
            obstacles_in_the_way(all_obstacles(world), player.z, move.step_height),
            lambda cx, cy: (
                cell_blocks(world.map, cx, cy, player.z, body, move.step_height, player.x, player.y)
                or (railed and _drops_away(world.map, cx, cy, player, move.step_height))
            ),
            walls,
        )

    if player_input.jump and player.on_ground:
        player.z_velocity = move.jump_speed
        player.on_ground = False
    _apply_gravity(world.map, player, move, dt, all_obstacles(world))

    def wants(pickup: Pickup) -> bool:
        return all(_meter(player, m) < METER_MAX[m] for m in pickup.effects)

    for pickup in take_pickups(world.pickups, player.x, player.y, player.z, wants):
        colour = pickup.pass_colour
        if colour is not None and colour not in world.passes:
            world.passes.add(colour)
            world.events.append(PassPickedUp(world.tick, player.id, colour))
        for meter, amount in pickup.effects.items():
            before = _meter(player, meter)
            after = min(METER_MAX[meter], before + amount)
            setattr(player, meter, after)
            world.events.append(
                Refreshed(world.tick, player.id, pickup.type, meter, after - before)
            )


def _meter(player: PlayerState, meter: str) -> int:
    value: int = getattr(player, meter)
    return value


def _ready_to_ride(world: World, positions: list[tuple[float, float]]) -> bool:
    """Is every player in a lift car that's here, with its doors shut (and
    not just arrived), that the team may send (its pass, if it's locked)?"""
    ready = {
        door.car
        for door in world.doors
        if door.kind == LIFT
        and door.car_here
        and door.open == 0.0
        and not door.opening
        and door.arrive_in == 0.0
        and door.unlocked_by(world.passes)
    }
    return all((math.floor(x), math.floor(y)) in ready for x, y in positions)


def body_height(player: PlayerState, move: MovementConfig) -> float:
    return move.crouch_height if player.crouching else move.height


def eye_height(player: PlayerState, move: MovementConfig) -> float:
    """Eye height above the feet, for the camera (and later, for aiming)."""
    return move.crouch_eye_height if player.crouching else move.eye_height


def cell_blocks(
    game_map: GameMap,
    cx: int,
    cy: int,
    feet: float,
    body: float,
    step: float,
    x: float | None = None,
    y: float | None = None,
) -> bool:
    """Can't move into a cell whose floor is too high to step up onto, or whose
    ceiling is too low for your body once you're on its floor. The floor is
    measured at the point of the cell nearest (x, y), which matters on stairs:
    you step onto the low end, and can't climb on from the side halfway up."""
    if x is None or y is None:
        floor = game_map.floor_height(cx, cy)  # inf for full walls
    else:
        floor = game_map.floor_at(_clamp_to(x, cx), _clamp_to(y, cy))
    if floor > feet + step + _EPS:
        return True
    return game_map.ceiling_height(cx, cy) - max(floor, feet) < body - _EPS


def on_railed_floor(game_map: GameMap, player: PlayerState, radius: float) -> bool:
    """Standing on a balcony (below the top of its railing)? Then its railings
    stop you walking off any edge that drops away."""
    for cx, cy in _cells_under(player.x, player.y, radius):
        if game_map.wall_at(cx, cy) == RAILING_WALL_TYPE and game_map.is_low_block(cx, cy):
            top = game_map.floor_height(cx, cy)
            if top - _EPS <= player.z < top + RAIL_HEIGHT:
                return True
    return False


def _drops_away(game_map: GameMap, cx: int, cy: int, player: PlayerState, step: float) -> bool:
    """Is the floor of cell (cx, cy), where you'd step into it, more than a step
    below your feet? (Stairs arriving at a balcony are level with it: fine.)"""
    floor = game_map.floor_at(_clamp_to(player.x, cx), _clamp_to(player.y, cy))
    return floor < player.z - step - _EPS


def support_height(game_map: GameMap, x: float, y: float, radius: float) -> float:
    """The highest floor under a collision box: what you're standing on. Sampled
    at the box's corners and centre (clamped into each cell under it), so a
    stair's slope is followed smoothly."""
    best = -math.inf
    points = [(x + sx * radius, y + sy * radius) for sx in (-1, 1) for sy in (-1, 1)]
    points.append((x, y))
    for cx, cy in _cells_under(x, y, radius):
        for px, py in points:
            best = max(best, game_map.floor_at(_clamp_to(px, cx), _clamp_to(py, cy)))
    return best


def _clamp_to(v: float, cell: int) -> float:
    """Clamp a coordinate into a cell (just inside its edges)."""
    return min(max(v, cell + _EPS), cell + 1 - _EPS)


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


def all_obstacles(world: World) -> list[Obstacle]:
    """The furniture, and the leaves of open cubicle doors."""
    return [*world.obstacles, *cubicle_door_leaves(world.cubicle_doors)]


def standing_height(
    game_map: GameMap,
    obstacles: list[Obstacle] | tuple[Obstacle, ...],
    x: float,
    y: float,
    radius: float,
    feet: float,
    step: float,
) -> float:
    """What a box at (x, y), feet at `feet`, stands on: the floor, or the top
    of furniture under it that it's up on (or can step onto)."""
    floor = support_height(game_map, x, y, radius)
    return max(floor, obstacle_support(obstacles, x, y, radius, feet, step))


def _apply_gravity(
    game_map: GameMap,
    player: PlayerState,
    move: MovementConfig,
    dt: float,
    obstacles: list[Obstacle] | tuple[Obstacle, ...] = (),
) -> None:
    support = standing_height(
        game_map, obstacles, player.x, player.y, move.radius, player.z, move.step_height
    )
    if (
        player.on_ground
        and player.z_velocity <= 0.0
        and abs(player.z - support) <= move.step_height + _EPS
    ):
        # Standing, climbing a step, or walking down stairs (a small drop is
        # followed rather than fallen, as in Doom): stay on the floor.
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

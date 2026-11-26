"""A game session: where the world lives, from the client's point of view.

LocalSession runs the simulation in-process (single-player);
NetworkSession (client/network.py) has the same interface, sending inputs
to a server and drawing its snapshots, so the game loop doesn't change.
"""

from typing import Protocol

from ok_doomer.client.boxes import PlacedBox, cubicle_door_boxes, door_boxes, switch_boxes
from ok_doomer.client.render import Camera, SpriteInstance, sprites_from_things
from ok_doomer.shared.building import floor_map_text
from ok_doomer.shared.events import Event, LevelComplete
from ok_doomer.shared.mapfile import GameMap
from ok_doomer.shared.player import PlayerInput, PlayerState
from ok_doomer.shared.sim import eye_height, step
from ok_doomer.shared.spawn import arrive_on_floor, spawn_player
from ok_doomer.shared.switches import POWER
from ok_doomer.shared.timestep import FixedTimestep
from ok_doomer.shared.world import World, drain_events

EYE_EASE_SPEED = 2.5  # walls per second: standing <-> crouching in ~0.1 s


class GameSession(Protocol):
    @property
    def game_map(self) -> GameMap: ...

    def update(self, real_dt: float, player_input: PlayerInput) -> list[Event]:
        """Advance by real elapsed time; return events for sound/HUD."""
        ...

    def camera(self, view_angle: float, pitch: float = 0.0) -> Camera:
        """Where to draw from this frame. The client owns its view direction."""
        ...

    def sprites(self) -> list[SpriteInstance]:
        """Things to draw this frame (the renderer skips names with no art)."""
        ...

    def boxes(self) -> list[PlacedBox]:
        """Moving geometry to draw this frame: doors, as they are now."""
        ...

    def passes(self) -> frozenset[str]:
        """The ID passes the team holds, for the HUD."""
        ...

    def power(self) -> bool:
        """Is the floor's power on (lights, lifts)?"""
        ...

    def spawn_angle(self) -> float:
        """The way the player faced when they arrived on this floor. The
        session moves on to the next floor itself, when the lift gets
        there: `game_map` is then a different map."""
        ...

    def ended(self) -> str | None:
        """Why the game's over (the server said goodbye, or went away), or
        None while it goes on."""
        ...

    def close(self) -> None:
        """Done with it (leave the server)."""
        ...


def static_sprites(world: World) -> list[SpriteInstance]:
    """The map's things that never move. Pickups aren't among them: they
    come from the world (they get taken)."""
    pickup_types = {p.type for p in world.pickups}
    return [s for s in sprites_from_things(world.map) if s.name not in pickup_types]


def lying_pickups(world: World) -> list[SpriteInstance]:
    return [SpriteInstance(p.type, p.x, p.y, p.z, p.angle) for p in world.pickups if not p.taken]


def world_boxes(world: World) -> list[PlacedBox]:
    """Moving geometry as it is now: doors, switches, cubicle doors."""
    doors = door_boxes(world.doors, world.passes)
    return doors + switch_boxes(world.switches) + cubicle_door_boxes(world.cubicle_doors)


def ease_eye(eye: float, player: PlayerState, world: World, real_dt: float) -> float:
    """Eye height eases towards standing/crouching height rather than
    snapping, so crouching dips the view over ~0.1 s."""
    target = eye_height(player, world.config.movement)
    change = EYE_EASE_SPEED * real_dt
    return eye + max(-change, min(change, target - eye))


class LocalSession:
    def __init__(self, world: World, player_name: str, player_id: int = 1) -> None:
        self._start(world, spawn_player(world, player_id, player_name))

    def _start(self, world: World, player: PlayerState) -> None:
        self.world, self.player = world, player
        self._spawn_angle = player.angle
        self.timestep = FixedTimestep(world.config.dt)
        self._prev = (self.player.x, self.player.y, self.player.z)
        self._eye = eye_height(self.player, world.config.movement)
        self._static_sprites = static_sprites(world)

    @property
    def game_map(self) -> GameMap:
        return self.world.map

    def update(self, real_dt: float, player_input: PlayerInput) -> list[Event]:
        events: list[Event] = []
        for _ in range(self.timestep.advance(real_dt)):
            self._prev = (self.player.x, self.player.y, self.player.z)
            step(self.world, {self.player.id: player_input})
            events += drain_events(self.world)
            arrived = next(
                (e.next_floor for e in events if isinstance(e, LevelComplete) and e.next_floor),
                None,
            )
            if arrived is not None:  # the lift's there: on to the next floor
                world = arrive_on_floor(self.world, floor_map_text(arrived))
                self._start(world, world.players[self.player.id])
                events += drain_events(world)
                break  # the rest of this frame's time goes to the new floor next frame
        self._eye = ease_eye(self._eye, self.player, self.world, real_dt)
        return events

    def camera(self, view_angle: float, pitch: float = 0.0) -> Camera:
        # Interpolate between the last two simulation states: rendering at 144 fps
        # with a 60 Hz simulation would otherwise judder.
        a = self.timestep.alpha
        px, py, pz = self._prev
        x = px + (self.player.x - px) * a
        y = py + (self.player.y - py) * a
        z = pz + (self.player.z - pz) * a
        return Camera(x, y, view_angle, z=z + self._eye, pitch=pitch)

    def sprites(self) -> list[SpriteInstance]:
        return self._static_sprites + lying_pickups(self.world)

    def boxes(self) -> list[PlacedBox]:
        return world_boxes(self.world)

    def passes(self) -> frozenset[str]:
        return frozenset(self.world.passes)

    def power(self) -> bool:
        return POWER in self.world.flags

    def spawn_angle(self) -> float:
        return self._spawn_angle

    def ended(self) -> str | None:
        return None

    def close(self) -> None:
        pass

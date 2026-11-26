"""A game session: where the world lives, from the client's point of view.

LocalSession runs the simulation in-process (single-player). Milestone 3 adds
a NetworkSession with the same interface that sends inputs to the server and
interpolates snapshots, so the game loop doesn't change.
"""

from typing import Protocol

from doomer.client.boxes import PlacedBox, cubicle_door_boxes, door_boxes, switch_boxes
from doomer.client.render import Camera, SpriteInstance, sprites_from_things
from doomer.shared.building import floor_map_path
from doomer.shared.events import Event
from doomer.shared.mapfile import GameMap, load_map
from doomer.shared.player import PlayerInput, PlayerState
from doomer.shared.sim import eye_height, step
from doomer.shared.switches import POWER
from doomer.shared.timestep import FixedTimestep
from doomer.shared.world import World, drain_events, new_world, spawn_player

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

    def change_floor(self, storey: str) -> None:
        """The lift's arrived: on to that floor of the building, the team
        in its lift car."""
        ...

    def spawn_angle(self) -> float:
        """The way the player faced when they arrived on this floor."""
        ...


class LocalSession:
    def __init__(self, world: World, player_name: str, player_id: int = 1) -> None:
        self._start(world, spawn_player(world, player_id, player_name))

    def _start(self, world: World, player: PlayerState) -> None:
        self.world, self.player = world, player
        self._spawn_angle = player.angle
        self.timestep = FixedTimestep(world.config.dt)
        self._prev = (self.player.x, self.player.y, self.player.z)
        # Eye height eases towards standing/crouching height rather than
        # snapping, so crouching dips the view over ~0.1 s.
        self._eye = eye_height(self.player, world.config.movement)
        # Map things never move; pickups come from the world (they get taken).
        pickup_types = {p.type for p in world.pickups}
        self._static_sprites = [
            s for s in sprites_from_things(world.map) if s.name not in pickup_types
        ]

    @property
    def game_map(self) -> GameMap:
        return self.world.map

    def update(self, real_dt: float, player_input: PlayerInput) -> list[Event]:
        for _ in range(self.timestep.advance(real_dt)):
            self._prev = (self.player.x, self.player.y, self.player.z)
            step(self.world, {self.player.id: player_input})
        target = eye_height(self.player, self.world.config.movement)
        change = EYE_EASE_SPEED * real_dt
        self._eye += max(-change, min(change, target - self._eye))
        return drain_events(self.world)

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
        lying = [
            SpriteInstance(p.type, p.x, p.y, p.z, p.angle)
            for p in self.world.pickups
            if not p.taken
        ]
        return self._static_sprites + lying

    def boxes(self) -> list[PlacedBox]:
        doors = door_boxes(self.world.doors, self.world.passes)
        cubicles = cubicle_door_boxes(self.world.cubicle_doors)
        return doors + switch_boxes(self.world.switches) + cubicles

    def passes(self) -> frozenset[str]:
        return frozenset(self.world.passes)

    def power(self) -> bool:
        return POWER in self.world.flags

    def change_floor(self, storey: str) -> None:
        """Load that floor (doomer/shared/building.py) and put the player in
        it, at its start (in the lift car they arrive in), keeping their
        health and coffee. Passes are a floor's own: the new one starts
        without."""
        before = self.player
        world = new_world(load_map(floor_map_path(storey)), self.world.mode, self.world.config)
        player = spawn_player(world, before.id, before.name)
        player.health, player.coffee = before.health, before.coffee
        self._start(world, player)

    def spawn_angle(self) -> float:
        return self._spawn_angle

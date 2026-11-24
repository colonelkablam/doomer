"""A game session: where the world lives, from the client's point of view.

LocalSession runs the simulation in-process (single-player). Milestone 3 adds
a NetworkSession with the same interface that sends inputs to the server and
interpolates snapshots, so the game loop doesn't change.
"""

from typing import Protocol

from doomer.client.boxes import PlacedBox, door_boxes
from doomer.client.render import Camera, SpriteInstance, sprites_from_things
from doomer.shared.events import Event
from doomer.shared.mapfile import GameMap
from doomer.shared.player import PlayerInput
from doomer.shared.sim import eye_height, step
from doomer.shared.timestep import FixedTimestep
from doomer.shared.world import World, drain_events, spawn_player

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


class LocalSession:
    def __init__(self, world: World, player_name: str, player_id: int = 1) -> None:
        self.world = world
        self.player = spawn_player(world, player_id, player_name)
        self.timestep = FixedTimestep(world.config.dt)
        self._prev = (self.player.x, self.player.y, self.player.z)
        # Eye height eases towards standing/crouching height rather than
        # snapping, so crouching dips the view over ~0.1 s.
        self._eye = eye_height(self.player, world.config.movement)
        # Map things never move; later, enemies and pickups come from the world.
        self._static_sprites = sprites_from_things(world.map)

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
        return self._static_sprites

    def boxes(self) -> list[PlacedBox]:
        return door_boxes(self.world.doors)

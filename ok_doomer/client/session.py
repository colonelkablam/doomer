"""A game session: where the world lives, from the client's point of view.

NetworkSession (client/network.py) sends inputs to a server and draws its
snapshots; LocalSession (client/local.py), single-player, is the same on
a server in the game's own process. The game loop sees only this
interface.
"""

from typing import TYPE_CHECKING, Protocol

from ok_doomer.client.boxes import PlacedBox, cubicle_door_boxes, door_boxes, switch_boxes
from ok_doomer.client.render import Camera, SpriteInstance, sprites_from_things
from ok_doomer.shared.events import Event
from ok_doomer.shared.mapfile import GameMap
from ok_doomer.shared.player import PlayerInput, PlayerState
from ok_doomer.shared.sim import eye_height
from ok_doomer.shared.world import World

if TYPE_CHECKING:
    from ok_doomer.client.network import NetStats

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

    def net_stats(self) -> "NetStats":
        """How the connection's doing (F3)."""
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

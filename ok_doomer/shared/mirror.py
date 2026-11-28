"""A client's copy of the server's world, kept in step by its messages.

Built from the Welcome (the map text, mode and config: the same World the
server built), then each Update's snapshot is applied onto it and each
NewFloor rebuilds it. It also numbers the inputs the client sends and
remembers the last the server says it applied (`ack`), which is what
client-side prediction would replay from.

Standard library only: the bot uses it, and so will the game client.
"""

from dataclasses import dataclass

from ok_doomer.shared.config import SimConfig
from ok_doomer.shared.events import Event
from ok_doomer.shared.mapfile import parse_map
from ok_doomer.shared.player import PlayerInput, PlayerState
from ok_doomer.shared.protocol import Bye, Input, Message, NewFloor, ProtocolError, Update, Welcome
from ok_doomer.shared.snapshot import apply
from ok_doomer.shared.world import World, new_world


@dataclass(slots=True)
class WorldMirror:
    player_id: int
    mode: str
    config: SimConfig
    world: World
    seq: int = 0  # the last input numbered
    ack: int = 0  # the last input the server applied
    bye: Bye | None = None  # the server said goodbye
    updates: int = 0  # Updates applied (on any floor)
    floors: int = 1  # worlds built: 1 + floor changes

    @classmethod
    def from_welcome(cls, welcome: Welcome) -> "WorldMirror":
        world = new_world(parse_map(welcome.map_text), welcome.mode, welcome.config)
        return cls(welcome.player_id, welcome.mode, welcome.config, world)

    @property
    def player(self) -> PlayerState | None:
        """You, once an Update has put you in the world."""
        return self.world.players.get(self.player_id)

    def next_input(self, player_input: PlayerInput) -> Input:
        """Number an input to send."""
        self.seq += 1
        return Input(self.seq, player_input)

    def handle(self, message: Message) -> list[Event]:
        """Take a message from the server; the events it brings (for the HUD
        and sound)."""
        if isinstance(message, Update):
            apply(self.world, message.snapshot)
            self.ack = message.ack
            self.updates += 1
            return list(message.events)
        if isinstance(message, NewFloor):
            self.world = new_world(parse_map(message.map_text), self.mode, self.config)
            self.floors += 1
            return []
        if isinstance(message, Bye):
            self.bye = message
            return []
        raise ProtocolError(f"a server doesn't send {message.TYPE} after welcoming you")

"""Single-player: the server in the game's own process.

The same GameServer the Pi runs, joined by a LocalLink instead of a socket:
messages go across as Python objects (no JSON), and the server ticks in
lockstep with the game, one tick for each input the client sends (it sends
one a tick, by its fixed timestep). So single-player has no latency, is
deterministic, and runs every line multiplayer does: spawning, the lift
and the next floor, snapshots, prediction. A bug in those shows up here
too. (Hosting a game from the client would be the same plus a listener.)
"""

import queue

from ok_doomer.client.network import NetworkSession
from ok_doomer.server.config import ServerConfig
from ok_doomer.server.game import GameServer
from ok_doomer.shared.config import SimConfig
from ok_doomer.shared.player import PlayerState
from ok_doomer.shared.protocol import PROTOCOL_VERSION, Bye, Hello, Input, Message, Reject
from ok_doomer.shared.world import World

# Ticks run in one frame at most, catching up after a stall (as the game
# always did single-player: beyond that it skips rather than spiral).
MAX_CATCH_UP = 8


class LocalLink:
    """A NetworkSession's link to a GameServer in the same process."""

    def __init__(self, server: GameServer, name: str) -> None:
        self.server = server
        welcome = server.join(Hello(PROTOCOL_VERSION, name))
        if isinstance(welcome, Reject):
            raise ValueError(welcome.reason)
        self.welcome = welcome
        self.inbox: queue.Queue[Message | None] = queue.Queue()  # unused: see received()
        self._out: list[Message | None] = [server.update_for(welcome.player_id)]

    def send(self, message: Message) -> None:
        pid = self.welcome.player_id
        if isinstance(message, Input):
            self.server.receive(pid, message)
            self._out += self.server.tick().get(pid, [])  # lockstep: a tick an input
        elif isinstance(message, Bye):
            self.server.leave(pid, message.reason)

    def received(self) -> list[Message | None]:
        out, self._out = self._out, []
        return out

    def close(self, reason: str = "quit") -> None:
        self.server.leave(self.welcome.player_id, reason)


class LocalSession(NetworkSession):
    """Single-player on `map_text` (a map file's text), in `mode`."""

    def __init__(
        self,
        map_text: str,
        player_name: str,
        mode: str = "coop",
        seed: int = 0,
        config: SimConfig | None = None,
    ) -> None:
        self.server = GameServer(ServerConfig(mode=mode, sim=config or SimConfig()), seed, map_text)
        super().__init__(LocalLink(self.server, player_name), max_sends=MAX_CATCH_UP)
        for message in self.connection.received():  # the first update: you're in the world
            self._take(message)

    @property
    def world(self) -> World:
        """The server's world: the real one (the drawn copy is the mirror's)."""
        return self.server.world

    @property
    def player(self) -> PlayerState:
        return self.server.world.players[self.mirror.player_id]

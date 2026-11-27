"""The server's core: the one real World, the clients in it, and each tick.

No sockets and no clock here. Whatever carries the messages (TCP, or a
direct link in single-player) calls `join`, `receive` and `leave` as
messages arrive, and `tick` at the tick rate; `tick` returns what to send
to whom. Timeouts are counted in ticks, so all of this is deterministic
and tested without a network.

Each tick, each client's oldest queued input is used (one per tick, so a
tap of "use" shorter than a tick isn't lost). More than `input_slack`
left waiting are merged (presses kept): a backlog would otherwise stand,
lag for good. A client with nothing queued
has their last input repeated for `hold_ticks` (late packets don't make
them stutter), then stands still (one who's dropped off doesn't walk on).
"""

import dataclasses
import logging
from collections import deque
from dataclasses import dataclass, field

from ok_doomer.server.config import ServerConfig
from ok_doomer.shared.building import floor_map_text
from ok_doomer.shared.events import Event, LevelComplete, PlayerJoined, PlayerLeft
from ok_doomer.shared.mapfile import parse_map
from ok_doomer.shared.player import PlayerInput
from ok_doomer.shared.protocol import (
    PROTOCOL_VERSION,
    Bye,
    Hello,
    Input,
    Message,
    NewFloor,
    Reject,
    Update,
    Welcome,
)
from ok_doomer.shared.sim import idle_input, step
from ok_doomer.shared.snapshot import capture
from ok_doomer.shared.spawn import arrive_on_floor, spawn_player
from ok_doomer.shared.world import World, drain_events, new_world

log = logging.getLogger(__name__)

MAX_NAME = 16

# What to send whom, after a tick: player id -> messages, in order.
Outbox = dict[int, list[Message]]


@dataclass(slots=True)
class Client:
    player_id: int
    name: str
    inputs: deque[Input] = field(default_factory=deque)
    last_input: PlayerInput | None = None
    held_for: int = 0  # ticks the last input has been repeated
    ack: int = 0  # the last input applied
    newest: int = 0  # the newest input queued (older ones arriving are stale)
    last_heard: int = 0  # GameServer.ticks when a message last came from them


def merge_inputs(earlier: Input, later: Input) -> Input:
    """Two ticks' inputs as one: the later one's movement and view, and a
    press of use or jump in either (so a tap isn't lost)."""
    a, b = earlier.input, later.input
    merged = dataclasses.replace(b, use=a.use or b.use, jump=a.jump or b.jump)
    return Input(later.seq, merged)


class GameServer:
    def __init__(
        self, config: ServerConfig | None = None, seed: int = 0, map_text: str | None = None
    ) -> None:
        """`map_text`: a map to start on instead of the building's floor
        `config.storey` (a test room, a deathmatch map)."""
        self.config = config or ServerConfig()
        self.map_text = map_text if map_text is not None else floor_map_text(self.config.storey)
        self.world: World = new_world(
            parse_map(self.map_text), self.config.mode, self.config.sim, seed
        )
        self.clients: dict[int, Client] = {}
        self._next_id = 1
        # Ticks since the server started: the world's tick starts again on
        # each floor, so timeouts count these.
        self.ticks = 0

    @property
    def timeout_ticks(self) -> int:
        return round(self.config.timeout_seconds * self.config.sim.tick_rate)

    def join(self, hello: Hello) -> Welcome | Reject:
        """A client's Hello: let them in (a player in the world, spawned at
        the start), or say why not."""
        if hello.version != PROTOCOL_VERSION:
            return Reject(f"version {hello.version}; this server speaks {PROTOCOL_VERSION}")
        if len(self.clients) >= self.config.max_players:
            return Reject(f"the server's full ({self.config.max_players} players)")
        name = " ".join(hello.name.split())[:MAX_NAME] or f"player {self._next_id}"
        player_id = self._next_id
        self._next_id += 1  # ids aren't reused: a late message can't reach a newcomer
        spawn_player(self.world, player_id, name)
        self.clients[player_id] = Client(player_id, name, last_heard=self.ticks)
        self.world.events.append(PlayerJoined(self.world.tick, player_id, name))
        log.info("%s joined as player %d", name, player_id)
        return Welcome(player_id, self.world.mode, self.world.config, self.map_text)

    def update_for(self, player_id: int) -> Update:
        """The world as it is, for a client that's just joined (so it's in
        the world before the next tick)."""
        return Update(self.clients[player_id].ack, capture(self.world))

    def receive(self, player_id: int, message: Message) -> None:
        """A message from a client that's joined."""
        client = self.clients.get(player_id)
        if client is None:
            return
        client.last_heard = self.ticks
        if isinstance(message, Input):
            if message.seq <= client.newest:
                return  # a repeat or out of order: already have newer
            client.newest = message.seq
            client.inputs.append(message)
            while len(client.inputs) > self.config.input_queue:
                client.inputs.popleft()  # sending too fast: drop the oldest
        elif isinstance(message, Bye):
            self.leave(player_id, message.reason or "left")
        else:
            log.warning("player %d sent a %s; ignored", player_id, message.TYPE)

    def leave(self, player_id: int, reason: str = "left") -> None:
        client = self.clients.pop(player_id, None)
        if client is None:
            return
        self.world.players.pop(player_id, None)
        self.world.events.append(PlayerLeft(self.world.tick, player_id, client.name, reason))
        log.info("%s (player %d) left: %s", client.name, player_id, reason)

    def tick(self) -> Outbox:
        """Advance the world one tick; what to send to each client."""
        out: Outbox = {pid: [] for pid in self.clients}
        for pid in [
            pid
            for pid, c in self.clients.items()
            if self.ticks - c.last_heard >= self.timeout_ticks
        ]:
            out[pid].append(Bye("timed out"))
            self.leave(pid, "timed out")
        inputs = {pid: self._next_input(c) for pid, c in self.clients.items()}
        step(self.world, inputs)
        events = drain_events(self.world)
        arrived = next(
            (e.next_floor for e in events if isinstance(e, LevelComplete) and e.next_floor), None
        )
        if arrived is not None:
            events += self._new_floor(arrived, out)
        snapshot = capture(self.world)
        for pid, client in self.clients.items():
            out[pid].append(Update(client.ack, snapshot, tuple(events)))
        self.ticks += 1
        return out

    def _next_input(self, client: Client) -> PlayerInput:
        if client.inputs:
            message = client.inputs.popleft()
            # Don't let a backlog stand: it would be lag for good, since one
            # input is used a tick and one arrives a tick. Merge the excess.
            while len(client.inputs) > self.config.input_slack:
                message = merge_inputs(message, client.inputs.popleft())
            client.ack = message.seq
            client.last_input, client.held_for = message.input, 0
            return message.input
        if client.last_input is not None and client.held_for < self.config.hold_ticks:
            client.held_for += 1
            return client.last_input
        return idle_input(self.world.players[client.player_id])

    def _new_floor(self, storey: str, out: Outbox) -> list[Event]:
        """The lift's arrived: everyone on to the next floor. Clients get
        the map before the first Update of it."""
        self.map_text = floor_map_text(storey)
        self.world = arrive_on_floor(self.world, self.map_text)
        for pid, client in self.clients.items():
            out[pid].append(NewFloor(self.map_text))
            # Inputs meant for the old floor don't play out on this one.
            client.inputs.clear()
            client.ack = client.newest
            client.last_input, client.held_for = None, 0
        log.info("on to floor %s", storey)
        return drain_events(self.world)

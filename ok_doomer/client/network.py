"""Playing on a server: the connection, and a GameSession that draws the
server's world.

The game loop is pygame's, not asyncio's, so the connection is a plain
socket with a reader thread: the thread blocks on the socket and queues
each message; the game thread drains the queue once a frame and sends
inputs itself. Nothing else is shared between the two.

Drawing: each Update's snapshot goes onto a WorldMirror (shared/mirror.py),
and the players' positions are also kept in a short history, on the
server's timeline (its tick x dt; receipt times would bunch up when two
updates arrive in one frame). The view is drawn a little behind the newest
snapshot, interpolating between the two either side: OWN_DELAY_TICKS for
you (no prediction yet: what you see is the server's word), and
OTHERS_DELAY_TICKS for everyone else, leaving room for jitter.
"""

import contextlib
import logging
import math
import queue
import socket
import threading
from collections import deque
from dataclasses import dataclass

from ok_doomer.client.boxes import PlacedBox
from ok_doomer.client.render import Camera, SpriteInstance
from ok_doomer.client.session import (
    ease_eye,
    lying_pickups,
    static_sprites,
    world_boxes,
)
from ok_doomer.client.sprites import player_sprite_name
from ok_doomer.shared.events import Event
from ok_doomer.shared.mapfile import GameMap
from ok_doomer.shared.mirror import WorldMirror
from ok_doomer.shared.modes import GAME_MODES
from ok_doomer.shared.player import PlayerInput
from ok_doomer.shared.protocol import (
    MAX_LINE,
    PROTOCOL_VERSION,
    Bye,
    Hello,
    Message,
    NewFloor,
    ProtocolError,
    Reject,
    Update,
    Welcome,
    decode,
    encode,
)
from ok_doomer.shared.sim import eye_height
from ok_doomer.shared.switches import POWER
from ok_doomer.shared.timestep import FixedTimestep

log = logging.getLogger(__name__)

CONNECT_SECONDS = 5.0
OWN_DELAY_TICKS = 1.0
OTHERS_DELAY_TICKS = 3.0
HISTORY = 32  # snapshots of positions kept (~0.5 s)
LOST = "the connection to the server was lost"


class JoinFailed(Exception):
    """Couldn't join: refused by the server, or no server there."""


class Connection:
    """A joined connection to a server: send from the game thread, receive
    from a queue the reader thread fills (None: the connection's gone)."""

    def __init__(self, sock: socket.socket, welcome: Welcome) -> None:
        self.sock, self.welcome = sock, welcome
        self.inbox: queue.Queue[Message | None] = queue.Queue()
        self._closed = False
        self._reader = threading.Thread(target=self._read, name="server-reader", daemon=True)
        self._reader.start()

    @classmethod
    def join(cls, host: str, port: int, name: str) -> "Connection":
        try:
            sock = socket.create_connection((host, port), timeout=CONNECT_SECONDS)
        except OSError as e:
            raise JoinFailed(f"can't reach {host}:{port}: {e}") from e
        try:
            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            sock.sendall(encode(Hello(PROTOCOL_VERSION, name)))
            answer = decode(sock.makefile("rb").readline(MAX_LINE + 1))
        except (OSError, ProtocolError) as e:
            sock.close()
            raise JoinFailed(f"{host}:{port} didn't answer as a server: {e}") from e
        if isinstance(answer, Reject):
            sock.close()
            raise JoinFailed(f"the server said no: {answer.reason}")
        if not isinstance(answer, Welcome):
            sock.close()
            raise JoinFailed(f"expected a welcome, got {answer.TYPE}")
        sock.settimeout(None)  # the reader blocks until there's something
        return cls(sock, answer)

    def _read(self) -> None:
        stream = self.sock.makefile("rb")
        try:
            while line := stream.readline(MAX_LINE + 1):
                self.inbox.put(decode(line))
        except (OSError, ProtocolError, ValueError) as e:
            if not self._closed:
                log.warning("connection: %s", e)
        self.inbox.put(None)

    def send(self, message: Message) -> None:
        if self._closed:
            return
        try:
            self.sock.sendall(encode(message))
        except OSError as e:
            log.warning("sending: %s", e)
            self._closed = True

    def received(self) -> list[Message | None]:
        """Everything that's arrived, without waiting."""
        out: list[Message | None] = []
        while True:
            try:
                out.append(self.inbox.get_nowait())
            except queue.Empty:
                return out

    def close(self, reason: str = "quit") -> None:
        if not self._closed:
            self.send(Bye(reason))
            self._closed = True
        with contextlib.suppress(OSError):
            self.sock.shutdown(socket.SHUT_RDWR)
        self.sock.close()


@dataclass(frozen=True, slots=True)
class Pose:
    x: float
    y: float
    z: float
    angle: float


def lerp_pose(a: Pose, b: Pose, t: float) -> Pose:
    """Between two poses; the angle the short way round."""
    turn = (b.angle - a.angle + math.pi) % math.tau - math.pi
    return Pose(
        a.x + (b.x - a.x) * t,
        a.y + (b.y - a.y) * t,
        a.z + (b.z - a.z) * t,
        (a.angle + turn * t) % math.tau,
    )


def pose_at(history: deque[tuple[float, dict[int, Pose]]], pid: int, when: float) -> Pose | None:
    """A player's pose at server time `when`, between the snapshots either
    side (the nearest one, outside them: no guessing ahead)."""
    times = [(t, poses[pid]) for t, poses in history if pid in poses]
    if not times:
        return None
    if when <= times[0][0]:
        return times[0][1]
    for (t0, a), (t1, b) in zip(times, times[1:], strict=False):
        if t0 <= when <= t1:
            return lerp_pose(a, b, (when - t0) / (t1 - t0) if t1 > t0 else 1.0)
    return times[-1][1]


class NetworkSession:
    """A GameSession on a server (client/session.py)."""

    def __init__(self, connection: Connection) -> None:
        self.connection = connection
        self.mirror = WorldMirror.from_welcome(connection.welcome)
        self.timestep = FixedTimestep(self.mirror.config.dt)
        self.history: deque[tuple[float, dict[int, Pose]]] = deque(maxlen=HISTORY)
        self._clock = 0.0  # local seconds, from the frame times given
        self._newest = (0.0, 0.0)  # (server time, local clock) of the newest update
        self._ended: str | None = None
        self._new_floor()

    @classmethod
    def join(cls, host: str, port: int, name: str) -> "NetworkSession":
        """Connect, join, and wait for the first update (so you're in the
        world before the first frame)."""
        session = cls(Connection.join(host, port, name))
        with contextlib.suppress(queue.Empty):
            session._take(session.connection.inbox.get(timeout=CONNECT_SECONDS))
        if session.mirror.player is None:
            session.close()
            raise JoinFailed("joined, but the server sent no world")
        session._eye = eye_height(session.mirror.player, session.mirror.config.movement)
        return session

    def _new_floor(self) -> None:
        self.history.clear()
        self._static = static_sprites(self.mirror.world)
        self._eye = self.mirror.config.movement.eye_height

    @property
    def game_map(self) -> GameMap:
        return self.mirror.world.map

    def update(self, real_dt: float, player_input: PlayerInput) -> list[Event]:
        self._clock += real_dt
        for _ in range(self.timestep.advance(real_dt)):  # one input a tick
            self.connection.send(self.mirror.next_input(player_input))
        events: list[Event] = []
        for message in self.connection.received():
            events += self._take(message)
        me = self.mirror.player
        if me is not None:
            self._eye = ease_eye(self._eye, me, self.mirror.world, real_dt)
        return events

    def _take(self, message: Message | None) -> list[Event]:
        if message is None:
            self._ended = self._ended or LOST
            return []
        try:
            events = self.mirror.handle(message)
        except ProtocolError as e:
            log.warning("server: %s", e)
            return []
        if isinstance(message, NewFloor):
            self._new_floor()
        elif isinstance(message, Update):
            when = message.snapshot.tick * self.mirror.config.dt
            poses = {p.id: Pose(p.x, p.y, p.z, p.angle) for p in self.mirror.world.players.values()}
            self.history.append((when, poses))
            self._newest = (when, self._clock)
        elif isinstance(message, Bye):
            self._ended = message.reason or "the server said goodbye"
        return events

    def _server_now(self) -> float:
        """The server's time now, by the newest update and the time since
        (capped: a stalled server doesn't run on)."""
        when, at = self._newest
        return when + min(self._clock - at, 2 * self.mirror.config.dt)

    def camera(self, view_angle: float, pitch: float = 0.0) -> Camera:
        dt = self.mirror.config.dt
        pose = pose_at(
            self.history, self.mirror.player_id, self._server_now() - OWN_DELAY_TICKS * dt
        )
        if pose is None:  # not in the world yet: the start
            start = self._start_thing()
            pose = Pose(start[0], start[1], 0.0, view_angle)
        return Camera(pose.x, pose.y, view_angle, z=pose.z + self._eye, pitch=pitch)

    def sprites(self) -> list[SpriteInstance]:
        when = self._server_now() - OTHERS_DELAY_TICKS * self.mirror.config.dt
        others = []
        for pid in self.mirror.world.players:
            if pid == self.mirror.player_id:
                continue
            pose = pose_at(self.history, pid, when)
            if pose is not None:
                others.append(
                    SpriteInstance(player_sprite_name(pid), pose.x, pose.y, pose.z, pose.angle)
                )
        return self._static + lying_pickups(self.mirror.world) + others

    def boxes(self) -> list[PlacedBox]:
        return world_boxes(self.mirror.world)

    def passes(self) -> frozenset[str]:
        return frozenset(self.mirror.world.passes)

    def power(self) -> bool:
        return POWER in self.mirror.world.flags

    def spawn_angle(self) -> float:
        me = self.mirror.player
        return me.angle if me is not None else self._start_thing()[2]

    def _start_thing(self) -> tuple[float, float, float]:
        spawns = self.game_map.things_of_type(GAME_MODES[self.mirror.mode].spawn_thing)
        return (spawns[0].x, spawns[0].y, spawns[0].angle) if spawns else (0.0, 0.0, 0.0)

    def ended(self) -> str | None:
        return self._ended

    def close(self) -> None:
        self.connection.close()

"""The game client on a server: joining, moving, seeing other players
(interpolated), the lift's new floor, and the server going away. A real
server on a thread, over localhost."""

import asyncio
import math
import threading
import time
from collections import deque
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import TypeVar

import pytest

from ok_doomer.bot.client import run_bot
from ok_doomer.client.network import (
    JoinFailed,
    NetworkSession,
    Pose,
    lerp_pose,
    pose_at,
)
from ok_doomer.client.sprites import SPRITES, player_sprite_name
from ok_doomer.server.config import ServerConfig
from ok_doomer.server.game import GameServer
from ok_doomer.server.net import TcpServer
from ok_doomer.shared.player import PlayerInput
from ok_doomer.shared.sim import RIDE_SECONDS

T = TypeVar("T")

ROOM = """\
[walls]
############
#..........#
#..........#
#..........#
############

[things]
coop_start 2.5 2.0 0
exit 10.5 1.5 0
"""


class ServerThread:
    """A TcpServer running on its own asyncio loop, in a thread."""

    def __init__(self, game: GameServer) -> None:
        self.loop = asyncio.new_event_loop()
        self.server = TcpServer(game)
        self.thread = threading.Thread(target=self.loop.run_forever, daemon=True)
        self.thread.start()
        self.call(self.server.start("127.0.0.1", 0))

    @property
    def port(self) -> int:
        return self.server.port

    def call(self, coroutine: "asyncio.Future[T] | asyncio.Task[T] | object") -> T:
        return asyncio.run_coroutine_threadsafe(coroutine, self.loop).result(5)  # type: ignore[arg-type]

    def do(self, fn: Callable[[], T]) -> T:
        """Run `fn` on the server's thread (the GameServer is its alone)."""

        async def run() -> T:
            return fn()

        return self.call(run())

    def stop(self) -> None:
        self.call(self.server.close())
        self.loop.call_soon_threadsafe(self.loop.stop)
        self.thread.join(5)
        self.loop.close()


@contextmanager
def _serving(game: GameServer | None = None) -> Iterator[ServerThread]:
    server = ServerThread(game or GameServer(map_text=ROOM))
    try:
        yield server
    finally:
        server.stop()


def _play(session: NetworkSession, seconds: float, player_input: PlayerInput) -> None:
    """Frames in real time, as the game loop runs them."""
    end = time.monotonic() + seconds
    last = time.monotonic()
    while time.monotonic() < end:
        time.sleep(1 / 120)
        now = time.monotonic()
        session.update(now - last, player_input)
        last = now


def test_lerp_takes_the_short_way_round() -> None:
    a, b = Pose(0.0, 0.0, 0.0, 0.1), Pose(2.0, 4.0, 1.0, math.tau - 0.1)
    mid = lerp_pose(a, b, 0.5)
    assert (mid.x, mid.y, mid.z) == (1.0, 2.0, 0.5)
    assert mid.angle == pytest.approx(0.0, abs=1e-9) or mid.angle == pytest.approx(math.tau)


def test_pose_at_interpolates_and_never_guesses_ahead() -> None:
    p0, p1 = Pose(0.0, 0.0, 0.0, 0.0), Pose(1.0, 0.0, 0.0, 0.0)
    history = deque([(1.0, {7: p0}), (2.0, {7: p1}), (3.0, {})])
    assert pose_at(history, 7, 1.25) == Pose(0.25, 0.0, 0.0, 0.0)
    assert pose_at(history, 7, 0.0) == p0  # before: the oldest
    assert pose_at(history, 7, 9.0) == p1  # after: the newest, not extrapolated
    assert pose_at(history, 8, 1.5) is None


def test_every_player_look_has_a_sprite() -> None:
    assert player_sprite_name(1) == "player_1" and player_sprite_name(9) == "player_1"
    assert all(player_sprite_name(i) in SPRITES for i in range(1, 20))


def test_join_walk_and_leave() -> None:
    with _serving() as server:
        session = NetworkSession.join("127.0.0.1", server.port, "ann")
        start = session.camera(0.0)
        assert session.game_map.name == server.do(lambda: server.server.game.world.map.name)
        _play(session, 0.5, PlayerInput(forward=1.0, angle=0.0))
        moved = session.camera(0.0)
        assert moved.x - start.x > 1.0  # ~1.75 cells in 0.5 s, less the delay
        assert moved.z == pytest.approx(0.6)  # eye height on a flat floor
        assert session.mirror.ack > 20
        session.close()
        for _ in range(50):
            if not server.do(lambda: list(server.server.game.clients)):
                break
            time.sleep(0.02)
        assert not server.do(lambda: list(server.server.game.clients))


def test_other_players_are_drawn_smoothly() -> None:
    """A bot wanders about; the client draws it as a player sprite, moving
    a little each frame rather than jumping a tick's worth at a time."""
    with _serving() as server:
        session = NetworkSession.join("127.0.0.1", server.port, "ann")
        bot = threading.Thread(
            target=lambda: asyncio.run(run_bot("127.0.0.1", server.port, "bot", 1, 1.5))
        )
        bot.start()
        xs: list[tuple[float, float]] = []
        end = time.monotonic() + 1.0
        last = time.monotonic()
        while time.monotonic() < end:
            time.sleep(1 / 240)  # a fast screen: several frames a tick
            now = time.monotonic()
            session.update(now - last, PlayerInput())
            last = now
            others = [s for s in session.sprites() if s.name.startswith("player_")]
            if others:
                (bot_sprite,) = others
                assert bot_sprite.name == player_sprite_name(2)
                xs.append((bot_sprite.x, bot_sprite.y))
        bot.join(5)
        session.close()
    steps = [math.dist(a, b) for a, b in zip(xs, xs[1:], strict=False)]
    moving = [s for s in steps if s > 0]
    assert len(moving) > 60  # it moves on most frames...
    walk_per_tick = 3.5 / 60
    assert max(steps) < 1.5 * walk_per_tick  # ...by less than a tick's worth each


def test_the_lift_moves_the_client_to_the_next_floor() -> None:
    with _serving(GameServer(ServerConfig(storey="G"))) as server:
        session = NetworkSession.join("127.0.0.1", server.port, "ann")
        assert session.game_map.storey == "G"
        dt = server.do(lambda: server.server.game.world.config.dt)

        def arrive() -> None:
            server.server.game.world.ride = RIDE_SECONDS - dt / 2

        server.do(arrive)
        _play(session, 0.3, PlayerInput())
        assert session.game_map.storey == "1"
        me = session.mirror.player
        assert me is not None and session.spawn_angle() == me.angle
        cam = session.camera(session.spawn_angle())
        assert (math.floor(cam.x), math.floor(cam.y)) == (math.floor(me.x), math.floor(me.y))
        session.close()


def test_the_server_going_away_ends_the_game() -> None:
    server = ServerThread(GameServer(map_text=ROOM))
    session = NetworkSession.join("127.0.0.1", server.port, "ann")
    server.stop()
    for _ in range(100):
        session.update(1 / 60, PlayerInput())
        if session.ended():
            break
        time.sleep(0.01)
    assert session.ended() is not None and "shutting down" in str(session.ended())
    session.close()


def test_joining_can_fail_with_a_reason() -> None:
    with _serving(GameServer(ServerConfig(max_players=1), map_text=ROOM)) as server:
        first = NetworkSession.join("127.0.0.1", server.port, "ann")
        with pytest.raises(JoinFailed, match="full"):
            NetworkSession.join("127.0.0.1", server.port, "bob")
        first.close()
        port = server.port
    with pytest.raises(JoinFailed, match="can't reach"):
        NetworkSession.join("127.0.0.1", port, "ann")  # nobody there now

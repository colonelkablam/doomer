"""The lag proxy (a bad network on one machine) and the client's network
stats (F3): through the proxy, round trips measure what was set, order is
kept, and you still move at once (prediction)."""

import asyncio
import random
import threading
import time

import pytest

from ok_doomer.client.local import LocalSession
from ok_doomer.client.network import NetworkSession
from ok_doomer.server.game import GameServer
from ok_doomer.server.lagproxy import LOSS_STALL_MS, Impairment, LagProxy
from ok_doomer.shared.player import PlayerInput
from ok_doomer.shared.protocol import MAX_LINE, PROTOCOL_VERSION, Hello, Update, decode, encode
from tests.test_network_session import ROOM, ServerThread


class ProxyThread:
    """A LagProxy in front of a server, on its own loop and thread."""

    def __init__(self, server_port: int, impairment: Impairment) -> None:
        self.loop = asyncio.new_event_loop()
        self.proxy = LagProxy(("127.0.0.1", server_port), impairment, seed=1)
        self.thread = threading.Thread(target=self.loop.run_forever, daemon=True)
        self.thread.start()
        asyncio.run_coroutine_threadsafe(self.proxy.start("127.0.0.1", 0), self.loop).result(5)

    def stop(self) -> None:
        asyncio.run_coroutine_threadsafe(self.proxy.close(), self.loop).result(5)
        self.loop.call_soon_threadsafe(self.loop.stop)
        self.thread.join(5)


def _play(session: NetworkSession, seconds: float, player_input: PlayerInput) -> None:
    end, last = time.monotonic() + seconds, time.monotonic()
    while time.monotonic() < end:
        time.sleep(1 / 120)
        now = time.monotonic()
        session.update(now - last, player_input)
        last = now


def test_a_delay_each_way_shows_as_the_round_trip() -> None:
    server = ServerThread(GameServer(map_text=ROOM))
    proxy = ProxyThread(server.port, Impairment(lag_ms=40))
    try:
        session = NetworkSession.join("127.0.0.1", proxy.proxy.port, "ann")
        x0 = session.camera(0.0).x
        session.update(1 / 60, PlayerInput(forward=1.0, angle=0.0))
        session.update(1 / 60, PlayerInput(forward=1.0, angle=0.0))
        assert session.camera(0.0).x > x0  # predicted: moving before the server's heard
        _play(session, 1.5, PlayerInput(forward=1.0, angle=0.0))
        stats = session.net_stats()
        assert stats.round_trip_ms is not None
        # 80 ms of wire, plus up to a tick or two of waiting each end.
        assert 80 <= stats.round_trip_ms <= 80 + 100  # (coarse timers on Windows CI)
        assert 45 <= stats.updates_per_second <= 75
        assert stats.unconfirmed >= 3  # ~80 ms of inputs in flight at 60 a second
        session.close()
    finally:
        proxy.stop()
        server.stop()


def test_jitter_never_reorders() -> None:
    """Updates arrive in the server's tick order, however each is delayed:
    TCP keeps order, and so does the proxy."""

    async def test(port: int) -> list[int]:
        reader, writer = await asyncio.open_connection("127.0.0.1", port, limit=MAX_LINE + 1)
        writer.write(encode(Hello(PROTOCOL_VERSION, "ann")))
        await reader.readline()  # welcome
        ticks: list[int] = []
        while len(ticks) < 60:
            message = decode(await asyncio.wait_for(reader.readline(), 3))
            if isinstance(message, Update):
                ticks.append(message.snapshot.tick)
        writer.close()
        return ticks

    server = ServerThread(GameServer(map_text=ROOM))
    proxy = ProxyThread(server.port, Impairment(lag_ms=10, jitter_ms=40, loss_percent=5))
    try:
        ticks = asyncio.run(test(proxy.proxy.port))
    finally:
        proxy.stop()
        server.stop()
    assert ticks == sorted(ticks) and len(set(ticks)) == len(ticks)


def test_loss_is_a_stall_not_a_gap() -> None:
    rng = random.Random(0)
    steady = Impairment(lag_ms=20, jitter_ms=10)
    assert all(0.020 <= steady.delay(rng) <= 0.030 for _ in range(200))
    lossy = Impairment(lag_ms=20, loss_percent=100)
    assert lossy.delay(rng) == pytest.approx(0.020 + LOSS_STALL_MS / 1000)


def test_single_player_has_no_round_trip_to_speak_of() -> None:
    session = LocalSession(ROOM, "ann")
    for _ in range(30):
        session.update(1 / 60, PlayerInput(forward=1.0))
    stats = session.net_stats()
    assert stats.round_trip_ms == 0.0 and stats.unconfirmed == 0 and stats.corrections == 0
    assert len(stats.lines()) == 3 and "round trip 0 ms" in stats.lines()[0]

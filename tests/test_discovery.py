"""Finding games on the LAN: servers answer Seeks with what they're
running; the finder lists them (once each, with a ping), forgets the
silent, and shows a game from other code as needing updating."""

import asyncio
import json
import socket
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager

import pytest

from ok_doomer.client.discovery import (
    FORGET_SECONDS,
    LOOPBACK,
    FoundGame,
    GameFinder,
    broadcast_targets,
)
from ok_doomer.server.config import ServerConfig
from ok_doomer.server.discovery import MIN_REPLY_GAP
from ok_doomer.server.game import GameServer
from ok_doomer.server.net import TcpServer
from ok_doomer.shared.protocol import (
    MAX_DATAGRAM,
    PROTOCOL_VERSION,
    GameInfo,
    Hello,
    Seek,
    decode,
    encode,
)
from tests.test_network_session import ROOM


@contextmanager
def _server(**config: object) -> Iterator[TcpServer]:
    """A server on its own thread, answering Seeks on a free UDP port."""
    loop = asyncio.new_event_loop()
    game = GameServer(ServerConfig(**config), map_text=ROOM)  # type: ignore[arg-type]
    server = TcpServer(game)
    thread = threading.Thread(target=loop.run_forever, daemon=True)
    thread.start()
    asyncio.run_coroutine_threadsafe(server.start("127.0.0.1", 0, 0), loop).result(5)
    try:
        yield server
    finally:
        asyncio.run_coroutine_threadsafe(server.close(), loop).result(5)
        loop.call_soon_threadsafe(loop.stop)
        thread.join(5)
        loop.close()


def _ask(port: int, message: bytes = encode(Seek(PROTOCOL_VERSION))) -> list[bytes]:
    """Send a datagram to a responder; what comes back within a moment."""
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        s.settimeout(0.3)
        s.sendto(message, (LOOPBACK, port))
        replies = []
        try:
            while True:
                replies.append(s.recvfrom(4096)[0])
        except TimeoutError:
            return replies


def _find(port: int, seconds: float = 0.4) -> list[FoundGame]:
    finder = GameFinder(port, targets=[LOOPBACK])
    try:
        finder.seek()
        end = time.monotonic() + seconds
        games: list[FoundGame] = []
        while time.monotonic() < end:
            games = finder.poll()
            time.sleep(0.01)
        return games
    finally:
        finder.close()


def test_a_server_says_what_its_running() -> None:
    with _server(name="Nick's desktop", max_players=6) as server:
        assert server.discovery_port is not None
        (reply,) = _ask(server.discovery_port)
        info = decode(reply)
        assert isinstance(info, GameInfo)
        assert (info.version, info.name, info.floor, info.mode) == (
            PROTOCOL_VERSION,
            "Nick's desktop",
            "GROUND FLOOR",
            "coop",
        )
        assert (info.players, info.max_players, info.port) == (0, 6, server.port)
        server.game.join(Hello(PROTOCOL_VERSION, "ann"))  # (a player, as a join would add)
        time.sleep(MIN_REPLY_GAP)
        (again,) = _ask(server.discovery_port)
        latest = decode(again)
        assert isinstance(latest, GameInfo) and latest.players == 1  # live, not a snapshot


def test_the_finder_lists_it_with_a_ping_and_a_way_to_join() -> None:
    with _server(name="desk") as server:
        assert server.discovery_port is not None
        (game,) = _find(server.discovery_port)
        assert game.joinable and game.address == LOOPBACK and game.info is not None
        assert game.info.port == server.port and 0 <= game.ping_ms < 400
        assert "desk" in game.describe() and f"{LOOPBACK}:{server.port}" in game.describe()


def test_a_responder_cant_be_made_to_flood() -> None:
    """Junk, oversized datagrams and other messages get nothing; one
    address gets a reply at most every MIN_REPLY_GAP."""
    with _server() as server:
        port = server.discovery_port
        assert port is not None
        assert _ask(port, b"hello?") == []
        assert _ask(port, encode(Hello(PROTOCOL_VERSION, "x"))) == []
        big = json.dumps({"type": "seek", "version": 3, "pad": "x" * MAX_DATAGRAM}).encode()
        assert _ask(port, big) == []
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.settimeout(0.3)
            time.sleep(MIN_REPLY_GAP)
            for _ in range(10):  # a burst
                s.sendto(encode(Seek(PROTOCOL_VERSION)), (LOOPBACK, port))
            replies = 0
            try:
                while True:
                    s.recvfrom(4096)
                    replies += 1
            except TimeoutError:
                pass
        assert replies == 1


def test_no_discovery_when_its_off() -> None:
    loop = asyncio.new_event_loop()
    plain = TcpServer(GameServer(map_text=ROOM))
    loop.run_until_complete(plain.start("127.0.0.1", 0))  # no discovery port given
    try:
        assert plain.discovery_port is None
    finally:
        loop.run_until_complete(plain.close())
        loop.close()


def _info(server_id: str = "abcd", name: str = "desk") -> bytes:
    return encode(
        GameInfo(PROTOCOL_VERSION, server_id, name, "GROUND FLOOR", "G", "coop", 1, 8, 5029)
    )


def test_one_game_heard_two_ways_is_listed_once_by_its_lan_address() -> None:
    finder = GameFinder(targets=[])
    try:
        finder._heard(_info(), LOOPBACK, 1.0)
        finder._heard(_info(), "192.168.1.20", 1.0)
        finder._heard(_info(), LOOPBACK, 1.1)  # again from this machine: keep the LAN one
        (game,) = finder.poll(1.2)
        assert game.address == "192.168.1.20"
        finder._heard(_info("efgh", "laptop"), "192.168.1.30", 1.2)
        assert [g.name for g in finder.poll(1.3)] == ["desk", "laptop"]
    finally:
        finder.close()


def test_a_silent_game_is_forgotten() -> None:
    finder = GameFinder(targets=[])
    try:
        finder._heard(_info(), "192.168.1.20", 10.0)
        assert finder.poll(10.0 + FORGET_SECONDS - 0.1)
        assert finder.poll(10.0 + FORGET_SECONDS + 0.1) == []
    finally:
        finder.close()


@pytest.mark.parametrize("version", [PROTOCOL_VERSION - 1, PROTOCOL_VERSION + 5])
def test_a_game_from_other_code_shows_as_needing_updating(version: int) -> None:
    """Its answer may not even decode here (fields changed): its version
    and name are read anyway, so it's listed rather than missing."""
    finder = GameFinder(targets=[])
    try:
        odd = {"type": "game", "version": version, "name": "old-laptop", "port": 5029, "new": 1}
        finder._heard(json.dumps(odd).encode(), "192.168.1.40", 1.0)
        finder._heard(b"not json", "192.168.1.41", 1.0)  # nonsense: not listed
        (game,) = finder.poll(1.0)
        assert not game.joinable and "needs updating" in game.describe()
        assert f"version {version}" in game.describe()
    finally:
        finder.close()


def test_seeks_go_to_everyone_each_local_network_and_this_machine() -> None:
    targets = broadcast_targets()
    assert targets[0] == "255.255.255.255" and targets[-1] == LOOPBACK
    assert all(t.endswith(".255") for t in targets[1:-1])

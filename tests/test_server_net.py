"""The TCP transport and the bot, over real sockets on localhost: the
handshake, updates at the tick rate, bad clients dropped, bots playing."""

import asyncio
import math
import signal
import socket
import subprocess
import sys
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from pathlib import Path

import pytest

from ok_doomer.bot.brain import Wanderer
from ok_doomer.bot.client import BotRejected, run_bot
from ok_doomer.server import net
from ok_doomer.server.config import ServerConfig
from ok_doomer.server.game import GameServer
from ok_doomer.server.net import TcpServer
from ok_doomer.shared.mirror import WorldMirror
from ok_doomer.shared.player import PlayerInput, PlayerState
from ok_doomer.shared.protocol import (
    MAX_LINE,
    PROTOCOL_VERSION,
    Bye,
    Hello,
    Message,
    Reject,
    Update,
    Welcome,
    decode,
    encode,
)

ROOM = """\
[walls]
############
#..........#
#..........#
#..........#
############

[things]
coop_start 2.5 2.5 0
exit 10.5 1.5 0
"""

Streams = tuple[asyncio.StreamReader, asyncio.StreamWriter]


@asynccontextmanager
async def _serving(**config: object) -> AsyncIterator[TcpServer]:
    server = TcpServer(GameServer(ServerConfig(**config), map_text=ROOM))  # type: ignore[arg-type]
    await server.start("127.0.0.1", 0)
    try:
        yield server
    finally:
        await server.close()


async def _connect(server: TcpServer) -> Streams:
    return await asyncio.open_connection("127.0.0.1", server.port, limit=MAX_LINE + 1)


async def _read(reader: asyncio.StreamReader) -> Message | None:
    line = await asyncio.wait_for(reader.readline(), 2.0)
    return decode(line) if line else None


async def _join(server: TcpServer, name: str = "ann") -> tuple[Streams, WorldMirror]:
    reader, writer = await _connect(server)
    writer.write(encode(Hello(PROTOCOL_VERSION, name)))
    welcome = await _read(reader)
    assert isinstance(welcome, Welcome)
    return (reader, writer), WorldMirror.from_welcome(welcome)


def _run(test: Callable[[], Awaitable[None]]) -> None:
    asyncio.run(asyncio.wait_for(test(), 10.0))


def test_join_and_get_updates_at_the_tick_rate() -> None:
    async def test() -> None:
        async with _serving() as server:
            (reader, writer), mirror = await _join(server)
            assert mirror.player_id == 1 and mirror.world.map.name == server.game.world.map.name
            loop = asyncio.get_running_loop()
            start = loop.time()
            while loop.time() - start < 0.5:
                message = await _read(reader)
                assert isinstance(message, Update)
                mirror.handle(message)
            assert 20 <= mirror.updates <= 40  # ~30 at 60 Hz
            writer.close()

    _run(test)


def test_inputs_move_the_player() -> None:
    async def test() -> None:
        async with _serving() as server:
            (reader, writer), mirror = await _join(server)
            # 8: as many as the server queues (a burst of more drops the oldest).
            for _ in range(8):
                writer.write(encode(mirror.next_input(PlayerInput(forward=1.0, angle=0.0))))
            await writer.drain()
            while mirror.ack < 8:
                message = await _read(reader)
                assert message is not None
                mirror.handle(message)
            me = mirror.player
            assert me is not None
            assert me.x == pytest.approx(2.5 + 8 * 3.5 / 60)  # 8 ticks at walking speed
            writer.close()

    _run(test)


def test_the_server_sets_tcp_nodelay() -> None:
    async def test() -> None:
        async with _serving() as server:
            (_, writer), mirror = await _join(server)
            sock = server.writers[mirror.player_id].get_extra_info("socket")
            assert sock.getsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY)
            writer.close()

    _run(test)


def test_a_wrong_version_is_rejected_and_hung_up_on() -> None:
    async def test() -> None:
        async with _serving() as server:
            reader, writer = await _connect(server)
            writer.write(encode(Hello(PROTOCOL_VERSION + 1, "old")))
            assert isinstance(await _read(reader), Reject)
            assert await _read(reader) is None  # hung up
            assert not server.game.clients
            writer.close()

    _run(test)


@pytest.mark.parametrize(
    "first", [b"hello?\n", encode(Bye("eh")), b'{"type": "hello", "version": 1}\n']
)
def test_anything_but_a_hello_first_is_dropped(first: bytes) -> None:
    async def test() -> None:
        async with _serving() as server:
            reader, writer = await _connect(server)
            writer.write(first)
            bye = await _read(reader)
            assert isinstance(bye, Bye) and "protocol error" in bye.reason
            assert await _read(reader) is None
            assert not server.game.clients
            writer.close()

    _run(test)


def test_a_bad_line_after_joining_drops_the_player() -> None:
    async def test() -> None:
        async with _serving() as server:
            (reader, writer), _ = await _join(server)
            writer.write(b'{"type": "input", "seq": 1, "angle": NaN}\n')
            while (message := await _read(reader)) is not None:
                last = message
            assert isinstance(last, Bye) and "protocol error" in last.reason
            assert not server.game.clients and not server.game.world.players
            writer.close()

    _run(test)


def test_no_hello_in_time_is_dropped(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(net, "HELLO_SECONDS", 0.2)

    async def test() -> None:
        async with _serving() as server:
            reader, writer = await _connect(server)
            assert await _read(reader) is None  # silence, then hung up on
            writer.close()

    _run(test)


def test_hanging_up_removes_the_player() -> None:
    async def test() -> None:
        async with _serving() as server:
            (_, writer), _ = await _join(server)
            (_, other), _ = await _join(server, "bob")
            writer.close()
            for _ in range(50):
                if list(server.game.clients) == [2]:
                    break
                await asyncio.sleep(0.02)
            assert list(server.game.world.players) == [2]
            other.close()

    _run(test)


def test_closing_the_server_says_goodbye() -> None:
    async def test() -> None:
        async with _serving() as server:
            (reader, writer), _ = await _join(server)
        while (message := await _read(reader)) is not None:
            last = message
        assert isinstance(last, Bye) and "shutting down" in last.reason
        writer.close()

    _run(test)


def test_two_bots_play_and_leave() -> None:
    async def test() -> None:
        async with _serving() as server:
            port = server.port
            a, b = await asyncio.gather(
                run_bot("127.0.0.1", port, "bot a", seed=1, seconds=1.0),
                run_bot("127.0.0.1", port, "bot b", seed=2, seconds=1.0),
            )
            for mirror in (a, b):
                assert mirror.updates > 30 and mirror.ack > 30
                assert sorted(mirror.world.players) == [1, 2]  # each sees both
                me = mirror.player
                assert me is not None and math.dist((me.x, me.y), (2.5, 2.5)) > 0.5
            for _ in range(50):  # they said bye
                if not server.game.clients:
                    break
                await asyncio.sleep(0.02)
            assert not server.game.clients

    _run(test)


def test_a_bot_turned_away_says_why() -> None:
    async def test() -> None:
        async with _serving(max_players=1) as server:
            (_, writer), _ = await _join(server)
            with pytest.raises(BotRejected, match="full"):
                await run_bot("127.0.0.1", server.port, seconds=0.5)
            writer.close()

    _run(test)


def test_a_stuck_wanderer_turns_and_presses_use() -> None:
    import random

    brain = Wanderer(random.Random(0))
    me = PlayerState(1, "bot", 1.5, 1.5, 0.0)
    first = brain.think(me)
    assert first.forward == 1.0 and first.angle == 0.0 and not first.use
    inputs = [brain.think(me) for _ in range(10)]  # not moving: stuck
    assert inputs[-1].use and inputs[-1].angle != 0.0


@pytest.mark.skipif(sys.platform == "win32", reason="SIGTERM is POSIX")
def test_sigterm_stops_the_server_saying_goodbye(tmp_path: Path) -> None:
    """As systemd stops it: a real server process, a client in it."""
    map_file = tmp_path / "room.map"
    map_file.write_text(ROOM)
    with socket.socket() as probe:  # a free port
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    proc = subprocess.Popen(
        [sys.executable, "-m", "ok_doomer.server", "--host", "127.0.0.1", "--port", str(port),
         "--map", str(map_file)],
        stderr=subprocess.PIPE, text=True,
    )  # fmt: skip

    async def test() -> None:
        for _ in range(100):  # until it's listening
            try:
                reader, writer = await asyncio.open_connection("127.0.0.1", port)
                break
            except OSError:
                await asyncio.sleep(0.05)
        writer.write(encode(Hello(PROTOCOL_VERSION, "ann")))
        assert isinstance(await _read(reader), Welcome)
        proc.send_signal(signal.SIGTERM)
        while (message := await _read(reader)) is not None:
            last = message
        assert isinstance(last, Bye) and "shutting down" in last.reason
        writer.close()

    try:
        _run(test)
        assert proc.wait(5) == 0
    finally:
        proc.kill()
        proc.wait()


def test_the_command_lines_refuse_bad_arguments(tmp_path: Path) -> None:
    from ok_doomer.bot.__main__ import main as bot_main
    from ok_doomer.server.__main__ import main as server_main

    assert server_main(["--map", str(tmp_path / "nowhere.map")]) == 1
    assert server_main(["--storey", "9"]) == 1  # no such floor
    assert bot_main(["--connect", "nonsense"]) == 1

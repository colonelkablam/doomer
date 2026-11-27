"""The TCP transport: asyncio, one JSON line per message (shared/protocol.py).

Everything runs on one asyncio thread, so the GameServer is never touched
by two things at once: connections call into it as lines arrive, and the
tick loop calls `tick()` at the tick rate and writes what it returns.

Per connection: TCP_NODELAY (send small packets at once, rather than
Nagle's algorithm holding them back to batch), a Hello within
HELLO_SECONDS or it's dropped, anything that won't decode ends it (a Bye
saying why), and a client so far behind that its unsent data passes
MAX_BUFFERED is dropped rather than buffered without end.
"""

import asyncio
import contextlib
import logging
import socket

from ok_doomer.server.discovery import new_server_id, server_name, start_responder
from ok_doomer.server.game import GameServer
from ok_doomer.shared.building import floor_name
from ok_doomer.shared.protocol import (
    MAX_LINE,
    PROTOCOL_VERSION,
    Bye,
    Encoder,
    GameInfo,
    Hello,
    Message,
    ProtocolError,
    Reject,
    Welcome,
    decode,
    encode,
)

log = logging.getLogger(__name__)

HELLO_SECONDS = 5.0
MAX_BUFFERED = 1 << 20  # bytes waiting to go to one client
MAX_BEHIND = 8  # ticks: further behind than this (a stall), skip ahead


class TcpServer:
    def __init__(self, game: GameServer) -> None:
        self.game = game
        self.server_id = new_server_id()
        self._discovery: asyncio.DatagramTransport | None = None
        self.writers: dict[int, asyncio.StreamWriter] = {}
        self._server: asyncio.Server | None = None
        self._ticker: asyncio.Task[None] | None = None

    @property
    def port(self) -> int:
        """The port it's listening on (useful when started on port 0)."""
        assert self._server is not None
        port: int = self._server.sockets[0].getsockname()[1]
        return port

    async def start(self, host: str, port: int, discovery_port: int | None = None) -> None:
        """Listen for players on TCP `port`; and, given `discovery_port`,
        answer those looking for games on the LAN (server/discovery.py)."""
        self._server = await asyncio.start_server(self._connection, host, port, limit=MAX_LINE + 1)
        self._ticker = asyncio.create_task(self._tick_loop())
        log.info("listening on %s:%d", host, self.port)
        if discovery_port is not None:
            self._discovery, _ = await start_responder(self.describe, host, discovery_port)

    @property
    def discovery_port(self) -> int | None:
        if self._discovery is None:
            return None
        port: int = self._discovery.get_extra_info("sockname")[1]
        return port

    def describe(self) -> GameInfo:
        """The game as it is now, for those looking for one."""
        world = self.game.world
        return GameInfo(
            PROTOCOL_VERSION,
            self.server_id,
            server_name(self.game.config.name),
            floor_name(world.map.storey),
            world.map.name,
            world.mode,
            len(self.game.clients),
            self.game.config.max_players,
            self.port,
        )

    async def close(self, reason: str = "the server's shutting down") -> None:
        if self._discovery is not None:
            self._discovery.close()
        if self._ticker is not None:
            self._ticker.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._ticker
        for pid in list(self.writers):
            self._send(pid, [Bye(reason)])
            self._drop(pid, reason)
        if self._server is not None:
            self._server.close()
            await self._server.wait_closed()

    async def serve_forever(self) -> None:
        assert self._ticker is not None
        await self._ticker

    # -- connections --------------------------------------------------------

    async def _connection(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        sock = writer.get_extra_info("socket")
        if sock is not None:
            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        peer = writer.get_extra_info("peername")
        pid: int | None = None
        try:
            hello = decode(await asyncio.wait_for(reader.readline(), HELLO_SECONDS))
            if not isinstance(hello, Hello):
                raise ProtocolError(f"expected hello, got {hello.TYPE}")
            answer: Welcome | Reject = self.game.join(hello)
            writer.write(encode(answer))
            if isinstance(answer, Reject):
                log.info("%s rejected: %s", peer, answer.reason)
                await writer.drain()
                return
            pid = answer.player_id
            self.writers[pid] = writer
            while line := await reader.readline():
                self.game.receive(pid, decode(line))
                if pid not in self.game.clients:  # they said bye
                    return
        except ProtocolError as e:
            log.warning("%s: %s", peer, e)
            with contextlib.suppress(ConnectionError):
                writer.write(encode(Bye(f"protocol error: {e}")))
        except (TimeoutError, asyncio.LimitOverrunError, ValueError) as e:
            log.warning("%s: %r", peer, e)  # slow hello, or a line too long
        except ConnectionError:
            pass
        finally:
            if pid is not None:
                self._drop(pid, "disconnected")
            writer.close()
            with contextlib.suppress(ConnectionError):
                await writer.wait_closed()

    def _drop(self, pid: int, reason: str) -> None:
        self.game.leave(pid, reason)
        writer = self.writers.pop(pid, None)
        if writer is not None:
            writer.close()

    def _send(self, pid: int, messages: list[Message], encoder: Encoder | None = None) -> None:
        writer = self.writers.get(pid)
        if writer is None or writer.is_closing():
            return
        encoder = encoder or Encoder()
        writer.write(b"".join(encoder.encode(m) for m in messages))
        if writer.transport.get_write_buffer_size() > MAX_BUFFERED:
            log.warning("player %d can't keep up; dropped", pid)
            self._drop(pid, "too far behind")

    # -- ticks --------------------------------------------------------------

    async def _tick_loop(self) -> None:
        """Tick at the tick rate, by the loop's monotonic clock: each tick is
        due a fixed `dt` after the last, so timing doesn't drift."""
        loop = asyncio.get_running_loop()
        dt = self.game.config.sim.dt
        due = loop.time()
        while True:
            out = self.game.tick()
            encoder = Encoder()  # the tick's snapshot encoded once, for everyone
            for pid, messages in out.items():
                self._send(pid, messages, encoder)
                if any(isinstance(m, Bye) for m in messages):  # timed out
                    self._drop(pid, "timed out")
            due += dt
            behind = loop.time() - due
            if behind > MAX_BEHIND * dt:
                log.warning("%.0f ms behind; skipping ahead", behind * 1000)
                due = loop.time()
            await asyncio.sleep(max(0.0, due - loop.time()))

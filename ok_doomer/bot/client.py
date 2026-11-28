"""A bot's connection: join, keep a WorldMirror in step, and send one input
a tick from its brain, until told to stop or the server says goodbye."""

import asyncio
import contextlib
import logging
import random
import socket

from ok_doomer.bot.brain import Wanderer
from ok_doomer.shared.mirror import WorldMirror
from ok_doomer.shared.protocol import (
    MAX_LINE,
    PROTOCOL_VERSION,
    Bye,
    Hello,
    Reject,
    Welcome,
    decode,
    encode,
)

log = logging.getLogger(__name__)


class BotRejected(Exception):
    pass


async def run_bot(
    host: str, port: int, name: str = "bot", seed: int = 0, seconds: float | None = None
) -> WorldMirror:
    """Play until `seconds` have passed (None: until the server hangs up);
    the bot's mirror of the world, as it ended."""
    reader, writer = await asyncio.open_connection(host, port, limit=MAX_LINE + 1)
    sock = writer.get_extra_info("socket")
    if sock is not None:
        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    try:
        writer.write(encode(Hello(PROTOCOL_VERSION, name)))
        welcome = decode(await reader.readline())
        if isinstance(welcome, Reject):
            raise BotRejected(welcome.reason)
        if not isinstance(welcome, Welcome):
            raise BotRejected(f"expected welcome, got {welcome.TYPE}")
        mirror = WorldMirror.from_welcome(welcome)
        log.info("%s joined as player %d", name, mirror.player_id)
        listening = asyncio.create_task(_listen(reader, mirror))
        try:
            await _play(writer, mirror, Wanderer(random.Random(seed)), seconds, listening)
        finally:
            listening.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await listening
        if mirror.bye is None:
            with contextlib.suppress(ConnectionError):
                writer.write(encode(Bye("bot done")))
                await writer.drain()
        return mirror
    finally:
        writer.close()
        with contextlib.suppress(ConnectionError):
            await writer.wait_closed()


async def _listen(reader: asyncio.StreamReader, mirror: WorldMirror) -> None:
    while mirror.bye is None and (line := await reader.readline()):
        mirror.handle(decode(line))


async def _play(
    writer: asyncio.StreamWriter,
    mirror: WorldMirror,
    brain: Wanderer,
    seconds: float | None,
    listening: "asyncio.Task[None]",
) -> None:
    loop = asyncio.get_running_loop()
    dt = mirror.config.dt
    start = due = loop.time()
    while not listening.done() and (seconds is None or loop.time() - start < seconds):
        me = mirror.player
        if me is not None:  # in the world (after the first update)
            writer.write(encode(mirror.next_input(brain.think(me))))
            await writer.drain()
        due += dt
        await asyncio.sleep(max(0.0, due - loop.time()))

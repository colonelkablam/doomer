"""Being found on the LAN: answer anyone looking for games.

A client looking for games broadcasts a Seek (UDP, DISCOVERY_PORT); every
server on the network that hears it answers the sender directly with a
GameInfo: its name, floor, players, and the TCP port to join on. Clients
ask rather than servers announcing: the list is there the moment someone
looks, and servers say nothing while nobody does.

UDP can be aimed anywhere with a forged sender, so a responder must not
be usable to flood someone: datagrams over MAX_DATAGRAM are ignored, a
reply is small, and any one address gets at most a reply every
MIN_REPLY_GAP seconds.
"""

import asyncio
import logging
import secrets
import socket
from collections.abc import Callable

from ok_doomer.shared.protocol import (
    MAX_DATAGRAM,
    GameInfo,
    ProtocolError,
    Seek,
    decode,
    encode,
)

log = logging.getLogger(__name__)

MIN_REPLY_GAP = 0.2  # seconds between replies to one address


class DiscoveryResponder(asyncio.DatagramProtocol):
    """Answers Seeks with what `describe()` says about the game now."""

    def __init__(self, describe: Callable[[], GameInfo]) -> None:
        self.describe = describe
        self.transport: asyncio.DatagramTransport | None = None
        self._last_reply: dict[str, float] = {}

    def connection_made(self, transport: asyncio.BaseTransport) -> None:
        assert isinstance(transport, asyncio.DatagramTransport)
        self.transport = transport

    def datagram_received(self, data: bytes, addr: tuple[str | object, ...]) -> None:
        if self.transport is None or len(data) > MAX_DATAGRAM:
            return
        try:
            message = decode(data)
        except ProtocolError:
            return
        if not isinstance(message, Seek):
            return
        host = str(addr[0])
        now = asyncio.get_running_loop().time()
        if now - self._last_reply.get(host, -1e9) < MIN_REPLY_GAP:
            return
        self._last_reply[host] = now
        if len(self._last_reply) > 1000:  # forget the long-quiet
            self._last_reply = {h: t for h, t in self._last_reply.items() if now - t < 10}
        self.transport.sendto(encode(self.describe()), addr)


async def start_responder(
    describe: Callable[[], GameInfo], host: str, port: int
) -> tuple[asyncio.DatagramTransport, DiscoveryResponder]:
    """Listen for Seeks on UDP `port` (0: any free one, for tests)."""
    loop = asyncio.get_running_loop()
    transport, protocol = await loop.create_datagram_endpoint(
        lambda: DiscoveryResponder(describe), local_addr=(host, port), family=socket.AF_INET
    )
    log.info("answering those looking for games on UDP port %d", port)
    return transport, protocol


def new_server_id() -> str:
    return secrets.token_hex(4)


def server_name(configured: str) -> str:
    """What the game's called in lists of games found."""
    return configured or socket.gethostname()

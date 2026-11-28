"""A bad network on one machine: a TCP proxy that delays what passes through.

    python -m ok_doomer.server.lagproxy --to 127.0.0.1:5029 --listen 5030 --lag 30 --jitter 10

then join port 5030 instead of 5029. Each direction is delayed by `lag`
ms plus up to `jitter` more, so a round trip is about twice `lag`. For
Wi-Fi to the Pi, try --lag 15 --jitter 10 --loss 1; for a bad day,
--lag 40 --jitter 30 --loss 3.

It plays by TCP's rules, as the game's connection does: nothing is ever
lost or reordered. A lost packet on a real network is sent again, and
everything behind it waits (head-of-line blocking), so `loss` is the
chance a chunk of data stalls for LOSS_STALL_MS, holding up what follows;
jitter delays chunks but never lets one overtake another.

Standard library only (asyncio), like the server.
"""

import argparse
import asyncio
import contextlib
import logging
import random
import socket
import sys
from dataclasses import dataclass

log = logging.getLogger(__name__)

LOSS_STALL_MS = 200.0  # about TCP's shortest retransmission timeout (Linux)
CHUNK = 64 * 1024


@dataclass(frozen=True, slots=True)
class Impairment:
    lag_ms: float = 0.0  # each way
    jitter_ms: float = 0.0  # up to this much more, each chunk
    loss_percent: float = 0.0  # chance a chunk stalls LOSS_STALL_MS (and all behind it)

    def delay(self, rng: random.Random) -> float:
        """Seconds to hold one chunk."""
        ms = self.lag_ms + rng.uniform(0.0, self.jitter_ms)
        if rng.random() * 100.0 < self.loss_percent:
            ms += LOSS_STALL_MS
        return ms / 1000.0


async def _pipe(
    reader: asyncio.StreamReader,
    writer: asyncio.StreamWriter,
    impairment: Impairment,
    rng: random.Random,
) -> None:
    """Copy one direction, each chunk delivered after its delay, in order:
    never before the chunk ahead of it."""
    loop = asyncio.get_running_loop()
    queue: asyncio.Queue[tuple[float, bytes]] = asyncio.Queue()

    async def deliver() -> None:
        while True:
            due, data = await queue.get()
            if not data:
                break
            await asyncio.sleep(max(0.0, due - loop.time()))
            writer.write(data)
            await writer.drain()

    delivering = asyncio.create_task(deliver())
    last_due = 0.0
    try:
        while data := await reader.read(CHUNK):
            last_due = max(last_due, loop.time() + impairment.delay(rng))  # TCP: in order
            await queue.put((last_due, data))
        await queue.put((last_due, b""))  # then close, after what's queued
        with contextlib.suppress(ConnectionError):
            await delivering
    except ConnectionError:
        pass
    finally:
        delivering.cancel()
        with contextlib.suppress(asyncio.CancelledError, ConnectionError):
            await delivering
        writer.close()


class LagProxy:
    def __init__(self, target: tuple[str, int], impairment: Impairment, seed: int = 0) -> None:
        self.target, self.impairment = target, impairment
        self.rng = random.Random(seed)
        self._server: asyncio.Server | None = None
        self._connections: set[asyncio.Task[None]] = set()

    @property
    def port(self) -> int:
        assert self._server is not None
        port: int = self._server.sockets[0].getsockname()[1]
        return port

    async def start(self, host: str, port: int) -> None:
        self._server = await asyncio.start_server(self._connection, host, port)

    async def close(self) -> None:
        """Stop listening, and end the connections going through."""
        if self._server is not None:
            self._server.close()
        for task in list(self._connections):
            task.cancel()
        await asyncio.gather(*self._connections, return_exceptions=True)
        if self._server is not None:
            await self._server.wait_closed()

    async def _connection(
        self, client_reader: asyncio.StreamReader, client_writer: asyncio.StreamWriter
    ) -> None:
        try:
            server_reader, server_writer = await asyncio.open_connection(*self.target)
        except OSError as e:
            log.warning("can't reach %s:%d: %s", *self.target, e)
            client_writer.close()
            return
        for w in (client_writer, server_writer):
            sock = w.get_extra_info("socket")
            if sock is not None:
                sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        task = asyncio.current_task()
        assert task is not None
        self._connections.add(task)
        try:
            await asyncio.gather(
                _pipe(client_reader, server_writer, self.impairment, self.rng),
                _pipe(server_reader, client_writer, self.impairment, self.rng),
            )
        finally:
            self._connections.discard(task)
            client_writer.close()
            server_writer.close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m ok_doomer.server.lagproxy", description=__doc__
    )
    parser.add_argument("--to", default="127.0.0.1:5029", metavar="HOST:PORT", help="the server")
    parser.add_argument("--listen", type=int, default=5030, metavar="PORT")
    parser.add_argument("--lag", type=float, default=30.0, metavar="MS", help="each way")
    parser.add_argument("--jitter", type=float, default=10.0, metavar="MS")
    parser.add_argument("--loss", type=float, default=0.0, metavar="PERCENT")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    host, _, port = args.to.rpartition(":")
    if not host or not port.isdigit():
        print("error: --to wants HOST:PORT", file=sys.stderr)
        return 1
    impairment = Impairment(args.lag, args.jitter, args.loss)

    async def run() -> None:
        proxy = LagProxy((host, int(port)), impairment)
        await proxy.start("127.0.0.1", args.listen)
        log.info(
            "join 127.0.0.1:%d for %s:%s with %.0f ms each way (+%.0f jitter, %.1f%% stalls)",
            proxy.port, host, port, args.lag, args.jitter, args.loss,
        )  # fmt: skip
        await asyncio.Event().wait()

    with contextlib.suppress(KeyboardInterrupt):
        try:
            asyncio.run(run())
        except OSError as e:
            print(f"error: {e}", file=sys.stderr)
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

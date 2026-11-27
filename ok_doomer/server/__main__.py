"""Run a server: python -m ok_doomer.server [--port 5029] [--map maps/x.map].

Starts on the building's ground floor (or the floor given), in co-op;
players join with `python -m ok_doomer.client --connect HOST:PORT` (or
bots: `python -m ok_doomer.bot --connect HOST:PORT`). Ctrl+C stops it.
"""

import argparse
import asyncio
import contextlib
import dataclasses
import logging
import signal
import sys
import threading
from collections.abc import Callable
from pathlib import Path

from ok_doomer.server.config import ServerConfig
from ok_doomer.server.game import GameServer
from ok_doomer.server.net import TcpServer
from ok_doomer.shared.config import ConfigError, load_toml
from ok_doomer.shared.mapfile import MapError, parse_map
from ok_doomer.shared.world import ModeNotSupported

LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m ok_doomer.server", description=__doc__)
    parser.add_argument("--config", type=Path, help="TOML file of server settings")
    parser.add_argument("--host", help="address to listen on (default: every interface)")
    parser.add_argument("--port", type=int)
    parser.add_argument("--storey", help="the building's floor to start on (G, 1, ...)")
    parser.add_argument("--mode", help="coop or deathmatch")
    parser.add_argument("--map", type=Path, help="a map file to play instead of the building")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--name", help="the game's name in lists of games found")
    parser.add_argument(
        "--exit-with-stdin",
        action="store_true",
        help="stop (cleanly) when standard input closes: run by a game hosting it",
    )
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )

    try:
        config = load_toml(ServerConfig, args.config) if args.config else ServerConfig()
        overrides = {k: getattr(args, k) for k in ("host", "port", "storey", "mode", "name")}
        config = dataclasses.replace(
            config, **{k: v for k, v in overrides.items() if v is not None}
        )
        map_text = args.map.read_text(encoding="utf-8") if args.map else None
        if map_text is not None:
            parse_map(map_text)  # fail here, not on the first join
        game = GameServer(config, args.seed, map_text)
    except (OSError, MapError, ConfigError, ModeNotSupported, KeyError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1

    async def serve() -> None:
        server = TcpServer(game)
        await server.start(
            config.host, config.port, config.discovery_port if config.discovery else None
        )
        serving = asyncio.ensure_future(server.serve_forever())
        loop = asyncio.get_running_loop()
        if sys.platform != "win32":  # systemd stops a service with SIGTERM: stop cleanly
            loop.add_signal_handler(signal.SIGTERM, serving.cancel)
        if args.exit_with_stdin:  # the game hosting this quit, crashed or was killed
            _on_stdin_closed(lambda: loop.call_soon_threadsafe(serving.cancel))
        try:
            await serving
        except asyncio.CancelledError:
            logging.getLogger(__name__).info("stopping")
        finally:
            await server.close()

    with contextlib.suppress(KeyboardInterrupt):
        try:
            asyncio.run(serve())
        except OSError as e:  # the port's taken, or the address isn't this machine's
            print(f"error: {e}", file=sys.stderr)
            return 1
    return 0


def _on_stdin_closed(then: Callable[[], object]) -> None:
    """Call `then` when standard input closes: a pipe from the process that
    started this one, which closes however that process ends. On a daemon
    thread, so it never keeps this process alive."""

    def wait() -> None:
        with contextlib.suppress(OSError, ValueError):
            while sys.stdin.buffer.read(4096):
                pass
        then()

    threading.Thread(target=wait, name="stdin-watch", daemon=True).start()


if __name__ == "__main__":
    sys.exit(main())

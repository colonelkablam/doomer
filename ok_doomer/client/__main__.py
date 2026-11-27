"""Play: python -m ok_doomer.client [map] [--mode coop|deathmatch] for
single-player, or --connect HOST:PORT to join a server."""

import argparse
import logging
import sys
from pathlib import Path

from ok_doomer.client.app import run
from ok_doomer.client.config import ClientConfig
from ok_doomer.client.network import JoinFailed, NetworkSession
from ok_doomer.client.session import GameSession, LocalSession
from ok_doomer.shared.config import ConfigError, load_toml
from ok_doomer.shared.mapfile import MapError, load_map
from ok_doomer.shared.modes import supported_modes
from ok_doomer.shared.world import ModeNotSupported, new_world

DEFAULT_MAP = Path(__file__).resolve().parents[2] / "maps" / "office.map"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m ok_doomer.client", description=__doc__)
    parser.add_argument("map", nargs="?", type=Path, default=DEFAULT_MAP)
    parser.add_argument("--mode", help="coop or deathmatch (default: first the map supports)")
    parser.add_argument("--name", default="player")
    parser.add_argument("--config", type=Path, help="TOML file overriding client settings")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--connect", metavar="HOST:PORT", help="join a server instead")
    parser.add_argument(
        "--no-predict", action="store_true", help="on a server: wait for it to move you"
    )
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    session: GameSession
    try:
        config = load_toml(ClientConfig, args.config) if args.config else ClientConfig()
        if args.connect:
            host, _, port = args.connect.rpartition(":")
            if not host or not port.isdigit():
                raise ConfigError("--connect wants HOST:PORT, e.g. 192.168.1.20:5029")
            session = NetworkSession.join(host, int(port), args.name, not args.no_predict)
        else:
            game_map = load_map(args.map)
            modes = supported_modes(game_map)
            mode = args.mode or (modes[0] if modes else "coop")
            session = LocalSession(new_world(game_map, mode, seed=args.seed), args.name)
    except (OSError, MapError, ConfigError, ModeNotSupported, JoinFailed) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1

    try:
        run(session, config, session.spawn_angle())
    finally:
        session.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())

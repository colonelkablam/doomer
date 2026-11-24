"""Single-player for now: python -m doomer.client [map] [--mode coop|deathmatch]."""

import argparse
import logging
import sys
from pathlib import Path

from doomer.client.app import run
from doomer.client.config import ClientConfig
from doomer.client.session import LocalSession
from doomer.shared.config import ConfigError, load_toml
from doomer.shared.mapfile import MapError, load_map
from doomer.shared.modes import supported_modes
from doomer.shared.world import ModeNotSupported, new_world

DEFAULT_MAP = Path(__file__).resolve().parents[2] / "maps" / "test.map"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m doomer.client", description=__doc__)
    parser.add_argument("map", nargs="?", type=Path, default=DEFAULT_MAP)
    parser.add_argument("--mode", help="coop or deathmatch (default: first the map supports)")
    parser.add_argument("--name", default="player")
    parser.add_argument("--config", type=Path, help="TOML file overriding client settings")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    try:
        config = load_toml(ClientConfig, args.config) if args.config else ClientConfig()
        game_map = load_map(args.map)
        modes = supported_modes(game_map)
        mode = args.mode or (modes[0] if modes else "coop")
        session = LocalSession(new_world(game_map, mode, seed=args.seed), args.name)
    except (OSError, MapError, ConfigError, ModeNotSupported) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1

    run(session, config, session.player.angle)
    return 0


if __name__ == "__main__":
    sys.exit(main())

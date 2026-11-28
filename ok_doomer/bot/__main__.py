"""Bots: python -m ok_doomer.bot --connect HOST:PORT [--count 3] [--seconds 60].

Each joins the server and wanders about (turning when it gets stuck). A
test player, and with --count, a load test.
"""

import argparse
import asyncio
import contextlib
import logging
import sys

from ok_doomer.bot.client import BotRejected, run_bot

LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m ok_doomer.bot", description=__doc__)
    parser.add_argument("--connect", default="127.0.0.1:5029", metavar="HOST:PORT")
    parser.add_argument("--count", type=int, default=1)
    parser.add_argument("--name", default="bot")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--seconds", type=float, help="how long to play (default: until stopped)")
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    host, _, port = args.connect.rpartition(":")
    if not host or not port.isdigit():
        print("error: --connect wants HOST:PORT, e.g. 127.0.0.1:5029", file=sys.stderr)
        return 1

    async def bots() -> None:
        names = (
            [args.name] if args.count == 1 else [f"{args.name} {i + 1}" for i in range(args.count)]
        )
        await asyncio.gather(
            *(run_bot(host, int(port), n, args.seed + i, args.seconds) for i, n in enumerate(names))
        )

    with contextlib.suppress(KeyboardInterrupt):
        try:
            asyncio.run(bots())
        except (OSError, BotRejected) as e:
            print(f"error: {e}", file=sys.stderr)
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

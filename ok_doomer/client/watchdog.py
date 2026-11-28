"""If the game freezes, say where: a hang leaves no message of its own.

Each frame `pet()`s the watchdog. If one doesn't come for HANG_SECONDS,
Python's faulthandler writes every thread's stack to ~/.ok-doomer/hang.log
(again each HANG_SECONDS while it's stuck), from a thread of its own, so
it works even when the game's thread is stuck inside SDL. Costs nothing
while frames keep coming: petting just moves the deadline on.
"""

import contextlib
import faulthandler
import logging
from pathlib import Path
from typing import IO

log = logging.getLogger(__name__)

HANG_SECONDS = 10.0


def hang_log_path() -> Path:
    return Path.home() / ".ok-doomer" / "hang.log"


class Watchdog:
    def __init__(self) -> None:
        self._file: IO[str] | None = None
        try:
            path = hang_log_path()
            path.parent.mkdir(parents=True, exist_ok=True)
            self._file = path.open("w", encoding="utf-8")
        except OSError as e:
            log.warning("no hang log: %s", e)

    def pet(self) -> None:
        """A frame's come: put the deadline back."""
        if self._file is not None:
            faulthandler.dump_traceback_later(HANG_SECONDS, repeat=True, file=self._file)

    def close(self) -> None:
        faulthandler.cancel_dump_traceback_later()
        if self._file is not None:
            with contextlib.suppress(OSError):
                self._file.close()
            self._file = None

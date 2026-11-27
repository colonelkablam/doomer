"""Hosting a game from the start screen: others on the LAN join it, and so
do you.

The game runs the dedicated server (`python -m ok_doomer.server`) as a
process of its own, and joins it over 127.0.0.1 like anyone else. Not the
in-process server single-player uses: that one ticks in step with your
inputs, a hosted one must keep real time for everyone; and in a thread of
this process it would share Python's lock with the renderer (measured,
another server in-process took a frame from 17 to 30 ms). As a process it
has a core of its own, is exactly the server a Pi runs (answering those
looking for games included), and can't outlive the game: it's started
with `--exit-with-stdin` and a pipe from here, which closes however this
process ends (quit, crash, killed), and it stops cleanly, telling everyone.

Its port is 5029 if free (else any free one: a dedicated server on this
machine already has 5029); its log goes to ~/.ok-doomer/host.log.
"""

import contextlib
import logging
import socket
import subprocess
import sys
import time
from pathlib import Path

from ok_doomer.client.discovery import local_addresses

log = logging.getLogger(__name__)

DEFAULT_PORT = 5029
START_SECONDS = 10.0  # for the server to be up, at most
STOP_SECONDS = 3.0  # for it to stop cleanly, before it's made to
PROJECT = Path(__file__).resolve().parents[2]


class HostFailed(Exception):
    """The server wouldn't start (with why, from its log)."""


def free_port(preferred: int = DEFAULT_PORT) -> int:
    """`preferred` if nothing's listening there, else any free port."""
    for port in (preferred, 0):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("", port))
            except OSError:
                continue
            bound: int = s.getsockname()[1]
            return bound
    raise HostFailed("no free port to host on")


def log_path() -> Path:
    return Path.home() / ".ok-doomer" / "host.log"


class HostedServer:
    def __init__(self, process: subprocess.Popen[bytes], port: int) -> None:
        self.process, self.port = process, port

    @classmethod
    def start(cls, name: str, preferred_port: int = DEFAULT_PORT) -> "HostedServer":
        """Start a server called `name` (in lists of games found); return
        once it accepts players."""
        port = free_port(preferred_port)
        path = log_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as log_file:
            process = subprocess.Popen(
                [
                    sys.executable,
                    "-m",
                    "ok_doomer.server",
                    "--port",
                    str(port),
                    "--name",
                    name,
                    "--exit-with-stdin",
                ],  # fmt: skip
                stdin=subprocess.PIPE,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                cwd=PROJECT,
            )
        hosted = cls(process, port)
        end = time.monotonic() + START_SECONDS
        while time.monotonic() < end:
            if process.poll() is not None:
                raise HostFailed(f"the server stopped: {_last_line(path)}")
            with contextlib.suppress(OSError), socket.create_connection(("127.0.0.1", port), 0.2):
                log.info("hosting on port %d", port)
                return hosted
            time.sleep(0.05)
        hosted.stop()
        raise HostFailed(f"the server didn't start in {START_SECONDS:.0f} s: {_last_line(path)}")

    def where(self) -> str:
        """Where others join: this machine's LAN address and the port."""
        addresses = local_addresses()
        return f"{addresses[0] if addresses else '127.0.0.1'}:{self.port}"

    def stop(self) -> None:
        """Stop it, cleanly (everyone told) if it will."""
        if self.process.poll() is not None:
            return
        with contextlib.suppress(OSError):
            assert self.process.stdin is not None
            self.process.stdin.close()  # it watches this: it stops
        try:
            self.process.wait(STOP_SECONDS)
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.process.wait()


def _last_line(path: Path) -> str:
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").strip().splitlines()
    except OSError:
        return "(no log)"
    return lines[-1] if lines else "(nothing logged)"

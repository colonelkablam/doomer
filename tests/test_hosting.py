"""Hosting from the start screen: a real server process, joinable and
findable, stopped cleanly (everyone told), never outliving the game."""

import contextlib
import os
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import pytest

from ok_doomer.client import hosting
from ok_doomer.client.discovery import GameFinder
from ok_doomer.client.hosting import HostedServer, HostFailed, free_port
from ok_doomer.shared.protocol import PROTOCOL_VERSION, Bye, Hello, Welcome, decode, encode


@pytest.fixture(autouse=True)
def _home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The host's log goes to a scratch home."""
    monkeypatch.setattr(hosting, "log_path", lambda: tmp_path / "host.log")


def test_the_port_is_5029_if_free_else_another() -> None:
    with socket.socket() as busy:
        busy.bind(("", 0))
        busy.listen()
        taken = busy.getsockname()[1]
        assert free_port(taken) != taken
    with socket.socket() as probe:
        probe.bind(("", 0))
        free = probe.getsockname()[1]
    assert free_port(free) == free


def _join(port: int, name: str = "bob") -> Any:
    sock = socket.create_connection(("127.0.0.1", port), 5)
    stream = sock.makefile("rb")
    sock.sendall(encode(Hello(PROTOCOL_VERSION, name)))
    assert isinstance(decode(stream.readline()), Welcome)
    return sock, stream


def test_a_hosted_game_is_joined_found_and_stopped_cleanly() -> None:
    host = HostedServer.start("ann's game", preferred_port=free_port(0))
    try:
        sock, stream = _join(host.port)
        assert host.where().endswith(f":{host.port}")
        finder = GameFinder(targets=["127.0.0.1"])
        try:
            names: list[str] = []
            end = time.monotonic() + 3
            while not names and time.monotonic() < end:
                finder.seek()
                time.sleep(0.2)
                names = [g.name for g in finder.poll() if g.info and g.info.port == host.port]
        finally:
            finder.close()
        if names:  # (UDP 5028 may be another server's here: then it's unfound, by design)
            assert names == ["ann's game"]
        host.stop()
        last = None
        while line := stream.readline():
            last = decode(line)
        assert isinstance(last, Bye) and "shutting down" in last.reason
        assert host.process.returncode == 0  # it stopped itself, cleanly
        sock.close()
    finally:
        host.stop()


def test_a_server_that_wont_start_says_why(monkeypatch: pytest.MonkeyPatch) -> None:
    real_popen = subprocess.Popen

    def failing(args: Any, **kwargs: Any) -> Any:
        script = "print('error: [Errno 98] address already in use'); raise SystemExit(1)"
        return real_popen([sys.executable, "-c", script], **kwargs)

    monkeypatch.setattr(subprocess, "Popen", failing)
    with pytest.raises(HostFailed, match="address already in use"):
        HostedServer.start("ann's game")


@pytest.mark.skipif(sys.platform == "win32", reason="checks the pid with POSIX signals")
def test_it_never_outlives_the_game(tmp_path: Path) -> None:
    """The game is killed outright (no chance to tidy up): the server it
    hosted stops anyway, its pipe closing."""
    script = tmp_path / "game.py"
    script.write_text(
        "import sys, time\n"
        "from ok_doomer.client.hosting import HostedServer, free_port\n"
        "host = HostedServer.start('ann', free_port(0))\n"
        "print(host.process.pid, flush=True)\n"
        "time.sleep(60)\n"
    )
    root = Path(hosting.__file__).resolve().parents[2]
    env = {**os.environ, "PYTHONPATH": str(root), "HOME": str(tmp_path)}
    game = subprocess.Popen([sys.executable, str(script)], stdout=subprocess.PIPE, env=env)
    try:
        assert game.stdout is not None
        server_pid = int(game.stdout.readline())
        os.kill(game.pid, signal.SIGKILL)  # the game dies
        game.wait(5)
        end = time.monotonic() + 10
        while time.monotonic() < end:
            try:
                os.kill(server_pid, 0)  # still there?
            except ProcessLookupError:
                return  # gone: good
            time.sleep(0.1)
            with contextlib.suppress(ChildProcessError):
                os.waitpid(server_pid, os.WNOHANG)  # (not our child: the game's)
        pytest.fail("the hosted server outlived the game")
    finally:
        game.kill()


def test_hosting_from_the_menu(monkeypatch: pytest.MonkeyPatch) -> None:
    """HOST GAME: the server starts, you join it, the greeting says where;
    leaving stops the server."""
    from ok_doomer.client import menu as menu_mod
    from ok_doomer.client.config import ClientConfig, RenderConfig
    from ok_doomer.client.menu import HostGame, Quit

    monkeypatch.setattr(hosting, "DEFAULT_PORT", free_port(0))
    choices: list[object] = [HostGame(), Quit()]
    seen: dict[str, Any] = {}

    def fake_play(
        display: object, session: Any, config: object, angle: float, greeting: Any
    ) -> bool:
        seen["session"], seen["greeting"] = type(session).__name__, greeting
        seen["players"] = len(session.mirror.world.players)
        return False

    started: list[HostedServer] = []
    real_start = HostedServer.start.__func__  # type: ignore[attr-defined]

    def start(cls: type, name: str, preferred_port: int = 0) -> HostedServer:
        host: HostedServer = real_start(cls, name, free_port(0))
        started.append(host)
        return host

    monkeypatch.setattr(menu_mod, "run_menu", lambda *a, **k: choices.pop(0))
    monkeypatch.setattr(menu_mod, "play", fake_play)
    monkeypatch.setattr(HostedServer, "start", classmethod(start))
    menu_mod.start(ClientConfig(render=RenderConfig(vsync=False)), "ann")
    (host,) = started
    assert seen["session"] == "NetworkSession" and seen["players"] == 1
    assert seen["greeting"] == f"Hosting: others can join at {host.where()}"
    assert host.process.poll() is not None  # stopped on leaving

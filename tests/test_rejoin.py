"""Rejoining: a player who leaves or drops and comes back with the same
name soon after gets their player back (where they were, health, coffee);
one back on a new connection before the old one timed out replaces it;
the start screen offers REJOIN GAME meanwhile."""

import asyncio
from typing import Any

import pytest

from ok_doomer.client import menu as menu_mod
from ok_doomer.client.config import ClientConfig, RenderConfig
from ok_doomer.client.hud import news_for
from ok_doomer.client.menu import REJOIN_SECONDS, JoinGame, Menu, Quit
from ok_doomer.client.network import LOST, NetworkSession
from ok_doomer.server.config import ServerConfig
from ok_doomer.server.game import GameServer
from ok_doomer.server.net import SHUTTING_DOWN, TcpServer
from ok_doomer.shared.events import PlayerJoined, PlayerLeft
from ok_doomer.shared.player import PlayerInput
from ok_doomer.shared.protocol import (
    MAX_LINE,
    PROTOCOL_VERSION,
    Hello,
    Input,
    Update,
    Welcome,
    decode,
    encode,
)
from ok_doomer.shared.world import drain_events

ROOM = """\
[walls]
##########
#........#
#........#
#........#
##########

[things]
coop_start 2.5 2.5 0
exit 8.5 1.5 0
"""


def _join(server: GameServer, name: str) -> int:
    welcome = server.join(Hello(PROTOCOL_VERSION, name))
    assert isinstance(welcome, Welcome)
    return welcome.player_id


def _walk(server: GameServer, pid: int, ticks: int) -> None:
    for seq in range(1, ticks + 1):
        server.receive(pid, Input(seq, PlayerInput(forward=1.0)))
        server.tick()


def _setup(**config: object) -> tuple[GameServer, int]:
    server = GameServer(ServerConfig(**config), map_text=ROOM)  # type: ignore[arg-type]
    ann = _join(server, "ann")
    _walk(server, ann, 30)
    me = server.world.players[ann]
    me.health, me.coffee = 40, 25
    return server, ann


def test_back_soon_with_the_same_name_and_youre_where_you_were() -> None:
    server, ann = _setup()
    before = server.world.players[ann]
    assert before.x > 3.5  # walked off the start
    server.leave(ann, "disconnected")
    for _ in range(60):
        server.tick()
    drain_events(server.world)
    again = _join(server, "ANN")  # the name's matched whatever its case
    back = server.world.players[again]
    assert again != ann and ann not in server.world.players  # a new id
    assert (back.x, back.y, back.angle) == (before.x, before.y, before.angle)
    assert (back.health, back.coffee) == (40, 25)
    joined = [e for e in server.world.events if isinstance(e, PlayerJoined)]
    assert joined[0].rejoined and not server.departed


def test_too_late_or_someone_else_and_its_a_fresh_start() -> None:
    server, ann = _setup(rejoin_seconds=1.0)
    server.leave(ann, "left")
    bob = _join(server, "bob")  # not ann: ann's kept still
    assert server.world.players[bob].health == 100 and "ann" in server.departed
    for _ in range(61):
        server.tick()
    late = server.world.players[_join(server, "ann")]
    assert late.x < 3.5 and late.health == 100  # at the start (beside bob, on it)


def test_a_new_floor_forgets_who_left() -> None:
    server, ann = _setup()
    _join(server, "bob")
    server.leave(ann, "left")
    server._new_floor("1", {pid: [] for pid in server.clients})
    assert not server.departed


def test_back_before_the_old_connection_timed_out_replaces_it() -> None:
    """A Wi-Fi blip: the server hasn't noticed the old connection's dead
    (it's only gone quiet) when ann joins again."""
    server, ann = _setup(reconnect_silent_seconds=1.0)
    bob = _join(server, "bob")
    for _ in range(61):  # ann silent a second; bob's still talking
        server.receive(bob, Input(server.ticks + 100, PlayerInput()))
        server.tick()
    where = server.world.players[ann].x
    drain_events(server.world)
    again = _join(server, "ann")
    assert ann not in server.clients and server.world.players[again].x == where
    events = server.world.events
    assert any(isinstance(e, PlayerLeft) and e.reason == "reconnected" for e in events)
    names = {1: "ann", 2: "bob"}
    assert [news_for(e, bob, names) for e in events] == [
        None,  # the old connection's going isn't news: the rejoining is
        None,  # (PlayerSpawned)
        "ann rejoined",
    ]
    second = _join(server, "bob")  # bob's talking: a second bob, not bob replaced
    assert bob in server.clients and second in server.clients


def test_back_where_someone_now_stands_and_youre_beside_it() -> None:
    server, ann = _setup()
    spot = server.world.players[ann]
    server.leave(ann, "left")
    cat = server.world.players[_join(server, "cat")]
    cat.x, cat.y = spot.x, spot.y
    back = server.world.players[_join(server, "ann")]
    assert (back.x, back.y) != (spot.x, spot.y) and back.health == 40


def test_over_tcp_a_dropped_connection_rejoins() -> None:
    async def hello(port: int) -> tuple[asyncio.StreamReader, asyncio.StreamWriter, int]:
        reader, writer = await asyncio.open_connection("127.0.0.1", port, limit=MAX_LINE + 1)
        writer.write(encode(Hello(PROTOCOL_VERSION, "ann")))
        welcome = decode(await reader.readline())
        assert isinstance(welcome, Welcome)
        return reader, writer, welcome.player_id

    async def test() -> None:
        game = GameServer(ServerConfig(discovery=False), map_text=ROOM)
        server = TcpServer(game)
        await server.start("127.0.0.1", 0)
        try:
            _, writer, first = await hello(server.port)
            await asyncio.sleep(0.1)
            game.world.players[first].health = 33
            writer.transport.abort()  # gone, no goodbye
            for _ in range(50):
                if first not in game.clients:
                    break
                await asyncio.sleep(0.02)
            reader, _, second = await hello(server.port)
            while not isinstance(decode(await reader.readline()), Update):
                pass
            assert game.world.players[second].health == 33
        finally:
            await server.close()

    asyncio.run(asyncio.wait_for(test(), 10.0))


# -- the start screen ---------------------------------------------------------


def test_dropped_rejoin_game_is_offered_selected_and_said_then_goes() -> None:
    menu = Menu("ann")
    game = JoinGame("192.168.1.20", 5029)
    menu.offer_rejoin(game, LOST, now=100.0)
    assert menu.items()[0].label == "REJOIN" and menu.selected == 0
    assert menu.message == LOST and menu.last_game() == "LAST GAME: 192.168.1.20:5029 - 30 S"
    menu.expire(112.4)
    assert menu.last_game().endswith(" - 18 S")
    assert menu.activate() == game
    menu.expire(100.0 + REJOIN_SECONDS - 1)
    assert menu.rejoin == game
    menu.selected = 2  # HOST GAME
    menu.expire(100.0 + REJOIN_SECONDS)
    assert menu.rejoin is None and menu.message == ""
    assert menu.items()[menu.selected].label == "HOST GAME"  # still on what you were


def test_left_on_purpose_rejoin_is_offered_but_not_selected() -> None:
    menu = Menu("ann")
    menu.offer_rejoin(JoinGame("pi", 5029), None, now=0.0)
    assert menu.items()[0].action == "rejoin" and menu.items()[menu.selected].action == "single"
    assert menu.message == ""


def test_after_a_game_joined_the_menu_offers_rejoin(monkeypatch: pytest.MonkeyPatch) -> None:
    menus: list[Menu] = []
    choices: list[object] = [JoinGame("pi", 5029), JoinGame("pi", 5029), Quit()]
    endings = [LOST, SHUTTING_DOWN]

    class FakeSession:
        def __init__(self) -> None:
            self.reason = endings.pop(0)

        def spawn_angle(self) -> float:
            return 0.0

        def ended(self) -> str:
            return self.reason

        def close(self) -> None:
            pass

    def spy(display: object, config: object, menu: Menu, port: int = 0) -> object:
        menus.append(menu)
        return choices.pop(0)

    offered: list[object] = []
    monkeypatch.setattr(menu_mod, "run_menu", spy)
    monkeypatch.setattr(menu_mod, "_connecting", lambda *a: None)
    monkeypatch.setattr(NetworkSession, "join", classmethod(lambda cls, *a, **k: FakeSession()))

    def fake_play(*a: Any, **k: Any) -> bool:
        offered.append(menus[-1].rejoin)
        return False

    monkeypatch.setattr(menu_mod, "play", fake_play)
    menu_mod.start(ClientConfig(render=RenderConfig(vsync=False)), "ann")
    # dropped: offered next time round; the server shut down: nothing to rejoin
    assert offered == [None, JoinGame("pi", 5029)] and menus[-1].rejoin is None

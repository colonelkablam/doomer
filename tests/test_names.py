"""Two players with the same name: the second is numbered (BOB002), and
told; numbered names still rejoin and reconnect as themselves."""

from ok_doomer.client.network import NetworkSession
from ok_doomer.server.config import ServerConfig
from ok_doomer.server.game import MAX_NAME, GameServer, numbered, same_name
from ok_doomer.shared.player import PlayerInput
from ok_doomer.shared.protocol import PROTOCOL_VERSION, Hello, Input, Reject, Welcome
from tests.test_network_session import ROOM, ServerThread


def _join(server: GameServer, name: str) -> int:
    welcome = server.join(Hello(PROTOCOL_VERSION, name))
    assert isinstance(welcome, Welcome)
    return welcome.player_id


def _name(server: GameServer, pid: int) -> str:
    return server.clients[pid].name


def _talk(server: GameServer, *pids: int, ticks: int = 1) -> None:
    """Ticks in which `pids` keep sending (everyone else falls silent)."""
    for _ in range(ticks):
        for pid in pids:
            server.receive(pid, Input(server.ticks + 1, PlayerInput()))
        server.tick()


def test_the_same_name_again_is_numbered_whatever_its_case() -> None:
    server = GameServer(map_text=ROOM)
    names = [_name(server, _join(server, n)) for n in ("bob", "Bob", "BOB", "ann")]
    assert names == ["bob", "Bob002", "BOB003", "ann"]  # BOB and bob look the same on screen
    long = "x" * MAX_NAME
    assert [_name(server, _join(server, long)) for _ in range(2)] == [long, "x" * 13 + "002"]


def test_a_number_freed_is_used_again() -> None:
    server = GameServer(map_text=ROOM)
    _join(server, "bob")
    second = _join(server, "bob")
    _join(server, "bob")
    server.leave(second)
    assert _name(server, _join(server, "cat")) == "cat"
    assert _name(server, _join(server, "bob")) == "bob002"


def test_numbered_names_are_recognised() -> None:
    assert numbered("bob", 1) == "bob" and numbered("bob", 12) == "bob012"
    assert same_name("bob002", "bob") and same_name("BOB", "bob")
    assert not same_name("bob2", "bob") and not same_name("bobby", "bob")
    assert not same_name("bob001", "bob")  # the first bob is plain bob


def test_bob002_drops_and_rejoins_as_himself() -> None:
    server = GameServer(map_text=ROOM)
    first = _join(server, "bob")
    second = _join(server, "bob")
    server.world.players[second].health = 40
    server.leave(second, "disconnected")
    back = _join(server, "bob")  # asks for bob, as before; bob's still in
    assert _name(server, back) == "bob002" and server.world.players[back].health == 40
    assert first in server.clients


def test_a_silent_bob002_is_replaced_not_a_playing_bob() -> None:
    server = GameServer(ServerConfig(reconnect_silent_seconds=1.0), map_text=ROOM)
    first = _join(server, "bob")
    second = _join(server, "bob")
    _talk(server, first, ticks=61)  # bob plays on; bob002's connection's died quietly
    back = _join(server, "bob")
    assert first in server.clients and second not in server.clients
    assert _name(server, back) == "bob002"


def test_a_full_server_lets_a_reconnecting_player_back() -> None:
    server = GameServer(ServerConfig(max_players=2, reconnect_silent_seconds=1.0), map_text=ROOM)
    ann = _join(server, "ann")
    _join(server, "bob")
    _talk(server, ann, ticks=61)  # bob's gone quiet
    assert isinstance(server.join(Hello(PROTOCOL_VERSION, "cat")), Reject)  # full
    back = _join(server, "bob")  # bob, back: his old self doesn't count
    assert _name(server, back) == "bob"


def test_the_client_is_told_its_name() -> None:
    server = ServerThread(GameServer(map_text=ROOM))
    try:
        sessions = [NetworkSession.join("127.0.0.1", server.port, n) for n in ("bob", "bob")]
        assert [s.renamed() for s in sessions] == [None, "bob002"]
        for s in sessions:
            s.close()
    finally:
        server.stop()

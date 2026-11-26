"""The server's core, without sockets: joining, inputs (one per tick,
queued, held when late), leaving and timeouts, updates, and the lift
taking everyone on to the next floor."""

import math

from ok_doomer.server.config import ServerConfig
from ok_doomer.server.game import GameServer
from ok_doomer.shared.building import floor_map_text
from ok_doomer.shared.events import LevelComplete, SwitchThrown
from ok_doomer.shared.mapfile import parse_map
from ok_doomer.shared.player import PlayerInput
from ok_doomer.shared.protocol import (
    PROTOCOL_VERSION,
    Bye,
    Hello,
    Input,
    Message,
    NewFloor,
    Reject,
    Update,
    Welcome,
    decode,
    encode,
)
from ok_doomer.shared.sim import RIDE_SECONDS
from ok_doomer.shared.snapshot import apply, capture
from ok_doomer.shared.switches import POWER
from ok_doomer.shared.world import new_world

SOUTH = math.pi / 2

# test_power.py's room: the breaker on the south wall at x = 2.5.
ROOM = """\
[walls]
############
#..........#
#..........#
#..........#
############

[things]
coop_start 2.5 1.5 90
exit 10.5 1.5 0
breaker 2.5 3.99 270
"""
AT_BREAKER = (2.5, 3.2)


def _server(**config: object) -> GameServer:
    return GameServer(ServerConfig(**config), map_text=ROOM)  # type: ignore[arg-type]


def _join(server: GameServer, name: str = "ann") -> int:
    welcome = server.join(Hello(PROTOCOL_VERSION, name))
    assert isinstance(welcome, Welcome)
    return welcome.player_id


def _updates(out: list[Message]) -> list[Update]:
    return [m for m in out if isinstance(m, Update)]


def test_joining_gets_the_map_and_a_player() -> None:
    server = _server()
    welcome = server.join(Hello(PROTOCOL_VERSION, "  Ann   Smith "))
    assert isinstance(welcome, Welcome)
    assert (welcome.player_id, welcome.mode, welcome.map_text) == (1, "coop", ROOM)
    assert welcome.config == server.world.config
    assert server.world.players[1].name == "Ann Smith"
    assert _join(server, "bob") == 2


def test_wrong_version_or_full_is_rejected() -> None:
    server = _server(max_players=1)
    assert isinstance(server.join(Hello(PROTOCOL_VERSION + 1, "x")), Reject)
    _join(server)
    full = server.join(Hello(PROTOCOL_VERSION, "y"))
    assert isinstance(full, Reject) and "full" in full.reason
    assert list(server.world.players) == [1]


def test_names_are_tidied_and_never_empty() -> None:
    server = _server()
    _join(server, "   ")
    _join(server, "x" * 40)
    assert server.world.players[1].name == "player 1"
    assert len(server.world.players[2].name) == 16


def test_one_input_a_tick_and_the_ack_follows() -> None:
    server = _server()
    pid = _join(server)
    for seq in (1, 2, 3):
        server.receive(pid, Input(seq, PlayerInput(forward=1.0, angle=0.0)))
    acks = [_updates(server.tick()[pid])[0].ack for _ in range(4)]
    assert acks == [1, 2, 3, 3]


def test_a_tap_shorter_than_a_tick_isnt_lost() -> None:
    """Use pressed and released between two ticks: both inputs are queued,
    so the press reaches the simulation."""
    server = _server()
    pid = _join(server)
    server.world.players[pid].x, server.world.players[pid].y = AT_BREAKER
    server.receive(pid, Input(1, PlayerInput(angle=SOUTH, use=True)))
    server.receive(pid, Input(2, PlayerInput(angle=SOUTH)))
    events = _updates(server.tick()[pid])[0].events
    assert any(isinstance(e, SwitchThrown) for e in events)
    assert POWER in server.world.flags


def test_stale_and_repeated_inputs_are_ignored() -> None:
    server = _server()
    pid = _join(server)
    server.receive(pid, Input(5, PlayerInput(forward=1.0)))
    server.receive(pid, Input(5, PlayerInput(forward=-1.0)))  # a repeat
    server.receive(pid, Input(4, PlayerInput(forward=-1.0)))  # late: older than 5
    assert [m.seq for m in server.clients[pid].inputs] == [5]


def test_flooding_drops_the_oldest_inputs() -> None:
    server = _server(input_queue=3)
    pid = _join(server)
    for seq in range(1, 11):
        server.receive(pid, Input(seq, PlayerInput()))
    assert [m.seq for m in server.clients[pid].inputs] == [8, 9, 10]


def test_a_late_input_is_held_then_the_player_stands_still() -> None:
    server = _server(hold_ticks=3)
    pid = _join(server)
    p = server.world.players[pid]
    server.receive(pid, Input(1, PlayerInput(forward=1.0, angle=0.0)))
    xs = []
    for _ in range(6):
        server.tick()
        xs.append(p.x)
    moves = [b - a for a, b in zip(xs, xs[1:], strict=False)]
    assert all(m > 0 for m in moves[:3])  # 1 tick of input, then 3 held
    assert moves[3:] == [0.0, 0.0]  # then standing


def test_leaving_removes_the_player() -> None:
    server = _server()
    a, b = _join(server), _join(server, "bob")
    server.receive(b, Bye("quit"))
    assert list(server.world.players) == [a] and list(server.clients) == [a]
    assert list(server.tick()) == [a]
    server.receive(b, Input(1, PlayerInput()))  # after leaving: ignored
    server.leave(b)  # twice: fine


def test_a_silent_client_times_out() -> None:
    server = _server(timeout_seconds=0.5)  # 30 ticks
    quiet, chatty = _join(server), _join(server, "bob")
    for seq in range(1, 40):
        server.receive(chatty, Input(seq, PlayerInput()))
        out = server.tick()
        if quiet not in server.clients:
            break
    assert seq == 31
    assert out[quiet] == [Bye("timed out")]
    assert list(server.world.players) == [chatty]


def test_timeouts_still_count_after_a_floor_change() -> None:
    """The new floor's world starts again from tick 0; timeouts don't."""
    server = GameServer(ServerConfig(storey="G", timeout_seconds=0.5))
    quiet, chatty = _join(server), _join(server, "bob")
    for seq in range(1, 101):  # both busy for 100 ticks
        server.receive(quiet, Input(seq, PlayerInput()))
        server.receive(chatty, Input(seq, PlayerInput()))
        server.tick()
    server.world.ride = RIDE_SECONDS - server.world.config.dt / 2
    server.receive(chatty, Input(101, PlayerInput()))
    server.tick()
    assert server.world.map.storey == "1"
    for seq in range(102, 140):  # now only bob talks
        server.receive(chatty, Input(seq, PlayerInput()))
        server.tick()
    assert list(server.clients) == [chatty]


def test_every_client_gets_the_same_snapshot_and_events() -> None:
    server = _server()
    a, b = _join(server), _join(server, "bob")
    server.receive(a, Input(1, PlayerInput()))
    out = server.tick()
    (ua,), (ub,) = _updates(out[a]), _updates(out[b])
    assert ua.snapshot == ub.snapshot and ua.events == ub.events
    assert (ua.ack, ub.ack) == (1, 0)  # each its own
    assert len([m for m in out[a] if isinstance(m, Update)]) == 1


def test_a_client_keeps_its_world_in_step_through_the_wire() -> None:
    """A client building its world from the Welcome and applying each
    Update (encoded and decoded, as over TCP) has the server's world."""
    server = _server()
    welcome = server.join(Hello(PROTOCOL_VERSION, "ann"))
    assert isinstance(welcome, Welcome)
    client_world = new_world(parse_map(welcome.map_text), welcome.mode, welcome.config)
    for seq in range(1, 90):
        server.receive(1, decode(encode(Input(seq, PlayerInput(forward=1.0, angle=seq * 0.1)))))
        for message in server.tick()[1]:
            received = decode(encode(message))
            assert isinstance(received, Update)
            apply(client_world, received.snapshot)
    assert capture(client_world) == capture(server.world)


def test_the_lift_takes_everyone_to_the_next_floor() -> None:
    server = GameServer(ServerConfig(storey="G"))
    a, b = _join(server), _join(server, "bob")
    server.world.players[a].health = 40
    for seq in (1, 2, 3):  # 3 inputs, 2 ticks: one's still queued when the lift arrives
        server.receive(a, Input(seq, PlayerInput(forward=1.0)))
    server.tick()
    # Everyone's aboard and the ride's nearly over.
    server.world.ride = RIDE_SECONDS - server.world.config.dt / 2
    out = server.tick()
    for pid in (a, b):
        kinds = [type(m) for m in out[pid]]
        assert kinds == [NewFloor, Update]  # the map first, then the floor's first update
    new_floor = out[a][0]
    assert isinstance(new_floor, NewFloor) and new_floor.map_text == floor_map_text("1")
    update = out[a][1]
    assert isinstance(update, Update)
    assert any(isinstance(e, LevelComplete) for e in update.events)
    assert server.world.map.storey == "1" and server.map_text == floor_map_text("1")
    assert server.world.players[a].health == 40 and sorted(server.world.players) == [a, b]
    # The input meant for the ground floor is dropped, and acked.
    assert update.ack == 3 and not server.clients[a].inputs
    # The update is of the new floor: it applies onto a world built from its map.
    apply(new_world(parse_map(new_floor.map_text), "coop"), update.snapshot)


def test_same_seed_same_inputs_same_game() -> None:
    def play() -> list[object]:
        server = GameServer(ServerConfig(mode="coop"), seed=4)
        _join(server)
        _join(server, "bob")
        sent: list[object] = []
        for seq in range(1, 60):
            server.receive(1, Input(seq, PlayerInput(forward=1.0, angle=seq * 0.05)))
            server.receive(2, Input(seq, PlayerInput(strafe=1.0, use=seq % 7 == 0)))
            sent.append(server.tick())
        return sent

    assert play() == play()

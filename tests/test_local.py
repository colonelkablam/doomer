"""Single-player on the in-process server: the same game as stepping the
world directly, no latency, events through to the game, and leaving."""

import math

from ok_doomer.client.local import LocalSession
from ok_doomer.shared.events import SwitchThrown
from ok_doomer.shared.mapfile import parse_map
from ok_doomer.shared.player import PlayerInput
from ok_doomer.shared.sim import step
from ok_doomer.shared.spawn import spawn_player
from ok_doomer.shared.switches import POWER
from ok_doomer.shared.world import new_world

DT = 1 / 60
SOUTH = math.pi / 2
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


def _inputs(n: int) -> list[PlayerInput]:
    return [
        PlayerInput(forward=1.0, strafe=math.sin(i * 0.1), angle=i * 0.02, jump=i % 40 == 0)
        for i in range(n)
    ]


def test_it_plays_exactly_as_stepping_the_world() -> None:
    """Through the server, snapshots and prediction, and back: the same
    numbers as calling `step` directly."""
    world = new_world(parse_map(ROOM), "coop")
    spawn_player(world, 1, "p")
    session = LocalSession(ROOM, "p")
    for player_input in _inputs(240):
        step(world, {1: player_input})
        session.update(DT, player_input)
    direct = world.players[1]
    cam = session.camera(0.0)
    assert (session.player.x, session.player.y, session.player.z) == (direct.x, direct.y, direct.z)
    assert (cam.x, cam.y) == (direct.x, direct.y)  # drawn there too (a whole tick in)
    assert session.predictor.corrections == 0


def test_no_latency_the_server_has_each_input_at_once() -> None:
    session = LocalSession(ROOM, "p")
    session.update(DT, PlayerInput(forward=1.0, angle=0.0))
    assert session.world.tick == 1 and session.mirror.ack == session.mirror.seq == 1
    assert session.mirror.player == session.player  # the drawn copy's up to date


def test_events_reach_the_game() -> None:
    session = LocalSession(ROOM, "p")
    session.player.x, session.player.y = 2.5, 3.2  # at the breaker
    session.update(DT, PlayerInput(angle=SOUTH))
    events = session.update(DT, PlayerInput(angle=SOUTH, use=True))
    assert [e.flag for e in events if isinstance(e, SwitchThrown)] == [POWER]
    assert session.power()  # and the drawn world has the power on


def test_its_a_server_with_one_client_who_leaves() -> None:
    session = LocalSession(ROOM, "ann")
    assert [c.name for c in session.server.clients.values()] == ["ann"]
    assert session.ended() is None
    session.close()
    assert not session.server.clients and not session.server.world.players


def test_a_long_frame_catches_up_as_single_player_always_did() -> None:
    session = LocalSession(ROOM, "p")
    session.update(5 * DT + DT / 2, PlayerInput(forward=1.0, angle=0.0))
    assert session.world.tick == 5  # five ticks in one frame, not capped at 2

"""Client-side prediction: you move at once whatever the latency, the server
and the prediction agree when nothing's in dispute, and when they don't,
the correction glides rather than snaps. A NetworkSession on a link
straight to a GameServer, with made-up latency, tick by tick."""

import math
import queue
from collections import deque

import pytest

from ok_doomer.client.network import NetworkSession
from ok_doomer.server.config import ServerConfig
from ok_doomer.server.game import GameServer
from ok_doomer.shared.player import PlayerInput
from ok_doomer.shared.prediction import CORRECT_SECONDS, SNAP_DISTANCE, Predictor
from ok_doomer.shared.protocol import PROTOCOL_VERSION, Hello, Message, Welcome

DT = 1 / 60
ROOM = """\
[walls]
############
#..........#
#..........#
#..........#
############

[things]
coop_start 2.5 2.0 0
exit 10.5 1.5 0
breaker 6.5 3.99 270
"""


class LaggyLink:
    """A NetworkSession's link to a GameServer in the same process, each
    message taking `latency` ticks each way. `tick()` runs the server one
    tick and delivers whatever's due."""

    def __init__(self, server: GameServer, latency: int) -> None:
        self.server, self.latency = server, latency
        welcome = server.join(Hello(PROTOCOL_VERSION, "ann"))
        assert isinstance(welcome, Welcome)
        self.welcome = welcome
        self.inbox: queue.Queue[Message | None] = queue.Queue()
        self.now = 0
        self.to_server: deque[tuple[int, Message]] = deque()
        self.to_client: deque[tuple[int, Message]] = deque()

    def send(self, message: Message) -> None:
        self.to_server.append((self.now + self.latency, message))

    def received(self) -> list[Message | None]:
        out: list[Message | None] = []
        while self.to_client and self.to_client[0][0] <= self.now:
            out.append(self.to_client.popleft()[1])
        return out

    def close(self, reason: str = "quit") -> None:
        pass

    def tick(self) -> None:
        while self.to_server and self.to_server[0][0] <= self.now:
            self.server.receive(self.welcome.player_id, self.to_server.popleft()[1])
        for message in self.server.tick().get(self.welcome.player_id, []):
            self.to_client.append((self.now + self.latency, message))
        self.now += 1


def _session(latency: int, predict: bool = True) -> tuple[NetworkSession, LaggyLink]:
    link = LaggyLink(GameServer(ServerConfig(), map_text=ROOM), latency)
    session = NetworkSession(link, predict)
    for _ in range(latency * 2 + 2):  # until the first update's in
        link.tick()
        session.update(DT, PlayerInput())
    assert session.mirror.player is not None
    return session, link


def _frame(session: NetworkSession, link: LaggyLink, player_input: PlayerInput) -> float:
    """One tick: the client sends, the server ticks, the client takes what's
    arrived. Where the camera is, after."""
    session.update(DT, player_input)
    link.tick()
    return session.camera(0.0).x


def _frames_to_move(session: NetworkSession, link: LaggyLink) -> int:
    x0 = session.camera(0.0).x
    frames = 1
    while _frame(session, link, PlayerInput(forward=1.0, angle=0.0)) == x0:
        frames += 1
    return frames


def _local_frames_to_move() -> int:
    """Single-player, the same room: how soon a key press shows."""
    from ok_doomer.client.session import LocalSession
    from ok_doomer.shared.mapfile import parse_map
    from ok_doomer.shared.world import new_world

    session = LocalSession(new_world(parse_map(ROOM), "coop"), "ann")
    x0 = session.camera(0.0).x
    frames = 1
    while True:
        session.update(DT, PlayerInput(forward=1.0, angle=0.0))
        if session.camera(0.0).x != x0:
            return frames
        frames += 1


@pytest.mark.parametrize("latency", [0, 3, 6, 12])  # up to 200 ms each way
def test_you_move_as_soon_as_in_single_player_whatever_the_latency(latency: int) -> None:
    session, link = _session(latency)
    assert _frames_to_move(session, link) == _local_frames_to_move() <= 2


@pytest.mark.parametrize("latency", [3, 6])
def test_without_prediction_you_wait_for_the_round_trip(latency: int) -> None:
    session, link = _session(latency, predict=False)
    assert _frames_to_move(session, link) >= 2 * latency


@pytest.mark.parametrize("latency", [0, 6])
def test_walking_about_the_server_agrees_with_the_prediction(latency: int) -> None:
    """Walking, turning, into a wall and along it, jumping, crouching: the
    same code both sides, so no corrections, and you end up exactly where
    the server has you."""
    session, link = _session(latency)
    script = (
        [PlayerInput(forward=1.0, angle=0.0)] * 120  # east, into the wall, stopped
        + [PlayerInput(forward=1.0, strafe=1.0, angle=0.3)] * 40  # sliding along it
        + [PlayerInput(forward=1.0, angle=math.pi, jump=True)] * 30  # back west, jumping
        + [PlayerInput(forward=-1.0, angle=2.0, crouch=True)] * 30
        + [PlayerInput()] * (2 * latency + 4)  # stand while it all arrives
    )
    for player_input in script:
        _frame(session, link, player_input)
    assert session.predictor.corrections == 0
    server_me = link.server.world.players[link.welcome.player_id]
    cam = session.camera(0.0)
    assert (cam.x, cam.y) == (server_me.x, server_me.y)


def test_a_disagreement_glides_rather_than_snaps() -> None:
    """The server puts you 0.3 cells from where you thought: the drawn
    position doesn't jump, then glides there over ~CORRECT_SECONDS."""
    session, link = _session(latency=3)
    me = link.server.world.players[link.welcome.player_id]
    before = session.camera(0.0).x
    me.x += 0.3  # the server's world moves you (a push, a door...)
    xs = [_frame(session, link, PlayerInput()) for _ in range(60)]
    assert session.predictor.corrections == 1
    steps = [b - a for a, b in zip([before, *xs], xs, strict=False)]
    assert max(abs(s) for s in steps) < 0.3 / 3  # no frame jumps far
    assert xs[-1] == pytest.approx(me.x, abs=1e-3)  # it gets there
    halfway = round(CORRECT_SECONDS / DT) + 3 * 2  # a fade time after it arrived
    assert 0.0 < xs[halfway] - before < 0.3


def test_a_big_disagreement_snaps() -> None:
    session, link = _session(latency=3)
    me = link.server.world.players[link.welcome.player_id]
    me.x += SNAP_DISTANCE * 3  # teleported
    for _ in range(8):
        _frame(session, link, PlayerInput())
    assert session.camera(0.0).x == pytest.approx(me.x)


def test_the_prediction_never_touches_the_world() -> None:
    """Use and pickups are the server's: predicting a press of use changes
    nothing in the client's copy of the world."""
    session, link = _session(latency=6)
    world = session.mirror.world
    assert session.predictor.player is not None
    session.predictor.player.x, session.predictor.player.y = 6.5, 3.2  # at the breaker
    session.update(DT, PlayerInput(angle=math.pi / 2, use=True))  # facing it, pressing use
    assert "power" not in world.flags and not world.switches[0].on


def test_a_predictor_with_no_player_waits() -> None:
    predictor = Predictor()
    assert predictor.position(0.5) is None
    predictor.fade(1.0)  # nothing to fade: fine

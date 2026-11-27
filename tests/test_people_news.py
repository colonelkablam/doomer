"""Who's here: join and leave messages, and names over heads (only for
people you can see)."""

import math
from collections.abc import Sequence

import pytest

from ok_doomer.client.config import RenderConfig
from ok_doomer.client.headless import use_dummy_video
from ok_doomer.client.hud import message_for, name_tags
from ok_doomer.client.network import NetworkSession
from ok_doomer.client.render import Camera, Renderer, SpriteInstance, make_frame_surface
from ok_doomer.client.sprites import PLAYER_SPRITE
from ok_doomer.server.config import ServerConfig
from ok_doomer.server.game import GameServer
from ok_doomer.shared.events import Event, PlayerJoined, PlayerLeft
from ok_doomer.shared.mapfile import parse_map
from ok_doomer.shared.player import PlayerInput
from ok_doomer.shared.protocol import PROTOCOL_VERSION, Bye, Hello, Input, Update, Welcome
from tests.test_prediction import LaggyLink

use_dummy_video()

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
"""


def _events(out: Sequence[object]) -> list[Event]:
    return [e for m in out if isinstance(m, Update) for e in m.events]


def test_the_server_says_who_joins_and_who_leaves_and_why() -> None:
    server = GameServer(ServerConfig(timeout_seconds=0.1), map_text=ROOM)
    ann = server.join(Hello(PROTOCOL_VERSION, "ann"))
    bob = server.join(Hello(PROTOCOL_VERSION, "bob"))
    assert isinstance(ann, Welcome) and isinstance(bob, Welcome)
    joined = _events(server.tick()[ann.player_id])
    assert [(e.player_id, e.name) for e in joined if isinstance(e, PlayerJoined)] == [
        (1, "ann"),
        (2, "bob"),
    ]
    server.receive(bob.player_id, Bye("quit"))
    left = [e for e in _events(server.tick()[ann.player_id]) if isinstance(e, PlayerLeft)]
    assert [(e.name, e.reason) for e in left] == [("bob", "quit")]
    for _ in range(20):  # ann goes quiet too: nobody left to tell but the log
        out = server.tick()
    assert not server.clients and out == {}


def test_a_time_out_is_news_to_everyone_else() -> None:
    server = GameServer(ServerConfig(timeout_seconds=0.1), map_text=ROOM)  # 6 ticks
    server.join(Hello(PROTOCOL_VERSION, "ann"))
    server.join(Hello(PROTOCOL_VERSION, "bob"))
    news: list[Event] = []
    for seq in range(1, 12):
        server.receive(1, Input(seq, PlayerInput()))  # ann keeps talking; bob doesn't
        news += _events(server.tick().get(1, []))
    left = [e for e in news if isinstance(e, PlayerLeft)]
    assert [(e.name, e.reason) for e in left] == [("bob", "timed out")]


def test_the_hud_says_it() -> None:
    assert message_for(PlayerJoined(0, 2, "Bob")) == "Bob joined"
    assert message_for(PlayerLeft(0, 2, "Bob", "quit")) == "Bob left"
    assert message_for(PlayerLeft(0, 2, "Bob", "timed out")) == "Bob timed out"
    assert message_for(PlayerLeft(0, 2, "Bob", "disconnected")) == "Bob lost connection"
    assert message_for(PlayerLeft(0, 2, "Bob", "bot done")) == "Bob left"  # any other reason


def test_you_hear_of_others_joining_not_yourself() -> None:
    link = LaggyLink(GameServer(ServerConfig(), map_text=ROOM), latency=0)
    session = NetworkSession(link)
    news: list[Event] = []
    for _ in range(3):
        news += session.update(1 / 60, PlayerInput())
        link.tick()
    link.server.join(Hello(PROTOCOL_VERSION, "bob"))
    for _ in range(3):
        news += session.update(1 / 60, PlayerInput())
        link.tick()
    assert [e.name for e in news if isinstance(e, PlayerJoined)] == ["bob"]
    session.sprites()  # a frame drawn (as the game loop does)
    assert [p[0] for p in session.people()] == ["bob"]  # and he's there to name


# A wall across the room at x = 6, with a doorway in the bottom row (y = 3).
WALLED = """\
[walls]
############
#.....#....#
#.....#....#
#..........#
############

[things]
coop_start 2.5 2.0 0
exit 10.5 1.5 0
"""


def _tags(
    people: list[tuple[str, float, float]], camera: Camera, map_text: str = WALLED
) -> list[str]:
    game_map = parse_map(map_text)
    config = RenderConfig()
    renderer = Renderer(config)
    sprites = [SpriteInstance(PLAYER_SPRITE, x, y) for _, x, y in people]
    renderer.render(make_frame_surface(config), game_map, camera, sprites)
    return [t.name for t in name_tags(renderer, people)]


def test_names_over_the_heads_of_people_you_can_see() -> None:
    east = Camera(1.5, 2.0, 0.0)  # looking east along the room
    assert _tags([("ann", 4.5, 2.0)], east) == ["ann"]
    assert _tags([("bob", 8.5, 1.5)], east) == []  # behind the wall
    in_line = Camera(1.5, 3.5, 0.0)  # in line with the doorway
    assert _tags([("cat", 8.5, 3.5)], in_line) == ["cat"]  # seen through it
    west = Camera(5.5, 2.0, math.pi)  # looking west: ann's behind the camera's back
    assert _tags([("ann", 7.5, 2.0)], west) == []


def test_no_names_far_off(monkeypatch: pytest.MonkeyPatch) -> None:
    from ok_doomer.client import hud

    camera = Camera(1.5, 3.5, 0.0)
    people = [("ann", 4.5, 3.5), ("bob", 9.5, 3.5)]  # 3 and 8 cells off
    monkeypatch.setattr(hud, "NAME_DISTANCE", 5.0)
    assert _tags(people, camera) == ["ann"]


def test_behind_the_counter_the_name_sits_on_what_shows() -> None:
    """The lobby's reception counter: someone standing just behind it shows
    from the chest up, and the name sits just over their head."""
    from ok_doomer.client.hud import NAME_GAP
    from ok_doomer.shared.building import floor_map_text

    game_map = parse_map(floor_map_text("G"))
    desk = game_map.things_of_type("reception_desk")[0]
    camera = Camera(desk.x - 3.0, desk.y, 0.0, z=0.6)  # in front of it, facing it
    x, y = desk.x + 0.6, desk.y  # just behind it
    config = RenderConfig()
    renderer = Renderer(config)
    sprite = SpriteInstance(PLAYER_SPRITE, x, y, 0.0, math.pi)
    renderer.render(make_frame_surface(config), game_map, camera, [sprite])
    (tag,) = name_tags(renderer, [("dev", x, y)])
    seen = renderer.seen[(x, y)]
    assert tag.row == seen.top - NAME_GAP

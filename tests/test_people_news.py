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


def test_names_are_drawn_in_green_pixel_capitals() -> None:
    """Into the frame (before scaling): the 3x5 font in phosphor green on a
    black panel, capitals, the bottom middle at the tag; clipped at the edge."""
    import numpy as np

    from ok_doomer.client.hud import NAME_BACK, NAME_COLOUR, NameTag, draw_name_tags

    pixels = np.full((320, 200), 7, dtype=np.uint8)
    draw_name_tags(pixels, [NameTag("bob", 100.0, 50.0)])
    green = np.argwhere(pixels == NAME_COLOUR)
    (u0, v0), (u1, v1) = green.min(axis=0), green.max(axis=0)
    assert (u1 - u0 + 1, v1 - v0 + 1) == (11, 5)  # "BOB": 3 letters of 3, gaps of 1; 5 high
    assert v1 == 49 and abs((u0 + u1) / 2 - 100) <= 1  # its bottom middle at the tag
    assert (pixels == NAME_BACK).sum() == 13 * 7 - len(green)  # a panel round it
    draw_name_tags(pixels, [NameTag("x" * 16, 2.0, 3.0)], scale=2)  # off the edge: clipped
    draw_name_tags(pixels, [NameTag("éé", 50.0, 50.0)])  # nothing it can draw: nothing


def test_on_joining_you_hear_whos_already_here() -> None:
    from ok_doomer.shared.events import PlayersHere

    server = GameServer(ServerConfig(), map_text=ROOM)
    server.join(Hello(PROTOCOL_VERSION, "Bob"))
    server.join(Hello(PROTOCOL_VERSION, "Cath"))
    link = LaggyLink(server, latency=0)  # ann joins third
    session = NetworkSession(link)
    news: list[Event] = []
    for _ in range(3):
        news += session.update(1 / 60, PlayerInput())
        link.tick()
    (here,) = [e for e in news if isinstance(e, PlayersHere)]
    assert here.names == ("Bob", "Cath")
    assert message_for(here) == "Here: Bob, Cath"
    assert not [e for e in news if isinstance(e, PlayerJoined)]  # not old news as new


def test_two_joining_at_once_both_show() -> None:
    """Bots started together join within a tick: both messages stack."""
    import numpy as np

    from ok_doomer.client.hud import MAX_MESSAGES, MESSAGE_SECONDS, Hud

    hud = Hud()
    hud.handle([PlayerJoined(5, 2, "bot 1"), PlayerJoined(5, 3, "bot 2")], now=10.0)
    assert hud.messages(10.0) == ["bot 1 joined", "bot 2 joined"]
    hud.handle([PlayerLeft(9, 2, "bot 1", "quit")], now=11.0)
    assert hud.messages(11.0) == ["bot 1 joined", "bot 2 joined", "bot 1 left"]
    assert hud.messages(10.0 + MESSAGE_SECONDS + 0.1) == ["bot 1 left"]  # each its own time
    hud.handle([PlayerJoined(9, i, f"p{i}") for i in range(10)], now=12.0)
    assert len(hud.messages(12.0)) == MAX_MESSAGES == 6  # the latest few
    fewer = Hud(max_messages=2)  # the setting
    fewer.handle([PlayerJoined(9, i, f"p{i}") for i in range(5)], now=12.0)
    assert fewer.messages(12.0) == ["p3 joined", "p4 joined"]
    hud.draw_frame(np.zeros((320, 200), dtype=np.uint8), 12.0)  # and they draw


def test_messages_are_drawn_in_5x7_green_stacked_and_wrapped() -> None:
    import numpy as np

    from ok_doomer.client.hud import NAME_COLOUR, Hud, wrap
    from ok_doomer.client.pixelart import FONT_5X7

    hud = Hud()
    hud.handle([PlayerJoined(1, 2, "Bob"), PlayerJoined(1, 3, "Cath")], now=0.0)
    pixels = np.zeros((320, 200), dtype=np.uint8)
    hud.draw_frame(pixels, now=1.0)
    rows = np.nonzero((pixels == NAME_COLOUR).any(axis=0))[0]
    lines = np.split(rows, np.nonzero(np.diff(rows) > 1)[0] + 1)
    assert [len(line) for line in lines] == [7, 7]  # two lines of 5x7 text
    assert lines[0][0] >= 200 // 6  # from a sixth of the way down
    long = "Here: " + ", ".join(f"Player {i}" for i in range(12))
    wrapped = wrap(long, FONT_5X7, 312)
    assert len(wrapped) > 1 and " ".join(wrapped) == long
    assert all(len(line) * 6 - 1 <= 312 for line in wrapped)


def test_every_message_can_be_written_in_the_5x7_font() -> None:
    """No HUD message loses letters to the font: every character any of
    them can produce is in it."""
    import itertools
    import typing
    from dataclasses import fields

    from ok_doomer.client.pixelart import FONT_5X7
    from ok_doomer.shared.dispensers import DISPENSERS
    from ok_doomer.shared.protocol import EVENT_TYPES

    samples: dict[object, list[object]] = {
        int: [1],
        float: [0.5],
        bool: [True, False],
        str: ["Bob"],
        str | None: ["1", None],
        tuple[str, ...]: [("Bob", "Cath")],
    }
    texts = set()
    for cls in EVENT_TYPES.values():
        hints = typing.get_type_hints(cls)
        names = [f.name for f in fields(cls)]
        options: list[list[typing.Any]] = []
        for name in names:
            if name == "dispenser":
                options.append(list(DISPENSERS))
            elif name in ("meter",):
                options.append(["health", "coffee"])
            elif name in ("reason",):
                options.append(["", "full", "empty", "no power", "timed out", "quit"])
            elif name in ("colour", "lock"):
                options.append(["blue", "red", "gold"])
            elif name == "flag":
                options.append(["power"])
            else:
                options.append(samples[hints[name]])
        for combo in itertools.product(*options):
            values: dict[str, typing.Any] = dict(zip(names, combo, strict=True))
            text = message_for(cls(**values))
            if text:
                texts.add(text)
    assert len(texts) > 15
    missing = {c for t in texts for c in t.upper()} - set(FONT_5X7)
    assert not missing, missing


def test_titles_and_the_banner_are_pixel_text_too() -> None:
    """A floor's name on arrival (twice the size), messages under it; the
    "complete" banner across a black band. Every floor and map name can be
    written in the font."""
    import numpy as np

    from ok_doomer.client.hud import NAME_BACK, NAME_COLOUR, Hud
    from ok_doomer.client.pixelart import FONT_5X7
    from ok_doomer.shared.building import FLOOR_LABELS, floor_name
    from ok_doomer.shared.events import LevelComplete, PlayersHere
    from ok_doomer.shared.mapfile import load_map
    from tests.test_mapfile import MAPS

    hud = Hud()
    hud.arrive(parse_map(ROOM), 0.0)
    hud.handle([PlayersHere(0, ("Bob",))], 0.1)
    pixels = np.zeros((320, 200), dtype=np.uint8)
    hud.draw_frame(pixels, 1.0)
    rows = np.nonzero((pixels == NAME_COLOUR).any(axis=0))[0]
    runs = [len(r) for r in np.split(rows, np.nonzero(np.diff(rows) > 1)[0] + 1)]
    assert runs == [14, 7, 7]  # the floor (5x7 at twice), the map's name, the message
    same = Hud()  # a map named as its floor is: the name once
    same.arrive(parse_map("name: Ground Floor\n" + ROOM), 0.0)
    pixels[:] = 0
    same.draw_frame(pixels, 1.0)
    assert (pixels == NAME_COLOUR).any(axis=0).sum() == 14

    hud.handle([LevelComplete(5, None)], 2.0)
    pixels[:] = 0
    hud.draw_frame(pixels, 2.0)
    assert (pixels[:, 100] == NAME_BACK).sum() + (pixels[:, 100] == NAME_COLOUR).sum() == 320

    names = [floor_name(label) for label in FLOOR_LABELS]
    names += [load_map(path).name for path in MAPS.glob("*.map")]
    missing = {c for name in names for c in name.upper()} - set(FONT_5X7)
    assert not missing, missing


def test_your_hud_tells_you_your_news_and_the_teams_not_others_doings() -> None:
    from ok_doomer.client.hud import Hud, news_for
    from ok_doomer.shared.events import (
        Dispensed,
        DoorLocked,
        LiftNoPower,
        PassPickedUp,
        SwitchThrown,
    )

    me, bob = 1, 2
    names = {1: "Ann", 2: "Bob"}
    theirs = [
        DoorLocked(5, 0, bob, "red"),
        Dispensed(5, bob, "water_cooler", "health", 5, 20),
        LiftNoPower(5, 0, bob),
    ]
    for event in theirs:
        assert news_for(event, me, names) is None  # a bot's stuck at a door: not your news
    assert news_for(DoorLocked(5, 0, me, "red"), me, names) == "Locked: you need the red pass"
    assert news_for(PassPickedUp(5, bob, "blue"), me, names) == "Bob picked up the blue pass"
    assert news_for(PassPickedUp(5, me, "blue"), me, names) == "Picked up the blue pass"
    assert news_for(SwitchThrown(5, 0, bob, "power"), me, names) == "The power is back on"
    assert news_for(PlayerJoined(5, 3, "Cath"), me, names) == "Cath joined"

    hud = Hud()
    hud.handle([DoorLocked(5, 0, bob, "red"), PlayerJoined(5, 3, "Cath")], 0.0, me, names)
    assert hud.messages(0.0) == ["Cath joined"]


def test_the_console_has_everyones_doings_with_who() -> None:
    from ok_doomer.client.console import LINES_KEPT, Console
    from ok_doomer.client.hud import console_line
    from ok_doomer.shared.events import DoorLocked, PlayerSpawned

    names = {1: "Ann", 2: "Bob"}
    assert (
        console_line(DoorLocked(7, 0, 2, "red"), names) == "[7] Bob: Locked: you need the red pass"
    )
    assert console_line(PlayerJoined(8, 2, "Bob"), names) == "[8] Bob joined"
    assert (
        console_line(PlayerSpawned(0, 9, 1.5, 2.0), names)
        == "[0] player 9: PlayerSpawned x=1.5 y=2.0"
    )
    console = Console()
    for i in range(LINES_KEPT + 5):
        console.add(f"line {i}")
    assert len(console.lines) == LINES_KEPT and console.lines[-1] == f"line {LINES_KEPT + 4}"


def test_names_are_remembered_after_leaving() -> None:
    link = LaggyLink(GameServer(ServerConfig(), map_text=ROOM), latency=0)
    session = NetworkSession(link)
    bob = link.server.join(Hello(PROTOCOL_VERSION, "Bob"))
    assert isinstance(bob, Welcome)
    for _ in range(3):
        session.update(1 / 60, PlayerInput())
        link.tick()
    link.server.receive(bob.player_id, Bye("quit"))
    for _ in range(3):
        session.update(1 / 60, PlayerInput())
        link.tick()
    assert session.me() == 1 and session.names() == {1: "ann", 2: "Bob"}
    assert bob.player_id not in session.mirror.world.players  # gone, but still named

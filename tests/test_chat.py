"""Chat: T, type, Enter. The server tidies and limits it and tells
everyone (the sender too) with a Chat event; the HUD shows it longer than
other news; typing doesn't move you."""

import asyncio

import numpy as np
import pygame

from ok_doomer.client.app import Display, leave_warning, play
from ok_doomer.client.chat import ChatLine
from ok_doomer.client.config import ClientConfig, RenderConfig
from ok_doomer.client.hud import CHAT_SECONDS, MESSAGE_SECONDS, NAME_COLOUR, Hud, console_line
from ok_doomer.client.local import LocalSession
from ok_doomer.server.config import ServerConfig
from ok_doomer.server.game import GameServer, tidy_chat
from ok_doomer.server.net import TcpServer
from ok_doomer.shared.events import Chat, PlayerJoined
from ok_doomer.shared.player import PlayerInput
from ok_doomer.shared.protocol import (
    MAX_CHAT,
    MAX_LINE,
    PROTOCOL_VERSION,
    Hello,
    Say,
    Update,
    Welcome,
    decode,
    encode,
)

ROOM = """\
[walls]
######
#....#
#....#
######

[things]
coop_start 2.5 2.5 0
exit 4.5 1.5 0
"""


def _chats(server: GameServer) -> dict[int, list[Chat]]:
    out = server.tick()
    return {
        pid: [e for m in msgs if isinstance(m, Update) for e in m.events if isinstance(e, Chat)]
        for pid, msgs in out.items()
    }


def _server(**config: object) -> tuple[GameServer, int, int]:
    server = GameServer(ServerConfig(**config), map_text=ROOM)  # type: ignore[arg-type]
    ann = server.join(Hello(PROTOCOL_VERSION, "ann"))
    bob = server.join(Hello(PROTOCOL_VERSION, "bob"))
    assert isinstance(ann, Welcome) and isinstance(bob, Welcome)
    return server, ann.player_id, bob.player_id


def test_everyone_hears_it_the_sender_too() -> None:
    server, ann, bob = _server()
    server.receive(ann, Say("anyone got the red pass?"))
    chats = _chats(server)
    assert chats[ann] == chats[bob] == [Chat(0, ann, "ann", "anyone got the red pass?")]


def test_its_tidied() -> None:
    assert tidy_chat("  hi\tthere \x1b[2Jbob\n ") == "hi there [2Jbob"  # no escapes for a log
    assert tidy_chat("café ☕") == "caf"
    assert len(tidy_chat("x" * 500)) == MAX_CHAT
    server, ann, _ = _server()
    server.receive(ann, Say(" \x07 "))  # nothing left: nothing said
    assert _chats(server)[ann] == []


def test_too_fast_and_its_dropped_until_the_window_passes() -> None:
    server, ann, bob = _server(chat_burst=3, chat_window_seconds=1.0)
    for i in range(5):
        server.receive(ann, Say(f"spam {i}"))
    assert [c.text for c in _chats(server)[bob]] == ["spam 0", "spam 1", "spam 2"]
    for _ in range(server.config.sim.tick_rate):
        server.tick()
    server.receive(ann, Say("sorry"))
    assert [c.text for c in _chats(server)[bob]] == ["sorry"]


def test_over_tcp() -> None:
    async def join(port: int, name: str) -> tuple[asyncio.StreamReader, asyncio.StreamWriter]:
        reader, writer = await asyncio.open_connection("127.0.0.1", port, limit=MAX_LINE + 1)
        writer.write(encode(Hello(PROTOCOL_VERSION, name)))
        assert isinstance(decode(await reader.readline()), Welcome)
        return reader, writer

    async def heard(reader: asyncio.StreamReader) -> Chat:
        while True:
            message = decode(await reader.readline())
            if isinstance(message, Update):
                for event in message.events:
                    if isinstance(event, Chat):
                        return event

    async def test() -> None:
        server = TcpServer(GameServer(ServerConfig(discovery=False), map_text=ROOM))
        await server.start("127.0.0.1", 0)
        try:
            (ra, wa), (rb, _) = await join(server.port, "ann"), await join(server.port, "bob")
            wa.write(encode(Say("hello bob")))
            for reader in (ra, rb):
                chat = await heard(reader)
                assert (chat.name, chat.text) == ("ann", "hello bob")
        finally:
            await server.close()

    asyncio.run(asyncio.wait_for(test(), 10.0))


def test_single_player_hears_itself() -> None:
    session = LocalSession(ROOM, "ann")
    try:
        session.say("is anyone there")
        events: list[object] = []
        for _ in range(3):
            events += session.update(1 / 60, PlayerInput())
        assert [e.text for e in events if isinstance(e, Chat)] == ["is anyone there"]
    finally:
        session.close()


def test_typing_a_line() -> None:
    line = ChatLine()
    assert line.key(pygame.K_a, "a") is None and not line.open  # closed: keys aren't typed
    line.start()
    for ch in "hi bobb":
        line.key(pygame.K_a, ch)
    line.key(pygame.K_BACKSPACE)
    line.key(pygame.K_a, "é")  # not in the font: not typed
    line.key(pygame.K_a, "\t")
    assert line.text == "hi bob"
    assert line.key(pygame.K_RETURN) == "hi bob" and not line.open
    line.start()
    line.key(pygame.K_a, "x")
    assert line.key(pygame.K_ESCAPE) is None and not line.open  # Esc: dropped
    line.start()
    assert line.key(pygame.K_RETURN) is None  # nothing typed: nothing sent
    line.start()
    for _ in range(MAX_CHAT + 10):
        line.key(pygame.K_a, "x")
    assert line.text == "x" * MAX_CHAT


def test_the_line_is_drawn_its_end_if_its_long() -> None:
    line = ChatLine()
    line.start()
    for _ in range(MAX_CHAT):
        line.key(pygame.K_a, "x")
    pixels = np.zeros((320, 200), dtype=np.uint8)
    line.draw(pixels)
    lit = np.nonzero((pixels == NAME_COLOUR).any(axis=1))[0]
    assert lit[0] >= 4 and lit[-1] < 320 - 2  # fits across
    assert (pixels[:, 180:] == NAME_COLOUR).any() and not (pixels[:, :150] == NAME_COLOUR).any()


def test_the_hud_shows_chat_by_name_for_longer() -> None:
    hud = Hud()
    names = {1: "ann", 2: "bob"}
    hud.handle([Chat(5, 2, "bob", "on my way"), PlayerJoined(5, 3, "cat")], 0.0, me=1, names=names)
    assert hud.messages(MESSAGE_SECONDS - 0.1) == ["bob: on my way", "cat joined"]
    assert hud.messages(CHAT_SECONDS - 0.1) == ["bob: on my way"]
    hud.handle([Chat(6, 1, "ann", "hi")], 0.0, me=1, names=names)
    assert "ann: hi" in hud.messages(0.1)  # yours too: it's what everyone saw
    assert console_line(Chat(6, 1, "ann", "hi"), names) == "[6] ann: hi"  # named once


def test_typing_in_the_game_sends_and_doesnt_walk() -> None:
    pygame.init()
    try:
        display = Display(RenderConfig(vsync=False))
        session = LocalSession(ROOM, "ann")
        said: list[str] = []
        session.say = said.append  # type: ignore[method-assign,assignment]
        start = (session.player.x, session.player.y)
        for key, ch in [(pygame.K_t, "t"), (pygame.K_w, "w"), (pygame.K_TAB, "\t")]:
            pygame.event.post(pygame.event.Event(pygame.KEYDOWN, key=key, unicode=ch, mod=0))
        for key, ch in [(pygame.K_RETURN, "\r"), (pygame.K_ESCAPE, "")]:
            pygame.event.post(pygame.event.Event(pygame.KEYDOWN, key=key, unicode=ch, mod=0))
        pygame.event.post(pygame.event.Event(pygame.QUIT))
        assert play(display, session, ClientConfig(render=RenderConfig(vsync=False)), 0.0)
        assert said == ["w"]  # T opened the line; Tab typed nothing; Enter sent
        assert (session.player.x, session.player.y) == start
        session.close()
        display.close()
    finally:
        pygame.quit()


def test_the_host_is_warned_before_leaving_ends_it_for_others() -> None:
    assert leave_warning(True, 2) == "Esc ends the game for 2 others - click to stay"
    assert leave_warning(True, 1) == "Esc ends the game for 1 other - click to stay"
    assert leave_warning(True, 0) == leave_warning(False, 3) == "Click to play - Esc to leave"

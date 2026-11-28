"""The start screen: what keys and clicks do, joining a game found or one
typed, your name remembered, and the whole loop (menu, game, menu)."""

from dataclasses import replace
from pathlib import Path

import numpy as np
import pygame
import pytest

from ok_doomer.client import menu as menu_mod
from ok_doomer.client import settings
from ok_doomer.client.app import Display
from ok_doomer.client.config import ClientConfig, RenderConfig
from ok_doomer.client.discovery import FoundGame
from ok_doomer.client.hud import NAME_COLOUR
from ok_doomer.client.menu import JoinGame, Menu, Quit, SinglePlayer, parse_address
from ok_doomer.shared.protocol import PROTOCOL_VERSION, GameInfo


@pytest.fixture(autouse=True)
def _home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Settings go to a scratch home, not the real one."""
    monkeypatch.setattr(settings, "settings_path", lambda: tmp_path / "settings.json")
    return tmp_path


def _game(name: str, address: str, version: int = PROTOCOL_VERSION) -> FoundGame:
    info = GameInfo(version, name, name, "GROUND FLOOR", "G", "coop", 1, 8, 5029)
    joinable = info if version == PROTOCOL_VERSION else None
    return FoundGame(address, version, name, joinable, 4.0, 0.0)


def _keys(menu: Menu, *keys: int | str) -> object:
    """Press keys (a str: type it); what the last chose."""
    choice = None
    for k in keys:
        if isinstance(k, str):
            for ch in k:
                choice = menu.key(pygame.K_a, ch)
        else:
            choice = menu.key(k)
    return choice


def test_single_player_and_quit() -> None:
    assert _keys(Menu("ann"), pygame.K_RETURN, pygame.K_RETURN) == SinglePlayer("coop")
    assert _keys(Menu("ann"), pygame.K_UP, pygame.K_RETURN) == Quit()  # round the top
    assert _keys(Menu("ann"), pygame.K_ESCAPE) == Quit()


def test_host_game_is_chosen() -> None:
    from ok_doomer.client.menu import HostGame

    assert _keys(Menu("ann"), pygame.K_DOWN, pygame.K_RETURN, pygame.K_RETURN) == HostGame("coop")


def test_campaign_or_deathmatch_coming() -> None:
    """Single player and host each ask which: the campaign, or deathmatch,
    greyed out until it's built; back goes back to what you'd chosen."""
    menu = Menu("ann")
    _keys(menu, pygame.K_DOWN, pygame.K_RETURN)  # HOST GAME
    assert menu.screen == "host" and menu.title() == "HOST A GAME"
    items = menu.items()
    assert [i.label for i in items] == ["CAMPAIGN", "DEATHMATCH - COMING", "BACK"]
    assert items[0].enabled and not items[1].enabled
    menu.move(1)  # past deathmatch: it can't be chosen
    assert menu.items()[menu.selected].label == "BACK"
    assert menu.activate(1) is None  # not even by the mouse
    _keys(menu, pygame.K_ESCAPE)
    assert menu.screen == "main" and menu.items()[menu.selected].label == "HOST GAME"


def test_games_found_say_their_mode() -> None:
    from ok_doomer.client.menu import game_line

    campaign, dm = _game("desk", "192.168.1.20"), _game("pi", "192.168.1.30")
    assert dm.info is not None
    dm = FoundGame(dm.address, dm.version, dm.name, replace(dm.info, mode="deathmatch"), 4.0, 0.0)
    assert "GROUND FLOOR" in game_line(campaign) and "DEATHMATCH" in game_line(dm)


def test_joining_a_game_found() -> None:
    menu = Menu("ann")
    _keys(menu, pygame.K_DOWN, pygame.K_DOWN, pygame.K_RETURN)  # JOIN GAME
    assert menu.screen == "join"
    assert menu.items()[0].label == "LOOKING FOR GAMES..." and not menu.items()[0].enabled
    assert menu.items()[menu.selected].label == "ENTER AN ADDRESS..."
    menu.set_games([_game("old-laptop", "192.168.1.40", 2), _game("desk", "192.168.1.20")])
    assert "needs updating" in menu.items()[0].label and not menu.items()[0].enabled
    menu.selected = 0
    menu.move(1)  # onto the one that can be joined
    assert _keys(menu, pygame.K_RETURN) == JoinGame("192.168.1.20", 5029)


def test_joining_by_address() -> None:
    menu = Menu("ann")
    menu.go("join")
    menu.selected = len(menu.items()) - 2  # ENTER AN ADDRESS...
    _keys(menu, pygame.K_RETURN)
    assert menu.screen == "address"
    assert _keys(menu, "192.168.1.20", pygame.K_RETURN) == JoinGame("192.168.1.20", 5029)
    _keys(menu, *[pygame.K_BACKSPACE] * 20)
    assert _keys(menu, "pi:5030", pygame.K_RETURN) == JoinGame("pi", 5030)
    _keys(menu, *[pygame.K_BACKSPACE] * 20, "host:port", pygame.K_RETURN)
    assert "not an address" in menu.message and menu.screen == "address"  # a bad port


def test_parse_address() -> None:
    assert parse_address("192.168.1.20") == ("192.168.1.20", 5029)
    assert parse_address(" 10.0.0.5:6000 ") == ("10.0.0.5", 6000)
    assert parse_address("raspberrypi.local") == ("raspberrypi.local", 5029)
    for bad in ("", ":5029", "host:", "host:99999", "a b"):
        assert parse_address(bad) is None, bad


def test_your_name_is_typed_and_remembered(_home: Path) -> None:
    menu = Menu(settings.load_name())
    assert menu.name == "player"
    menu.selected = 3  # NAME
    _keys(menu, pygame.K_RETURN, *[pygame.K_BACKSPACE] * 10, "Ws Nick!", pygame.K_RETURN)
    assert menu.name == "Ws Nick"  # W and S typed as letters; "!" isn't in the font
    assert menu.screen == "main" and settings.load_name() == "Ws Nick"
    assert Menu(settings.load_name()).name == "Ws Nick"  # next time


def test_a_broken_settings_file_means_defaults(_home: Path) -> None:
    (_home / "settings.json").write_text("{not json")
    assert settings.load_name() == "player"
    settings.save_name("x" * 40)
    assert settings.load_name() == "x" * settings.MAX_NAME


def test_esc_goes_back_a_screen_at_a_time() -> None:
    menu = Menu("ann")
    menu.go("join")
    menu.go("address")
    assert _keys(menu, pygame.K_ESCAPE) is None and menu.screen == "join"
    assert _keys(menu, pygame.K_ESCAPE) is None and menu.screen == "main"
    assert _keys(menu, pygame.K_ESCAPE) == Quit()


def test_its_drawn_and_the_mouse_finds_the_items() -> None:
    menu = Menu("ann")
    pixels = np.zeros((320, 200), dtype=np.uint8)
    menu.draw(pixels)
    assert (pixels == NAME_COLOUR).any() and len(menu.rows) == 5
    top, bottom = menu.rows[2]
    assert menu.item_at((top + bottom) / 2) == 2 and menu.item_at(0) is None
    assert menu.activate(menu.item_at((top + bottom) / 2)) is None and menu.screen == "join"


def _post_keys(*keys: int) -> None:
    for k in keys:
        pygame.event.post(pygame.event.Event(pygame.KEYDOWN, key=k, unicode="", mod=0))


def test_the_menu_runs_in_a_window() -> None:
    pygame.init()
    try:
        display = Display(RenderConfig(vsync=False))
        menu = Menu("ann")
        _post_keys(pygame.K_DOWN, pygame.K_DOWN, pygame.K_RETURN)  # JOIN GAME: looks for games
        _post_keys(pygame.K_ESCAPE, pygame.K_ESCAPE)  # back (onto JOIN GAME); quit
        assert menu_mod.run_menu(display, ClientConfig(), menu, port=0) == Quit()
    finally:
        pygame.quit()


def test_start_goes_menu_game_menu(monkeypatch: pytest.MonkeyPatch) -> None:
    """Single player, leave the game (back to the menu), quit."""
    choices: list[object] = [SinglePlayer(), Quit()]
    played: list[str] = []

    def fake_play(
        display: object,
        session: object,
        config: object,
        angle: float,
        greeting: object = None,
        hosting: bool = False,
    ) -> bool:
        assert not hosting
        played.append(type(session).__name__)
        return False  # left the game: back to the menu

    monkeypatch.setattr(menu_mod, "run_menu", lambda *a, **k: choices.pop(0))
    monkeypatch.setattr(menu_mod, "play", fake_play)
    menu_mod.start(ClientConfig(render=RenderConfig(vsync=False)), "ann")
    assert played == ["LocalSession"] and choices == []


def test_a_failed_join_says_why_on_the_menu(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[str] = []

    def spy(display: object, config: object, menu: Menu, port: int = 0) -> object:
        seen.append(menu.message)
        if len(seen) == 1:
            return JoinGame("127.0.0.1", 1)  # nobody there
        return Quit()

    monkeypatch.setattr(menu_mod, "run_menu", spy)
    menu_mod.start(ClientConfig(render=RenderConfig(vsync=False)), "ann")
    assert seen[0] == "" and "can't reach 127.0.0.1:1" in seen[1]


def test_everything_the_menu_writes_is_in_the_font() -> None:
    """The selection mark, the cursor, every label and hint: no character
    silently missing (as ">" and "_" once were)."""
    from ok_doomer.client.pixelart import FONT_5X7

    texts = []
    for screen in ("main", "single", "host", "join", "address", "name"):
        menu = Menu("ann")
        menu.go(screen, "192.168.1.20:5029")
        menu.set_games([_game("desk", "192.168.1.20"), _game("old", "1.2.3.4", 2)])
        texts += [f"> {i.label}" for i in menu.items()] + [menu.title()]
    texts.append("ARROWS AND ENTER, OR THE MOUSE - ESC GOES BACK")
    missing = {c for t in texts for c in t.upper()} - set(FONT_5X7)
    assert not missing, missing


def test_the_items_line_up_down_the_left_the_mark_before_them() -> None:
    from ok_doomer.client.menu import DIM, LEFT

    menu = Menu("ann")
    pixels = np.zeros((320, 200), dtype=np.uint8)
    menu.draw(pixels)

    def first_column(item: int, colour: int, after: int = 0) -> int:
        top, bottom = menu.rows[item]
        cols = np.nonzero((pixels[after:, top:bottom] == colour).any(axis=1))[0]
        return after + int(cols[0])

    mark = 2 * 6 * 2  # "> ": two characters of 5x7 with their gaps, at twice
    assert LEFT <= first_column(0, NAME_COLOUR) < LEFT + mark  # the mark, at the left
    words = first_column(0, NAME_COLOUR, after=LEFT + mark - 2)  # SINGLE PLAYER
    assert words == LEFT + mark == first_column(2, DIM)  # JOIN GAME, level with it
    from ok_doomer.client.hud import NAME_BACK

    on_navy = np.full((320, 200), 7, dtype=np.uint8)
    menu.draw(on_navy)
    assert not (on_navy == NAME_BACK).any()  # bare words: no black panels or outlines

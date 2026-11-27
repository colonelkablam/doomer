"""The start screen: single player, host a game, join one, your name.

Green terminal text over an 80s office block at night, turning slowly
(client/skyline.py). Keys (arrows or W/S,
Enter, Esc) or the mouse. JOIN GAME lists the games found on the LAN
(client/discovery.py), live, and takes an address typed instead. Your name
is remembered (client/settings.py). Leaving a game comes back here.

`Menu` is the state (which screen, what's selected, what's being typed)
and what keys and clicks do to it, separate from pygame's loop, so it's
tested without a window; `run_menu` is the loop; `start` is the whole
program: the menu, a game, the menu again.
"""

import contextlib
import logging
from dataclasses import dataclass

import pygame

from ok_doomer.client.app import Display, _is_fullscreen_toggle, play
from ok_doomer.client.config import ClientConfig
from ok_doomer.client.discovery import FoundGame, GameFinder
from ok_doomer.client.hud import NAME_COLOUR, draw_text, wrap
from ok_doomer.client.local import LocalSession
from ok_doomer.client.network import JoinFailed, NetworkSession
from ok_doomer.client.palette import GREEN, GREY, U8Array, colour_index
from ok_doomer.client.pixelart import FONT_5X7
from ok_doomer.client.render import make_frame_surface
from ok_doomer.client.session import GameSession
from ok_doomer.client.settings import MAX_NAME, load_name, save_name
from ok_doomer.client.skyline import Skyline
from ok_doomer.shared.building import floor_map_text
from ok_doomer.shared.protocol import DISCOVERY_PORT

log = logging.getLogger(__name__)

DEFAULT_PORT = 5029
SEEK_EVERY = 1.0  # seconds between asks, on the join screen
DIM = colour_index(GREEN, 9)  # items not selected
OFF = colour_index(GREY, 7)  # items you can't choose
NAME_CHARS = set("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 -'.")


@dataclass(frozen=True, slots=True)
class SinglePlayer:
    pass


@dataclass(frozen=True, slots=True)
class JoinGame:
    host: str
    port: int


@dataclass(frozen=True, slots=True)
class Quit:
    pass


Choice = SinglePlayer | JoinGame | Quit


@dataclass(frozen=True, slots=True)
class Item:
    label: str
    action: str
    enabled: bool = True


def parse_address(text: str) -> tuple[str, int] | None:
    """ "192.168.1.20" or "192.168.1.20:5029" -> (host, port); None if not."""
    host, colon, port = text.strip().rpartition(":")
    if not colon:
        host, port = text.strip(), str(DEFAULT_PORT)
    if not host or " " in host or not port.isdigit() or not 0 < int(port) < 65536:
        return None
    return host, int(port)


def game_line(game: FoundGame) -> str:
    """A found game, short enough for the screen."""
    if game.info is None or not game.joinable:
        return f"{game.name}  (needs updating)"
    i = game.info
    return f"{i.name}  {i.floor}  {i.players}/{i.max_players}  {game.ping_ms:.0f} MS"


class Menu:
    def __init__(self, name: str) -> None:
        self.screen = "main"  # main, join, address, name
        self.selected = 0
        self.name = name
        self.typing = ""  # the address or name being typed
        self.address = ""  # the last address joined by hand
        self.message = ""  # a line under the items: why a join failed
        self.games: list[FoundGame] = []
        self.rows: list[tuple[int, int]] = []  # each item's rows, as last drawn (for the mouse)

    def items(self) -> list[Item]:
        if self.screen == "main":
            return [
                Item("SINGLE PLAYER", "single"),
                Item("HOST GAME (SOON)", "host", enabled=False),
                Item("JOIN GAME", "join"),
                Item(f"NAME: {self.name}", "name"),
                Item("QUIT", "quit"),
            ]
        if self.screen == "join":
            found = [Item(game_line(g), f"game {i}", g.joinable) for i, g in enumerate(self.games)]
            if not found:
                found = [Item("LOOKING FOR GAMES...", "none", enabled=False)]
            return [*found, Item("ENTER AN ADDRESS...", "address"), Item("BACK", "back")]
        label = "ADDRESS" if self.screen == "address" else "NAME"
        return [Item(f"{label}: {self.typing}_", "done"), Item("BACK", "back")]

    def title(self) -> str:
        return {
            "main": "",
            "join": "GAMES ON THIS NETWORK",
            "address": "JOIN BY ADDRESS (HOST:PORT)",
            "name": "YOUR NAME",
        }[self.screen]

    def go(self, screen: str, typing: str = "") -> None:
        self.screen, self.typing, self.selected = screen, typing, 0
        self.message = "" if screen != "main" else self.message
        self._settle()

    def move(self, step: int) -> None:
        """Up (-1) or down (+1), past what can't be chosen."""
        items = self.items()
        for _ in items:
            self.selected = (self.selected + step) % len(items)
            if items[self.selected].enabled:
                return

    def _settle(self) -> None:
        """Keep the selection on something that can be chosen (the list of
        games changes under it)."""
        items = self.items()
        self.selected = min(self.selected, len(items) - 1)
        if not items[self.selected].enabled:
            self.move(1)

    def set_games(self, games: list[FoundGame]) -> None:
        self.games = games
        self._settle()

    def activate(self, index: int | None = None) -> Choice | None:
        items = self.items()
        index = self.selected if index is None else index
        if not 0 <= index < len(items) or not items[index].enabled:
            return None
        self.selected = index
        action = items[index].action
        if action == "single":
            return SinglePlayer()
        if action == "quit":
            return Quit()
        if action == "join":
            self.go("join")
        elif action == "name":
            self.go("name", self.name)
        elif action == "address":
            self.go("address", self.address)
        elif action == "back":
            self.back()
        elif action.startswith("game "):
            game = self.games[int(action.split()[1])]
            assert game.info is not None
            return JoinGame(game.address, game.info.port)
        elif action == "done":
            return self._done_typing()
        return None

    def _done_typing(self) -> Choice | None:
        if self.screen == "name":
            self.name = " ".join(self.typing.split())[:MAX_NAME] or self.name
            save_name(self.name)
            self.go("main")
            self.selected = 3
            return None
        address = parse_address(self.typing)
        if address is None:
            self.message = "That's not an address: try 192.168.1.20 or 192.168.1.20:5029"
            return None
        self.address = self.typing.strip()
        return JoinGame(*address)

    def back(self) -> Choice | None:
        if self.screen == "main":
            return Quit()
        self.go("join" if self.screen == "address" else "main")
        return None

    def key(self, key: int, text: str = "") -> Choice | None:
        """A key pressed (`text`: the character it typed, if any)."""
        if key == pygame.K_ESCAPE:
            return self.back()
        if key in (pygame.K_UP, pygame.K_w) and not self._typing_letters(text):
            self.move(-1)
        elif key in (pygame.K_DOWN, pygame.K_s) and not self._typing_letters(text):
            self.move(1)
        elif key in (pygame.K_RETURN, pygame.K_KP_ENTER):
            return self.activate()
        elif self.screen in ("address", "name"):
            if key == pygame.K_BACKSPACE:
                self.typing = self.typing[:-1]
            elif text and len(self.typing) < (MAX_NAME if self.screen == "name" else 40):
                self.typing += text if _typeable(self.screen, text) else ""
        return None

    def _typing_letters(self, text: str) -> bool:
        """W and S are letters while typing, not up and down."""
        return self.screen in ("address", "name") and bool(text)

    def item_at(self, row: float) -> int | None:
        for i, (top, bottom) in enumerate(self.rows):
            if top <= row <= bottom:
                return i
        return None

    def draw(self, pixels: U8Array) -> None:
        """The screen into the frame: the title, the items (the selected
        bright and marked, the rest dim, the unchoosable grey), any message."""
        w, h = pixels.shape
        draw_text(pixels, "OK DOOMER", FONT_5X7, w / 2, 32, scale=3)
        row = 46
        if self.title():
            row += draw_text(pixels, self.title(), FONT_5X7, w / 2, row + 7, colour=DIM) + 6
        scale = 2 if self.screen == "main" else 1
        self.rows = []
        for i, item in enumerate(self.items()):
            chosen = i == self.selected and item.enabled
            label = f"> {item.label}" if chosen else item.label
            colour = NAME_COLOUR if chosen else DIM if item.enabled else OFF
            height = draw_text(
                pixels, label, FONT_5X7, w / 2, row + 7 * scale, scale, colour=colour
            )
            self.rows.append((row, row + height))
            row += height + (4 if scale > 1 else 2)
        for line in wrap(self.message, FONT_5X7, w - 16):
            row += draw_text(pixels, line, FONT_5X7, w / 2, row + 10) + 1
        hint = "ARROWS AND ENTER, OR THE MOUSE - ESC GOES BACK"
        draw_text(pixels, hint, FONT_5X7, w / 2, h - 4, colour=DIM)


def _typeable(screen: str, text: str) -> bool:
    """Can `text` be typed into this screen's box? A name: what the font
    has. An address: letters, digits, dots, colons, hyphens."""
    if screen == "name":
        return text.upper() in NAME_CHARS
    return text.isalnum() or text in ".:-"


def _free_the_mouse() -> None:
    pygame.event.set_grab(False)
    pygame.mouse.set_visible(True)
    with contextlib.suppress(pygame.error):
        pygame.mouse.set_relative_mode(False)


def run_menu(
    display: Display, config: ClientConfig, menu: Menu, port: int = DISCOVERY_PORT
) -> Choice:
    """Show the menu until something's chosen."""
    rc = config.render
    frame = make_frame_surface(rc)
    skyline = Skyline(rc.width, rc.height)
    clock = pygame.time.Clock()
    finder: GameFinder | None = None
    next_seek = 0.0
    _free_the_mouse()
    try:
        while True:
            clock.tick(min(rc.max_fps or 60, 60))
            now = pygame.time.get_ticks() / 1000.0
            for event in pygame.event.get():
                choice: Choice | None = None
                if event.type == pygame.QUIT:
                    return Quit()
                if event.type == pygame.KEYDOWN and _is_fullscreen_toggle(event):
                    display.toggle_fullscreen()
                elif event.type == pygame.KEYDOWN:
                    choice = menu.key(event.key, event.unicode)
                elif event.type == pygame.MOUSEMOTION:
                    index = menu.item_at(display.to_frame(event.pos)[1])
                    if index is not None and menu.items()[index].enabled:
                        menu.selected = index
                elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    index = menu.item_at(display.to_frame(event.pos)[1])
                    if index is not None:
                        choice = menu.activate(index)
                if choice is not None:
                    return choice
            if menu.screen == "join":  # look for games while the list is up
                if finder is None:
                    finder = GameFinder(port)
                if now >= next_seek:
                    finder.seek()
                    next_seek = now + SEEK_EVERY
                menu.set_games(finder.poll())
            elif finder is not None:
                finder.close()
                finder, next_seek = None, 0.0
            pixels = pygame.surfarray.pixels2d(frame)
            skyline.draw(pixels, now)  # the office block at night, turning
            menu.draw(pixels)
            del pixels
            display.show(frame)
            pygame.display.flip()
    finally:
        skyline.close()
        if finder is not None:
            finder.close()


def _connecting(display: Display, config: ClientConfig, text: str) -> None:
    """A frame saying what's happening, while joining (it can take seconds)."""
    frame = make_frame_surface(config.render)
    pixels = pygame.surfarray.pixels2d(frame)
    draw_text(pixels, text, FONT_5X7, config.render.width / 2, config.render.height // 2)
    del pixels
    display.show(frame)
    pygame.display.flip()


def start(config: ClientConfig, name: str | None = None, predict: bool = True) -> None:
    """The game: the start screen, a game, back to the start screen, until
    you quit."""
    pygame.init()
    try:
        display = Display(config.render)
        menu = Menu(name or load_name())
        while True:
            choice = run_menu(display, config, menu)
            session: GameSession
            if isinstance(choice, Quit):
                return
            if isinstance(choice, SinglePlayer):
                session = LocalSession(floor_map_text("G"), menu.name)
            else:
                _connecting(display, config, f"JOINING {choice.host}:{choice.port}...")
                try:
                    session = NetworkSession.join(choice.host, choice.port, menu.name, predict)
                except JoinFailed as e:
                    menu.message = str(e)
                    continue
            try:
                quit_all = play(display, session, config, session.spawn_angle())
            finally:
                session.close()
            if quit_all:
                return
            menu.go("main")
    finally:
        pygame.quit()

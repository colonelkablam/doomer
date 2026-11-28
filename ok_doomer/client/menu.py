"""The start screen: single player, host a game, join one, your name.

Green terminal text on navy, down the left; on the right one of the
game's own objects turning (client/showcase.py). Keys (arrows or W/S,
Enter, Esc) or the mouse. JOIN GAME lists the games found on the LAN
(client/discovery.py), live, and takes an address typed instead. Your name
is remembered (client/settings.py). Leaving a game comes back here; after
a game joined, REJOIN GAME is offered for as long as the server keeps
your player (ServerConfig.rejoin_seconds): back where you were.

`Menu` is the state (which screen, what's selected, what's being typed)
and what keys and clicks do to it, separate from pygame's loop, so it's
tested without a window; `run_menu` is the loop; `start` is the whole
program: the menu, a game, the menu again.
"""

import contextlib
import logging
import math
from dataclasses import dataclass

import pygame

from ok_doomer.client.app import Display, _is_fullscreen_toggle, play
from ok_doomer.client.config import ClientConfig
from ok_doomer.client.discovery import FoundGame, GameFinder
from ok_doomer.client.hosting import HostedServer, HostFailed
from ok_doomer.client.hud import DIM_COLOUR, NAME_COLOUR, draw_text, wrap
from ok_doomer.client.local import LocalSession
from ok_doomer.client.network import JoinFailed, NetworkSession
from ok_doomer.client.palette import BLUE, GREY, U8Array, colour_index
from ok_doomer.client.pixelart import FONT_5X7
from ok_doomer.client.render import make_frame_surface
from ok_doomer.client.session import GameSession
from ok_doomer.client.settings import MAX_NAME, load_name, save_name
from ok_doomer.client.showcase import Showcase
from ok_doomer.server.config import ServerConfig
from ok_doomer.server.net import SHUTTING_DOWN
from ok_doomer.shared.building import floor_map_text
from ok_doomer.shared.protocol import DISCOVERY_PORT

log = logging.getLogger(__name__)

DEFAULT_PORT = 5029
SEEK_EVERY = 1.0  # seconds between asks, on the join screen
DIM = DIM_COLOUR  # items not selected
OFF = colour_index(GREY, 7)  # items you can't choose
BACKGROUND = colour_index(BLUE, 1)  # navy: green reads well on it
LEFT = 14  # where the items start, frame pixels from the left
OBJECT_AT = (250, 112)  # the turning object's middle
NAME_CHARS = set("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 -'.")
REJOIN_SECONDS = ServerConfig().rejoin_seconds  # as long as a server keeps you
INDENT = 12  # REJOIN, under its heading


@dataclass(frozen=True, slots=True)
class SinglePlayer:
    mode: str = "coop"  # the campaign (shared/modes.py names it co-op)


@dataclass(frozen=True, slots=True)
class HostGame:
    mode: str = "coop"


@dataclass(frozen=True, slots=True)
class JoinGame:
    host: str
    port: int


@dataclass(frozen=True, slots=True)
class Quit:
    pass


Choice = SinglePlayer | HostGame | JoinGame | Quit


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


# What the player calls each mode (the code's names: shared/modes.py).
MODE_NAMES = {"coop": "CAMPAIGN", "deathmatch": "DEATHMATCH"}
# Modes that can be chosen yet; the rest are shown, greyed: coming.
PLAYABLE = {"coop"}


def game_line(game: FoundGame) -> str:
    """A found game, short enough for the screen (its mode is the heading
    it's under): a campaign by the floor it's on, deathmatch by its map."""
    if game.info is None or not game.joinable:
        return f"{game.name}  (needs updating)"
    i = game.info
    where = i.floor if i.mode == "coop" else i.map_name
    return f"{i.name}  {where}  {i.players}/{i.max_players}  {game.ping_ms:.0f} MS"


def game_heading(game: FoundGame) -> str:
    """The heading a found game's listed under: its mode."""
    if game.info is None or not game.joinable:
        return "OTHER VERSIONS"  # can't be joined: the mode doesn't matter
    return MODE_NAMES.get(game.info.mode, game.info.mode.upper())


def game_items(games: list[FoundGame]) -> list["Item"]:
    """The games found, under a heading for each mode there are games of
    (campaign, deathmatch, then any others), in the order found within."""
    order = [*MODE_NAMES.values()]
    headings = sorted(
        {game_heading(g) for g in games},
        key=lambda h: (order.index(h) if h in order else len(order), h),
    )
    items = []
    for heading in headings:
        items.append(Item(heading, "heading", enabled=False))
        items += [
            Item(game_line(g), f"game {i}", g.joinable)
            for i, g in enumerate(games)
            if game_heading(g) == heading
        ]
    return items


class Menu:
    def __init__(self, name: str) -> None:
        self.screen = "main"  # main, single, host (which mode), join, address, name
        self.selected = 0
        self.name = name
        self.typing = ""  # the address or name being typed
        self.address = ""  # the last address joined by hand
        self.message = ""  # a line under the items: why a join failed
        self.games: list[FoundGame] = []
        self.rows: list[tuple[int, int]] = []  # each item's rows, as last drawn (for the mouse)
        self.rejoin: JoinGame | None = None  # the game just left, while it keeps your player
        self.rejoin_until = 0.0
        self._rejoin_message = ""
        self.now = 0.0  # as of the last expire(): for REJOIN's countdown

    def offer_rejoin(self, game: JoinGame, dropped: str | None, now: float) -> None:
        """You've just left `game`: REJOIN GAME, at the top, for a while.
        `dropped`: why, if you didn't choose to (then it's selected, and
        said)."""
        self.rejoin, self.rejoin_until, self.now = game, now + REJOIN_SECONDS, now
        self.go("main")
        if dropped:
            self.message = self._rejoin_message = dropped  # the countdown says the rest
        else:
            self.selected = 1  # you left: SINGLE PLAYER, as usual

    def expire(self, now: float) -> None:
        """The server's forgotten you by now: no more REJOIN."""
        self.now = now
        if self.rejoin is not None and now >= self.rejoin_until:
            action = self.items()[self.selected].action
            self.rejoin = None
            if self.message == self._rejoin_message:
                self.message = ""
            if self.screen == "main":
                actions = [i.action for i in self.items()]
                self.selected = actions.index(action) if action in actions else 0

    def items(self) -> list[Item]:
        if self.screen == "main":
            rejoin = [Item("REJOIN", "rejoin")] if self.rejoin is not None else []
            return [
                *rejoin,
                Item("SINGLE PLAYER", "single"),
                Item("HOST GAME", "host"),
                Item("JOIN GAME", "join"),
                Item(f"NAME: {self.name}", "name"),
                Item("QUIT", "quit"),
            ]
        if self.screen in ("single", "host"):
            return [
                *(
                    Item(
                        name if mode in PLAYABLE else f"{name} - COMING",
                        f"mode {mode}",
                        mode in PLAYABLE,
                    )  # fmt: skip
                    for mode, name in MODE_NAMES.items()
                ),
                Item("BACK", "back"),
            ]
        if self.screen == "join":
            found = game_items(self.games)
            if not found:
                found = [Item("LOOKING FOR GAMES...", "none", enabled=False)]
            return [*found, Item("ENTER AN ADDRESS...", "address"), Item("BACK", "back")]
        label = "ADDRESS" if self.screen == "address" else "NAME"
        return [Item(f"{label}: {self.typing}_", "done"), Item("BACK", "back")]

    def last_game(self) -> str:
        """REJOIN's heading: where, and how long the server keeps you."""
        assert self.rejoin is not None
        left = max(0, math.ceil(self.rejoin_until - self.now))
        return f"LAST GAME: {self.rejoin.host}:{self.rejoin.port} - {left} S"

    def title(self) -> str:
        return {
            "main": "",
            "single": "SINGLE PLAYER",
            "host": "HOST A GAME",
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
        if action in ("single", "host"):
            self.go(action)  # which mode
            return None
        if action.startswith("mode "):
            mode = action.split()[1]
            return SinglePlayer(mode) if self.screen == "single" else HostGame(mode)
        if action == "quit":
            return Quit()
        if action == "rejoin":
            return self.rejoin
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
            self.selected = [i.action for i in self.items()].index("name")
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
        came_from = self.screen
        self.go("join" if self.screen == "address" else "main")
        if self.screen == "main":  # back on what you'd chosen
            actions = [i.action for i in self.items()]
            if came_from in actions:
                self.selected = actions.index(came_from)
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

    def shows_object(self) -> bool:
        """The turning object beside the menu (main and name screens; the
        lists of games and the address box want the width)."""
        return self.screen in ("main", "name")  # (the modes' long labels want the width)

    def draw(self, pixels: U8Array) -> None:
        """The screen into the frame: the title in the middle; the items down
        the left (the selected bright and marked, the rest dim, the
        unchoosable grey); any message under them. Bare green on the
        navy, no panels."""
        w, h = pixels.shape

        def text(
            words: str,
            column: float,
            bottom: int,
            scale: int = 1,
            centre: bool = True,
            colour: int = NAME_COLOUR,
        ) -> int:
            return draw_text(
                pixels,
                words,
                FONT_5X7,
                column,
                bottom,
                scale,
                panel=False,
                centre=centre,
                colour=colour,
                outline=False,
            )

        text("OK DOOMER", w / 2, 32, 3)
        row = 48
        if self.title():
            row += text(self.title(), w / 2, row + 6, colour=DIM) + 8
        scale = 2 if self.screen in ("main", "single", "host") else 1  # short lists: big
        self.rows = []
        for i, item in enumerate(self.items()):
            chosen = i == self.selected and item.enabled
            label = f"> {item.label}" if chosen else f"  {item.label}"
            colour = NAME_COLOUR if chosen else DIM if item.enabled else OFF
            if item.action == "heading":  # the games under it are indented by the mark
                label, colour = item.label, DIM
                row += 3 if i else 0
            elif item.action == "address" and self.games:
                row += 8  # apart from the list of games: not one of them
            left = LEFT
            if item.action == "rejoin":  # a block of its own: the game you were in
                row += text(self.last_game(), LEFT, row + 6, centre=False, colour=DIM) + 3
                left += INDENT
            height = text(label, left, row + 7 * scale - 1, scale, centre=False, colour=colour)
            self.rows.append((row, row + height))
            gap = 6 if scale > 1 and self.rejoin is None else 4  # REJOIN's block: closer
            row += height + gap + (4 if item.action == "rejoin" else 0)
        for line in wrap(self.message, FONT_5X7, w - 2 * LEFT):
            row += text(line, LEFT, row + 6, centre=False) + 2
        text("ARROWS AND ENTER, OR THE MOUSE - ESC GOES BACK", w / 2, h - 5, colour=DIM)


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
    showcase = Showcase()  # one of the game's objects, at random
    clock = pygame.time.Clock()
    finder: GameFinder | None = None
    next_seek = 0.0
    _free_the_mouse()
    try:
        while True:
            clock.tick(min(rc.max_fps or 60, 60))
            now = pygame.time.get_ticks() / 1000.0
            menu.expire(now)
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
            pixels[:] = BACKGROUND
            if menu.shows_object():
                showcase.draw(pixels, now, OBJECT_AT)
            menu.draw(pixels)
            del pixels
            display.show(frame)
            pygame.display.flip()
    finally:
        showcase.close()
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
    display: Display | None = None
    try:
        display = Display(config.render)
        menu = Menu(name or load_name())
        while True:
            choice = run_menu(display, config, menu)
            session: GameSession
            host: HostedServer | None = None
            greeting = None
            if isinstance(choice, Quit):
                return
            try:
                if isinstance(choice, SinglePlayer):
                    session = LocalSession(floor_map_text("G"), menu.name)
                elif isinstance(choice, HostGame):
                    _connecting(display, config, "STARTING YOUR GAME...")
                    host = HostedServer.start(f"{menu.name}'s game")
                    session = NetworkSession.join("127.0.0.1", host.port, menu.name, predict)
                    greeting = f"Hosting: others can join at {host.where()}"
                else:
                    _connecting(display, config, f"JOINING {choice.host}:{choice.port}...")
                    session = NetworkSession.join(choice.host, choice.port, menu.name, predict)
            except (HostFailed, JoinFailed) as e:
                if host is not None:
                    host.stop()
                menu.message = str(e)
                continue
            try:
                angle = session.spawn_angle()
                quit_all = play(display, session, config, angle, greeting, hosting=host is not None)
                dropped = session.ended()
            finally:
                session.close()
                if host is not None:
                    host.stop()  # everyone's told the game's over
            if quit_all:
                return
            if isinstance(choice, JoinGame) and dropped != SHUTTING_DOWN:
                menu.offer_rejoin(choice, dropped, pygame.time.get_ticks() / 1000.0)
            else:
                if isinstance(choice, JoinGame):
                    menu.rejoin = None  # that game's over: nothing to rejoin
                menu.go("main")
    finally:
        if display is not None:
            display.close()
        pygame.quit()

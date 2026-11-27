"""The heads-up display: messages about what just happened, and the ID passes
the team holds. Drawn on the window, over the scaled-up 3D view.

What to say about an event (`message_for`) is a pure function, so it's tested
without a display; Hud only keeps messages on screen for a while and draws.
"""

from collections.abc import Collection, Iterable
from dataclasses import dataclass

import numpy as np
import pygame

from ok_doomer.client.boxes import pass_card_rgba
from ok_doomer.client.palette import GREEN, GREY, U8Array, colour_index
from ok_doomer.client.pixelart import FONT_3X5, FONT_5X7, BoolArray
from ok_doomer.client.render import Renderer
from ok_doomer.shared.building import floor_name
from ok_doomer.shared.dispensers import DISPENSERS
from ok_doomer.shared.events import (
    CubicleDoorUsed,
    Dispensed,
    DoorLocked,
    Event,
    LevelComplete,
    LiftCalled,
    LiftNoPower,
    PassPickedUp,
    PlayerJoined,
    PlayerLeft,
    PlayersHere,
    Refreshed,
    SwitchThrown,
)
from ok_doomer.shared.mapfile import GameMap

MESSAGE_SECONDS = 3.0
MAX_MESSAGES = 4  # shown at once, stacked
TITLE_SECONDS = 3.0  # a floor's name, as you arrive on it
PASS_COLOURS_RGB = {"blue": (90, 140, 240), "red": (230, 90, 70), "gold": (240, 210, 90)}
TEXT = (255, 255, 255)
SHADOW = (0, 0, 0)


_CARDS: dict[str, pygame.Surface] = {}
CARD_SIZE = (30, 48)  # pixels: the card's proportions in the world (it's taller than wide)


def _card_image(colour: str) -> pygame.Surface:
    """A held pass: the same card as the one you picked up (boxes.pass_card_rgba)."""
    if colour not in _CARDS:
        rgba = pass_card_rgba(colour)
        image = pygame.Surface((64, 64), pygame.SRCALPHA)
        pygame.surfarray.pixels3d(image)[...] = rgba[..., :3]
        pygame.surfarray.pixels_alpha(image)[...] = rgba[..., 3]
        _CARDS[colour] = pygame.transform.scale(image, CARD_SIZE)
    return _CARDS[colour]


def message_for(event: Event) -> str | None:
    if isinstance(event, Dispensed):
        name = DISPENSERS[event.dispenser].name
        name = name[0].upper() + name[1:]
        if event.amount:
            return f"{event.meter.capitalize()} +{event.amount}"
        if event.reason == "no power":
            return f"{name} has no power"
        if event.reason == "empty":
            return f"{name} has nothing left for you"
        return f"{event.meter.capitalize()}'s full"
    if isinstance(event, Refreshed):
        return f"{event.meter.capitalize()} +{event.amount}" if event.amount else None
    if isinstance(event, PassPickedUp):
        return f"Picked up the {event.colour} pass"
    if isinstance(event, DoorLocked):
        return f"Locked: you need the {event.lock} pass"
    if isinstance(event, SwitchThrown) and event.flag == "power":
        return "The power is back on"
    if isinstance(event, CubicleDoorUsed) and event.engaged:
        return "Engaged: someone's in there"
    if isinstance(event, LiftNoPower):
        return "The lift has no power"
    if isinstance(event, LiftCalled):
        return "The lift's on its way"
    if isinstance(event, PlayerJoined):
        return f"{event.name} joined"
    if isinstance(event, PlayersHere):
        return "Here: " + ", ".join(event.names)
    if isinstance(event, PlayerLeft):
        return f"{event.name} {_LEFT.get(event.reason, 'left')}"
    return None


# How someone left, as the HUD says it.
_LEFT = {
    "left": "left",
    "quit": "left",
    "timed out": "timed out",
    "disconnected": "lost connection",
    "too far behind": "lost connection",
}


NAME_DISTANCE = 12.0  # cells: further off, no name over their head
NAME_GAP = 2  # frame pixels between the name and the top of what's seen of them
MIN_SEEN = 6  # frame pixels of someone showing before they get a name


@dataclass(frozen=True, slots=True)
class NameTag:
    """A name to draw over someone's head: (column, row) in the frame, the
    bottom middle of the text."""

    name: str
    column: float
    row: float


def name_tags(renderer: Renderer, people: list[tuple[str, float, float]]) -> list[NameTag]:
    """Names for the people the frame just rendered shows (name, x, y): over
    the top of what's seen of them, if enough is, and they're near enough.
    The renderer knows what it drew of each (Renderer.seen), after walls and
    desks hid their parts: so someone behind a counter gets a name only if
    their head shows over it, and the name sits on it."""
    tags = []
    for name, x, y in people:
        seen = renderer.seen.get((x, y))
        if seen is None or seen.pixels < MIN_SEEN or seen.depth > NAME_DISTANCE:
            continue
        tags.append(NameTag(name, seen.column, seen.top - NAME_GAP))
    return tags


NAME_COLOUR = colour_index(GREEN, 15)  # phosphor green, the brightest the palette has
NAME_BACK = colour_index(GREY, 0)  # black: an outline, or a panel behind


Font = dict[str, tuple[str, ...]]  # glyph rows of "#" and "." (pixelart)


def draw_name_tags(
    pixels: U8Array, tags: list[NameTag], scale: int = 1, panel: bool = True
) -> None:
    """Names into the frame (palette indices, [column][row]), before it's
    scaled up, so they're chunky pixels like the rest: the 3x5 font in
    phosphor green, on a black panel (or outlined), the bottom middle of
    each at its tag's (column, row)."""
    for tag in tags:
        draw_text(pixels, tag.name, FONT_3X5, tag.column, round(tag.row) - 1, scale, panel)


def draw_text(
    pixels: U8Array,
    text: str,
    font: Font,
    column: float,
    bottom: int,
    scale: int = 1,
    panel: bool = True,
) -> int:
    """`text` into the frame in `font` (capitals; what it can't draw, a
    space), phosphor green on a black panel (or outlined): centred on
    `column`, its last row at `bottom`. How tall it came out (0: nothing)."""
    text = "".join(c if c in font else " " for c in text.upper()).strip()
    if not text:
        return 0
    w, h = pixels.shape
    glyphs = text_mask(text, font, scale)
    tw, th = glyphs.shape
    left, top = round(column - tw / 2), bottom + 1 - th
    border = 1
    back = np.ones((tw + 2 * border, th + 2 * border), dtype=bool)
    if not panel:  # just round the letters: grow them by a pixel
        back = np.zeros_like(back)
        for du in (0, 1, 2):
            for dv in (0, 1, 2):
                back[du : du + tw, dv : dv + th] |= glyphs
    _blit(pixels, back, left - border, top - border, NAME_BACK, w, h)
    _blit(pixels, glyphs, left, top, NAME_COLOUR, w, h)
    return int(th) + 2 * border


def text_width(text: str, font: Font, scale: int = 1) -> int:
    """Pixels across: each glyph and a column between."""
    glyph = len(next(iter(font.values()))[0])
    return (len(text) * (glyph + 1) - 1) * scale


def text_mask(text: str, font: Font, scale: int = 1) -> BoolArray:
    """[column][row]: where the text's pixels are."""
    sample = next(iter(font.values()))
    glyph, rows = len(sample[0]), len(sample)
    mask = np.zeros((text_width(text, font, scale), rows * scale), dtype=bool)
    for i, ch in enumerate(text):
        for gy, row in enumerate(font[ch]):
            for gx, bit in enumerate(row):
                if bit == "#":
                    u = (i * (glyph + 1) + gx) * scale
                    mask[u : u + scale, gy * scale : gy * scale + scale] = True
    return mask


def wrap(text: str, font: Font, width: int, scale: int = 1) -> list[str]:
    """Lines of `text`, broken between words, each no wider than `width`
    pixels (a word longer than that goes alone)."""
    lines: list[str] = []
    for word in text.split():
        if lines and text_width(f"{lines[-1]} {word}", font, scale) <= width:
            lines[-1] += f" {word}"
        else:
            lines.append(word)
    return lines


def _blit(
    pixels: U8Array, mask: BoolArray, left: int, top: int, index: int, w: int, h: int
) -> None:
    """Set `index` where `mask` is, placed at (left, top), clipped to the frame."""
    mw, mh = mask.shape
    u0, v0 = max(0, -left), max(0, -top)
    u1, v1 = min(mw, w - left), min(mh, h - top)
    if u0 >= u1 or v0 >= v1:
        return
    region = pixels[left + u0 : left + u1, top + v0 : top + v1]
    region[mask[u0:u1, v0:v1]] = index


class Hud:
    def __init__(self, font: pygame.font.Font) -> None:
        self.font = font
        self._messages: list[tuple[str, float]] = []  # (text, until), oldest first
        self.complete = False  # the top floor built is done: a banner until you quit
        self._storey = "G"
        self._title: tuple[str, str] | None = None  # a floor's name, on arrival
        self._title_until = 0.0

    def arrive(self, game_map: GameMap, now: float) -> None:
        """A floor's begun: its name for a few seconds."""
        self._storey = game_map.storey
        self._title = (floor_name(game_map.storey), game_map.name)
        self._title_until = now + TITLE_SECONDS
        self._messages = []

    def handle(self, events: Iterable[Event], now: float) -> None:
        """Take this frame's events: each message shows for MESSAGE_SECONDS,
        the last MAX_MESSAGES stacked (two players joining at once: both)."""
        for event in events:
            if isinstance(event, LevelComplete) and event.next_floor is None:
                self.complete = True  # the top of what's built
            text = message_for(event)
            if text is not None:
                self._messages.append((text, now + MESSAGE_SECONDS))
        self._messages = [m for m in self._messages if m[1] > now][-MAX_MESSAGES:]

    def messages(self, now: float) -> list[str]:
        """What's showing, oldest first."""
        return [text for text, until in self._messages if now < until]

    def draw_messages(self, pixels: U8Array, now: float, scale: int = 1) -> None:
        """The messages into the frame (before it's scaled up): the 5x7 font,
        green on black panels like the names, stacked down from a sixth of
        the way down the screen, each wrapped to fit."""
        w, h = pixels.shape
        row = h // 6
        for text in self.messages(now):
            for line in wrap(text, FONT_5X7, w - 8, scale):
                row += draw_text(pixels, line, FONT_5X7, w / 2, row + 7 * scale, scale) + 1

    def draw(self, window: pygame.Surface, passes: Collection[str], now: float) -> None:
        width, height = window.get_size()
        if self.complete:
            banner = pygame.Rect(0, height // 2 - 50, width, 100)
            shade = pygame.Surface(banner.size, pygame.SRCALPHA)
            shade.fill((0, 0, 0, 170))
            window.blit(shade, banner)
            done = f"{floor_name(self._storey)} COMPLETE"
            self._text(window, done, (width // 2, height // 2 - 24), centre=True)
            self._text(window, "More floors to come", (width // 2, height // 2 + 4), centre=True)
            self._text(window, "Esc to quit", (width // 2, height // 2 + 30), centre=True)
        elif self._title is not None and now < self._title_until:
            storey, name = self._title
            self._text(window, storey, (width // 2, height // 3 - 14), centre=True)
            self._text(window, name, (width // 2, height // 3 + 12), centre=True)
        # Passes held: a card in each colour, bottom left.
        for i, colour in enumerate(c for c in PASS_COLOURS_RGB if c in passes):
            card = _card_image(colour)
            window.blit(card, (16 + i * (card.get_width() + 8), height - card.get_height() - 12))

    def _text(self, window: pygame.Surface, text: str, at: tuple[int, int], centre: bool) -> None:
        for offset, colour in (((2, 2), SHADOW), ((0, 0), TEXT)):
            image = self.font.render(text, True, colour)
            rect = image.get_rect(center=at) if centre else image.get_rect(topleft=at)
            window.blit(image, rect.move(offset))

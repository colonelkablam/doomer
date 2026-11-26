"""The heads-up display: messages about what just happened, and the ID passes
the team holds. Drawn on the window, over the scaled-up 3D view.

What to say about an event (`message_for`) is a pure function, so it's tested
without a display; Hud only keeps messages on screen for a while and draws.
"""

from collections.abc import Collection, Iterable

import pygame

from doomer.client.boxes import pass_card_rgba
from doomer.shared.building import floor_name
from doomer.shared.dispensers import DISPENSERS
from doomer.shared.events import (
    CubicleDoorUsed,
    Dispensed,
    DoorLocked,
    Event,
    LevelComplete,
    LiftCalled,
    LiftNoPower,
    PassPickedUp,
    Refreshed,
    SwitchThrown,
)
from doomer.shared.mapfile import GameMap

MESSAGE_SECONDS = 3.0
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
    return None


class Hud:
    def __init__(self, font: pygame.font.Font) -> None:
        self.font = font
        self._message: str | None = None
        self._until = 0.0
        self.complete = False  # the top floor built is done: a banner until you quit
        self._storey = "G"
        self._title: tuple[str, str] | None = None  # a floor's name, on arrival
        self._title_until = 0.0

    def arrive(self, game_map: GameMap, now: float) -> None:
        """A floor's begun: its name for a few seconds."""
        self._storey = game_map.storey
        self._title = (floor_name(game_map.storey), game_map.name)
        self._title_until = now + TITLE_SECONDS
        self._message = None

    def handle(self, events: Iterable[Event], now: float) -> None:
        """Take this frame's events; the latest message wins."""
        for event in events:
            if isinstance(event, LevelComplete) and event.next_floor is None:
                self.complete = True  # the top of what's built
            text = message_for(event)
            if text is not None:
                self._message, self._until = text, now + MESSAGE_SECONDS

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
        if self._message is not None and now < self._until:
            self._text(window, self._message, (width // 2, height // 6), centre=True)
        # Passes held: a card in each colour, bottom left.
        for i, colour in enumerate(c for c in PASS_COLOURS_RGB if c in passes):
            card = _card_image(colour)
            window.blit(card, (16 + i * (card.get_width() + 8), height - card.get_height() - 12))

    def _text(self, window: pygame.Surface, text: str, at: tuple[int, int], centre: bool) -> None:
        for offset, colour in (((2, 2), SHADOW), ((0, 0), TEXT)):
            image = self.font.render(text, True, colour)
            rect = image.get_rect(center=at) if centre else image.get_rect(topleft=at)
            window.blit(image, rect.move(offset))

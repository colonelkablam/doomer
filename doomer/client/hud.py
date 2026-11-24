"""The heads-up display: messages about what just happened, and the ID passes
the team holds. Drawn on the window, over the scaled-up 3D view.

What to say about an event (`message_for`) is a pure function, so it's tested
without a display; Hud only keeps messages on screen for a while and draws.
"""

from collections.abc import Collection, Iterable

import pygame

from doomer.shared.events import DoorLocked, Event, PassPickedUp, SwitchThrown

MESSAGE_SECONDS = 3.0
PASS_COLOURS_RGB = {"blue": (90, 140, 240), "red": (230, 90, 70), "gold": (240, 210, 90)}
TEXT = (255, 255, 255)
SHADOW = (0, 0, 0)


def message_for(event: Event) -> str | None:
    if isinstance(event, PassPickedUp):
        return f"Picked up the {event.colour} pass"
    if isinstance(event, DoorLocked):
        return f"Locked: you need the {event.lock} pass"
    if isinstance(event, SwitchThrown) and event.flag == "power":
        return "The power is back on"
    return None


class Hud:
    def __init__(self, font: pygame.font.Font) -> None:
        self.font = font
        self._message: str | None = None
        self._until = 0.0

    def handle(self, events: Iterable[Event], now: float) -> None:
        """Take this frame's events; the latest message wins."""
        for event in events:
            text = message_for(event)
            if text is not None:
                self._message, self._until = text, now + MESSAGE_SECONDS

    def draw(self, window: pygame.Surface, passes: Collection[str], now: float) -> None:
        width, height = window.get_size()
        if self._message is not None and now < self._until:
            self._text(window, self._message, (width // 2, height // 6), centre=True)
        # Passes held: a card in each colour, bottom left.
        for i, colour in enumerate(c for c in PASS_COLOURS_RGB if c in passes):
            card = pygame.Rect(16 + i * 34, height - 46, 26, 34)
            pygame.draw.rect(window, (235, 230, 215), card, border_radius=3)
            pygame.draw.rect(window, PASS_COLOURS_RGB[colour], card.inflate(0, -24).move(0, -12))
            pygame.draw.rect(window, SHADOW, card, width=1, border_radius=3)

    def _text(self, window: pygame.Surface, text: str, at: tuple[int, int], centre: bool) -> None:
        for offset, colour in (((2, 2), SHADOW), ((0, 0), TEXT)):
            image = self.font.render(text, True, colour)
            rect = image.get_rect(center=at) if centre else image.get_rect(topleft=at)
            window.blit(image, rect.move(offset))

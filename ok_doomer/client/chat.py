"""Typing a chat message: T opens the line, Enter sends it, Esc drops it.

Keys go to the line while it's open, not to the controls (the game loop
polls them with `keyboard=False`), so typing "wasd" doesn't walk you off.
What can be typed is what the 5x7 font can draw, so what you see is what
everyone gets. The server tidies it again (shared/protocol.py, Say).
"""

import pygame

from ok_doomer.client.hud import draw_text, text_width
from ok_doomer.client.palette import U8Array
from ok_doomer.client.pixelart import FONT_5X7
from ok_doomer.shared.protocol import MAX_CHAT

PROMPT = "Say: "
CURSOR = "_"


class ChatLine:
    def __init__(self) -> None:
        self.text: str | None = None  # None: closed

    @property
    def open(self) -> bool:
        return self.text is not None

    def start(self) -> None:
        self.text = ""

    def key(self, key: int, unicode: str = "") -> str | None:
        """A key pressed while the line's open: what to send (Enter, with
        something typed), else None."""
        if self.text is None:
            return None
        if key in (pygame.K_RETURN, pygame.K_KP_ENTER):
            text, self.text = self.text.strip(), None
            return text or None
        if key == pygame.K_ESCAPE:
            self.text = None
        elif key == pygame.K_BACKSPACE:
            self.text = self.text[:-1]
        elif (
            len(unicode) == 1
            and unicode.isprintable()
            and unicode.upper() in FONT_5X7
            and len(self.text) < MAX_CHAT
        ):
            self.text += unicode
        return None

    def draw(self, pixels: U8Array, scale: int = 1) -> None:
        """The line being typed, bottom left; the end of it, if it's longer
        than the screen is wide."""
        if self.text is None:
            return
        w, h = pixels.shape
        shown = self.text
        while shown and text_width((PROMPT + shown + CURSOR).upper(), FONT_5X7, scale) > w - 8:
            shown = shown[1:]
        line = PROMPT + shown + CURSOR
        draw_text(pixels, line, FONT_5X7, 4, h - 4, scale, centre=False)

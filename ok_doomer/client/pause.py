"""The pause menu: Esc in a game (or the window losing focus).

Over the game, darkened: RESUME, and leaving, worded for what leaving
does. Single-player, the game truly pauses (its server ticks only as
inputs come: client/local.py). On a server it can't: the game goes on
and you stand still, still in it. Leaving a game you joined can be undone
for a while (REJOIN, on the start screen); leaving one you host ends it
for everyone, so it says how many.

`PauseMenu` is the state and what keys and clicks do; the game loop
(client/app.py) feeds it and draws it.
"""

from dataclasses import dataclass

import pygame

from ok_doomer.client.hud import DIM_COLOUR as DIM
from ok_doomer.client.hud import NAME_COLOUR, draw_text, text_width
from ok_doomer.client.palette import COLORMAP, LIGHT_LEVELS, U8Array
from ok_doomer.client.pixelart import FONT_5X7

DARKEN = LIGHT_LEVELS * 5 // 8  # the game behind, this far towards black


@dataclass(frozen=True, slots=True)
class Resume:
    pass


@dataclass(frozen=True, slots=True)
class Leave:
    pass


PauseChoice = Resume | Leave


def leave_label(solo: bool, hosting: bool, others: int) -> str:
    """What leaving is called: what it does."""
    if solo:
        return "QUIT TO MENU"
    if hosting:
        return f"END GAME FOR {others} OTHER{'S' if others != 1 else ''}" if others else "END GAME"
    return "LEAVE GAME"


class PauseMenu:
    def __init__(self, solo: bool, hosting: bool = False) -> None:
        self.solo, self.hosting = solo, hosting
        self.selected = 0  # RESUME: Esc then Enter, or Esc twice, and you're back
        self.rows: list[tuple[int, int]] = []  # each item's rows, as last drawn (the mouse)

    def items(self, others: int = 0) -> list[str]:
        return ["RESUME", leave_label(self.solo, self.hosting, others)]

    def key(self, key: int, text: str = "") -> PauseChoice | None:
        if key == pygame.K_ESCAPE:
            return Resume()
        if key in (pygame.K_UP, pygame.K_w, pygame.K_DOWN, pygame.K_s):
            self.selected = 1 - self.selected
        elif key in (pygame.K_RETURN, pygame.K_KP_ENTER, pygame.K_SPACE):
            return self.activate()
        return None

    def activate(self, index: int | None = None) -> PauseChoice | None:
        index = self.selected if index is None else index
        if index not in (0, 1):
            return None
        self.selected = index
        return Resume() if index == 0 else Leave()

    def item_at(self, row: float) -> int | None:
        for i, (top, bottom) in enumerate(self.rows):
            if top <= row <= bottom:
                return i
        return None

    def draw(self, pixels: U8Array, others: int = 0) -> None:
        """The game darkened; the title and items over it, centred."""
        pixels[:] = COLORMAP[DARKEN][pixels]
        w, h = pixels.shape

        def text(words: str, bottom: int, scale: int, colour: int = NAME_COLOUR) -> int:
            return draw_text(
                pixels, words, FONT_5X7, w / 2, bottom, scale, panel=False, colour=colour
            )

        row = h // 4
        row += text("PAUSED" if self.solo else "MENU", row + 13, 2) + 4
        if not self.solo:  # it can't pause: say so
            row += text("THE GAME GOES ON - YOU STAND STILL", row + 6, 1, DIM) + 4
        row += 10
        self.rows = []
        for i, label in enumerate(self.items(others)):
            chosen = i == self.selected
            height = text(label, row + 13, 2, NAME_COLOUR if chosen else DIM)
            if chosen:  # the mark, before the words (centred without it)
                left = w / 2 - text_width(label, FONT_5X7, 2) / 2 - 2 * 12
                draw_text(pixels, ">", FONT_5X7, left, row + 13, 2, panel=False, centre=False)
            self.rows.append((row, row + height))
            row += height + 8

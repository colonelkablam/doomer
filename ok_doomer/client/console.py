"""The console: every event, everyone's, as it happens (the backquote key
shows it). For debugging: the HUD tells you your own news and the team's;
this tells you what everybody did, with who. Also logged at debug level.
In the system font: it's a tool, not part of the game."""

import logging
from collections import deque

import pygame

log = logging.getLogger(__name__)

LINES_KEPT = 200
LINES_SHOWN = 16


class Console:
    def __init__(self) -> None:
        self.lines: deque[str] = deque(maxlen=LINES_KEPT)
        self.visible = False

    def add(self, line: str) -> None:
        self.lines.append(line)
        log.debug("%s", line)

    def draw(self, window: pygame.Surface, font: pygame.font.Font) -> None:
        """The latest lines, on a dark box across the bottom of the window."""
        if not self.visible:
            return
        shown = list(self.lines)[-LINES_SHOWN:]
        step = font.get_linesize()
        height = step * LINES_SHOWN + 12
        box = pygame.Surface((window.get_width(), height), pygame.SRCALPHA)
        box.fill((0, 0, 0, 190))
        top = window.get_height() - height
        window.blit(box, (0, top))
        first = top + 6 + (LINES_SHOWN - len(shown)) * step  # the newest at the bottom
        for i, line in enumerate(shown):
            image = font.render(line, True, (190, 230, 190))
            window.blit(image, (8, first + i * step))

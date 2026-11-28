"""The pause menu: Esc in a game. RESUME first; leaving worded for what it
does; single-player truly paused, a game on a server not."""

from collections.abc import Iterator
from typing import Any

import numpy as np
import pygame
import pytest

from ok_doomer.client.app import Display, play
from ok_doomer.client.config import ClientConfig, RenderConfig
from ok_doomer.client.hud import NAME_COLOUR
from ok_doomer.client.local import LocalSession
from ok_doomer.client.pause import Leave, PauseMenu, Resume, leave_label
from ok_doomer.client.pixelart import FONT_5X7

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


def test_leaving_says_what_it_does() -> None:
    assert leave_label(solo=True, hosting=False, others=0) == "QUIT TO MENU"
    assert leave_label(solo=False, hosting=False, others=3) == "LEAVE GAME"
    assert leave_label(solo=False, hosting=True, others=0) == "END GAME"
    assert leave_label(solo=False, hosting=True, others=1) == "END GAME FOR 1 OTHER"
    assert leave_label(solo=False, hosting=True, others=2) == "END GAME FOR 2 OTHERS"


def test_resume_is_first_so_esc_twice_or_esc_enter_goes_back() -> None:
    assert PauseMenu(True).key(pygame.K_ESCAPE) == Resume()
    assert PauseMenu(True).key(pygame.K_RETURN) == Resume()
    menu = PauseMenu(False)
    assert menu.key(pygame.K_DOWN) is None and menu.key(pygame.K_RETURN) == Leave()
    menu.key(pygame.K_w)  # back up: W as well as the arrow
    assert menu.activate() == Resume() and menu.activate(5) is None


def test_its_drawn_over_the_game_darkened_and_the_mouse_finds_it() -> None:
    menu = PauseMenu(False, hosting=True)
    game = np.full((320, 200), 7 * 16 + 15, dtype=np.uint8)  # bright, everywhere
    pixels = game.copy()
    menu.draw(pixels, others=2)
    assert (pixels != game).mean() > 0.9  # darkened, nearly all of it
    assert (pixels == NAME_COLOUR).any() and len(menu.rows) == 2
    top, bottom = menu.rows[1]
    assert menu.item_at((top + bottom) / 2) == 1 and menu.item_at(0) is None
    words = {c for w in menu.items(2) + ["MENU", "THE GAME GOES ON - YOU STAND STILL"] for c in w}
    assert not words - set(FONT_5X7) - {" "}


def _frames(
    monkeypatch: pytest.MonkeyPatch, script: list[list[Any]], seen: list[int], session: Any
) -> None:
    """pygame.event.get, a frame at a time: each frame's events from the
    script (keys by key code), and the world's tick noted."""
    frames: Iterator[list[Any]] = iter(script)
    real_get = pygame.event.get

    def get(*args: Any, **kwargs: Any) -> list[Any]:
        real_get()
        seen.append(session.server.world.tick)
        keys = next(frames, [pygame.QUIT])
        return [
            pygame.event.Event(pygame.KEYDOWN, key=k, unicode="", mod=0)
            if isinstance(k, int) and k != pygame.QUIT
            else pygame.event.Event(pygame.QUIT)
            for k in keys
        ]

    monkeypatch.setattr(pygame.event, "get", get)


@pytest.mark.parametrize("solo", [True, False])
def test_single_player_pauses_a_server_game_goes_on(
    monkeypatch: pytest.MonkeyPatch, solo: bool
) -> None:
    pygame.init()
    try:
        display = Display(RenderConfig(vsync=False))
        session = LocalSession(ROOM, "ann")
        monkeypatch.setattr(session, "solo", lambda: solo)  # as if on a server
        idle: list[Any] = [[]] * 8
        script = [*idle, [pygame.K_ESCAPE], *idle, [pygame.K_RETURN], *idle]
        script += [[pygame.K_ESCAPE], [pygame.K_DOWN], [pygame.K_RETURN]]
        seen: list[int] = []
        _frames(monkeypatch, script, seen, session)
        config = ClientConfig(render=RenderConfig(vsync=False, max_fps=60))
        assert play(display, session, config, 0.0) is False  # left, not quit
        before, during, after = seen[2:9], seen[10:18], seen[19:27]
        assert before[-1] > before[0] and after[-1] > after[0]  # running
        assert (during[-1] == during[0]) == solo  # paused only single-player
        session.close()
        display.close()
    finally:
        pygame.quit()

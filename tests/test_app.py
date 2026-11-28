"""Smoke test: the real windowed loop runs (on the dummy video driver) and quits."""

import pygame
import pytest

from ok_doomer.client.app import run
from ok_doomer.client.config import ClientConfig, ControlsConfig, RenderConfig
from ok_doomer.client.input import KeyboardMouseInput
from ok_doomer.client.local import LocalSession
from ok_doomer.shared.events import Event
from ok_doomer.shared.player import PlayerInput
from tests.test_mapfile import MAPS


class QuitAfter(LocalSession):
    def __init__(self, frames: int, *args: object, **kwargs: object) -> None:
        super().__init__(*args, **kwargs)  # type: ignore[arg-type]
        self.frames_left = frames

    def update(self, real_dt: float, player_input: PlayerInput) -> list[Event]:
        self.frames_left -= 1
        if self.frames_left == 0:
            pygame.event.post(pygame.event.Event(pygame.QUIT))
        return super().update(real_dt, player_input)


def test_game_loop_runs_and_quits() -> None:
    session = QuitAfter(5, (MAPS / "test.map").read_text(), "smoke")
    run(session, ClientConfig(render=RenderConfig(vsync=False)), session.player.angle)
    assert session.frames_left <= 0


def test_mouse_capture_falls_back_when_relative_mode_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def no_relative_mode(_: bool) -> None:
        raise pygame.error("No relative mode implementation available")

    monkeypatch.setattr(pygame.mouse, "set_relative_mode", no_relative_mode)
    pygame.init()
    try:
        pygame.display.set_mode((64, 40))
        controls = KeyboardMouseInput(ControlsConfig(), 0.0, 0.4)
        controls.set_captured(True)
        assert controls.captured
        assert pygame.event.get_grab()
        controls.poll(0.016)  # must not raise
        controls.set_captured(False)
        assert not pygame.event.get_grab()
    finally:
        pygame.quit()


@pytest.mark.parametrize(
    "window, integer, expected",
    [
        ((1280, 800), True, (0, 0, 1280, 800)),  # the default window: exactly 4x
        ((1920, 1080), True, (160, 40, 1600, 1000)),  # 5x, black borders
        ((1920, 1080), False, (96, 0, 1728, 1080)),  # as large as fits: 5.4x
        ((2560, 1440), True, (160, 20, 2240, 1400)),  # 7x: height decides (1440 / 200)
        ((300, 150), True, (0, 0, 300, 150)),  # smaller than the game: shrinks, keeps 1
    ],
)
def test_the_picture_fills_the_window_keeping_its_shape(
    window: tuple[int, int], integer: bool, expected: tuple[int, int, int, int]
) -> None:
    from ok_doomer.client.app import view_rect

    assert tuple(view_rect(window, (320, 200), integer)) == expected


class ToggleThenQuit(QuitAfter):
    def update(self, real_dt: float, player_input: PlayerInput) -> list[Event]:
        if self.frames_left == 3:  # F11 part way through: full screen, then quit
            pygame.event.post(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_F11, mod=0))
        return super().update(real_dt, player_input)


@pytest.mark.parametrize("start_fullscreen", [False, True])
def test_full_screen_starts_and_toggles(start_fullscreen: bool) -> None:
    session = ToggleThenQuit(5, (MAPS / "test.map").read_text(), "smoke")
    config = ClientConfig(render=RenderConfig(vsync=False, fullscreen=start_fullscreen))
    run(session, config, session.player.angle)
    assert session.frames_left <= 0


class NetStatsThenQuit(QuitAfter):
    def update(self, real_dt: float, player_input: PlayerInput) -> list[Event]:
        if self.frames_left == 4:  # F3: the network stats in the corner
            pygame.event.post(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_F3, mod=0))
        return super().update(real_dt, player_input)


def test_network_stats_show_with_f3() -> None:
    session = NetStatsThenQuit(6, (MAPS / "test.map").read_text(), "smoke")
    run(session, ClientConfig(render=RenderConfig(vsync=False)), session.player.angle)
    assert session.frames_left <= 0


class ConsoleThenQuit(QuitAfter):
    def update(self, real_dt: float, player_input: PlayerInput) -> list[Event]:
        if self.frames_left == 4:  # the backquote key: the console
            pygame.event.post(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_BACKQUOTE, mod=0))
        return super().update(real_dt, player_input)


def test_the_console_shows_with_backquote() -> None:
    session = ConsoleThenQuit(6, (MAPS / "test.map").read_text(), "smoke")
    run(session, ClientConfig(render=RenderConfig(vsync=False)), session.player.angle)
    assert session.frames_left <= 0

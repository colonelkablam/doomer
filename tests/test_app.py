"""Smoke test: the real windowed loop runs (on the dummy video driver) and quits."""

import pygame
import pytest

from doomer.client.app import run
from doomer.client.config import ClientConfig, ControlsConfig, RenderConfig
from doomer.client.input import KeyboardMouseInput
from doomer.client.render import Camera
from doomer.client.session import LocalSession
from doomer.shared.events import Event
from doomer.shared.mapfile import load_map
from doomer.shared.player import PlayerInput
from doomer.shared.world import new_world
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

    def camera(self, view_angle: float) -> Camera:
        return super().camera(view_angle)


def test_game_loop_runs_and_quits() -> None:
    session = QuitAfter(5, new_world(load_map(MAPS / "test.map"), "coop"), "smoke")
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
        controls = KeyboardMouseInput(ControlsConfig(), 0.0)
        controls.set_captured(True)
        assert controls.captured
        assert pygame.event.get_grab()
        controls.poll(0.016)  # must not raise
        controls.set_captured(False)
        assert not pygame.event.get_grab()
    finally:
        pygame.quit()

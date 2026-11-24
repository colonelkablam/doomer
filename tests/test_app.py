"""Smoke test: the real windowed loop runs (on the dummy video driver) and quits."""

import pygame

from doomer.client.app import run
from doomer.client.config import ClientConfig, RenderConfig
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

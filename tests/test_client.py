import math

import pygame
import pytest

from doomer.client.config import ClientConfig, ControlsConfig, RenderConfig
from doomer.client.input import Action, compose_input, turn
from doomer.client.palette import WALL_COLOURS
from doomer.client.render import Camera, Renderer
from doomer.client.screenshot import render_frame
from doomer.client.session import LocalSession
from doomer.shared.config import from_dict
from doomer.shared.mapfile import parse_map
from doomer.shared.player import PlayerInput
from doomer.shared.world import new_world

ROOM = parse_map(
    """\
[walls]
#########
#.......2
#.......2
#.......2
#########

[things]
coop_start 1.5 2.5 0
exit 7.5 2.5 0
"""
)
CFG = RenderConfig()


def column_height(frame: pygame.Surface, col: int) -> int:
    """Number of wall-coloured pixels in a column (neither ceiling nor floor)."""
    flat = {CFG.ceiling_colour[:3], CFG.floor_colour[:3]}
    return sum(1 for y in range(CFG.height) if tuple(frame.get_at((col, y)))[:3] not in flat)


def test_wall_ahead_has_its_colour_at_the_horizon() -> None:
    frame = render_frame(ROOM, Camera(4.5, 2.5, 0.0))
    centre = tuple(frame.get_at((CFG.width // 2, CFG.height // 2)))[:3]
    r, g, b = WALL_COLOURS[2]
    # Shaded, so darker but in proportion: red dominates.
    assert centre[0] > centre[1] and centre[0] > centre[2]
    assert centre[0] <= r
    assert tuple(frame.get_at((0, 0)))[:3] == CFG.ceiling_colour
    assert tuple(frame.get_at((0, CFG.height - 1)))[:3] == CFG.floor_colour


def test_wall_height_is_inversely_proportional_to_distance() -> None:
    # The east wall's face is at x = 8.
    near = column_height(render_frame(ROOM, Camera(5.5, 2.5, 0.0)), CFG.width // 2)  # 2.5 away
    far = column_height(render_frame(ROOM, Camera(3.0, 2.5, 0.0)), CFG.width // 2)  # 5.0 away
    assert near == pytest.approx(2 * far, abs=2)
    assert near == pytest.approx(Renderer(CFG).projection / 2.5, abs=1)


def test_no_fisheye_flat_wall_has_equal_height_across_screen() -> None:
    # Facing a flat wall square-on, perpendicular distance is the same in every
    # column, so every column must be the same height.
    # The room is 3 cells tall, so from 4 cells away the east wall spans only the
    # middle ~58% of the screen; outside that, rays hit the side walls.
    frame = render_frame(ROOM, Camera(4.0, 2.5, 0.0))
    heights = {column_height(frame, c) for c in (80, 120, 160, 200, 240)}
    assert max(heights) - min(heights) <= 1


def test_compose_input() -> None:
    assert compose_input({Action.FORWARD, Action.STRAFE_LEFT}, 1.0) == PlayerInput(1.0, -1.0, 1.0)
    assert compose_input({Action.FORWARD, Action.BACK}, 0.0) == PlayerInput(0.0, 0.0, 0.0)


def test_turn_mouse_right_is_clockwise_and_wraps() -> None:
    c = ControlsConfig(mouse_sensitivity=0.01, turn_speed=1.0)
    assert turn(0.0, set(), 10, 0.0, c) == pytest.approx(0.1)
    assert turn(0.0, {Action.TURN_LEFT}, 0, 0.5, c) == pytest.approx(math.tau - 0.5)


def test_local_session_steps_at_fixed_rate_and_interpolates() -> None:
    session = LocalSession(new_world(ROOM, "coop"), "test")
    dt = session.world.config.dt
    start_x = session.player.x
    session.update(dt * 2.5, PlayerInput(forward=1.0, angle=0.0))
    assert session.world.tick == 2
    step_len = session.world.config.movement.walk_speed * dt
    assert session.player.x == pytest.approx(start_x + 2 * step_len)
    # Camera sits halfway between tick 1 and tick 2.
    cam = session.camera(0.3)
    assert cam.x == pytest.approx(start_x + 1.5 * step_len)
    assert cam.angle == 0.3


def test_client_config_from_toml_dict() -> None:
    cfg = from_dict(
        ClientConfig,
        {"render": {"window_scale": 4, "floor_colour": [1, 2, 3]}, "controls": {"forward": ["i"]}},
    )
    assert cfg.render.window_scale == 4
    assert cfg.render.floor_colour == (1, 2, 3)
    assert cfg.controls.forward == ("i",)

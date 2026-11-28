import math

import pygame
import pytest

from ok_doomer.client.config import ClientConfig, ControlsConfig, RenderConfig
from ok_doomer.client.input import Action, compose_input, look, turn
from ok_doomer.client.local import LocalSession
from ok_doomer.client.palette import GREEN, MAGENTA, PINK, WOOD
from ok_doomer.client.render import Camera, Renderer
from ok_doomer.client.screenshot import render_frame
from ok_doomer.shared.config import from_dict
from ok_doomer.shared.mapfile import parse_map
from ok_doomer.shared.player import PlayerInput

ROOM_TEXT = """\
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
ROOM = parse_map(ROOM_TEXT)
# Ceiling and floor from ramps the wood panelling (wall type 2) never uses, so
# any pixel in another ramp is wall.
CFG = RenderConfig(ceiling_colour=(250, 130, 190), floor_colour=(110, 200, 90))
CEILING_RAMP, FLOOR_RAMP = PINK, GREEN


def render(camera: Camera) -> pygame.Surface:
    return render_frame(ROOM, camera, CFG, things=False)


def ramp_at(frame: pygame.Surface, x: int, y: int) -> int:
    return int(frame.get_at_mapped((x, y))) >> 4


def column_height(frame: pygame.Surface, col: int) -> int:
    """Number of wall pixels in a column (neither ceiling nor floor)."""
    flat = {CEILING_RAMP, FLOOR_RAMP}
    return sum(1 for y in range(CFG.height) if ramp_at(frame, col, y) not in flat)


def test_frame_is_8_bit_with_wall_texture_at_horizon() -> None:
    frame = render(Camera(4.5, 2.5, 0.0))
    assert frame.get_bitsize() == 8
    assert ramp_at(frame, CFG.width // 2, CFG.height // 2) == WOOD  # type 2 = wood panelling
    assert ramp_at(frame, 0, 0) == CEILING_RAMP
    assert ramp_at(frame, 0, CFG.height - 1) == FLOOR_RAMP


def test_floor_darkens_towards_the_horizon() -> None:
    frame = render(Camera(4.5, 2.5, math.pi))  # facing the long way down the room
    near = int(frame.get_at_mapped((0, CFG.height - 1))) & 0x0F
    far = int(frame.get_at_mapped((0, CFG.height // 2 + 2))) & 0x0F
    assert near > far


def test_unknown_wall_type_uses_missing_texture() -> None:
    # 'z' (wall type 35) has no texture registered.
    odd = parse_map("[walls]\n#####\n#...z\n#####\n")
    frame = Renderer(CFG).render_indices(odd, Camera(1.5, 1.5, 0.0))
    assert (int(frame[CFG.width // 2, CFG.height // 2]) >> 4) == MAGENTA


def test_wall_height_is_inversely_proportional_to_distance() -> None:
    # The east wall's face is at x = 8.
    near = column_height(render(Camera(5.5, 2.5, 0.0)), CFG.width // 2)  # 2.5 away
    far = column_height(render(Camera(3.0, 2.5, 0.0)), CFG.width // 2)  # 5.0 away
    assert near == pytest.approx(2 * far, abs=2)
    assert near == pytest.approx(Renderer(CFG).projection / 2.5, abs=1)


def test_no_fisheye_flat_wall_has_equal_height_across_screen() -> None:
    # Facing a flat wall square-on, perpendicular distance is the same in every
    # column, so every column must be the same height.
    # The room is 3 cells tall, so from 4 cells away the east wall spans only the
    # middle ~half of the screen; outside that, rays hit the side walls.
    frame = render(Camera(4.0, 2.5, 0.0))
    heights = {column_height(frame, c) for c in (100, 130, 160, 190, 220)}
    assert max(heights) - min(heights) <= 1


def test_compose_input() -> None:
    assert compose_input({Action.FORWARD, Action.STRAFE_LEFT}, 1.0) == PlayerInput(1.0, -1.0, 1.0)
    assert compose_input({Action.FORWARD, Action.BACK}, 0.0) == PlayerInput(0.0, 0.0, 0.0)


def test_turn_mouse_right_is_clockwise_and_wraps() -> None:
    c = ControlsConfig(mouse_sensitivity=0.01, turn_speed=1.0)
    assert turn(0.0, set(), 10, 0.0, c) == pytest.approx(0.1)
    assert turn(0.0, {Action.TURN_LEFT}, 0, 0.5, c) == pytest.approx(math.tau - 0.5)


def test_local_session_steps_at_fixed_rate_and_interpolates() -> None:
    session = LocalSession(ROOM_TEXT, "test")
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


def test_compose_input_carries_pitch_jump_and_crouch() -> None:
    inp = compose_input({Action.JUMP, Action.CROUCH}, 0.5, pitch=0.2)
    assert (inp.angle, inp.pitch, inp.jump, inp.crouch) == (0.5, 0.2, True, True)


def test_mouse_look_up_down_inverts_and_clamps() -> None:
    c = ControlsConfig(mouse_sensitivity=0.01)
    assert look(0.0, -10, c, 1.0) == pytest.approx(0.1)  # mouse forward (dy < 0): look up
    inverted = ControlsConfig(mouse_sensitivity=0.01, invert_mouse_y=True)
    assert look(0.0, -10, inverted, 1.0) == pytest.approx(-0.1)
    assert look(0.0, -1000, c, 0.4) == pytest.approx(0.4)  # clamped


def test_session_camera_follows_feet_and_eases_eye_height() -> None:
    session = LocalSession(ROOM_TEXT, "test")
    move = session.world.config.movement
    dt = session.world.config.dt
    assert session.camera(0.0).z == pytest.approx(move.eye_height)
    session.update(dt, PlayerInput(crouch=True))
    eased = session.camera(0.0).z
    assert move.crouch_eye_height < eased < move.eye_height  # on its way down
    for _ in range(30):
        session.update(dt, PlayerInput(crouch=True))
    assert session.camera(0.0, pitch=0.1).z == pytest.approx(move.crouch_eye_height)
    assert session.camera(0.0, pitch=0.1).pitch == 0.1


def test_renderer_caches_follow_the_map_not_its_id() -> None:
    # Render many maps of different sizes in turn: a cache keyed on id() could
    # serve a freed map's table to a new map that reused its id.
    renderer = Renderer(CFG)
    for width in range(5, 25):
        m = parse_map(
            "[walls]\n" + "#" * width + "\n#" + "." * (width - 2) + "#\n" + "#" * width + "\n"
        )
        renderer.render_indices(m, Camera(1.5, 1.5, 0.0))
        assert renderer._map_arrays is not None
        assert renderer._map_arrays.walls.shape == (3, width)

"""Power: the breaker, level flags, and emergency lighting."""

import math

import numpy as np
import pytest

from doomer.client.config import LightingConfig, RenderConfig
from doomer.client.lighting import bake_light_map
from doomer.client.palette import OCHRE
from doomer.client.render import Camera, Renderer
from doomer.shared.events import SwitchThrown
from doomer.shared.mapfile import parse_map
from doomer.shared.player import PlayerInput
from doomer.shared.sim import step
from doomer.shared.switches import POWER
from doomer.shared.world import drain_events, new_world, spawn_player

SOUTH, NORTH = math.pi / 2, -math.pi / 2

# A room, x 1..10, y 1..3. The breaker on the south wall (its face is y = 4)
# at x = 2.5; a main-circuit panel at x = 3.5 and an emergency one at x = 8.5.
ROOM = """\
[walls]
############
#..........#
#..........#
#..........#
############

[things]
coop_start 2.5 1.5 90
exit 10.5 1.5 0
breaker 2.5 3.99 270
ceiling_light 3.5 2.5 0
ceiling_light 8.5 2.5 0 circuit=emergency
"""


def test_a_floor_with_a_breaker_starts_without_power_one_without_with() -> None:
    assert POWER not in new_world(parse_map(ROOM), "coop").flags
    no_breaker = ROOM.replace("breaker 2.5 3.99 270\n", "")
    assert POWER in new_world(parse_map(no_breaker), "coop").flags


def test_throwing_the_breaker_restores_the_power_once() -> None:
    world = new_world(parse_map(ROOM), "coop")
    p = spawn_player(world, 1, "p")
    p.x, p.y = 2.5, 3.2  # in front of it, facing it
    step(world, {1: PlayerInput(angle=SOUTH, use=True)})
    assert POWER in world.flags and world.switches[0].on
    events = drain_events(world)
    assert [e.flag for e in events if isinstance(e, SwitchThrown)] == [POWER]
    step(world, {1: PlayerInput(angle=SOUTH)})
    step(world, {1: PlayerInput(angle=SOUTH, use=True)})  # again: nothing more
    assert not any(isinstance(e, SwitchThrown) for e in drain_events(world))


@pytest.mark.parametrize("x, y, angle", [(2.5, 3.2, NORTH), (2.5, 1.5, SOUTH)])
def test_you_must_face_the_breaker_and_be_near_it(x: float, y: float, angle: float) -> None:
    world = new_world(parse_map(ROOM), "coop")
    p = spawn_player(world, 1, "p")
    p.x, p.y = x, y
    step(world, {1: PlayerInput(angle=angle, use=True)})
    assert POWER not in world.flags


def test_the_breaker_cabinet_blocks_you() -> None:
    world = new_world(parse_map(ROOM), "coop")
    p = spawn_player(world, 1, "p")
    p.x, p.y = 2.5, 2.0
    for _ in range(90):
        step(world, {1: PlayerInput(forward=1.0, angle=SOUTH)})
    assert p.y == pytest.approx(3.99 - 0.16 - world.config.movement.radius)


def test_without_power_only_emergency_panels_light_the_floor() -> None:
    cfg = LightingConfig(lift=1.0)
    light = bake_light_map(parse_map(ROOM), cfg)
    dark = light.without_power()

    def at(m: object, x: float) -> float:
        return float(m.at(np.array([x]), np.array([2.5]))[0])  # type: ignore[attr-defined]

    assert at(light, 3.5) > cfg.ambient + 0.3  # under the main panel, powered
    assert at(dark, 3.5) < cfg.unpowered_ambient + 0.05  # dead without power
    assert at(dark, 8.5) > cfg.unpowered_ambient + 0.3  # the emergency panel still lights
    assert at(dark, 8.5) < at(light, 8.5)  # (a lower ambient, though)


def test_dead_panels_and_mains_signs_go_dark_without_power() -> None:
    from doomer.client.render import SpriteInstance

    room = parse_map(ROOM.replace("[things]", "[ceiling]\n" + "gggggggggggg\n" * 5 + "\n[things]"))
    r = Renderer(RenderConfig(), LightingConfig())
    sign = SpriteInstance("welcome_sign", 10.99, 2.0, 0.0, math.pi)
    camera = Camera(5.0, 2.0, 0.0, pitch=0.3)
    on = r.render_indices(room, camera, [sign], power=True).copy()
    off = r.render_indices(room, camera, [sign], power=False).copy()

    def lit_brass(frame: np.ndarray) -> int:  # the letters' ramp, brightly lit
        return int((((frame >> 4) == OCHRE) & ((frame & 0x0F) >= 11)).sum())

    assert lit_brass(on) > 100
    assert lit_brass(off) < lit_brass(on) // 4


def test_hud_says_the_power_is_back() -> None:
    from doomer.client.hud import message_for

    assert message_for(SwitchThrown(0, 0, 1, POWER)) == "The power is back on"

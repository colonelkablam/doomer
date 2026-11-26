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
    assert p.y == pytest.approx(3.99 - 0.24 - world.config.movement.radius)  # its footprint


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


def test_indicator_lamps_blink_and_go_dark_without_mains() -> None:
    from doomer.client.boxes import Box, lit_now

    lamp = Box((0, 0.01), (0, 0.01), (0, 0.01), (0,) * 5, indicator=True, blink=1.0)
    seen = {lit_now(lamp, 0.0, 0.0, t / 10, power=True) for t in range(10)}
    assert seen == {True, False}  # on half the time
    mains = Box((0, 0.01), (0, 0.01), (0, 0.01), (0,) * 5, indicator=True, mains=True)
    assert lit_now(mains, 0, 0, 0.0, power=True) and not lit_now(mains, 0, 0, 0.0, power=False)


def test_the_breaker_blinks_red_until_thrown_then_shows_steady_green() -> None:
    import dataclasses

    from doomer.client.boxes import switch_boxes
    from doomer.shared.switches import switches_from_map

    (off,) = switches_from_map(parse_map(ROOM))
    on = dataclasses.replace(off, on=True)
    (red,) = [p.box for p in switch_boxes([off]) if p.box.indicator]
    (green,) = [p.box for p in switch_boxes([on]) if p.box.indicator]
    assert red.blink > 0 and green.blink == 0 and not red.mains  # on its own battery
    assert red.y != green.y  # two lamps side by side


def test_server_lamps_blink_out_of_step_and_only_with_power() -> None:
    from doomer.client.boxes import BOX_MODELS

    for kind in ("tape", "disk"):
        cabinet, *lamps = BOX_MODELS[f"server:{kind}"]
        assert len(lamps) == 4 and all(lp.indicator and lp.mains for lp in lamps)
        assert len({(lp.blink, lp.blink_phase) for lp in lamps}) == 4  # each its own timer
        assert all(lp.x[0] >= cabinet.x[1] for lp in lamps)  # on its front


def test_the_plant_room_breaker_and_servers_stand_on_solid_walls() -> None:
    from doomer.shared.mapfile import load_map

    office = load_map("maps/office.map")
    for t in office.things_of_type("breaker") + office.things_of_type("server"):
        bx, by = t.x - 0.05 * math.cos(t.angle), t.y - 0.05 * math.sin(t.angle)
        cell = (math.floor(bx), math.floor(by))
        assert office.is_full_wall(*cell) and office.walls[cell[1]][cell[0]] != 4  # not a window

"""Turning keyboard and mouse state into PlayerInput.

The pygame-specific part (KeyboardMouseInput) is thin; the logic that matters
(compose_input) is a pure function so it can be tested without a display.
"""

import logging
import math
from collections.abc import Set
from enum import Enum
from typing import Protocol

import pygame

from doomer.client.config import ControlsConfig
from doomer.shared.player import PlayerInput

log = logging.getLogger(__name__)


class Action(Enum):
    FORWARD = "forward"
    BACK = "back"
    STRAFE_LEFT = "strafe_left"
    STRAFE_RIGHT = "strafe_right"
    TURN_LEFT = "turn_left"
    TURN_RIGHT = "turn_right"
    JUMP = "jump"
    CROUCH = "crouch"


class InputSource(Protocol):
    """Anything that produces a player's input each frame: keyboard, bot, replay."""

    def poll(self, real_dt: float) -> PlayerInput: ...


def compose_input(actions: Set[Action], angle: float, pitch: float = 0.0) -> PlayerInput:
    forward = (Action.FORWARD in actions) - (Action.BACK in actions)
    strafe = (Action.STRAFE_RIGHT in actions) - (Action.STRAFE_LEFT in actions)
    return PlayerInput(
        forward=float(forward),
        strafe=float(strafe),
        angle=angle,
        pitch=pitch,
        jump=Action.JUMP in actions,
        crouch=Action.CROUCH in actions,
    )


def turn(
    angle: float, actions: Set[Action], mouse_dx: float, real_dt: float, c: ControlsConfig
) -> float:
    """New view angle. Positive angle is clockwise, so mouse right / turn right adds."""
    keys = (Action.TURN_RIGHT in actions) - (Action.TURN_LEFT in actions)
    angle += mouse_dx * c.mouse_sensitivity + keys * c.turn_speed * real_dt
    return angle % math.tau


def look(pitch: float, mouse_dy: float, c: ControlsConfig, limit: float) -> float:
    """New pitch. Pulling the mouse back (positive dy) looks up, unless inverted."""
    pitch += (mouse_dy if c.invert_mouse_y else -mouse_dy) * c.mouse_sensitivity
    return max(-limit, min(limit, pitch))


class KeyboardMouseInput:
    """Reads pygame keyboard and relative mouse state. Requires pygame.init()."""

    def __init__(self, controls: ControlsConfig, angle: float, pitch_limit: float) -> None:
        self.controls = controls
        self.angle = angle
        self.pitch = 0.0
        self.pitch_limit = pitch_limit
        self.captured = False
        # SDL relative mode where available; otherwise grab + hidden cursor +
        # warp back to the window centre each frame (some VMs, remote desktops).
        self._warp_fallback = False
        self._bindings: list[tuple[int, Action]] = []
        for action in Action:
            for key_name in getattr(controls, action.value):
                self._bindings.append((pygame.key.key_code(key_name), action))

    def set_captured(self, captured: bool) -> None:
        """Capture hides the cursor and reports raw mouse motion for mouse look."""
        self.captured = captured
        if not self._warp_fallback:
            try:
                pygame.mouse.set_relative_mode(captured)
            except pygame.error as e:
                log.warning("relative mouse mode unavailable (%s); using cursor warping", e)
                self._warp_fallback = True
        if self._warp_fallback:
            pygame.event.set_grab(captured)
            pygame.mouse.set_visible(not captured)
            self._recentre()
        pygame.mouse.get_rel()  # discard motion accumulated while not captured

    def poll(self, real_dt: float) -> PlayerInput:
        pressed = pygame.key.get_pressed()
        actions = {action for code, action in self._bindings if pressed[code]}
        mouse_dx, mouse_dy = pygame.mouse.get_rel() if self.captured else (0, 0)
        if self.captured and self._warp_fallback:
            self._recentre()
        self.angle = turn(self.angle, actions, mouse_dx, real_dt, self.controls)
        self.pitch = look(self.pitch, mouse_dy, self.controls, self.pitch_limit)
        return compose_input(actions, self.angle, self.pitch)

    def _recentre(self) -> None:
        surface = pygame.display.get_surface()
        if surface is not None:
            pygame.mouse.set_pos(surface.get_rect().center)
            pygame.mouse.get_rel()  # the warp itself registers as motion; drop it

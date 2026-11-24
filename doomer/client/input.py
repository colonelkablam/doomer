"""Turning keyboard and mouse state into PlayerInput.

The pygame-specific part (KeyboardMouseInput) is thin; the logic that matters
(compose_input) is a pure function so it can be tested without a display.
"""

import math
from collections.abc import Set
from enum import Enum
from typing import Protocol

import pygame

from doomer.client.config import ControlsConfig
from doomer.shared.player import PlayerInput


class Action(Enum):
    FORWARD = "forward"
    BACK = "back"
    STRAFE_LEFT = "strafe_left"
    STRAFE_RIGHT = "strafe_right"
    TURN_LEFT = "turn_left"
    TURN_RIGHT = "turn_right"


class InputSource(Protocol):
    """Anything that produces a player's input each frame: keyboard, bot, replay."""

    def poll(self, real_dt: float) -> PlayerInput: ...


def compose_input(actions: Set[Action], angle: float) -> PlayerInput:
    forward = (Action.FORWARD in actions) - (Action.BACK in actions)
    strafe = (Action.STRAFE_RIGHT in actions) - (Action.STRAFE_LEFT in actions)
    return PlayerInput(forward=float(forward), strafe=float(strafe), angle=angle)


def turn(
    angle: float, actions: Set[Action], mouse_dx: float, real_dt: float, c: ControlsConfig
) -> float:
    """New view angle. Positive angle is clockwise, so mouse right / turn right adds."""
    keys = (Action.TURN_RIGHT in actions) - (Action.TURN_LEFT in actions)
    angle += mouse_dx * c.mouse_sensitivity + keys * c.turn_speed * real_dt
    return angle % math.tau


class KeyboardMouseInput:
    """Reads pygame keyboard and relative mouse state. Requires pygame.init()."""

    def __init__(self, controls: ControlsConfig, angle: float) -> None:
        self.controls = controls
        self.angle = angle
        self.captured = False
        self._bindings: list[tuple[int, Action]] = []
        for action in Action:
            for key_name in getattr(controls, action.value):
                self._bindings.append((pygame.key.key_code(key_name), action))

    def set_captured(self, captured: bool) -> None:
        """Capture hides the cursor and reports raw mouse motion for mouse look."""
        self.captured = captured
        pygame.mouse.set_relative_mode(captured)
        pygame.mouse.get_rel()  # discard motion accumulated while not captured

    def poll(self, real_dt: float) -> PlayerInput:
        pressed = pygame.key.get_pressed()
        actions = {action for code, action in self._bindings if pressed[code]}
        mouse_dx = pygame.mouse.get_rel()[0] if self.captured else 0
        self.angle = turn(self.angle, actions, mouse_dx, real_dt, self.controls)
        return compose_input(actions, self.angle)

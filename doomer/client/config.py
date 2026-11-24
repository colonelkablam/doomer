from dataclasses import dataclass, field

Colour = tuple[int, ...]


@dataclass(frozen=True, slots=True)
class RenderConfig:
    width: int = 320  # internal resolution; scaled up to the window
    height: int = 200
    window_scale: int = 3
    fov_degrees: float = 75.0  # horizontal
    max_fps: int = 144  # 0 = uncapped
    vsync: bool = True
    shade_distance: float = 14.0  # walls fade to their darkest at this distance
    # Mapped to the nearest palette entry.
    ceiling_colour: Colour = (205, 200, 180)  # off-white ceiling tiles
    floor_colour: Colour = (120, 108, 96)  # grey-brown office carpet
    show_fps: bool = True


@dataclass(frozen=True, slots=True)
class ControlsConfig:
    mouse_sensitivity: float = 0.0025  # radians per pixel of mouse movement
    turn_speed: float = 2.6  # radians per second for keyboard turning
    # Key names as understood by pygame.key.key_code().
    forward: tuple[str, ...] = ("w", "up")
    back: tuple[str, ...] = ("s", "down")
    strafe_left: tuple[str, ...] = ("a",)
    strafe_right: tuple[str, ...] = ("d",)
    turn_left: tuple[str, ...] = ("left",)
    turn_right: tuple[str, ...] = ("right",)


@dataclass(frozen=True, slots=True)
class ClientConfig:
    render: RenderConfig = field(default_factory=RenderConfig)
    controls: ControlsConfig = field(default_factory=ControlsConfig)

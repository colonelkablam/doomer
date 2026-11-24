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
class LightingConfig:
    """The baked light map (doomer/client/lighting.py). Light values are
    brightness factors: 1 = full brightness, before distance fade."""

    enabled: bool = True  # False: distance fade only (as before lighting; slow machines)
    ambient: float = 0.3  # light everywhere, with no panel in reach
    panel_intensity: float = 1.0  # one ceiling panel, at distance 0
    falloff_radius: float = 1.2  # cells: a panel's light halves at this distance
    cutoff: float = 6.0  # cells: panels light nothing further away than this
    samples_per_cell: int = 4  # light map resolution
    # Distance fade with lighting on, replacing RenderConfig.shade_distance:
    # gentler, since the light map does the local darkening and the two
    # multiply (at 14 cells, lit rooms go murky in the distance).
    shade_distance: float = 20.0
    # Ceilings lower than a storey (the soffit over a doorway) sit in shadow:
    # they take the plain light times this, not the square root ceilings get.
    soffit_shade: float = 0.85


@dataclass(frozen=True, slots=True)
class ControlsConfig:
    mouse_sensitivity: float = 0.0025  # radians per pixel of mouse movement
    invert_mouse_y: bool = False  # True: push the mouse forward to look down
    turn_speed: float = 2.6  # radians per second for keyboard turning
    # Key names as understood by pygame.key.key_code().
    forward: tuple[str, ...] = ("w", "up")
    back: tuple[str, ...] = ("s", "down")
    strafe_left: tuple[str, ...] = ("a",)
    strafe_right: tuple[str, ...] = ("d",)
    turn_left: tuple[str, ...] = ("left",)
    turn_right: tuple[str, ...] = ("right",)
    jump: tuple[str, ...] = ("space",)
    crouch: tuple[str, ...] = ("left ctrl", "c")
    use: tuple[str, ...] = ("e",)  # open and close doors


@dataclass(frozen=True, slots=True)
class ClientConfig:
    render: RenderConfig = field(default_factory=RenderConfig)
    controls: ControlsConfig = field(default_factory=ControlsConfig)
    lighting: LightingConfig = field(default_factory=LightingConfig)

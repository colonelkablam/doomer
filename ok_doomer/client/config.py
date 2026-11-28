from dataclasses import dataclass, field

Colour = tuple[int, ...]


@dataclass(frozen=True, slots=True)
class RenderConfig:
    width: int = 320  # internal resolution; scaled up to the window
    height: int = 200
    window_scale: int = 4  # the window starts at width x height times this (resizable)
    fullscreen: bool = False  # start full screen (F11 or Alt+Enter toggles it)
    # Scale the picture by a whole number (every game pixel the same size,
    # crisp; black borders fill the rest), or as large as fits (fills more of
    # the screen, but pixels come out unevenly sized).
    integer_scaling: bool = True
    fov_degrees: float = 75.0  # horizontal
    max_fps: int = 144  # 0 = uncapped
    vsync: bool = True
    shade_distance: float = 14.0  # walls fade to their darkest at this distance
    # Floors and ceilings are textured by area (ok_doomer/client/areas.py). Set
    # both colours to draw them flat instead (mapped to the nearest palette
    # entry): cheaper, and tests use it to tell floor from ceiling.
    ceiling_colour: Colour | None = None
    floor_colour: Colour | None = None
    show_fps: bool = True
    # Other players' names over their heads: the terminals' green pixel font,
    # `name_scale` frame pixels a font pixel (1: 3x5 letters), on a black
    # panel (False: a black outline round the letters).
    name_scale: int = 1
    name_panel: bool = True
    # HUD messages ("Bob joined", "Picked up the blue pass"): the 5x7 pixel
    # font, green on black, this many frame pixels a font pixel.
    message_scale: int = 1
    max_messages: int = 6  # stacked at once; each shows for 3 s


@dataclass(frozen=True, slots=True)
class LightingConfig:
    """The baked light map (ok_doomer/client/lighting.py). Light values are
    brightness factors: 1 = full brightness, before distance fade."""

    enabled: bool = True  # False: distance fade only (as before lighting; slow machines)
    ambient: float = 0.3  # light everywhere, with no panel in reach
    # With the power off (a floor whose breaker isn't thrown yet): the ambient,
    # and only the emergency lights: exit signs and `circuit=emergency` panels.
    unpowered_ambient: float = 0.1
    panel_intensity: float = 1.0  # one ceiling panel, at distance 0
    exit_sign_intensity: float = 0.7  # an exit sign's glow, as a fraction of a panel's
    # Light is raised to this power (1 = as baked): below 1 lifts the dim
    # end, where the palette's 16 shades are far apart (shade 2 -> 3 is a
    # third brighter), so dark rooms are less blotchy. Bright light hardly
    # changes (0.9^0.8 = 0.92).
    lift: float = 0.65
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

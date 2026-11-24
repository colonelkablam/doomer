"""Registry of thing types that can appear in a map's [things] section.

Adding a new thing (a pickup, a door, a monster) starts with registering its
type here. Behaviour lives elsewhere and looks things up by type name. How a
thing looks is the client's business (doomer/client/sprites.py), keyed by the
same name.
"""

from collections.abc import Mapping
from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class Footprint:
    """Part of a piece of furniture's floor plan, in the thing's own frame
    (x towards its front, the way its angle faces; y to its right), cells;
    and the height of its top (walls), which things placed on it rest on.
    doomer/shared/furniture.py turns them into obstacles and surfaces."""

    x: tuple[float, float]
    y: tuple[float, float]
    top: float


# ID passes, by colour: staff, facilities, executive (docs/GAME.md). A pass
# thing is a pickup; a door's `lock` names the pass that opens it.
PASS_COLOURS = ("blue", "red", "gold")


@dataclass(frozen=True, slots=True)
class ThingType:
    name: str
    description: str
    blocking: bool = False  # stops players (and later shots and enemies)
    radius: float = 0.0  # half-width of the square footprint, in cells
    # Properties a map may give this thing (`key=value` after its angle), each
    # with the values allowed. Anything else is a map error.
    properties: Mapping[str, tuple[str, ...]] = field(default_factory=dict)
    # A blocking thing's floor plan, if not the square `radius`: rectangles,
    # turned to its angle (snapped to quarter turns).
    footprints: tuple[Footprint, ...] = ()


THING_TYPES: dict[str, ThingType] = {}


def register_thing_type(thing_type: ThingType) -> ThingType:
    if thing_type.name in THING_TYPES:
        raise ValueError(f"thing type already registered: {thing_type.name}")
    if thing_type.blocking and thing_type.radius <= 0:
        raise ValueError(f"blocking thing type {thing_type.name} needs a radius")
    THING_TYPES[thing_type.name] = thing_type
    return thing_type


# Markers
COOP_START = register_thing_type(ThingType("coop_start", "Co-op player start"))
EXIT = register_thing_type(ThingType("exit", "Co-op level exit (shown as an exit sign)"))
DM_SPAWN = register_thing_type(ThingType("dm_spawn", "Deathmatch spawn point"))

# Office furniture (docs/GAME.md)
for _name, _description, _blocking, _radius in (
    # Radii match the sprite widths in doomer/client/sprites.py (64 px = 1 cell).
    ("desk", "Desk with a beige PC and CRT", True, 0.32),
    ("water_cooler", "Water cooler", True, 0.12),
    ("filing_cabinet", "Filing cabinet", True, 0.16),
    ("ficus", "Potted ficus", True, 0.16),
    # Drawn as a panel set into its cell's ceiling (not a sprite).
    ("ceiling_light", "Fluorescent ceiling light panel", False, 0.0),
    # Sits on a desk-pod cell (a low block), so it needs no footprint of its own.
    ("terminal", "Beige PC and CRT, for standing on a desk pod", False, 0.0),
    # Drawn as solid boxes (client/boxes.py); faces its map angle, like terminal.
    ("mac", "Apple Macintosh SE/30 with keyboard and mouse, for a desk", False, 0.0),
):
    register_thing_type(ThingType(_name, _description, _blocking, _radius))

# The reception desk (drawn by doomer/client/boxes.py; its dimensions match):
# angle = the way its front faces (the visitors' side). A transaction ledge
# along the front, and the work surface behind it, at their own heights:
# things placed on it (a computer, a pass) rest on whichever they're over.
RECEPTION_DESK = register_thing_type(
    ThingType(
        "reception_desk",
        "Reception desk: a ledge for visitors, a work surface behind",
        blocking=True,
        radius=0.3,
        footprints=(
            Footprint((0.14, 0.31), (-0.92, 0.92), 0.425),  # the ledge
            Footprint((-0.30, 0.14), (-0.92, 0.92), 0.27),  # the work surface
        ),
    )
)

# Signs: drawn by doomer/client/boxes.py. Stand it at a wall's face (just in
# front of it, in the room); angle = the way it faces, out of the wall.
WELCOME_SIGN = register_thing_type(
    ThingType("welcome_sign", "Brass letters: Welcome to Boomer Industries")
)

# Doors (doomer/shared/doors.py): in a doorway cell; angle = the way it opens.
# `lock=<colour>`: only opens for a team holding that pass.
for _name, _description in (
    ("door_hinged", "Hinged office door: press use to open or close"),
    ("door_double", "Double office doors: press use to open or close"),
    ("door_sliding", "Sliding glass doors: open by themselves as you approach"),
):
    register_thing_type(ThingType(_name, _description, properties={"lock": PASS_COLOURS}))

# ID passes (doomer/shared/pickups.py): walk over one to pick it up; the team
# shares it. Drawn as a sprite on whatever it lies on.
for _colour in PASS_COLOURS:
    register_thing_type(ThingType(f"pass_{_colour}", f"{_colour.capitalize()} ID pass"))

"""Registry of thing types that can appear in a map's [things] section.

Adding a new thing (a pickup, a door, a monster) starts with registering its
type here. Behaviour lives elsewhere and looks things up by type name. How a
thing looks is the client's business (doomer/client/sprites.py), keyed by the
same name.
"""

from collections.abc import Mapping
from dataclasses import dataclass, field

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

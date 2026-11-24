"""Registry of thing types that can appear in a map's [things] section.

Adding a new thing (a pickup, a door, a monster) starts with registering its
type here. Behaviour lives elsewhere and looks things up by type name. How a
thing looks is the client's business (doomer/client/sprites.py), keyed by the
same name.
"""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ThingType:
    name: str
    description: str
    blocking: bool = False  # stops players (and later shots and enemies)
    radius: float = 0.0  # half-width of the square footprint, in cells


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
):
    register_thing_type(ThingType(_name, _description, _blocking, _radius))

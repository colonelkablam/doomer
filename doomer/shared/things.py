"""Registry of thing types that can appear in a map's [things] section.

Adding a new thing (a pickup, a door, a monster) starts with registering its
type here. Behaviour lives elsewhere and looks things up by type name.
"""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ThingType:
    name: str
    description: str


THING_TYPES: dict[str, ThingType] = {}


def register_thing_type(thing_type: ThingType) -> ThingType:
    if thing_type.name in THING_TYPES:
        raise ValueError(f"thing type already registered: {thing_type.name}")
    THING_TYPES[thing_type.name] = thing_type
    return thing_type


COOP_START = register_thing_type(ThingType("coop_start", "Co-op player start"))
EXIT = register_thing_type(ThingType("exit", "Co-op level exit"))
DM_SPAWN = register_thing_type(ThingType("dm_spawn", "Deathmatch spawn point"))

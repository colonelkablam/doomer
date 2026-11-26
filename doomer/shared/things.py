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
    # Sits on a desk-pod cell (a low block), so it needs no footprint of its own.
    ("terminal", "Beige PC and CRT, for standing on a desk pod", False, 0.0),
    # Drawn as solid boxes (client/boxes.py); faces its map angle, like terminal.
    ("mac", "Apple Macintosh SE/30 with keyboard and mouse, for a desk", False, 0.0),
    # The same: a brown 1980s desk phone (boxes), facing whoever uses it.
    ("phone", "1980s desk phone, for a desk", False, 0.0),
    # And the boss's late-80s PC: a green screen on a flat system unit.
    ("pc", "Late-1980s business PC with a green screen, for a desk", False, 0.0),
    # An office swivel chair (boxes): blocks, a small footprint you can't
    # stand on; angle = the way its sitter faces.
    ("chair", "Office swivel chair", True, 0.1),
):
    # Things with screens are on a circuit: the main one (dark in a power
    # cut) unless `circuit=emergency`.
    screen = _name in ("terminal", "mac", "pc")
    properties = {"circuit": ("main", "emergency")} if screen else {}
    register_thing_type(ThingType(_name, _description, _blocking, _radius, properties))

# Lobby seating (drawn by doomer/client/boxes.py, sizes matching): a modular
# sofa in runs of two or three seats and a corner unit (backs on two sides),
# to make an L; angle = the way the sitters face, its back(s) behind them
# (a corner's second back on its left as you sit: model -y). Each seat
# 0.28 wide, 0.32 deep. A coffee table (things rest on its top) and a
# small fern in a bowl, for a table or a desk.
for _name, _description, _seats in (
    ("sofa_two", "Modular sofa: two seats", 2),
    ("sofa_three", "Modular sofa: three seats", 3),
    ("sofa_corner", "Modular sofa: a corner unit", 1),
):
    _half = 0.14 * _seats
    register_thing_type(
        ThingType(
            _name,
            _description,
            blocking=True,
            radius=max(0.16, _half),
            footprints=(Footprint((-0.16, 0.16), (-_half, _half), 0.29),),
        )
    )
COFFEE_TABLE = register_thing_type(
    ThingType(
        "coffee_table",
        "Low coffee table, amber glass on chrome legs",
        blocking=True,
        radius=0.25,
        footprints=(Footprint((-0.25, 0.25), (-0.15, 0.15), 0.15),),
    )
)
FERN = register_thing_type(ThingType("fern", "A small fern in a copper bowl"))

# A poster or picture on a wall (drawn as a box by doomer/client/boxes.py),
# `kind=` which; plan: the floor's fire plan, drawn from the map, a red dot
# where it hangs. Put it at the wall's face, its angle out of the wall.
POSTER_KINDS = (
    "synergy",
    "teamwork",
    "excellence",
    "employee_mar",
    "employee_apr",
    "employee_may",
    "founder",
    "accident",
    "high_voltage",
    "plan",
)
POSTER = register_thing_type(
    ThingType("poster", "A poster or picture on a wall", properties={"kind": POSTER_KINDS})
)

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

# The boss's desk: absurdly big (two cells long), a leather-topped slab on two
# carved pedestals, a high-backed chair behind it (model -x: the boss sits
# facing +x, towards visitors). Things on it rest on its top.
EXECUTIVE_DESK = register_thing_type(
    ThingType(
        "executive_desk",
        "The boss's desk: two cells of mahogany and leather, a throne behind it",
        blocking=True,
        radius=1.0,
        footprints=(
            Footprint((-0.46, 0.46), (-1.0, 1.0), 0.33),  # the top
            Footprint((-1.0, -0.62), (-0.22, 0.22), 0.55),  # the chair
        ),
    )
)

# A fluorescent light panel set into its cell's ceiling (drawn as a panel,
# not a sprite). `circuit=emergency`: stays lit when the power's off
# (doomer/shared/switches.py); main-circuit panels go dark.
CEILING_LIGHT = register_thing_type(
    ThingType(
        "ceiling_light",
        "Fluorescent ceiling light panel",
        properties={"circuit": ("main", "emergency")},
    )
)

# The plant room's breaker: use it to restore power (doomer/shared/switches.py).
# Stand it just in front of a wall's face, angle out of the wall.
# Blocks like furniture: an isolator box on the wall, 0.16 deep (its lever
# out to 0.24), 0.44 wide, up to 0.8 (drawn by doomer/client/boxes.py, whose
# sizes match). Put it on a solid wall, not a window.
BREAKER = register_thing_type(
    ThingType(
        "breaker",
        "Main breaker: restores the power",
        blocking=True,
        radius=0.22,
        footprints=(Footprint((0.0, 0.24), (-0.22, 0.22), 0.8),),
    )
)

# A late-80s minicomputer cabinet (drawn by doomer/client/boxes.py, sizes
# matching): kind=tape (reels behind glass) or disk (drive drawers); its
# lamps blink while there's power. Stand it at a wall's face, angle out of
# the wall; a row of them side by side 0.38 apart. Blocks.
SERVER = register_thing_type(
    ThingType(
        "server",
        "Minicomputer cabinet, its lamps blinking with power",
        blocking=True,
        radius=0.19,
        properties={"kind": ("tape", "disk")},
        footprints=(Footprint((0.0, 0.36), (-0.19, 0.19), 0.74),),
    )
)

# Signs: drawn by doomer/client/boxes.py. Stand it at a wall's face (just in
# front of it, in the room); angle = the way it faces, out of the wall.
WELCOME_SIGN = register_thing_type(
    ThingType("welcome_sign", "Lit brass letters: Boomer Industries")
)
# A lit green EXIT sign over a main door, on its own battery: the emergency
# light in a power cut (a dim pool of light round it). Stand it at the face
# of the wall over the doorway, on the side people leave from. `kind=`
# where it's mounted: door (the default: just over a door's head, on the
# wall over a doorway) or transom (higher, on the transom bar at the storey
# line over glazed doors that run up to it, as the entrance's do).
EXIT_SIGN_HEIGHTS = {"door": 0.83, "transom": 1.0}  # walls: its middle
EXIT_SIGN = register_thing_type(
    ThingType(
        "exit_sign",
        "Emergency exit sign over a door",
        properties={"kind": tuple(EXIT_SIGN_HEIGHTS)},
    )
)

# Doors (doomer/shared/doors.py): in a doorway cell; angle = the way it opens.
# `lock=<colour>`: only opens for a team holding that pass.
for _name, _description in (
    ("door_hinged", "Hinged office door: press use to open or close"),
    ("door_double", "Double office doors: press use to open or close"),
    ("door_sliding", "Sliding glass doors: open by themselves as you approach"),
    ("door_lift", "Lift doors: use to call the lift (needs power); a car behind"),
):
    register_thing_type(ThingType(_name, _description, properties={"lock": PASS_COLOURS}))

# ID passes (doomer/shared/pickups.py): walk over one to pick it up; the team
# shares it. Drawn as a sprite on whatever it lies on.
for _colour in PASS_COLOURS:
    register_thing_type(ThingType(f"pass_{_colour}", f"{_colour.capitalize()} ID pass"))

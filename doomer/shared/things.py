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
# A vending machine (drawn by doomer/client/boxes.py, sizes matching): a
# health dispenser (doomer/shared/dispensers.py), a snack a use, on the main
# circuit. Stand it at a solid wall's face, angle out of the wall. Blocks.
VENDING_MACHINE = register_thing_type(
    ThingType(
        "vending_machine",
        "Snack vending machine: health, with the power on",
        blocking=True,
        radius=0.18,
        footprints=(Footprint((0.0, 0.3), (-0.18, 0.18), 0.7),),
    )
)
# A coffee cup on a desk: a pickup (doomer/shared/pickups.py), +5 coffee.
COFFEE_CUP = register_thing_type(ThingType("coffee_cup", "A cup of coffee: +5 coffee"))
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
    "gents",  # washroom door plates (on the corridor wall beside the door)
    "ladies",
    "wash_hands",  # NOW WASH YOUR HANDS
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

# Washroom fittings (drawn by doomer/client/boxes.py, sizes matching), each
# against a wall (`at=`), +x out of it. A run of cubicles: laminate
# partitions raised off the floor, a toilet and a door each (one open, one
# ENGAGED), the doors opening on use (doomer/shared/cubicles.py),
# `kind=gents` or `ladies`; a run of urinals with screens between; a vanity
# unit with basins and a mirror over it. Runs of two to four (cubicles to
# five), like the sofas.
# Cubicles are big enough for a player (0.5 wide): five across a 3-cell wall.
CUBICLE_WIDTH, CUBICLE_DEPTH, CUBICLE_TOP = 0.6, 0.9, 0.8
PARTITION_HALF = 0.012  # half a partition's thickness
# The front's fixed returns: a strip each side of every partition, so each
# door is narrower than its cubicle (0.52: still wider than a player).
RETURN_WIDTH, RETURN_DEPTH = 0.028, 0.02
# The toilet in each cubicle, in the run's frame (x out of the wall): the pan
# with its lid down (stand on it: a jump), the cistern against the wall.
PAN_X, PAN_HALF, PAN_TOP = (0.06, 0.36), 0.1, 0.18
CISTERN_X, CISTERN_HALF, CISTERN_TOP = (0.0, 0.08), 0.12, 0.38
URINAL_PITCH, URINAL_DEPTH = 0.36, 0.26  # a screen's depth
BASIN_PITCH, BASIN_DEPTH, BASIN_TOP = 0.3, 0.26, 0.3
CUBICLE_RUNS = {f"cubicles_{w}": n for n, w in ((2, "two"), (3, "three"), (4, "four"), (5, "five"))}


def _cubicle_footprints(n: int) -> tuple[Footprint, ...]:
    """A run's floor plan: each partition (thin) and its returns across the
    front, and each toilet's pan and cistern. The doors are the World's
    (doomer/shared/cubicles.py); the loo roll holders are too small to
    matter."""
    half = n * CUBICLE_WIDTH / 2
    out = []
    for k in range(n + 1):
        y = half - k * CUBICLE_WIDTH
        y0, y1 = max(y - PARTITION_HALF, -half), min(y + PARTITION_HALF, half)
        out.append(Footprint((0.0, CUBICLE_DEPTH), (y0, y1), CUBICLE_TOP))
        reach = PARTITION_HALF + RETURN_WIDTH
        r0, r1 = max(y - reach, -half), min(y + reach, half)
        out.append(Footprint((CUBICLE_DEPTH - RETURN_DEPTH, CUBICLE_DEPTH), (r0, r1), CUBICLE_TOP))
    for k in range(n):
        cy = half - (k + 0.5) * CUBICLE_WIDTH
        out.append(Footprint(PAN_X, (cy - PAN_HALF, cy + PAN_HALF), PAN_TOP))
        out.append(Footprint(CISTERN_X, (cy - CISTERN_HALF, cy + CISTERN_HALF), CISTERN_TOP))
    return tuple(out)


for _name, _n in CUBICLE_RUNS.items():
    register_thing_type(
        ThingType(
            _name,
            f"Toilet cubicles: a run of {_name.split('_')[1]}",
            blocking=True,
            radius=_n * CUBICLE_WIDTH / 2,
            # Whose they are: the shoes under the ENGAGED door.
            properties={"kind": ("gents", "ladies")},
            footprints=_cubicle_footprints(_n),
        )
    )
for _n, _word in ((2, "two"), (3, "three"), (4, "four")):
    _half = _n * URINAL_PITCH / 2
    register_thing_type(
        ThingType(
            f"urinals_{_word}",
            f"Urinals: a run of {_word} with screens between",
            blocking=True,
            radius=_half,
            footprints=(Footprint((0.0, URINAL_DEPTH), (-_half, _half), 0.6),),
        )
    )
    _half = _n * BASIN_PITCH / 2
    register_thing_type(
        ThingType(
            f"basins_{_word}",
            f"Vanity unit with {_word} basins, a mirror over it",
            blocking=True,
            radius=_half,
            footprints=(Footprint((0.0, BASIN_DEPTH), (-_half, _half), BASIN_TOP),),
        )
    )
# An office bin (drawn by doomer/client/boxes.py, sizes matching): a
# rectangular box, an open top, RUBBISH on its front and back. `kind=` its
# style: steel (charcoal), wood (brown wood-effect), orange (plastic),
# chrome (brushed steel, a swing flap on top: washrooms). Small, but you
# bump into it (and could hop onto it).
BIN_KINDS = ("steel", "wood", "orange", "chrome")
BIN_HALF_X, BIN_HALF_Y, BIN_TOP = 0.065, 0.09, 0.19
# A tall bin (`bin_tall`, the same styles): ~80 cm, for lounges, corridors,
# washrooms and plant rooms rather than beside a desk.
BIN_TALL_HALF_X, BIN_TALL_HALF_Y, BIN_TALL_TOP = 0.085, 0.11, 0.3
BIN_SIZES = {
    "bin": (BIN_HALF_X, BIN_HALF_Y, BIN_TOP),
    "bin_tall": (BIN_TALL_HALF_X, BIN_TALL_HALF_Y, BIN_TALL_TOP),
}
for _name, (_hx, _hy, _top) in BIN_SIZES.items():
    register_thing_type(
        ThingType(
            _name,
            "An office bin" if _name == "bin" else "A tall bin",
            blocking=True,
            radius=_hy,
            properties={"kind": BIN_KINDS},
            footprints=(Footprint((-_hx, _hx), (-_hy, _hy), _top),),
        )
    )
HAND_DRYER = register_thing_type(ThingType("hand_dryer", "Hand dryer on a wall"))
WET_FLOOR_SIGN = register_thing_type(
    ThingType("wet_floor_sign", "A yellow CAUTION WET FLOOR A-frame", blocking=True, radius=0.08)
)

# ID passes (doomer/shared/pickups.py): walk over one to pick it up; the team
# shares it. Drawn as a sprite on whatever it lies on.
for _colour in PASS_COLOURS:
    register_thing_type(ThingType(f"pass_{_colour}", f"{_colour.capitalize()} ID pass"))

"""The building: its floors, bottom to top, and which have maps.

Each map is one floor (its `floor:` header names which); the lifts run
between them. A lift's floor indicator counts through these labels as the
car travels; taking the lift up from a floor loads the next floor that has
a map. Config as data: adding a floor is a map file and a line here. The
server will use this to move the team on, as single-player does now.
"""

from pathlib import Path

# Bottom to top: three basements (car parks, the post room), the ground
# floor, five office floors, the penthouse.
FLOOR_LABELS = ("B3", "B2", "B1", "G", "1", "2", "3", "4", "5", "P")

MAPS_DIR = Path(__file__).resolve().parents[2] / "maps"
# The floors built so far: label -> map file (in MAPS_DIR).
FLOOR_MAPS = {"G": "office.map", "1": "floor1.map"}


def floor_index(label: str) -> int:
    """A floor's place in the building, 0 at the bottom."""
    return FLOOR_LABELS.index(label)


def floor_label(index: float) -> str:
    """The label for a (possibly in-between) position: the nearest floor."""
    return FLOOR_LABELS[max(0, min(len(FLOOR_LABELS) - 1, round(index)))]


def next_floor(label: str) -> str | None:
    """The next floor up that has a map, or None (the top of what's built)."""
    for above in FLOOR_LABELS[floor_index(label) + 1 :]:
        if above in FLOOR_MAPS:
            return above
    return None


def floor_name(label: str) -> str:
    """What a floor's called on screen: GROUND FLOOR, FLOOR 1, BASEMENT 2,
    PENTHOUSE."""
    if label == "G":
        return "GROUND FLOOR"
    if label == "P":
        return "PENTHOUSE"
    if label.startswith("B"):
        return f"BASEMENT {label[1:]}"
    return f"FLOOR {label}"


def floor_map_path(label: str) -> Path:
    return MAPS_DIR / FLOOR_MAPS[label]

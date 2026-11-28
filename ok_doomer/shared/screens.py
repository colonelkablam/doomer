"""Cubicle screens: thin fabric panels standing on screened desk pairs ('j').

One down the middle of each pair (between its two sitters, perpendicular to
the cell's facing) and one across the pair at each cell edge, shared edges
once, so every sitter has a three-sided booth. They're part of the map's
geometry, worked out here once so the simulation (collision) and the client
(drawing, ok_doomer/client/boxes.py) agree on where they are.

Collision follows the railing rule: a screen stops you only while your feet
are between the desk top and the screen's top, i.e. walking about on the
desks. From the floor the desk itself stops you. Jumping over one needs
headroom: under a storey-high ceiling a player on a desk can only rise 0.08
walls before bumping their head, so the screens are only jumpable in tall
rooms.
"""

from typing import NamedTuple

from ok_doomer.shared.mapfile import GameMap

SCREENED_DESK_WALL_TYPE = 19  # 'j'
SCREEN_HEIGHT = 0.21  # walls above the desk top (~57 cm: tops at ~1.25 m, seated head height)
SCREEN_HALF = 0.01  # half-thickness, cells (~5 cm thick)


class Screen(NamedTuple):
    """A screen: a thin panel in the plane x = `at` (axis 0) or y = `at`
    (axis 1), from `lo` to `hi` along the other axis, standing on a desk top
    at height `base` (walls)."""

    axis: int
    at: float
    lo: float
    hi: float
    base: float


def desk_screens(game_map: GameMap) -> list[Screen]:
    """Every screen in the map. Neighbouring pairs in a row (same type, same
    facing axis) share the screen on their common edge."""

    def screened(x: int, y: int) -> bool:
        return game_map.wall_at(x, y) == SCREENED_DESK_WALL_TYPE and game_map.is_low_block(x, y)

    def same_row(x: int, y: int, facing: int) -> bool:
        return screened(x, y) and game_map.facing_at(x, y) % 2 == facing % 2

    out = []
    for cy, row in enumerate(game_map.walls):
        for cx in range(len(row)):
            if not screened(cx, cy):
                continue
            base = game_map.floor_height(cx, cy)
            facing = game_map.facing_at(cx, cy)
            if facing % 2 == 0:
                # Sitters north and south: the pair meets along y = middle;
                # the row runs along x, so the edge screens are planes x = const.
                out.append(Screen(1, cy + 0.5, cx, cx + 1, base))
                out.append(Screen(0, cx, cy, cy + 1, base))
                if not same_row(cx + 1, cy, facing):
                    out.append(Screen(0, cx + 1, cy, cy + 1, base))
            else:
                out.append(Screen(0, cx + 0.5, cy, cy + 1, base))
                out.append(Screen(1, cy, cx, cx + 1, base))
                if not same_row(cx, cy + 1, facing):
                    out.append(Screen(1, cy + 1, cx, cx + 1, base))
    return out


def screens_at_height(screens: tuple[Screen, ...], feet: float) -> list[Screen]:
    """The screens that block someone whose feet are at `feet`: those whose
    desk top they're standing on or above, but below the screen's top."""
    eps = 1e-9
    return [s for s in screens if s.base - eps <= feet < s.base + SCREEN_HEIGHT]

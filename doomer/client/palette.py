"""Flat wall colours per wall type, until textures arrive (milestone 1b)."""

WALL_COLOURS: dict[int, tuple[int, int, int]] = {
    1: (150, 150, 155),  # grey stone
    2: (170, 70, 50),  # brick red
    3: (60, 120, 170),  # blue metal
    4: (90, 150, 70),  # mossy green
    5: (180, 150, 70),
    6: (130, 80, 150),
    7: (80, 160, 160),
    8: (190, 110, 60),
    9: (200, 200, 200),
}
UNKNOWN_WALL = (255, 0, 255)  # loud magenta, so a missing entry is obvious

"""Raised floors (low blocks): desks render with a face and a top, hide what's
behind them, and things can stand on them."""

import math

import numpy as np

from doomer.client.config import RenderConfig
from doomer.client.palette import GREY, SAND, STEEL
from doomer.client.render import EYE_HEIGHT, Camera, Renderer, SpriteInstance, sprites_from_things
from doomer.shared.mapfile import GameMap, parse_map

# A desk pod (wall type 3: partition-fronted, laminate top) two cells deep, at
# height 2/8 = 0.25, in front of a beige wall.
ROOM_TEXT = """\
[walls]
#########
#.......#
#...33..#
#.......#
#########

[floor]
000000000
000000000
000022000
000000000
000000000

[things]
coop_start 1.5 2.5 0
exit 7.5 2.5 0
terminal 4.5 2.5 0
"""
ROOM = parse_map(ROOM_TEXT)
CFG = RenderConfig()
R = Renderer(CFG)
CAM = Camera(1.5, 2.5, 0.0)  # facing east; desk front face 2.5 cells away
# A column whose ray meets the desk front at y = 2.25 (texture u = 0.25), clear
# of the partition texture's aluminium trim at u = 0 and 0.5. Perpendicular
# depths don't depend on the column.
COL = round(CFG.width / 2 + (-0.25 / 2.5) * R.projection)
CENTRE = CFG.width // 2  # sprites straight ahead are centred here
DESK_TOP = 0.25


def row_for(height: float, depth: float) -> float:
    """Screen row of a point at `height` (in walls) and perpendicular `depth`."""
    return CFG.height // 2 + (EYE_HEIGHT - height) * R.projection / depth


def test_front_face_top_surface_and_wall_behind() -> None:
    frame = R.render_indices(ROOM, CAM)
    near, far = 2.5, 4.5  # desk spans x = 4..6; camera at x = 1.5
    face_row = int((row_for(DESK_TOP, near) + row_for(0.0, near)) / 2)
    top_row = int((row_for(DESK_TOP, far) + row_for(DESK_TOP, near)) / 2)
    wall_row = int(row_for(DESK_TOP, far)) - 3  # just above the desk's far edge
    assert frame[COL, face_row] >> 4 == STEEL  # partition fabric front
    assert frame[COL, top_row] >> 4 == SAND  # beige laminate top
    assert frame[COL, wall_row] >> 4 == SAND  # beige office wall (type 1)
    # Depths in the z-buffer match the geometry.
    assert math.isclose(R.zbuffer[COL, face_row], near, rel_tol=1e-6)
    assert near < R.zbuffer[COL, top_row] < far
    assert math.isclose(R.zbuffer[COL, wall_row], 6.5, rel_tol=1e-6)  # wall face at x = 8


def test_desk_hides_bottom_of_a_sprite_behind_it() -> None:
    behind = [SpriteInstance("filing_cabinet", 6.5, 2.5)]  # depth 5, just past the desk
    with_cab = R.render_indices(ROOM, CAM, behind)
    without = R.render_indices(ROOM, CAM)
    changed = np.flatnonzero(with_cab[CENTRE] != without[CENTRE])  # sorted rows
    assert len(changed) > 0
    # Nothing drawn below the desk's far top edge: that's where the desk is.
    assert int(changed[-1]) < row_for(DESK_TOP, 4.5)


def test_thing_on_a_desk_stands_on_its_top() -> None:
    (terminal,) = [s for s in sprites_from_things(ROOM) if s.name == "terminal"]
    assert terminal.z == DESK_TOP
    frame = R.render_indices(ROOM, CAM, [terminal])
    without = R.render_indices(ROOM, CAM)
    changed = np.flatnonzero(frame[CENTRE] != without[CENTRE])  # sorted rows
    assert len(changed) > 0
    # The terminal's lowest pixel is on the desk top at its depth (3), not the floor.
    assert abs(int(changed[-1]) - row_for(DESK_TOP, 3.0)) <= 1.5


def test_looking_over_the_desk_from_the_side_shows_no_gaps() -> None:
    # Every pixel is drawn something; the z-buffer starts infinite only for floor
    # and ceiling, never for a column the desk occupies.
    frame = R.render_indices(ROOM, Camera(2.0, 1.5, math.radians(30)))
    assert frame.shape == (CFG.width, CFG.height)
    assert np.isfinite(R.zbuffer[:, CFG.height // 2]).all()  # a wall at every column


def test_ceiling_hung_things_ignore_desks_below() -> None:
    above = parse_map(ROOM_TEXT.replace("exit 7.5 2.5 0", "exit 5.5 2.5 0"))
    by_name = {s.name: s for s in sprites_from_things(above)}
    assert by_name["exit"].z == 0.0
    assert by_name["terminal"].z == DESK_TOP


# --- open desks (masked textures) --------------------------------------------------

OPEN_DESK = parse_map(
    """\
[walls]
#########
#.......#
#...a...#
#.......#
#########

[floor]
000000000
000000000
000020000
000000000
000000000
"""
)
# Camera at x = 1.5 facing east: desk front face at x = 4 (2.5 away), back face
# at x = 5 (3.5 away), wall at x = 8 (6.5 away). Office desk texture: legs
# flush with the face edges at u = 0..2 and 61..63 (of 64), open below the slab.


def column_for(y_at_x: float, x: float) -> int:
    """Screen column of the ray from the camera through (x, y_at_x)."""
    return round(CFG.width / 2 + (y_at_x - 2.5) / (x - 1.5) * R.projection)


def test_open_desk_shows_the_floor_through_the_gap_under_it() -> None:
    frame = R.render_indices(OPEN_DESK, CAM)
    col = column_for(2.5, 4.0)  # straight ahead: u = 0.5, between the legs
    gap_row = int(row_for(0.08, 2.5))  # under the slab, just above the floor at the front
    slab_row = int(row_for(DESK_TOP - 0.01, 2.5))
    # Looking low through the gap you see the floor just beyond the desk (the
    # wall's base is higher up the screen, behind the desktop). Floor = nothing
    # drawn over the background, so the z-buffer is still infinite there.
    assert math.isinf(R.zbuffer[col, gap_row])
    # ...in the desk's contact shadow: same colour ramp as the open floor, darker.
    floor = int(R.background(R.horizon, EYE_HEIGHT)[col, gap_row])
    shadowed = int(frame[col, gap_row])
    assert shadowed >> 4 == floor >> 4
    assert (shadowed & 0x0F) < (floor & 0x0F)
    assert math.isclose(R.zbuffer[col, slab_row], 2.5, rel_tol=1e-6)  # the desk's edge
    assert frame[col, slab_row] >> 4 == SAND


def test_open_desk_front_and_back_legs() -> None:
    frame = R.render_indices(OPEN_DESK, CAM)
    # Front leg: u = 1.5/64 on the front face.
    front = column_for(2.0 + 1.5 / 64, 4.0)
    row = int(row_for(0.1, 2.5))
    assert frame[front, row] >> 4 == GREY
    assert math.isclose(R.zbuffer[front, row], 2.5, rel_tol=1e-6)
    # Back leg: aim at u = 1.5/64 on the back face; the ray passes through the
    # gap in the front face (u ~ 0.18 there) and hits the leg 3.5 away.
    # It's only visible below the front slab's underside (height 12/64 at 2.5
    # away) and above the back face's floor line (3.5 away): pick the middle.
    back = column_for(2.0 + 1.5 / 64, 5.0)
    row = int((row_for(12 / 64, 2.5) + row_for(0.0, 3.5)) / 2)
    assert frame[back, row] >> 4 == GREY
    assert math.isclose(R.zbuffer[back, row], 3.5, rel_tol=1e-3)


# --- per-side textures (desk facing) ----------------------------------------------


def steel_desk_room(facing: str) -> GameMap:
    return parse_map(
        "[walls]\n#########\n#.......#\n#...c...#\n#.......#\n#########\n\n"
        "[floor]\n000000000\n000000000\n000020000\n\n"
        f"[facing]\n.........\n.........\n....{facing}....\n"
    )


def under_desk_is_open(game_map: GameMap, camera: Camera, depth: float) -> bool:
    """Is the middle of the face straight ahead see-through below the rail?"""
    R.render_indices(game_map, camera)
    row = int(CFG.height // 2 + (EYE_HEIGHT - 0.06) * R.projection / depth)
    return bool(R.zbuffer[CFG.width // 2, row] > depth + 0.5)


def test_steel_desk_sides_follow_its_facing() -> None:
    # Facing south, its pedestal end is its own left: the east side. Camera 2.5
    # from the face in each case.
    south = steel_desk_room("s")
    from_east = Camera(7.5, 2.5, math.pi)  # looking west at the east face
    from_west = Camera(1.5, 2.5, 0.0)  # looking east at the west face
    assert not under_desk_is_open(south, from_east, 2.5)  # solid pedestal side
    assert under_desk_is_open(south, from_west, 2.5)  # open end: just legs
    # Turn the desk to face north and the two ends swap.
    north = steel_desk_room("n")
    assert under_desk_is_open(north, from_east, 2.5)
    assert not under_desk_is_open(north, from_west, 2.5)


def test_executive_desk_is_solid_except_the_kneehole_at_the_front() -> None:
    room = parse_map(
        "[walls]\n#########\n#.......#\n#.......#\n#.......#\n#...b...#\n"
        "#.......#\n#.......#\n#.......#\n#########\n\n"
        "[floor]\n000000000\n000000000\n000000000\n000000000\n000020000\n\n"
        "[facing]\n.........\n.........\n.........\n.........\n....n....\n"
    )
    # Desk at cell (4, 4) facing north: kneehole on the north face, solid
    # modesty panel at the back (south) and solid sides. Cameras 2.5 away.
    assert under_desk_is_open(room, Camera(4.5, 1.5, math.pi / 2), 2.5)  # front
    assert not under_desk_is_open(room, Camera(4.5, 7.5, -math.pi / 2), 2.5)  # back
    assert not under_desk_is_open(room, Camera(7.5, 4.5, math.pi), 2.5)  # side
    assert not under_desk_is_open(room, Camera(1.5, 4.5, 0.0), 2.5)  # side


def test_desk_pair_ends_are_half_pedestal_half_open_from_either_end() -> None:
    pair = parse_map(
        "[walls]\n#########\n#.......#\n#...d...#\n#.......#\n#########\n\n"
        "[floor]\n000000000\n000000000\n000020000\n\n"
        "[facing]\n.........\n.........\n....n....\n"
    )
    depth = 2.5
    row = int(CFG.height // 2 + (EYE_HEIGHT - 0.06) * R.projection / depth)  # near the floor
    for camera in (Camera(7.5, 2.5, math.pi), Camera(1.5, 2.5, 0.0)):  # east end, west end
        R.render_indices(pair, camera)

        def see_through(u: float) -> bool:
            col = round(CFG.width / 2 + (u - 0.5) * R.projection / depth)
            return bool(R.zbuffer[col, row] > depth + 0.5)

        assert not see_through(0.25)  # left half: one desk's pedestal end panel
        assert see_through(0.75)  # right half: the other desk's open end

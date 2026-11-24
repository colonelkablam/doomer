"""Raised floors (low blocks): desks render with a face and a top, hide what's
behind them, and things can stand on them."""

import math

import numpy as np

from doomer.client.config import RenderConfig
from doomer.client.palette import SAND, STEEL
from doomer.client.render import EYE_HEIGHT, Camera, Renderer, SpriteInstance, sprites_from_things
from doomer.shared.mapfile import parse_map

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
    above = parse_map(
        ROOM_TEXT.replace("terminal 4.5 2.5 0", "terminal 4.5 2.5 0\nceiling_light 5.5 2.5 0")
    )
    by_name = {s.name: s for s in sprites_from_things(above)}
    assert by_name["ceiling_light"].z == 0.0
    assert by_name["terminal"].z == DESK_TOP

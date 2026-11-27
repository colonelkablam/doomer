"""People as voxel models in poses: the office worker (other players, until
the characters are drawn: GAME.md), standing, walking, crouching, jumping.

Each pose is a model baked into its own VIEWS frames (models.py), and a
separate sprite name, so the renderer needs nothing new: the game picks the
pose's name (client/animation.py) and the renderer the view, as for any
baked sprite.

Model axes (models.py): x out of his front, y to his right, z up; one
voxel = 1/64 of a wall (~4 cm). He's ~1.8 m (44 voxels), slim.

Built in three parts so a pose can move them:
- the upper body (belt up: torso, tie, head, hair), drawn upright with the
  hip at HIP_Z, then lowered to the pose's hip height and leaned forward
  about the hip (a true rotation, resampled, for the crouch: a lean by
  shear would keep its height);
- the legs, each a thigh and a shin as straight segments from the hip,
  swung forward (+) or back, the knee bent (+: the shin folds back);
- the arms, from the shoulders (where the upper body's lean put them),
  swung forward or back.
The hip's height is whatever puts the lower foot on the floor, unless a
pose holds it up (jumping: the feet tucked under, off the ground).
"""

import math
from dataclasses import dataclass

import numpy as np

from ok_doomer.client.models import GRID, VoxelModel
from ok_doomer.client.palette import BLUE, BRICK, GREY, OCHRE, SAND, WOOD

HIP_Z = 20  # standing: the hip joint's height (the belt sits on it)
THIGH = 9  # voxels, hip to knee
SHIN = 9  # knee to ankle
SHOE = 2  # ankle to floor
SHOULDER = (0.5, 33.0)  # standing: x, z of the shoulder joints
ARM = 13  # shoulder to wrist (the sleeve)
HAND = 4
LEFT_Y, RIGHT_Y = (-4, -1), (1, 4)  # legs
LEFT_ARM_Y, RIGHT_ARM_Y = (-7, -5), (5, 7)
TROUSERS = (BLUE, 3.5)
SHIRT = (GREY, 13.5)
SKIN = (SAND, 12.0)
_HALF = GRID // 2


@dataclass(frozen=True, slots=True)
class Leg:
    swing: float = 0.0  # degrees: the thigh forward (+) or back from straight down
    knee: float = 0.0  # degrees: the shin folded back from the thigh's line


@dataclass(frozen=True, slots=True)
class Stance:
    left: Leg = Leg()
    right: Leg = Leg()
    left_arm: float = 0.0  # degrees forward (+) or back
    right_arm: float = 0.0
    lean: float = 0.0  # degrees: the upper body forward about the hip
    hip: float | None = None  # hold the hip at this height (None: feet on the floor)


# A stride: left foot forward, legs passing (the right swinging through,
# its foot lifted), right foot forward, passing again. Arms swing against
# the legs.
WALK = (
    Stance(Leg(25, 5), Leg(-20, 10), left_arm=-20, right_arm=20),
    Stance(Leg(0, 0), Leg(10, 45), left_arm=0, right_arm=0),
    Stance(Leg(-20, 10), Leg(25, 5), left_arm=20, right_arm=-20),
    Stance(Leg(10, 45), Leg(0, 0), left_arm=0, right_arm=0),
)
STAND = Stance()
# Squatting, thighs level, leaning forward: 33 voxels (the game's crouch is 27:
# the head is big, and a lean steep enough to reach 27 looks like falling).
CROUCH = Stance(Leg(90, 120), Leg(85, 115), left_arm=25, right_arm=15, lean=30)
# In the air: knees drawn up, feet off the ground, arms up a little.
JUMP = Stance(Leg(40, 70), Leg(15, 60), left_arm=30, right_arm=30, hip=HIP_Z)


def office_worker(stance: Stance = STAND) -> VoxelModel:
    """A man in a white shirt and red tie, navy trousers, brown shoes, a
    dark quiff, in `stance`."""
    hip = stance.hip if stance.hip is not None else _hip_height(stance)
    m = VoxelModel(smooth=2)  # leaned and swung parts are staircases
    _place_upper_body(m, _upper_body(), hip, stance.lean)
    m.box((-2, 2), (-4, 4), (round(hip) - 4, round(hip)), *TROUSERS)  # the seat
    _leg(m, LEFT_Y, hip, stance.left)
    _leg(m, RIGHT_Y, hip, stance.right)
    shoulder = _rotated(SHOULDER, hip, stance.lean)
    _arm(m, LEFT_ARM_Y, shoulder, stance.left_arm)
    _arm(m, RIGHT_ARM_Y, shoulder, stance.right_arm)
    return m


def _hip_height(stance: Stance) -> float:
    """High enough that the lower foot's sole is on the floor."""
    return SHOE + max(_leg_drop(stance.left), _leg_drop(stance.right))


def _leg_drop(leg: Leg) -> float:
    a, b = math.radians(leg.swing), math.radians(leg.swing - leg.knee)
    return THIGH * math.cos(a) + SHIN * math.cos(b)


def _upper_body() -> VoxelModel:
    """Belt up, upright, the hip at HIP_Z."""
    m = VoxelModel()
    hair = (GREY, 2.5)
    m.box((-2, 3), (-4, 4), (20, 21), GREY, 2.0)  # belt
    m.box((2, 3), (-1, 1), (20, 21), OCHRE, 10.0)  # buckle
    m.box((-2, 3), (-5, 5), (21, 33), GREY, 14.0)  # white shirt
    # Red tie down the front, a darker knot; the collar either side.
    m.box((3, 4), (-1, 1), (23, 31), BRICK, 8.0)
    m.box((3, 4), (0, 1), (22, 23), BRICK, 8.0)  # its point
    m.box((3, 4), (-1, 1), (31, 33), BRICK, 6.0)  # knot
    m.box((2, 4), (-3, -1), (32, 33), GREY, 15.0)
    m.box((2, 4), (1, 3), (32, 33), GREY, 15.0)
    # Neck and head.
    m.box((-1, 2), (-2, 2), (33, 35), SAND, 11.0)
    m.box((-3, 3), (-3, 3), (35, 42), SAND, 13.0)
    for y in ((-4, -3), (3, 4)):
        m.box((-1, 1), y, (37, 40), SAND, 11.5)  # ears
    m.box((3, 4), (-1, 1), (37, 39), SAND, 12.0)  # nose
    m.box((2, 3), (-2, -1), (39, 40), GREY, 1.0)  # eyes
    m.box((2, 3), (1, 2), (39, 40), GREY, 1.0)
    m.box((2, 3), (-3, -1), (40, 41), *hair)  # eyebrows
    m.box((2, 3), (1, 3), (40, 41), *hair)
    m.box((2, 3), (-1, 1), (36, 37), SAND, 8.0)  # mouth
    # Dark hair: a quiff over the front, short at the back and sides.
    m.box((-4, 3), (-3, 3), (42, 44), *hair)
    m.box((0, 4), (-2, 2), (43, 45), *hair)  # quiff
    m.box((-4, -3), (-3, 3), (36, 42), *hair)  # back
    for y in ((-4, -3), (3, 4)):
        m.box((-3, 1), y, (40, 43), *hair)  # sides, above the ears
    return m


def _rotated(point: tuple[float, float], hip: float, lean: float) -> tuple[float, float]:
    """Where a point of the upright upper body (x, z) ends up: lowered so
    its hip is at `hip`, leaned `lean` degrees forward about it."""
    x, z = point
    h = z - HIP_Z
    t = math.radians(lean)
    return x * math.cos(t) + h * math.sin(t), hip - x * math.sin(t) + h * math.cos(t)


def _place_upper_body(m: VoxelModel, upper: VoxelModel, hip: float, lean: float) -> None:
    """Copy the upper body into `m`, moved by `_rotated`: each voxel of `m`
    takes the upper body's voxel that lands there (nearest; resampled
    backwards, so no holes)."""
    t = math.radians(lean)
    xs = np.arange(GRID) - _HALF + 0.5  # voxel centres, model x
    zs = np.arange(GRID) + 0.5
    tx, tz = np.meshgrid(xs, zs - hip, indexing="ij")  # target, relative to the hip
    # Undo the lean: source x and height above the hip.
    sx = tx * math.cos(t) - tz * math.sin(t)
    sh = tx * math.sin(t) + tz * math.cos(t)
    ix = np.floor(sx + _HALF).astype(np.intp)
    iz = np.floor(sh + HIP_Z).astype(np.intp)
    ok = (ix >= 0) & (ix < GRID) & (iz >= HIP_Z) & (iz < GRID)
    ix, iz = ix.clip(0, GRID - 1), iz.clip(0, GRID - 1)
    for y in range(GRID):
        ramp = upper.ramp[:, y, :][ix, iz]
        take = ok & (ramp >= 0)
        m.ramp[:, y, :][take] = ramp[take]
        m.shade[:, y, :][take] = upper.shade[:, y, :][ix, iz][take]


def _segment(
    m: VoxelModel,
    y: tuple[int, int],
    start: tuple[float, float],
    end: tuple[float, float],
    width: tuple[float, float],
    colour: tuple[int, float],
) -> None:
    """A limb from `start` to `end` (x, z): a run of slabs, one per half
    voxel along it, each `width` (x from, to: about the line) wide, so a
    steep swing leaves no gaps."""
    (x0, z0), (x1, z1) = start, end
    steps = max(1, math.ceil(2 * math.hypot(x1 - x0, z1 - z0)))
    for k in range(steps + 1):
        f = k / steps
        x, z = x0 + (x1 - x0) * f, z0 + (z1 - z0) * f
        zi = math.floor(z)
        m.box((round(x + width[0]), round(x + width[1])), y, (zi, zi + 1), *colour)


def _leg(m: VoxelModel, y: tuple[int, int], hip: float, leg: Leg) -> None:
    a, b = math.radians(leg.swing), math.radians(leg.swing - leg.knee)
    knee = (THIGH * math.sin(a), hip - THIGH * math.cos(a))
    ankle = (knee[0] + SHIN * math.sin(b), knee[1] - SHIN * math.cos(b))
    _segment(m, y, (0.0, hip - 0.5), knee, (-2, 2), TROUSERS)
    _segment(m, y, knee, ankle, (-2, 2), TROUSERS)
    ax, az = ankle
    sole = math.floor(az) - SHOE
    m.box((round(ax - 2), round(ax + 4)), y, (sole, sole + SHOE), WOOD, 4.0)  # shoe
    m.box((round(ax - 2), round(ax + 4)), y, (sole, sole + 1), WOOD, 2.0)  # its sole


def _arm(m: VoxelModel, y: tuple[int, int], shoulder: tuple[float, float], swing: float) -> None:
    a = math.radians(swing)
    sx, sz = shoulder
    wrist = (sx + ARM * math.sin(a), sz - ARM * math.cos(a))
    hand = (wrist[0] + HAND * math.sin(a), wrist[1] - HAND * math.cos(a))
    _segment(m, y, (sx, sz - 0.5), wrist, (-1.5, 1.5), SHIRT)
    _segment(m, y, wrist, hand, (-1.5, 1.5), SKIN)

"""Other players' poses: the walk cycle advancing by distance, standing,
crouching and jumping, and the posed models themselves."""

import math

import numpy as np
import numpy.typing as npt
import pytest

from ok_doomer.client.animation import STILL_FRAMES, STRIDE_CELLS, Stride
from ok_doomer.client.models import VIEWS
from ok_doomer.client.network import Pose, lerp_pose
from ok_doomer.client.people import CROUCH, JUMP, STAND, WALK, office_worker
from ok_doomer.client.sprites import (
    PLAYER_CROUCH,
    PLAYER_JUMP,
    PLAYER_SPRITE,
    PLAYER_WALK,
    SPRITES,
)

STEP = STRIDE_CELLS / len(PLAYER_WALK)  # travel per pose of the cycle


def _walk(stride: Stride, steps: list[float], facing: float = 0.0) -> list[str]:
    """Move east by each of `steps` (negative: west), a frame each."""
    x = 0.0
    names = [stride.pose(x, 0.0, facing, False, False)]
    for d in steps:
        x += d
        names.append(stride.pose(x, 0.0, facing, False, False))
    return names


def test_standing_still_is_standing() -> None:
    stride = Stride()
    assert {stride.pose(1.0, 1.0, 0.0, False, False) for _ in range(10)} == {PLAYER_SPRITE}


def test_the_cycle_advances_a_pose_a_quarter_stride() -> None:
    names = _walk(Stride(), [STEP] * 8)[1:]
    assert names == [*PLAYER_WALK[1:], *PLAYER_WALK, PLAYER_WALK[0]]


def test_by_distance_not_frames() -> None:
    """The same ground covered in more, smaller frames: the same pose."""
    coarse = _walk(Stride(), [STEP * 0.9] * 3)[-1]
    fine = _walk(Stride(), [STEP * 0.9 / 4] * 12)[-1]
    assert coarse == fine


def test_walking_backwards_runs_the_cycle_backwards() -> None:
    names = _walk(Stride(), [-STEP] * 4, facing=0.0)[1:]
    assert names == [PLAYER_WALK[3], PLAYER_WALK[2], PLAYER_WALK[1], PLAYER_WALK[0]]


def test_strafing_steps_forwards() -> None:
    stride = Stride()
    stride.pose(0.0, 0.0, 0.0, False, False)
    assert stride.pose(0.0, STEP * 1.1, 0.0, False, False) == PLAYER_WALK[1]  # sideways (y)


def test_a_quiet_frame_isnt_a_stop_but_a_few_are() -> None:
    stride = Stride()
    walking = _walk(stride, [STEP * 0.5] * 3)[-1]
    assert walking in PLAYER_WALK
    names = [stride.pose(1.5 * STEP, 0.0, 0.0, False, False) for _ in range(STILL_FRAMES)]
    assert names[: STILL_FRAMES - 1] == [walking] * (STILL_FRAMES - 1)  # held mid-stride
    assert names[-1] == PLAYER_SPRITE  # then standing


def test_crouching_and_jumping_have_their_own_poses() -> None:
    stride = Stride()
    assert stride.pose(0.0, 0.0, 0.0, True, False) == PLAYER_CROUCH
    assert stride.pose(0.1, 0.0, 0.0, False, True) == PLAYER_JUMP
    assert stride.pose(0.2, 0.0, 0.0, True, True) == PLAYER_JUMP  # a crouch-jump: in the air


def test_every_pose_is_baked_from_every_side() -> None:
    for name in (PLAYER_SPRITE, *PLAYER_WALK, PLAYER_CROUCH, PLAYER_JUMP):
        assert len(SPRITES[name].index) == VIEWS
    side = VIEWS // 4  # side on, where legs show most
    frames = [SPRITES[n].index[side] for n in (PLAYER_SPRITE, *PLAYER_WALK)]
    for i, a in enumerate(frames):
        for b in frames[i + 1 :]:
            assert (a != b).any()  # each pose looks different


def _extent(model_ramp: npt.NDArray[np.int8]) -> tuple[int, int]:
    zs = [int(z) for z in np.nonzero((model_ramp >= 0).any(axis=(0, 1)))[0]]
    return min(zs), max(zs) + 1


def test_poses_stand_on_the_floor_at_their_heights() -> None:
    stand, crouch, jump = (_extent(office_worker(s).ramp) for s in (STAND, CROUCH, JUMP))
    assert stand == (0, 45)  # ~1.8 m, a quiff on top
    assert crouch[0] == 0 and crouch[1] < 36  # squatting: well down
    assert jump[0] > 0  # feet off the ground
    for stance in WALK:  # a foot on the floor all through the stride
        assert _extent(office_worker(stance).ramp)[0] == 0


def test_interpolated_poses_take_crouch_and_air_from_the_nearer_snapshot() -> None:
    a = Pose(0.0, 0.0, 0.0, 0.0, crouching=False, airborne=False)
    b = Pose(1.0, 0.0, 0.5, 0.0, crouching=True, airborne=True)
    assert not lerp_pose(a, b, 0.4).crouching and lerp_pose(a, b, 0.6).airborne
    assert lerp_pose(a, b, 0.5).z == pytest.approx(0.25)


def test_smoothed_shading_removes_stripes_on_a_lean() -> None:
    """The crouch's leaned shirt front: faceted normals light its voxel
    steps alternately (stripes); smoothed, neighbouring rows shade alike."""
    from ok_doomer.client.models import bake

    model = office_worker(CROUCH)
    smooth = bake(model).index[0]
    model.smooth = 0
    faceted = bake(model).index[0]

    def row_changes(index: npt.NDArray[np.uint8]) -> int:
        shirt = index[30:34, 28:40]  # columns down the middle of the shirt front
        return int((np.diff(shirt.astype(int), axis=1) != 0).sum())

    assert row_changes(smooth) < row_changes(faceted)
    assert not math.isnan(float(smooth.mean()))

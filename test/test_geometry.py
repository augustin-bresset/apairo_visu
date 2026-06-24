"""Tests for the pure-numpy geometry helpers (no Open3D required)."""

import numpy as np
import pytest

from apairo_visu import geometry as g


def test_to_numpy_passthrough_and_tensor_like():
    arr = np.arange(6).reshape(2, 3)
    assert g.to_numpy(arr) is arr

    class FakeTensor:
        def numpy(self):
            return np.array([1.0, 2.0])

    np.testing.assert_array_equal(g.to_numpy(FakeTensor()), [1.0, 2.0])
    np.testing.assert_array_equal(g.to_numpy([3, 4, 5]), [3, 4, 5])


# A minimal perspective-like projection: the clip w-component picks up z, so a
# point in front of the camera (z > 0) has w > 0, behind has w < 0 -- enough to
# exercise the in_front / NDC logic without a full Open3D camera.
_PERSP = np.array([
    [1.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, 0.0],
    [0.0, 0.0, 1.0, 0.0],
])


def test_project_to_screen_centers_origin():
    view = np.eye(4)
    xyz = np.array([[0.0, 0.0, 1.0]])  # w = z = 1
    sx, sy, valid = g.project_to_screen(xyz, view, _PERSP, width=100, height=80)
    assert valid[0]
    assert sx[0] == pytest.approx(50.0)
    assert sy[0] == pytest.approx(40.0)


def test_project_to_screen_marks_behind_camera_invalid():
    view = np.eye(4)
    xyz = np.array([[0.0, 0.0, -1.0]])  # w = z = -1 -> behind
    _, _, valid = g.project_to_screen(xyz, view, _PERSP, 100, 80)
    assert not valid[0]


def test_pick_nearest_returns_closest_within_radius():
    view = np.eye(4)
    # Both in front; one projects to centre (50,40), the other off to the side.
    xyz = np.array([[0.0, 0.0, 1.0], [0.5, 0.0, 1.0]])
    idx = g.pick_nearest(xyz, view, _PERSP, 100, 80, px=50, py=40, radius=15)
    assert idx == 0


def test_pick_nearest_none_when_far():
    view = np.eye(4)
    xyz = np.array([[0.0, 0.0, 1.0]])
    assert g.pick_nearest(xyz, view, _PERSP, 100, 80, px=0, py=0, radius=5) is None


def test_pick_nearest_empty():
    assert g.pick_nearest(np.empty((0, 3)), np.eye(4), np.eye(4), 10, 10, 0, 0) is None


def test_poses_to_local_identity_reference():
    poses = [np.eye(4) for _ in range(3)]
    for i, t in enumerate(poses):
        t[:3, 3] = [i, 0, 0]
    local = g.poses_to_local(poses, ref_pose=poses[0])
    np.testing.assert_allclose(local, [[0, 0, 0], [1, 0, 0], [2, 0, 0]])


def test_poses_to_local_relative_to_moving_reference():
    poses = [np.eye(4), np.eye(4)]
    poses[0][:3, 3] = [5, 0, 0]
    poses[1][:3, 3] = [7, 0, 0]
    local = g.poses_to_local(poses, ref_pose=poses[1])
    # In frame of pose[1], pose[0] sits 2 m behind along x.
    np.testing.assert_allclose(local[0], [-2, 0, 0])
    np.testing.assert_allclose(local[1], [0, 0, 0])


def test_has_extent():
    assert g.has_extent(np.array([[0, 0, 0], [1, 0, 0]]))
    assert not g.has_extent(np.zeros((5, 3)))
    assert not g.has_extent(np.empty((0, 3)))

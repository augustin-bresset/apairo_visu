"""Tests for pose normalisation and loading."""

import numpy as np
import pytest

from apairo_visu import load_poses, pose_to_matrix
from apairo_visu.poses import quat_to_rotation


def test_pose_to_matrix_4x4_passthrough():
    t = np.eye(4)
    t[:3, 3] = [1, 2, 3]
    np.testing.assert_array_equal(pose_to_matrix(t), t)


def test_pose_to_matrix_3x4_lifts_to_4x4():
    p34 = np.hstack([np.eye(3), np.array([[1.0], [2.0], [3.0]])])
    out = pose_to_matrix(p34)
    assert out.shape == (4, 4)
    np.testing.assert_array_equal(out[3], [0, 0, 0, 1])
    np.testing.assert_array_equal(out[:3, 3], [1, 2, 3])


def test_pose_to_matrix_quaternion_vector():
    # identity quaternion (qw=1) -> rotation is identity
    p = np.array([4.0, 5.0, 6.0, 0.0, 0.0, 0.0, 1.0])
    out = pose_to_matrix(p)
    np.testing.assert_allclose(out[:3, :3], np.eye(3), atol=1e-12)
    np.testing.assert_array_equal(out[:3, 3], [4, 5, 6])


def test_pose_to_matrix_ignores_trailing_columns():
    # odometry rows often carry velocity after the 7 pose values
    p = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 9.9, 8.8])
    out = pose_to_matrix(p)
    np.testing.assert_array_equal(out[:3, 3], [1, 0, 0])


def test_pose_to_matrix_rejects_bad_shape():
    with pytest.raises(ValueError):
        pose_to_matrix(np.zeros((2, 2)))


def test_quat_to_rotation_is_orthonormal():
    r = quat_to_rotation(0.0, 0.0, np.sin(np.pi / 4), np.cos(np.pi / 4))  # 90° about z
    np.testing.assert_allclose(r @ r.T, np.eye(3), atol=1e-12)
    np.testing.assert_allclose(r @ [1, 0, 0], [0, 1, 0], atol=1e-12)


def test_load_poses_over_fake_dataset():
    class FakeSample:
        def __init__(self, p):
            self.data = {"poses": p}

    class FakeDataset:
        def __init__(self, mats):
            self._mats = mats

        def __len__(self):
            return len(self._mats)

        def __getitem__(self, i):
            return FakeSample(self._mats[i])

    mats = [np.hstack([np.eye(3), np.array([[i], [0], [0]], float)]) for i in range(3)]
    poses = load_poses(FakeDataset(mats), key="poses")
    assert len(poses) == 3
    assert all(p.shape == (4, 4) for p in poses)
    np.testing.assert_array_equal([p[0, 3] for p in poses], [0, 1, 2])

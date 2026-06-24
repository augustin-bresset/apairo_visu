"""Pose loading helpers -- turn an apairo pose channel into 4x4 matrices.

Datasets store ego-poses in several shapes (RELLIS: ``(3, 4)``; quaternion
odometry: ``(7,)`` ``[x y z qx qy qz qw]``; already-homogeneous ``(4, 4)``).
:func:`load_poses` reads a pose channel frame-by-frame and normalises every
entry to a ``(4, 4)`` ``T_world_sensor`` matrix the viewer can overlay.
"""

from __future__ import annotations

import numpy as np

from .geometry import to_numpy


def quat_to_rotation(qx: float, qy: float, qz: float, qw: float) -> np.ndarray:
    """Unit quaternion ``[qx, qy, qz, qw]`` -> ``(3, 3)`` rotation matrix."""
    x, y, z, w = qx, qy, qz, qw
    return np.array([
        [1 - 2 * (y * y + z * z),     2 * (x * y - z * w),     2 * (x * z + y * w)],
        [    2 * (x * y + z * w), 1 - 2 * (x * x + z * z),     2 * (y * z - x * w)],
        [    2 * (x * z - y * w),     2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ], dtype=np.float64)


def pose_to_matrix(p: np.ndarray) -> np.ndarray:
    """Normalise one pose entry to a ``(4, 4)`` homogeneous matrix.

    Accepts:
        * ``(4, 4)`` -- returned as float64;
        * ``(3, 4)`` -- bottom row ``[0, 0, 0, 1]`` appended;
        * ``(n,)`` with ``n >= 7`` -- the first 7 entries are read as
          ``[x, y, z, qx, qy, qz, qw]`` (translation + quaternion); trailing
          columns, e.g. linear/angular velocity in odometry channels, are
          ignored.
    """
    p = np.asarray(p, dtype=np.float64)
    if p.shape == (4, 4):
        return p
    if p.shape == (3, 4):
        t = np.eye(4)
        t[:3, :] = p
        return t
    if p.ndim == 1 and p.shape[0] >= 7:
        t = np.eye(4)
        t[:3, :3] = quat_to_rotation(*p[3:7])
        t[:3, 3] = p[:3]
        return t
    raise ValueError(
        f"Unsupported pose shape {p.shape}; expected (4,4), (3,4) or (n>=7,)."
    )


def load_poses(dataset, key: str = "poses") -> list[np.ndarray]:
    """Load every frame's pose from ``dataset`` as a list of ``(4, 4)`` matrices.

    Reads ``dataset[i].data[key]`` for each frame and normalises it via
    :func:`pose_to_matrix`.  For apairo datasets backed by a text/​stacked
    loader this is plain slicing -- no per-frame disk I/O.

    Args:
        dataset: Any apairo synchronous dataset exposing ``dataset[i].data``.
        key: Channel holding the pose (default ``"poses"``).

    Returns:
        List of ``(4, 4)`` ``T_world_sensor`` matrices, one per frame.
    """
    return [pose_to_matrix(to_numpy(dataset[i].data[key])) for i in range(len(dataset))]

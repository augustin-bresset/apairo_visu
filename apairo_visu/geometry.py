"""Geometry helpers for the viewer -- pure numpy, with lazy Open3D builders.

The numpy functions (:func:`to_numpy`, :func:`project_to_screen`,
:func:`poses_to_local`) carry no Open3D dependency and are unit-tested in
isolation.  The ``make_*`` builders import Open3D lazily so this module stays
importable in a headless environment.
"""

from __future__ import annotations

import numpy as np


def to_numpy(t) -> np.ndarray:
    """Coerce a torch tensor / array-like to a numpy array."""
    if hasattr(t, "numpy"):
        return t.numpy()
    return np.asarray(t)


def project_to_screen(
    xyz: np.ndarray,
    view: np.ndarray,
    proj: np.ndarray,
    width: int,
    height: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Project world points to viewport pixel coordinates.

    Args:
        xyz: ``(N, 3)`` world-space points.
        view: ``(4, 4)`` world -> camera matrix (``camera.get_view_matrix()``).
        proj: ``(4, 4)`` camera -> clip matrix (``camera.get_projection_matrix()``).
        width: Viewport width in pixels.
        height: Viewport height in pixels.

    Returns:
        ``(sx, sy, valid)`` where ``sx`` / ``sy`` are ``(N,)`` float pixel
        coordinates (origin top-left) and ``valid`` is an ``(N,)`` bool mask of
        points in front of the camera and inside the NDC frustum.
    """
    xyz = np.asarray(xyz, dtype=np.float64)
    n = len(xyz)
    vp = np.asarray(proj, dtype=np.float64) @ np.asarray(view, dtype=np.float64)
    clip = (vp @ np.hstack([xyz, np.ones((n, 1))]).T).T  # (N, 4)

    w = clip[:, 3]
    in_front = w > 0
    w_safe = np.where(in_front, w, 1.0)
    ndc_x = clip[:, 0] / w_safe
    ndc_y = clip[:, 1] / w_safe

    sx = (ndc_x + 1.0) * 0.5 * width
    sy = (1.0 - ndc_y) * 0.5 * height  # NDC y=+1 -> top -> screen y=0

    valid = in_front & (np.abs(ndc_x) <= 1.0) & (np.abs(ndc_y) <= 1.0)
    return sx, sy, valid


def pick_nearest(
    xyz: np.ndarray,
    view: np.ndarray,
    proj: np.ndarray,
    width: int,
    height: int,
    px: int,
    py: int,
    radius: float = 15.0,
) -> int | None:
    """Index of the point whose projection is nearest pixel ``(px, py)``.

    Returns ``None`` when no visible point falls within ``radius`` pixels.
    """
    if xyz is None or len(xyz) == 0:
        return None
    sx, sy, valid = project_to_screen(xyz[:, :3], view, proj, width, height)
    dist2 = (sx - px) ** 2 + (sy - py) ** 2
    dist2[~valid] = np.inf
    idx = int(np.argmin(dist2))
    return idx if dist2[idx] <= radius * radius else None


def poses_to_local(poses: np.ndarray, ref_pose: np.ndarray) -> np.ndarray:
    """Express the translations of ``poses`` in the frame of ``ref_pose``.

    Args:
        poses: Sequence of ``(4, 4)`` world poses (``T_world_sensor``).
        ref_pose: The ``(4, 4)`` pose defining the local frame.

    Returns:
        ``(M, 3)`` translations of each pose expressed in ``ref_pose``'s frame.
    """
    origins = np.array([p[:3, 3] for p in poses], dtype=np.float64)
    t_inv = np.linalg.inv(np.asarray(ref_pose, dtype=np.float64))
    h = np.hstack([origins, np.ones((len(origins), 1))])
    return (t_inv @ h.T).T[:, :3]


def has_extent(pts: np.ndarray, eps: float = 1e-6) -> bool:
    """True when ``pts`` spans more than ``eps`` total along its axes."""
    if len(pts) == 0:
        return False
    return bool((pts.max(axis=0) - pts.min(axis=0)).sum() > eps)


# ---------------------------------------------------------------------------
# Open3D builders (imported lazily -- not needed for the pure helpers above)
# ---------------------------------------------------------------------------


def make_point_cloud(xyz: np.ndarray, colors: np.ndarray):
    """Build an ``open3d.geometry.PointCloud`` from XYZ + RGB arrays."""
    import open3d as o3d

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.asarray(xyz, dtype=np.float64))
    pcd.colors = o3d.utility.Vector3dVector(colors)
    return pcd


def make_lineset(pts: np.ndarray, edges: list, color: list):
    """Build an ``open3d.geometry.LineSet`` from points, edges and one colour."""
    import open3d as o3d

    ls = o3d.geometry.LineSet()
    ls.points = o3d.utility.Vector3dVector(np.asarray(pts, dtype=np.float64))
    ls.lines = o3d.utility.Vector2iVector(edges)
    ls.colors = o3d.utility.Vector3dVector(np.tile(color, (len(edges), 1)))
    return ls

"""Visualise a TartanDrive sequence with apairo_visu.

Usage:
    python examples/view_tartandrive.py
    python examples/view_tartandrive.py --seq ~/tartandrive_data/turnpike_.../turnpike_...
    python examples/view_tartandrive.py --lidar velodyne_1
    python examples/view_tartandrive.py --lidar livox
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

import apairo
import apairo_visu

# ---------------------------------------------------------------------------
# Default path — adjust if your data lives elsewhere
# ---------------------------------------------------------------------------
_DEFAULT_SEQ = (
    Path.home()
    / "tartandrive_data"
    / "turnpike_2023-09-12-12-39-19"
    / "turnpike_2023-09-12-12-39-19"
)


# ---------------------------------------------------------------------------
# Pose helpers
# ---------------------------------------------------------------------------

def _quat_to_rot(qx: float, qy: float, qz: float, qw: float) -> np.ndarray:
    x, y, z, w = qx, qy, qz, qw
    return np.array([
        [1 - 2*(y*y + z*z),  2*(x*y - z*w),  2*(x*z + y*w)],
        [    2*(x*y + z*w),  1 - 2*(x*x + z*z),  2*(y*z - x*w)],
        [    2*(x*z - y*w),  2*(y*z + x*w), 1 - 2*(x*x + y*y)],
    ], dtype=np.float64)


def load_poses(seq_dir: Path, lidar_timestamps: np.ndarray) -> list[np.ndarray]:
    """Load super_odom poses and synchronise to LiDAR timestamps (nearest neighbour)."""
    odom_dir = seq_dir / "super_odom"
    odom_ts = np.loadtxt(odom_dir / "timestamps.txt")
    odom = np.load(odom_dir / "odometry.npy")   # (M, 7+): x y z qx qy qz qw ...

    # For each LiDAR timestamp, find the closest odometry timestamp
    indices = np.abs(odom_ts[:, None] - lidar_timestamps[None, :]).argmin(axis=0)

    poses = []
    for idx in indices:
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = _quat_to_rot(*odom[idx, 3:7])
        T[:3, 3] = odom[idx, 0:3]
        poses.append(T)

    return poses


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(description="TartanDrive LiDAR viewer")
    p.add_argument("--seq", default=str(_DEFAULT_SEQ), help="Sequence directory")
    p.add_argument("--lidar", default="velodyne_0",
                   choices=["velodyne_0", "velodyne_1", "livox"],
                   help="LiDAR channel to display")
    p.add_argument("--no-poses", action="store_true", help="Skip trajectory overlay")
    p.add_argument("--idx", type=int, default=0, help="Starting frame index")
    args = p.parse_args()

    seq_dir = Path(args.seq)
    lidar_key = args.lidar

    if not seq_dir.is_dir():
        raise SystemExit(f"Sequence directory not found: {seq_dir}")

    # --- Load dataset (creates .apairo on first run) ---
    print(f"Loading {lidar_key} from {seq_dir.name} …")
    ds = apairo.TartanKittiDataset(seq_dir, keys=[lidar_key])
    print(f"  {len(ds)} scans")

    # --- Load poses ---
    poses = None
    if not args.no_poses:
        try:
            lidar_ts = ds.timestamps[lidar_key]
            poses = load_poses(seq_dir, lidar_ts)
            print(f"  {len(poses)} poses loaded from super_odom")
        except Exception as e:
            print(f"  [WARN] Could not load poses: {e}")

    # --- Viewer config ---
    # TartanDrive velodyne scans are (N, 3) XYZ — no labels, no intensity column.
    # Defaults to Height (viridis on Z) which works well out of the box.
    view_cfg = apairo_visu.ViewConfig(
        point_key=lidar_key,
        label_key=None,
        intensity_channel=3,   # not present in (N,3) data → falls back to Height
    )

    print("Launching viewer … (← → or H/L to navigate, B for BEV, J for trajectory)")
    apairo_visu.LidarViewer.launch(
        ds,
        view_cfg=view_cfg,
        label_cfg=None,
        poses=poses,
        start_idx=args.idx,
    )


if __name__ == "__main__":
    main()

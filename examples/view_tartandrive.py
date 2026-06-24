"""Visualise a TartanDrive sequence with apairo_visu.

TartanDrive is asynchronous (each sensor fires at its own rate).  We let apairo
resample it onto the LiDAR clock with ``synchronize()`` -- so every frame holds
both the scan and the nearest odometry pose -- and overlay the trajectory with
``apairo_visu.load_poses``.  No hand-rolled timestamp matching or quaternion
maths.

Usage:
    python examples/view_tartandrive.py
    python examples/view_tartandrive.py --seq ~/tartandrive_data/turnpike_.../turnpike_...
    python examples/view_tartandrive.py --lidar velodyne_1
    python examples/view_tartandrive.py --lidar livox --no-poses
"""

from __future__ import annotations

import argparse
from pathlib import Path

import apairo
import apairo_visu

# ---------------------------------------------------------------------------
# Default path -- adjust if your data lives elsewhere
# ---------------------------------------------------------------------------
_DEFAULT_SEQ = (
    Path.home()
    / "tartandrive_data"
    / "turnpike_2023-09-12-12-39-19"
    / "turnpike_2023-09-12-12-39-19"
)

# Channel holding x y z qx qy qz qw (+ trailing velocity columns, ignored).
_ODOM_KEY = "super_odom"


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

    # --- Load async sequence, then resample onto the LiDAR clock --------------
    # synchronize() returns a synchronous view: each frame carries the scan plus
    # the nearest-in-time odometry sample (creates .apairo on first run).
    keys = [lidar_key] if args.no_poses else [lidar_key, _ODOM_KEY]
    print(f"Loading {', '.join(keys)} from {seq_dir.name} …")
    ds_async = apairo.TartanKittiDataset(seq_dir, keys=keys)
    ds = ds_async.synchronize(reference=lidar_key, method="nearest")
    print(f"  {len(ds)} synchronised frames")

    # --- Trajectory overlay from the odometry channel ------------------------
    poses = None
    if not args.no_poses:
        try:
            poses = apairo_visu.load_poses(ds, key=_ODOM_KEY)
            print(f"  {len(poses)} poses from {_ODOM_KEY!r}")
        except Exception as e:  # noqa: BLE001 -- overlay is optional
            print(f"  [WARN] Could not load poses: {e}")

    # TartanDrive velodyne scans are (N, 3) XYZ -- no labels, no intensity column.
    # Defaults to Height (viridis on Z), which works well out of the box.
    view_cfg = apairo_visu.ViewConfig(
        point_key=lidar_key,
        label_key=None,
        intensity_channel=3,   # not present in (N,3) data -> falls back to Height
    )

    print("Launching viewer … (<- -> or H/L to navigate, B for BEV, J for trajectory)")
    apairo_visu.LidarViewer.launch(
        ds,
        view_cfg=view_cfg,
        label_cfg=None,
        poses=poses,
        start_idx=args.idx,
    )


if __name__ == "__main__":
    main()

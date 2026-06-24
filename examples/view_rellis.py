"""Visualise a RELLIS-3D sequence with apairo_visu.

Usage:
    python examples/view_rellis.py
    python examples/view_rellis.py --root ~/data/rellis
    python examples/view_rellis.py --ontology ~/data/rellis/Rellis_3D_ontology/ontology2.yaml
    python examples/view_rellis.py --no-poses --idx 10
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import yaml

import apairo
import apairo_visu
from apairo_visu import Pipeline

_DEFAULT_ROOT = Path.home() / "data" / "rellis"
_DEFAULT_ONTOLOGY = _DEFAULT_ROOT / "Rellis_3D_ontology" / "ontology.yaml"


def load_label_mapping(ontology_path: str | Path) -> dict[int, str]:
    """Return a {label_id: name} mapping parsed from a RELLIS ontology file.

    Handles both ontology formats shipped with RELLIS-3D:

    * ``ontology.yaml``  -- a YAML list whose first element is the id->name dict.
    * ``ontology2.yaml`` -- a YAML dict with a ``learning_map`` key.
    """
    path = Path(ontology_path)
    with path.open() as f:
        data: Any = yaml.safe_load(f)

    if isinstance(data, list):
        id_to_name: dict[int, str] = {int(k): str(v) for k, v in data[0].items()}
    elif isinstance(data, dict):
        raw = data.get("learning_map") or data.get("other_map")
        if raw is None:
            raise ValueError(f"Cannot find 'learning_map' or 'other_map' key in {path}")
        id_to_name = {int(k): str(v) for k, v in raw.items()}
    else:
        raise ValueError(f"Unexpected ontology format in {path}")

    return id_to_name


# ---------------------------------------------------------------------------
# Preprocessing steps  (Pipeline-compatible: (pts, labels) -> (pts, labels))
# ---------------------------------------------------------------------------

def range_filter(
    pts: np.ndarray,
    labels: np.ndarray | None,
    max_r: float = 50.0,
) -> tuple[np.ndarray, np.ndarray | None]:
    mask = np.linalg.norm(pts[:, :3], axis=1) < max_r
    return pts[mask], labels[mask] if labels is not None else None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(description="RELLIS-3D LiDAR viewer")
    p.add_argument("--root", default=str(_DEFAULT_ROOT), help="Dataset root directory")
    p.add_argument("--ontology", default=str(_DEFAULT_ONTOLOGY),
                   help="Path to ontology.yaml or ontology2.yaml")
    p.add_argument("--no-poses", action="store_true", help="Skip trajectory overlay")
    p.add_argument("--idx", type=int, default=0, help="Starting frame index")
    args = p.parse_args()

    root = Path(args.root)
    if not root.is_dir():
        raise SystemExit(f"Dataset root not found: {root}")

    # --- Label mapping ---
    mapping = load_label_mapping(args.ontology)
    print("Label mapping loaded:")
    for label_id, name in sorted(mapping.items()):
        print(f"  {label_id:2d}: {name}")

    # --- Load dataset (poses key included for trajectory) ---
    print(f"\nLoading RELLIS-3D from {root} …")
    ds = apairo.Rellis3DDataset(root, keys=["lidar", "labels", "poses"])
    print(f"  {len(ds)} scans")

    # --- Poses: apairo stores RELLIS poses as 3x4; load_poses lifts to 4x4 ---
    poses = None
    if not args.no_poses:
        poses = apairo_visu.load_poses(ds, key="poses")
        print(f"  {len(poses)} poses loaded")

    # --- Pipeline (single viewport, range-filtered) ---
    pipelines = [Pipeline("RELLIS-3D", [range_filter])]

    # --- Label + view config ---
    label_cfg = apairo_visu.load_label_config("rellis")
    view_cfg  = apairo_visu.ViewConfig(
        point_key="lidar",
        label_key="labels",
        intensity_channel=3,
    )

    print("Launching viewer … (<- -> or H/L to navigate, B for BEV, J for trajectory, T to cycle colours)")
    apairo_visu.LidarViewer.launch(
        ds,
        view_cfg=view_cfg,
        label_cfg=label_cfg,
        poses=poses,
        start_idx=args.idx,
        pipelines=pipelines,
    )


if __name__ == "__main__":
    main()

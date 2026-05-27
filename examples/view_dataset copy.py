"""Example: launch the LiDAR viewer on a GOOSE-3D dataset."""

import numpy as np
import apairo
import apairo_visu

# ---- Load dataset --------------------------------------------------------
dataset = apairo.Goose3DDataset(
    root_dir="/path/to/goose",
    keys=["lidar", "labels"],
    split="val",
)

# ---- Label config (built-in) ---------------------------------------------
label_cfg = apairo_visu.load_label_config("goose")

# ---- (Optional) load poses -----------------------------------------------
# poses is a list of 4×4 numpy arrays, one per frame.
# If your dataset has no pose data, simply omit it.
poses = None  # e.g. np.load("poses.npy")  → list(poses)

# ---- Specify which keys to use (defaults match GOOSE / RELLIS / SemanticKITTI)
view_cfg = apairo_visu.ViewConfig(
    point_key="lidar",
    label_key="labels",
    intensity_channel=3,
)

# ---- Launch (blocks until window is closed) ------------------------------
apairo_visu.LidarViewer.launch(
    dataset,
    view_cfg=view_cfg,
    label_cfg=label_cfg,
    poses=poses,
    start_idx=0,
)

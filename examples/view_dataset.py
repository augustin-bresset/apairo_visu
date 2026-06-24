"""Example: launch the LiDAR viewer on a GOOSE-3D dataset."""

import apairo
import apairo_visu

# ---- Load dataset --------------------------------------------------------
dataset = apairo.Goose3DDataset(
    "/path/to/goose",
    keys=["lidar", "labels"],
    split="val",
)

# ---- Label config (built-in) ---------------------------------------------
label_cfg = apairo_visu.load_label_config("goose")

# ---- (Optional) trajectory overlay ---------------------------------------
# For a dataset with a pose channel, load_poses normalises it to 4x4 matrices:
#     poses = apairo_visu.load_poses(dataset, key="poses")
poses = None

# ---- Which keys to read from each Sample ---------------------------------
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

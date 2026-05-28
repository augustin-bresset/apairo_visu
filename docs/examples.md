# Examples

## GOOSE-3D

```python
import apairo
import apairo_visu

ds  = apairo.Goose3DDataset("/data/goose", keys=["lidar", "labels"], split="val")
cfg = apairo_visu.load_label_config("goose")

apairo_visu.LidarViewer.launch(ds, label_cfg=cfg)
```

## RELLIS-3D

```python
import apairo
import apairo_visu

ds  = apairo.Rellis3DDataset("/data/rellis")
cfg = apairo_visu.load_label_config("rellis")

apairo_visu.LidarViewer.launch(ds, label_cfg=cfg)
```

## SemanticKITTI

```python
import apairo
import apairo_visu

ds  = apairo.SemanticKittiDataset("/data/kitti", split="train")
cfg = apairo_visu.load_label_config("semantic_kitti")

apairo_visu.LidarViewer.launch(ds, label_cfg=cfg, start_idx=100)
```

## TartanDrive (velodyne, no labels)

Point clouds from TartanDrive are `(N, 3)` XYZ arrays with no intensity column. The viewer defaults to Height mode.

```python
import apairo
import apairo_visu

seq_dir = "/data/tartan/turnpike_2023-09-12-12-39-19/turnpike_2023-09-12-12-39-19"
ds = apairo.TartanKittiDataset(seq_dir, keys=["velodyne_0"])

view_cfg = apairo_visu.ViewConfig(point_key="velodyne_0", label_key=None)
apairo_visu.LidarViewer.launch(ds, view_cfg=view_cfg)
```

See [`examples/view_tartandrive.py`](../examples/view_tartandrive.py) for a complete script including pose loading and trajectory overlay.

## Custom dataset + custom label config

Any dataset that returns a `Sample` with a point cloud tensor works with `LidarViewer`. For a custom dataset, provide a `ViewConfig` and optionally a `label_cfg` dict.

```python
import apairo_visu

# Custom dataset: sample.data["cloud"] → (N, 4) float32, sample.data["sem"] → (N,) int64
view_cfg = apairo_visu.ViewConfig(
    point_key="cloud",
    label_key="sem",
    intensity_channel=3,
)

label_cfg = apairo_visu.load_label_config("/path/to/my_classes.yaml")

apairo_visu.LidarViewer.launch(my_dataset, view_cfg=view_cfg, label_cfg=label_cfg)
```

## Using NearestSyncDataset to access multiple channels

When you need both the point cloud and another channel (e.g. odometry) in each sample:

```python
import apairo
import apairo_visu
import numpy as np

ds_raw = apairo.TartanKittiDataset(seq_dir, keys=["velodyne_0", "super_odom"])
ds = apairo_visu.NearestSyncDataset(ds_raw, reference_key="velodyne_0")

# sample.data now contains both keys, synchronized to the LiDAR timestamps
sample = ds[0]
print(sample.data["velodyne_0"].shape)   # (N, 3)
print(sample.data["super_odom"].shape)   # (7,)
```

---

## Pipeline comparison

Use `pipelines=` to compare multiple processing strategies side-by-side.  
See [`examples/view_pipelines.py`](../examples/view_pipelines.py) for the full runnable script.

### Validate a preprocessing step

```python
import apairo_visu
from apairo_visu import Pipeline
import numpy as np

def range_filter(pts, labels, min_r=1.0, max_r=50.0):
    r = np.linalg.norm(pts[:, :3], axis=1)
    mask = (r >= min_r) & (r <= max_r)
    return pts[mask], labels[mask] if labels is not None else None

apairo_visu.LidarViewer.launch(ds, label_cfg=cfg, pipelines=[
    Pipeline("Raw"),
    Pipeline("Range filter", [range_filter]),
])
```

### Compare two models

```python
apairo_visu.LidarViewer.launch(ds, label_cfg=cfg, pipelines=[
    Pipeline("Ground truth"),
    Pipeline("Model A", [preprocess, model_a]),
    Pipeline("Model B", [preprocess, model_b]),
])
```

Each pipeline step is a callable `(pts, labels) → (pts, labels)`:

```python
def model_a(pts, labels):
    logits = net_a(torch.from_numpy(pts).cuda())
    pred   = logits.argmax(dim=-1).cpu().numpy().astype(np.int64)
    return pts, pred
```

Pipelines run in **parallel** — each viewport updates as soon as its pipeline finishes.  Elapsed time per pipeline is shown in the left panel.

Use the **Active pipelines** checkboxes in the panel to show or hide individual viewports at runtime.  Unchecking a pipeline stops computing it on subsequent frames; re-checking re-runs it on the current frame immediately.  This lets you focus on any subset without restarting the script.

### Compare preprocessing strategies + model

```python
apairo_visu.LidarViewer.launch(ds, label_cfg=cfg, pipelines=[
    Pipeline("Raw"),
    Pipeline("Preprocess A",    [preprocess_a]),
    Pipeline("Preprocess A + model", [preprocess_a, model]),
    Pipeline("Preprocess B + model", [preprocess_b, model]),
])
```

---

## CLI

```bash
# GOOSE-3D
python -m apairo_visu --dataset goose --root /data/goose --split val

# SemanticKITTI, starting at frame 200
python -m apairo_visu --dataset semantic_kitti --root /data/kitti --idx 200

# Custom label config
python -m apairo_visu --dataset rellis --root /data/rellis --cfg my_colors.yaml

# No labels (intensity/height only)
python -m apairo_visu --dataset goose --root /data/goose --no-labels
```

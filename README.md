# apairo_visu

Interactive 3D LiDAR visualisation for [apairo](../apairo) datasets.

`apairo_visu` extends apairo with an Open3D-based viewer that works natively with any `AbstractDataset`. Features:

- Semantic label colouring, height (viridis), and intensity display modes
- Per-class filter and distribution panel
- Trajectory overlay from pose matrices
- **Multi-pipeline comparison**: run preprocessing and/or model inference on each frame and compare N viewports side-by-side, with pipelines executing in parallel

## Installation

```bash
cd ~/dev/apairo_visu
python -m venv .venv && source .venv/bin/activate
pip install -e ../apairo   # local dependency
pip install -e .
```

## Quick start

```python
import apairo
import apairo_visu

ds  = apairo.Goose3DDataset("/data/goose", split="val")
cfg = apairo_visu.load_label_config("goose")

apairo_visu.LidarViewer.launch(ds, label_cfg=cfg)
```

## Pipeline comparison

Compare preprocessing strategies or model predictions side-by-side.  
Each `Pipeline` is a named sequence of `(pts, labels) → (pts, labels)` callables.

```python
from apairo_visu import Pipeline

apairo_visu.LidarViewer.launch(ds, label_cfg=cfg, pipelines=[
    Pipeline("Ground truth"),
    Pipeline("Model A", [preprocess, model_a]),
    Pipeline("Model B", [preprocess, model_b]),
])
```

Pipelines run in parallel — each viewport updates as soon as its pipeline finishes.  
See [`examples/view_pipelines.py`](examples/view_pipelines.py) for a full runnable example.

## CLI

```bash
python -m apairo_visu --dataset goose --root /data/goose --split val
python -m apairo_visu --dataset rellis --root /data/rellis
python -m apairo_visu --dataset semantic_kitti --root /data/kitti --split train --idx 50
```

## Keyboard shortcuts

| Key | Action |
|-----|--------|
| `→` / `L` | Next frame |
| `←` / `H` | Previous frame |
| `T` | Cycle colour mode (Semantic → Intensity → Height) |
| `B` | Bird's-eye (top-down) view |
| `R` | Reset camera (all viewports) |
| `J` | Toggle trajectory overlay |

## Documentation

- [Getting started](docs/getting_started.md)
- [LidarViewer API](docs/viewer.md)
- [Label configurations](docs/label_configs.md)
- [NearestSyncDataset](docs/sync.md)
- [Examples](docs/examples.md)

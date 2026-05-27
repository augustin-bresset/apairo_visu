# apairo_visu

Interactive 3D LiDAR visualisation for [apairo](../apairo) datasets.

`apairo_visu` extends apairo with an Open3D-based viewer that works natively with any `AbstractDataset`. It supports semantic label colouring, height/intensity display modes, per-class filtering, and trajectory overlays from pose data.

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

ds = apairo.Goose3DDataset("/data/goose", split="val")
cfg = apairo_visu.load_label_config("goose")

apairo_visu.LidarViewer.launch(ds, label_cfg=cfg)
```

### CLI

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
| `R` | Reset camera |
| `J` | Toggle trajectory overlay |

## Documentation

- [Getting started](docs/getting_started.md)
- [LidarViewer API](docs/viewer.md)
- [Label configurations](docs/label_configs.md)
- [NearestSyncDataset](docs/sync.md)
- [Examples](docs/examples.md)

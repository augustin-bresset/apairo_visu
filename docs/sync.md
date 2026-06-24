# Synchronising async datasets

Some apairo datasets (e.g. `TartanKittiDataset`) are **asynchronous**: each
channel lives on its own timeline, so `dataset[i]` returns a `Sample` with a
single key -- the *i*-th event in the merged timeline. The viewer expects
**synchronous** frames where `dataset[i]` carries every channel at once.

You do **not** need a viewer-specific wrapper for this: apairo ships a
`synchronize()` method on every dataset. (Earlier versions of `apairo_visu`
shipped a `NearestSyncDataset` helper -- it has been removed in favour of the
built-in, which is faster, lazier and supports interpolation and tolerances.)

## Usage

```python
import apairo
import apairo_visu

# Async dataset with two channels on independent clocks
ds_async = apairo.TartanKittiDataset(seq_dir, keys=["velodyne_0", "super_odom"])

# Resample onto the velodyne clock: each frame now carries both channels
ds = ds_async.synchronize(reference="velodyne_0", method="nearest")

sample = ds[0]
pts  = sample.data["velodyne_0"]   # (N, 3) point cloud
odom = sample.data["super_odom"]   # odometry at the nearest timestamp

apairo_visu.LidarViewer.launch(
    ds,
    view_cfg=apairo_visu.ViewConfig(point_key="velodyne_0", label_key=None),
    poses=apairo_visu.load_poses(ds, key="super_odom"),   # trajectory overlay
)
```

## `synchronize()` in one line

| Argument | Meaning |
|---|---|
| `reference` | Channel name providing the clock (or an array of timestamps, or `None` for the lowest-frequency channel). |
| `method` | `"nearest"` (closest event), `"latest"` (last event with `t ≤ t_ref`, online-style), a custom `(channel_ts, ref_ts) -> indices` callable, or an `Interpolator`. |
| `tolerance` | Max `\|t − t_ref\|` in seconds; frames with any channel out of tolerance are dropped. |

The result is a normal synchronous apairo view -- `len()`, random access,
`filter`, `select`, `cache` and PyTorch `DataLoader` all work on it, and so does
`LidarViewer`. See the apairo docs for the full contract.

## Trajectory overlay

A synchronised odometry channel pairs naturally with the viewer's trajectory
overlay: `apairo_visu.load_poses(ds, key="super_odom")` normalises each
`[x, y, z, qx, qy, qz, qw]` row (trailing columns ignored) to a `4×4`
`T_world_sensor` matrix. See [`examples/view_tartandrive.py`](../examples/view_tartandrive.py).

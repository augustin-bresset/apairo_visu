# NearestSyncDataset

Utility for working with apairo's async (multi-channel, time-stamped) datasets in a viewer context.

## Motivation

Async apairo datasets (like `TartanKittiDataset`) store multiple channels on independent timelines. `dataset[i]` returns a `Sample` with **one key only** -- the i-th event in the merged timeline, which could be from any channel. This makes sequential frame-by-frame navigation difficult.

`NearestSyncDataset` wraps an async dataset and presents it as a **synchronous** dataset aligned to one reference channel. Each `dataset[i]` returns a complete `Sample` with all channels, synchronized by nearest-timestamp matching.

## Usage

```python
import apairo
import apairo_visu

# Load async dataset with two channels
ds_raw = apairo.TartanKittiDataset(seq_dir, keys=["velodyne_0", "super_odom"])

# Wrap: align everything to the velodyne_0 timeline
ds = apairo_visu.NearestSyncDataset(ds_raw, reference_key="velodyne_0")

# Now ds[i] returns a Sample with both keys, synchronized
sample = ds[0]
pts   = sample.data["velodyne_0"]   # (N, 3) point cloud
odom  = sample.data["super_odom"]   # (7,)   odometry at nearest timestamp
```

## How it works

At construction time, for each non-reference key the wrapper pre-computes an index mapping:

```
for each reference frame i:
    frame_idx[key][i] = argmin |timestamps[key] - reference_timestamps[i]|
```

At access time, `__getitem__(i)` calls `dataset.loaders[key][frame_idx[key][i]]` for every key. No data is cached -- each access re-reads from the underlying loaders.

## API

```python
class NearestSyncDataset:
    def __init__(self, dataset, reference_key: str): ...
    def __len__(self) -> int: ...
    def __getitem__(self, idx: int) -> Sample: ...
```

The returned object supports `len()`, index access, and iteration -- the same interface expected by `LidarViewer`.

## Caveats

- **Temporal lag**: nearest-neighbor matching introduces up to half the inter-frame interval of the non-reference channel. For a 100 Hz odometry matched to a 10 Hz LiDAR, the maximum lag is 5 ms -- negligible for most use cases.
- **End-of-sequence boundary**: the last few reference frames may be matched to the same final non-reference frame if the channels do not have exactly the same duration.
- **Large datasets**: the index mapping is built entirely in memory at construction time. For a dataset with N reference frames and M non-reference frames, the construction cost is O(N x M) time and O(N) memory per non-reference key.

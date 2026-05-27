"""Synchronisation utilities for apairo async datasets."""

from __future__ import annotations

import numpy as np

from apairo.core.sample import Sample


class NearestSyncDataset:
    """Aligns an async apairo dataset to a single reference timeline.

    An async dataset (e.g. ``TartanKittiDataset``) interleaves events from
    multiple channels in one flat timeline.  This wrapper presents it as a
    *synchronous* dataset: ``self[i]`` returns one sample per reference-key
    frame, with every other key matched to its nearest timestamp.

    The wrapped dataset must expose ``loaders``, ``timestamps``, and ``keys``
    (all present on :class:`~apairo.dataset.kitti.KittiDataset` and its
    subclasses).

    Args:
        dataset:       Any apairo async dataset with ``loaders`` and
                       ``timestamps`` attributes.
        reference_key: Channel that defines the output timeline (typically the
                       highest-frequency or primary sensor, e.g. ``"velodyne_0"``).

    Example::

        ds_raw = TartanKittiDataset(seq_dir, keys=["velodyne_0", "super_odom"])
        ds = NearestSyncDataset(ds_raw, reference_key="velodyne_0")
        sample = ds[0]
        # sample.data["velodyne_0"]  → tensor (N, 3)
        # sample.data["super_odom"] → tensor (7,)  nearest-matched
    """

    def __init__(self, dataset, reference_key: str) -> None:
        if reference_key not in dataset.keys:
            raise KeyError(
                f"reference_key '{reference_key}' not found in dataset keys: {dataset.keys}"
            )
        self._dataset = dataset
        self._ref_key = reference_key

        ref_ts = dataset.timestamps[reference_key]   # (N,)
        self._ref_ts = ref_ts

        # For each key, build a (N,) array mapping reference frame i → loader index
        self._frame_idx: dict[str, np.ndarray] = {}
        for key in dataset.keys:
            ts = dataset.timestamps[key]             # (M,)
            if key == reference_key:
                self._frame_idx[key] = np.arange(len(ref_ts), dtype=np.intp)
            else:
                # nearest-neighbour: argmin over |ts - ref_ts[i]| for each i
                idx = np.abs(ts[:, None] - ref_ts[None, :]).argmin(axis=0)
                self._frame_idx[key] = idx.astype(np.intp)

    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._ref_ts)

    def __getitem__(self, idx: int) -> Sample:
        if not 0 <= idx < len(self):
            raise IndexError(f"Index {idx} out of range [0, {len(self)})")
        data = {
            key: self._dataset.loaders[key][int(self._frame_idx[key][idx])]
            for key in self._dataset.keys
        }
        return Sample(data=data, timestamp=float(self._ref_ts[idx]))

    def __iter__(self):
        self._pos = 0
        return self

    def __next__(self) -> Sample:
        if self._pos >= len(self):
            raise StopIteration
        sample = self[self._pos]
        self._pos += 1
        return sample

    @property
    def keys(self) -> list[str]:
        return list(self._dataset.keys)

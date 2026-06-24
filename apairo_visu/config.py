"""View and label configuration for :mod:`apairo_visu`.

These types carry *no* Open3D dependency, so they can be imported and built in
a headless context (e.g. while assembling pipelines inside a training script)
without pulling in the GUI stack.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

_CONFIGS_DIR = Path(__file__).parent / "configs"

BUILTIN_CONFIGS = {"goose", "rellis", "semantic_kitti"}


@dataclass
class ViewConfig:
    """Mapping from :class:`apairo.Sample` keys to viewer inputs.

    Tells the viewer which arrays to extract from ``sample.data`` on each frame.
    All fields correspond to keys in the ``Sample.data`` dict returned by
    ``dataset[idx]``.

    Attributes:
        point_key: Key for the point cloud array, shape ``(N, C)`` float32.
            The first three columns must be X, Y, Z.
        label_key: Key for the per-point semantic label array, shape ``(N,)``
            int64.  Set to ``None`` to disable label-based display modes; the
            viewer then starts in **Height** mode.
        intensity_channel: Column index of the intensity channel.  Ignored when
            the array has fewer than ``intensity_channel + 1`` columns (falls
            back silently to Height mode).
    """

    point_key: str = "lidar"
    label_key: str | None = "labels"
    intensity_channel: int = 3


def load_label_config(name_or_path: str | Path) -> dict:
    """Load a label config dict from a built-in name or a YAML file path.

    Args:
        name_or_path: One of ``"goose"``, ``"rellis"``, ``"semantic_kitti"``,
            or a path to a custom YAML file.

    Returns:
        Dict with keys ``color_map``, ``semantic_map``, and optionally
        ``traversable_map`` and ``ignore_index``.
    """
    p = Path(name_or_path)
    if not p.suffix:
        p = _CONFIGS_DIR / f"{name_or_path}.yaml"
    with open(p) as f:
        return yaml.safe_load(f)

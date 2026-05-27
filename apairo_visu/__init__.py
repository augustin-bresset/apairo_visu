"""apairo_visu — interactive 3-D LiDAR visualisation for apairo datasets."""

from pathlib import Path

import yaml

from .viewer import LidarViewer, ViewConfig
from .sync import NearestSyncDataset

_CONFIGS_DIR = Path(__file__).parent / "configs"

BUILTIN_CONFIGS = {"goose", "rellis", "semantic_kitti"}


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


__all__ = [
    "LidarViewer",
    "ViewConfig",
    "NearestSyncDataset",
    "load_label_config",
    "BUILTIN_CONFIGS",
]

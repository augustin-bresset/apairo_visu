"""CLI entry-point for apairo_visu.

Examples:
    python -m apairo_visu --dataset goose --root /data/goose --split train
    python -m apairo_visu --dataset rellis --root /data/rellis
    python -m apairo_visu --dataset semantic_kitti --root /data/kitti --split val
    python -m apairo_visu --dataset goose --root /data/goose --cfg my_colors.yaml --idx 50
"""

from __future__ import annotations

import argparse

from . import LidarViewer, ViewConfig, load_label_config

_DATASET_CLASSES = {
    "goose": ("apairo", "Goose3DDataset"),
    "rellis": ("apairo", "Rellis3DDataset"),
    "semantic_kitti": ("apairo", "SemanticKittiDataset"),
}

_DATASET_KEYS = {
    "goose": ("lidar", "labels"),
    "rellis": ("lidar", "labels"),
    "semantic_kitti": ("lidar", "labels"),
}


def _load_dataset(name: str, root: str, split: str | None):
    if name not in _DATASET_CLASSES:
        raise ValueError(
            f"Unknown dataset '{name}'. Known: {sorted(_DATASET_CLASSES)}"
        )
    module_name, class_name = _DATASET_CLASSES[name]
    import importlib
    mod = importlib.import_module(module_name)
    cls = getattr(mod, class_name)
    kwargs = {}
    if split is not None:
        kwargs["split"] = split
    return cls(root, **kwargs)


def main() -> None:
    p = argparse.ArgumentParser(description="apairo_visu -- LiDAR dataset viewer")
    p.add_argument("--dataset", required=True, help="Dataset name (goose | rellis | semantic_kitti)")
    p.add_argument("--root", required=True, help="Path to dataset root directory")
    p.add_argument("--split", default=None, help="Dataset split (train | val | test)")
    p.add_argument("--cfg", default=None, help="Path to a custom label YAML config (default: built-in)")
    p.add_argument("--idx", type=int, default=0, help="Starting frame index")
    p.add_argument("--no-labels", action="store_true", help="Disable label loading")
    args = p.parse_args()

    dataset = _load_dataset(args.dataset, args.root, args.split)

    cfg_path = args.cfg or args.dataset
    label_cfg = load_label_config(cfg_path) if not args.no_labels else None

    point_key, label_key = _DATASET_KEYS.get(args.dataset, ("lidar", "labels"))
    view_cfg = ViewConfig(
        point_key=point_key,
        label_key=None if args.no_labels else label_key,
    )

    print(f"Dataset: {args.dataset}  ({len(dataset)} frames)")
    print(f"Starting at frame {args.idx}")
    LidarViewer.launch(dataset, view_cfg=view_cfg, label_cfg=label_cfg, start_idx=args.idx)


if __name__ == "__main__":
    main()

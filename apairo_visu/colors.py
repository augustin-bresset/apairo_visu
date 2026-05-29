"""Color utilities for LiDAR point cloud visualisation."""

from __future__ import annotations

import numpy as np


def hex_to_rgb(hex_str: str) -> list[int]:
    h = hex_str.lstrip("#")
    return [int(h[i : i + 2], 16) for i in (0, 2, 4)]


def normalize_color_map(color_map: dict) -> dict[int, list[int]]:
    """Normalize color_map to {int_id: [r, g, b]} with values in [0, 255]."""
    out = {}
    for k, v in color_map.items():
        if isinstance(v, str):
            out[int(k)] = hex_to_rgb(v)
        else:
            out[int(k)] = [int(c) for c in v]
    return out


def labels_to_colors(
    labels: np.ndarray, color_map: dict[int, list[int]]
) -> np.ndarray:
    """Map label array to (N, 3) float64 RGB in [0, 1]. Unknown labels -> gray."""
    default = [128, 128, 128]
    colors = np.array(
        [color_map.get(int(l), default) for l in labels], dtype=np.float64
    )
    return colors / 255.0


def auto_color_map(num_classes: int) -> dict[int, list[int]]:
    """Generate a visually distinct color map via matplotlib tab20."""
    import matplotlib.pyplot as plt

    cmap = plt.get_cmap("tab20")
    return {i: [int(c * 255) for c in cmap(i % 20)[:3]] for i in range(num_classes)}


def height_colors(z: np.ndarray) -> np.ndarray:
    """Map Z coordinates to (N, 3) float64 viridis colors."""
    import matplotlib.pyplot as plt

    z_min, z_max = z.min(), z.max()
    span = z_max - z_min
    norm = (z - z_min) / span if span > 0 else np.zeros_like(z)
    cmap = plt.get_cmap("viridis")
    return cmap(norm)[:, :3]


def intensity_colors(intensity: np.ndarray) -> np.ndarray:
    """Map intensity values to (N, 3) float64 grayscale."""
    i_min, i_max = intensity.min(), intensity.max()
    span = i_max - i_min
    norm = (intensity - i_min) / span if span > 0 else np.zeros_like(intensity)
    return np.stack([norm, norm, norm], axis=1).astype(np.float64)

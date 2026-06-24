"""Tests for colour utilities."""

import numpy as np

from apairo_visu.colors import (
    hex_to_rgb,
    height_colors,
    intensity_colors,
    labels_to_colors,
    normalize_color_map,
)


def test_hex_to_rgb():
    assert hex_to_rgb("#ff0080") == [255, 0, 128]
    assert hex_to_rgb("00ff00") == [0, 255, 0]


def test_normalize_color_map_mixed_inputs():
    cmap = normalize_color_map({0: "#ffffff", "1": [10, 20, 30]})
    assert cmap[0] == [255, 255, 255]
    assert cmap[1] == [10, 20, 30]
    assert all(isinstance(k, int) for k in cmap)


def test_labels_to_colors_known_and_unknown():
    cmap = {0: [255, 0, 0], 1: [0, 255, 0]}
    out = labels_to_colors(np.array([0, 1, 99]), cmap)
    assert out.shape == (3, 3)
    np.testing.assert_allclose(out[0], [1.0, 0.0, 0.0])
    np.testing.assert_allclose(out[1], [0.0, 1.0, 0.0])
    np.testing.assert_allclose(out[2], [128 / 255] * 3)  # unknown -> gray


def test_height_colors_shape_and_range():
    z = np.linspace(-2, 5, 50)
    out = height_colors(z)
    assert out.shape == (50, 3)
    assert out.min() >= 0.0 and out.max() <= 1.0


def test_height_colors_constant_input():
    out = height_colors(np.full(8, 3.0))  # zero span must not divide by zero
    assert out.shape == (8, 3)
    assert np.isfinite(out).all()


def test_intensity_colors_is_grayscale():
    out = intensity_colors(np.array([0.0, 0.5, 1.0]))
    assert out.shape == (3, 3)
    # grayscale: all three channels equal per row
    np.testing.assert_allclose(out[:, 0], out[:, 1])
    np.testing.assert_allclose(out[:, 1], out[:, 2])

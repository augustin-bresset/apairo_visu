# Label configurations

A label config is a dict that maps class IDs to colours and names. It is used by the viewer to colour point cloud labels in Semantic mode and to populate the per-class filter panel.

## Structure

```yaml
# Required
color_map:
  0:  [0, 0, 0]        # class id -> [R, G, B]  (0–255)
  1:  '#ff0000'        # hex strings are also accepted
  23: [170, 170, 170]

semantic_map:
  0:  unlabeled
  1:  car
  23: asphalt

# Optional
traversable_map:
  - 23   # class ids considered traversable (used for future traversability display)

ignore_index: 0        # class id to ignore in stats (e.g. unlabeled)
```

Both `color_map` and `semantic_map` accept non-contiguous integer keys -- useful for datasets like RELLIS-3D where class IDs jump (0, 1, 3, 4, 5, …).

## Loading a config

### Built-in configs

Three configs are bundled with `apairo_visu`:

```python
cfg = apairo_visu.load_label_config("goose")           # GOOSE-3D (64 classes)
cfg = apairo_visu.load_label_config("rellis")          # RELLIS-3D (20 classes)
cfg = apairo_visu.load_label_config("semantic_kitti")  # SemanticKITTI (34 IDs)
```

### Custom YAML file

```python
cfg = apairo_visu.load_label_config("/path/to/my_dataset.yaml")
```

The function detects whether the argument is a built-in name (no file extension) or a file path (has extension or is an existing path).

### Inline dict

You can also pass a dict directly to `LidarViewer.launch`:

```python
label_cfg = {
    "color_map":    {0: [0,0,0], 1: [255,0,0]},
    "semantic_map": {0: "background", 1: "object"},
}
apairo_visu.LidarViewer.launch(ds, label_cfg=label_cfg)
```

## Built-in configs

### GOOSE-3D (`"goose"`)

64 outdoor classes including asphalt, soil, vegetation, infrastructure, and vehicles. Traversable classes: `asphalt` (23), `soil` (31), `low_grass` (50), `high_grass` (51).

### RELLIS-3D (`"rellis"`)

20 off-road classes with non-contiguous IDs (0, 1, 3, 4, …, 34). Traversable classes: `dirt` (1), `grass` (3), `asphalt` (10), `concrete` (23), `puddle` (31), `mud` (33).

### SemanticKITTI (`"semantic_kitti"`)

34 original label IDs covering urban driving scenes. Traversable classes: `road` (40), `parking` (44), `sidewalk` (48), `other-ground` (49), `terrain` (72).

## Auto-coloring

When `label_cfg=None`, the viewer auto-generates 32 distinct colours using matplotlib's `tab20` colormap. Class names are shown as their integer IDs. This is useful for exploring unlabeled or custom datasets.

```python
apairo_visu.LidarViewer.launch(ds)   # no label_cfg -> auto colors
```

# LidarViewer

The main class of `apairo_visu`. Builds an Open3D GUI window that lets you navigate a dataset frame by frame.

## API

### `LidarViewer.launch` (static)

```python
LidarViewer.launch(
    dataset,
    view_cfg: ViewConfig | None = None,
    label_cfg: dict | None = None,
    poses: list[np.ndarray] | None = None,
    start_idx: int = 0,
)
```

Creates the application and **blocks** until the window is closed.

| Parameter | Description |
|---|---|
| `dataset` | Any apairo dataset — must support `dataset[idx]` returning a `Sample`. |
| `view_cfg` | Which keys to read from each `Sample`. Defaults to `point_key="lidar"`, `label_key="labels"`. |
| `label_cfg` | Dict with `color_map`, `semantic_map`, optional `traversable_map`. Use `load_label_config()` for built-in configs. |
| `poses` | Optional list of 4×4 `np.ndarray` (T_world_sensor), one per frame. Enables the trajectory overlay. |
| `start_idx` | Frame index to display first. |

### `ViewConfig`

```python
@dataclass
class ViewConfig:
    point_key: str = "lidar"          # key for the point cloud in sample.data
    label_key: str | None = "labels"  # key for per-point labels, or None
    intensity_channel: int = 3        # column index for intensity
```

## Display modes

Three colour modes are available, cycled with `T` or via the combo box in the panel:

| Mode | Description | Requires |
|---|---|---|
| **Semantic** | Per-class colour from `label_cfg` | `label_key` + `label_cfg` |
| **Intensity** | Grayscale from the intensity column | `intensity_channel < point_cloud.shape[1]` |
| **Height** | Viridis colormap on the Z coordinate | always available |

When `label_key=None`, the viewer starts in **Height** mode. When the intensity channel is out of bounds (e.g. `(N, 3)` point clouds without an intensity column), Intensity mode falls back to Height.

## Panel layout

```
┌──────────────────┬──────────────────────────────────────┐
│  NAVIGATION      │                                      │
│  frame / total   │                                      │
│  point count     │                                      │
│  [< Prev][Next>] │                                      │
│  [BEV]  [Reset]  │         3-D scene                    │
│                  │                                      │
│  COLOUR MODE [T] │                                      │
│  [Semantic ▼]    │                                      │
│                  │                                      │
│  OVERLAYS        │                                      │
│  ☑ Trajectory[J] │                                      │
│                  │                                      │
│  CLASS DISTRIB.  │                                      │
│  asphalt  34.2%  │                                      │
│  soil      8.1%  │                                      │
│  ...             │                                      │
│                  │                                      │
│  FILTER CLASSES  │                                      │
│  [Show][Hide]    │                                      │
│  ■ 23: asphalt   │                                      │
│  ■ 31: soil      │                                      │
│  ...             │                                      │
└──────────────────┴──────────────────────────────────────┘
```

The **Overlays** section and **Filter classes** section are only shown when the corresponding data is available (poses and labels, respectively).

## Keyboard shortcuts

| Key | Action |
|---|---|
| `→` or `L` | Next frame |
| `←` or `H` | Previous frame |
| `T` | Cycle colour mode |
| `B` | Bird's-eye (top-down) view |
| `R` | Reset camera to bounding box |
| `J` | Toggle trajectory overlay (when poses provided) |

## Trajectory overlay

When `poses` is provided, the viewer draws:
- **Blue line** — past trajectory (frames 0 to current)
- **Orange line** — future trajectory (frames current to end)

Both are rendered in the **current sensor frame**: all trajectory waypoints (world-frame origins from the pose matrices) are transformed into the coordinate system of the current LiDAR scan using `T_sensor_world = inv(poses[current_idx])`.

```python
poses = [T_world_sensor_0, T_world_sensor_1, ...]   # list of (4, 4) float64
apairo_visu.LidarViewer.launch(ds, poses=poses)
```

## Programmatic construction

You can instantiate the viewer without launching the app (useful for testing or embedding):

```python
app = gui.Application.instance
app.initialize()

viewer = apairo_visu.LidarViewer(dataset, view_cfg, label_cfg, poses)
viewer._build_window()
app.run()
```

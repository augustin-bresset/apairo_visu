"""Interactive 3-D LiDAR viewer for apairo datasets — multi-pipeline edition.

Usage (single viewport, no transform — backward compatible):
    from apairo_visu import LidarViewer, ViewConfig, load_label_config
    cfg = load_label_config("goose")
    LidarViewer.launch(dataset, label_cfg=cfg)

Usage (compare multiple pipelines side-by-side):
    from apairo_visu import LidarViewer, Pipeline, load_label_config
    cfg = load_label_config("goose")
    LidarViewer.launch(dataset, label_cfg=cfg, pipelines=[
        Pipeline("Raw",     []),
        Pipeline("Model A", [preprocess, model_a]),
        Pipeline("Model B", [preprocess, model_b]),
    ])

Each Pipeline step is a callable: (pts: ndarray, labels: ndarray|None) -> (pts, labels).
Pipelines execute in parallel via a thread pool; each viewport updates as soon as
its pipeline finishes (useful when inference is slow).

Keyboard shortcuts:
    Right / L   next frame
    Left  / H   previous frame
    R           reset camera (all viewports)
    B           bird's-eye (top-down) view
    T           cycle colour mode  (Semantic → Intensity → Height)
    J           toggle trajectory overlay
"""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import open3d as o3d
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering

from .colors import (
    auto_color_map,
    height_colors,
    intensity_colors,
    labels_to_colors,
    normalize_color_map,
)

PANEL_W = 290
POINT_SIZE = 2.5
DISPLAY_MODES = ["Semantic", "Intensity", "Height"]
BANNER_H = 22   # pixels — pipeline-name banner above each viewport

C_TRAJ_PAST   = [0.20, 0.60, 1.00]
C_TRAJ_FUTURE = [1.00, 0.60, 0.10]


# ---------------------------------------------------------------------------
# Public data types
# ---------------------------------------------------------------------------


@dataclass
class ViewConfig:
    """Mapping from :class:`apairo.Sample` keys to viewer inputs.

    Tells the viewer which tensors to extract from ``sample.data`` on each
    frame.  All fields correspond to keys in the ``Sample.data`` dict returned
    by ``dataset[idx]``.

    Attributes:
        point_key: Key for the point cloud tensor, shape ``(N, C)`` float32.
            The first three columns must be X, Y, Z.
        label_key: Key for the per-point semantic label tensor, shape ``(N,)``
            int64.  Set to ``None`` to disable label-based display modes; the
            viewer will then start in **Height** mode.
        intensity_channel: Column index of the intensity channel.  Ignored when
            the tensor has fewer than ``intensity_channel + 1`` columns (falls
            back silently to Height mode).
    """

    point_key: str = "lidar"
    label_key: str | None = "labels"
    intensity_channel: int = 3


@dataclass
class Pipeline:
    """Named sequence of per-frame transforms applied before rendering.

    Each step is a callable with signature::

        step(pts: np.ndarray, labels: np.ndarray | None)
            -> tuple[np.ndarray, np.ndarray | None]

    Steps are applied in order; the output of each step is passed as input to
    the next.  Pass an empty ``steps`` list (the default) to display the raw
    frame without any transform.

    One :class:`Pipeline` maps to one viewport in the viewer.  When multiple
    pipelines are given to :meth:`LidarViewer.launch`, they run concurrently on
    a shared thread pool and each viewport updates as soon as its pipeline
    finishes — useful when inference is slow.

    Attributes:
        name: Label shown in the viewport banner and in the timing panel.
        steps: Ordered list of transform callables.

    Examples::

        Pipeline("Raw")                                  # no transform
        Pipeline("Range filter", [range_filter])         # preprocessing only
        Pipeline("Model A", [preprocess, model_a])       # preprocess → inference
    """

    name: str
    steps: list[Callable] = field(default_factory=list)

    def run(
        self,
        pts: np.ndarray,
        labels: np.ndarray | None,
    ) -> tuple[np.ndarray, np.ndarray | None]:
        for step in self.steps:
            pts, labels = step(pts, labels)
        return pts, labels


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _to_numpy(t) -> np.ndarray:
    if hasattr(t, "numpy"):
        return t.numpy()
    return np.asarray(t)


def _make_lineset(pts: np.ndarray, edges: list, color: list) -> o3d.geometry.LineSet:
    ls = o3d.geometry.LineSet()
    ls.points = o3d.utility.Vector3dVector(pts.astype(np.float64))
    ls.lines  = o3d.utility.Vector2iVector(edges)
    ls.colors = o3d.utility.Vector3dVector(np.tile(color, (len(edges), 1)))
    return ls


# ---------------------------------------------------------------------------
# Viewer
# ---------------------------------------------------------------------------


class LidarViewer:
    """Interactive 3-D LiDAR viewer for any apairo AbstractDataset.

    Supports one or more named pipelines displayed side-by-side for comparison.
    Each pipeline is a sequence of ``(pts, labels) → (pts, labels)`` callables
    applied before rendering.  Pipelines run in parallel via a thread pool; each
    viewport updates as soon as its pipeline finishes.

    Args:
        dataset:   Any apairo synchronous dataset (SynchronousDataset subclass).
        view_cfg:  Which keys to extract from each Sample.  Defaults to
                   ``point_key="lidar"`` and ``label_key="labels"``.
        label_cfg: Dict with ``color_map``, ``semantic_map``, and optionally
                   ``traversable_map``.  Use ``load_label_config(name)`` for
                   built-in configs.  Pass ``None`` to skip label colouring.
        poses:     Optional list of 4×4 numpy pose matrices (T_world_sensor),
                   one per frame, for a trajectory overlay (viewport 0 only).
        start_idx: First frame to display.
        pipelines: List of :class:`Pipeline` objects — one viewport per entry.
                   Defaults to ``[Pipeline("Raw", [])]`` (current frame, no transform).
    """

    def __init__(
        self,
        dataset,
        view_cfg: ViewConfig | None = None,
        label_cfg: dict | None = None,
        poses: list[np.ndarray] | None = None,
        start_idx: int = 0,
        pipelines: list[Pipeline] | None = None,
    ) -> None:
        self.dataset     = dataset
        self.cfg         = view_cfg or ViewConfig()
        self.current_idx = start_idx
        self._poses      = poses
        self._pipelines  = pipelines if pipelines is not None else [Pipeline("Raw", [])]

        # Colour / label setup
        if label_cfg is not None:
            self.color_map   = normalize_color_map(label_cfg["color_map"])
            self.semantic_map = {
                int(k): v for k, v in label_cfg.get("semantic_map", {}).items()
            }
        else:
            n = 32
            self.color_map    = auto_color_map(n)
            self.semantic_map = {i: str(i) for i in range(n)}

        self._trav_ids: set[int] = set()
        if label_cfg:
            self._trav_ids = {int(i) for i in label_cfg.get("traversable_map", [])}

        self._class_ids        = sorted(self.semantic_map.keys())
        self.active_classes: set[int] = set(self._class_ids)
        self._has_labels       = self.cfg.label_key is not None
        self._display_mode: int = 0 if self._has_labels else 2

        # Per-viewport state (one entry per pipeline)
        n = len(self._pipelines)
        self._scenes:            list[gui.SceneWidget]        = []
        self._banner_labels:     list[gui.Label]              = []
        self._mats:              list[rendering.MaterialRecord] = []
        self._cached_pts:        list[np.ndarray | None]      = [None] * n
        self._cached_labels:     list[np.ndarray | None]      = [None] * n
        self._camera_initialized: list[bool]                  = [False] * n

        # Parallel execution
        self._executor   = ThreadPoolExecutor(max_workers=max(n, 1))
        self._refresh_id = 0  # bumped on each navigation; discards stale callbacks

        # Shared panel widget refs
        self._window:        gui.Window | None    = None
        self._panel:         gui.Vert  | None     = None
        self._lbl_frame:     gui.Label | None     = None
        self._lbl_npts:      gui.Label | None     = None
        self._lbl_stats:     gui.Label | None     = None
        self._timing_labels: list[gui.Label]      = []
        self._checkboxes:    dict[int, gui.Checkbox] = {}
        self._mode_combo:    gui.Combobox | None  = None
        self._cb_traj:       gui.Checkbox | None  = None
        self._show_traj:     bool                 = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @staticmethod
    def launch(
        dataset,
        view_cfg: ViewConfig | None = None,
        label_cfg: dict | None = None,
        poses: list[np.ndarray] | None = None,
        start_idx: int = 0,
        pipelines: list[Pipeline] | None = None,
    ) -> None:
        """Create the GUI application and block until the window is closed.

        Args:
            dataset:   Any apairo dataset — must support ``dataset[idx]``
                       returning a ``Sample`` with a ``data`` dict.
            view_cfg:  Which keys to read from each ``Sample``.  Defaults to
                       ``point_key="lidar"`` and ``label_key="labels"``.
            label_cfg: Dict with ``color_map``, ``semantic_map``, and optionally
                       ``traversable_map``.  Use :func:`load_label_config` for
                       built-in configs.  Pass ``None`` to skip label colouring.
            poses:     Optional list of 4×4 ``np.ndarray`` (T_world_sensor), one
                       per frame.  Enables the trajectory overlay in viewport 0.
            start_idx: Frame index to display first.
            pipelines: List of :class:`Pipeline` objects — one viewport per entry.
                       Pipelines run in parallel; each viewport updates as soon as
                       its pipeline finishes.  Defaults to
                       ``[Pipeline("Raw", [])]``.
        """
        app = gui.Application.instance
        app.initialize()
        viewer = LidarViewer(dataset, view_cfg, label_cfg, poses, start_idx, pipelines)
        viewer._build_window()
        app.run()

    # ------------------------------------------------------------------
    # Window construction
    # ------------------------------------------------------------------

    def _build_window(self) -> None:
        app = gui.Application.instance
        w   = app.create_window("apairo — LiDAR Viewer", 1500, 900)
        self._window = w
        em = w.theme.font_size

        multi = len(self._pipelines) > 1

        # --- N viewports (one per pipeline) ---
        for pipeline in self._pipelines:
            mat = rendering.MaterialRecord()
            mat.shader     = "defaultUnlit"
            mat.point_size = POINT_SIZE
            self._mats.append(mat)

            scene = gui.SceneWidget()
            scene.scene = rendering.Open3DScene(w.renderer)
            scene.scene.set_background([0.08, 0.08, 0.08, 1.0])
            scene.set_on_key(self._on_key)
            self._scenes.append(scene)
            w.add_child(scene)

            if multi:
                banner = gui.Label(f"  {pipeline.name}")
                banner.text_color = gui.Color(0.9, 0.85, 0.5)
                self._banner_labels.append(banner)
                w.add_child(banner)

        # --- Left panel ---
        panel = gui.Vert(int(0.4 * em), gui.Margins(int(0.6 * em)))

        # Navigation
        panel.add_child(self._section_label("Navigation", em))

        self._lbl_frame = gui.Label("— / —")
        self._lbl_frame.text_color = gui.Color(0.85, 0.85, 0.85)
        panel.add_child(self._lbl_frame)

        self._lbl_npts = gui.Label("Points: —")
        self._lbl_npts.text_color = gui.Color(0.6, 0.6, 0.6)
        panel.add_child(self._lbl_npts)

        nav_row = gui.Horiz(int(0.3 * em))
        btn_prev = gui.Button("< Prev"); btn_prev.set_on_clicked(self._on_prev)
        btn_next = gui.Button("Next >"); btn_next.set_on_clicked(self._on_next)
        nav_row.add_stretch(); nav_row.add_child(btn_prev)
        nav_row.add_child(btn_next); nav_row.add_stretch()
        panel.add_child(nav_row)

        cam_row = gui.Horiz(int(0.3 * em))
        btn_bev   = gui.Button("BEV  [B]");   btn_bev.set_on_clicked(self._look_bev)
        btn_reset = gui.Button("Reset  [R]"); btn_reset.set_on_clicked(self._reset_camera)
        cam_row.add_stretch(); cam_row.add_child(btn_bev)
        cam_row.add_child(btn_reset); cam_row.add_stretch()
        panel.add_child(cam_row)

        if multi:
            btn_sync = gui.Button("Sync cam")
            btn_sync.set_on_clicked(self._sync_cameras)
            row = gui.Horiz(int(0.3 * em))
            row.add_stretch(); row.add_child(btn_sync); row.add_stretch()
            panel.add_child(row)

        panel.add_child(gui.Label(""))

        # Colour mode
        panel.add_child(self._section_label("Colour mode  [T]", em))
        combo = gui.Combobox()
        for m in DISPLAY_MODES:
            combo.add_item(m)
        combo.selected_index = self._display_mode
        combo.set_on_selection_changed(self._on_mode_changed)
        self._mode_combo = combo
        panel.add_child(combo)

        panel.add_child(gui.Label(""))

        # Trajectory overlay (when poses provided, shown in viewport 0 only)
        if self._poses is not None:
            panel.add_child(self._section_label("Overlays", em))
            cb_traj = gui.Checkbox("Trajectory  [J]")
            cb_traj.checked = False
            cb_traj.set_on_checked(self._on_traj_toggle)
            self._cb_traj = cb_traj
            panel.add_child(cb_traj)
            panel.add_child(gui.Label(""))

        # Per-pipeline timing (only in multi-pipeline mode)
        if multi:
            panel.add_child(self._section_label("Pipelines", em))
            for pipeline in self._pipelines:
                lbl = gui.Label(f"{pipeline.name}: —")
                lbl.text_color = gui.Color(0.7, 0.7, 0.7)
                self._timing_labels.append(lbl)
                panel.add_child(lbl)
            panel.add_child(gui.Label(""))

        # Class distribution (from pipeline 0)
        if self._has_labels:
            panel.add_child(self._section_label("Class distribution", em))
            self._lbl_stats = gui.Label("—")
            self._lbl_stats.text_color = gui.Color(0.75, 0.75, 0.75)
            panel.add_child(self._lbl_stats)
            panel.add_child(gui.Label(""))

            # Class filter (applied globally to all viewports)
            panel.add_child(self._section_label("Filter classes", em))

            toggle_row = gui.Horiz(int(0.3 * em))
            btn_show = gui.Button("Show all"); btn_show.set_on_clicked(self._on_show_all)
            btn_hide = gui.Button("Hide all"); btn_hide.set_on_clicked(self._on_hide_all)
            toggle_row.add_child(btn_show); toggle_row.add_child(btn_hide)
            panel.add_child(toggle_row)

            scroll = gui.ScrollableVert(
                int(0.3 * em), gui.Margins(0, 0, int(0.3 * em), 0)
            )
            for cls_id in self._class_ids:
                name = self.semantic_map.get(cls_id, str(cls_id))
                rgb  = self.color_map.get(cls_id, [128, 128, 128])
                tile = np.full((14, 14, 3), rgb, dtype=np.uint8)

                cb = gui.Checkbox(f"{cls_id}: {name}")
                cb.checked = True
                cb.set_on_checked(
                    lambda checked, cid=cls_id: self._on_class_toggle(cid, checked)
                )
                self._checkboxes[cls_id] = cb

                row = gui.Horiz(int(0.2 * em))
                row.add_child(gui.ImageWidget(o3d.geometry.Image(tile)))
                row.add_child(cb)
                scroll.add_child(row)

            panel.add_child(scroll)

        w.add_child(panel)
        self._panel = panel
        w.set_on_layout(self._on_layout)

        self._refresh()

    @staticmethod
    def _section_label(text: str, em: float) -> gui.Label:
        lbl = gui.Label(text.upper())
        lbl.text_color = gui.Color(0.5, 0.8, 1.0)
        return lbl

    def _on_layout(self, _ctx) -> None:
        r     = self._window.content_rect
        self._panel.frame = gui.Rect(r.x, r.y, PANEL_W, r.height)

        n          = len(self._scenes)
        multi      = n > 1
        vp_total_w = r.width - PANEL_W
        vp_w       = vp_total_w / n
        lh         = BANNER_H if multi else 0

        for i, scene in enumerate(self._scenes):
            x = r.x + PANEL_W + i * vp_w
            if multi:
                self._banner_labels[i].frame = gui.Rect(x, r.y, vp_w, lh)
            scene.frame = gui.Rect(x, r.y + lh, vp_w, r.height - lh)

    # ------------------------------------------------------------------
    # Navigation event handlers
    # ------------------------------------------------------------------

    def _on_next(self) -> None:
        self.current_idx = (self.current_idx + 1) % len(self.dataset)
        self._refresh()

    def _on_prev(self) -> None:
        self.current_idx = (self.current_idx - 1) % len(self.dataset)
        self._refresh()

    def _on_mode_changed(self, _text: str, idx: int) -> None:
        self._display_mode = idx
        for i, (pts, labels) in enumerate(zip(self._cached_pts, self._cached_labels)):
            if pts is not None:
                self._update_cloud(i, pts, labels)

    def _on_traj_toggle(self, checked: bool) -> None:
        self._show_traj = checked
        if checked:
            self._update_trajectory()
        else:
            self._scenes[0].scene.remove_geometry("traj_past")
            self._scenes[0].scene.remove_geometry("traj_future")

    def _on_class_toggle(self, cls_id: int, checked: bool) -> None:
        if checked:
            self.active_classes.add(cls_id)
        else:
            self.active_classes.discard(cls_id)
        self._recolor_all()

    def _on_show_all(self) -> None:
        self.active_classes = set(self._class_ids)
        for cb in self._checkboxes.values():
            cb.checked = True
        self._recolor_all()

    def _on_hide_all(self) -> None:
        self.active_classes = set()
        for cb in self._checkboxes.values():
            cb.checked = False
        self._recolor_all()

    def _recolor_all(self) -> None:
        for i, (pts, labels) in enumerate(zip(self._cached_pts, self._cached_labels)):
            if pts is not None:
                self._update_cloud(i, pts, labels)

    # Resolve key-down enum once, handling Open3D API differences across versions
    _KEY_DOWN = getattr(gui.KeyEvent, "DOWN", None) or gui.KeyEvent.Type.DOWN

    def _on_key(self, event) -> int:
        if event.type == self._KEY_DOWN:
            k = event.key
            if k in (gui.KeyName.RIGHT, ord("l"), ord("L")):
                self._on_next(); return gui.Widget.EventCallbackResult.HANDLED
            if k in (gui.KeyName.LEFT, ord("h"), ord("H")):
                self._on_prev(); return gui.Widget.EventCallbackResult.HANDLED
            if k in (ord("r"), ord("R")):
                self._reset_camera(); return gui.Widget.EventCallbackResult.HANDLED
            if k in (ord("b"), ord("B")):
                self._look_bev(); return gui.Widget.EventCallbackResult.HANDLED
            if k in (ord("t"), ord("T")):
                self._cycle_mode(); return gui.Widget.EventCallbackResult.HANDLED
            if k in (ord("j"), ord("J")) and self._cb_traj is not None:
                self._cb_traj.checked = not self._cb_traj.checked
                self._on_traj_toggle(self._cb_traj.checked)
                return gui.Widget.EventCallbackResult.HANDLED
        return gui.Widget.EventCallbackResult.IGNORED

    def _cycle_mode(self) -> None:
        self._display_mode = (self._display_mode + 1) % len(DISPLAY_MODES)
        if self._mode_combo is not None:
            self._mode_combo.selected_index = self._display_mode
        self._recolor_all()

    # ------------------------------------------------------------------
    # Data loading & parallel pipeline execution
    # ------------------------------------------------------------------

    def _refresh(self) -> None:
        sample = self.dataset[self.current_idx]

        pts_raw = sample.data.get(self.cfg.point_key)
        if pts_raw is None:
            return
        pts = _to_numpy(pts_raw).astype(np.float32)

        labels: np.ndarray | None = None
        if self.cfg.label_key:
            lbl_raw = sample.data.get(self.cfg.label_key)
            if lbl_raw is not None:
                labels = _to_numpy(lbl_raw).astype(np.int64)

        self._lbl_frame.text = f"{self.current_idx + 1} / {len(self.dataset)}"
        self._lbl_npts.text  = f"Points: {len(pts):,}"

        for i, lbl in enumerate(self._timing_labels):
            lbl.text = f"{self._pipelines[i].name}: …"

        # Bump ID so any in-flight callbacks from the previous frame are discarded
        self._refresh_id += 1
        current_id = self._refresh_id

        def _run(i: int) -> None:
            t0 = time.perf_counter()
            p_pts, p_labels = self._pipelines[i].run(
                pts.copy(),
                labels.copy() if labels is not None else None,
            )
            elapsed_ms = (time.perf_counter() - t0) * 1000
            gui.Application.instance.post_to_main_thread(
                self._window,
                lambda i=i, p=p_pts, l=p_labels, t=elapsed_ms:
                    self._on_pipeline_done(current_id, i, p, l, t),
            )

        for i in range(len(self._pipelines)):
            self._executor.submit(_run, i)

    def _on_pipeline_done(
        self,
        refresh_id: int,
        i: int,
        pts: np.ndarray,
        labels: np.ndarray | None,
        elapsed_ms: float,
    ) -> None:
        if refresh_id != self._refresh_id:
            return  # user navigated away; discard stale result

        self._cached_pts[i]    = pts
        self._cached_labels[i] = labels

        if i == 0 and labels is not None and self._lbl_stats is not None:
            self._lbl_stats.text = self._build_stats_text(labels)

        if i < len(self._timing_labels):
            self._timing_labels[i].text = f"{self._pipelines[i].name}: {elapsed_ms:.0f} ms"

        self._update_cloud(i, pts, labels)

        if self._show_traj and self._poses is not None and i == 0:
            self._update_trajectory()

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def _update_cloud(self, i: int, pts: np.ndarray, labels: np.ndarray | None) -> None:
        scene = self._scenes[i].scene

        if self._camera_initialized[i]:
            scene.remove_geometry("cloud")

        xyz  = pts[:, :3]
        mask = np.ones(len(xyz), dtype=bool)

        if self._display_mode == 0 and labels is not None:
            mask = np.isin(labels, list(self.active_classes))

        xyz_f = xyz[mask].astype(np.float64)
        if len(xyz_f) == 0:
            return

        colors = self._compute_colors(pts[mask], labels[mask] if labels is not None else None)

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(xyz_f)
        pcd.colors = o3d.utility.Vector3dVector(colors)
        scene.add_geometry("cloud", pcd, self._mats[i])

        if not self._camera_initialized[i]:
            self._setup_camera(i)
            self._camera_initialized[i] = True

    def _compute_colors(self, pts: np.ndarray, labels: np.ndarray | None) -> np.ndarray:
        mode = self._display_mode

        if mode == 0 and labels is not None:
            return labels_to_colors(labels, self.color_map)

        if mode == 1 and pts.shape[1] > self.cfg.intensity_channel:
            return intensity_colors(pts[:, self.cfg.intensity_channel])

        return height_colors(pts[:, 2])

    def _update_trajectory(self) -> None:
        if self._poses is None:
            return
        scene = self._scenes[0].scene
        scene.remove_geometry("traj_past")
        scene.remove_geometry("traj_future")

        pose_cur = self._poses[self.current_idx]
        T_inv    = np.linalg.inv(pose_cur)

        def _world_to_local(poses_slice):
            origins = np.array([p[:3, 3] for p in poses_slice])
            h       = np.hstack([origins, np.ones((len(origins), 1))])
            return (T_inv @ h.T).T[:, :3]

        mat_line = rendering.MaterialRecord()
        mat_line.shader     = "unlitLine"
        mat_line.line_width = 2.0

        if self.current_idx > 0:
            past_pts = _world_to_local(self._poses[: self.current_idx + 1])
            edges    = [[j, j + 1] for j in range(len(past_pts) - 1)]
            if edges:
                scene.add_geometry(
                    "traj_past", _make_lineset(past_pts, edges, C_TRAJ_PAST), mat_line
                )

        if self.current_idx < len(self._poses) - 1:
            future_pts = _world_to_local(self._poses[self.current_idx:])
            edges      = [[j, j + 1] for j in range(len(future_pts) - 1)]
            if edges:
                scene.add_geometry(
                    "traj_future", _make_lineset(future_pts, edges, C_TRAJ_FUTURE), mat_line
                )

    # ------------------------------------------------------------------
    # Camera
    # ------------------------------------------------------------------

    def _setup_camera(self, i: int) -> None:
        pts = self._cached_pts[i]
        if pts is None:
            return
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts[:, :3].astype(np.float64))
        bounds = pcd.get_axis_aligned_bounding_box()
        self._scenes[i].setup_camera(60, bounds, bounds.get_center())

    def _reset_camera(self) -> None:
        for i in range(len(self._scenes)):
            self._setup_camera(i)

    def _sync_cameras(self) -> None:
        """Reset all viewports to the bounding box of pipeline 0."""
        if self._cached_pts[0] is None:
            return
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(
            self._cached_pts[0][:, :3].astype(np.float64)
        )
        bounds = pcd.get_axis_aligned_bounding_box()
        for scene in self._scenes:
            scene.setup_camera(60, bounds, bounds.get_center())

    def _look_bev(self) -> None:
        pts_list = [p for p in self._cached_pts if p is not None]
        if not pts_list:
            return
        z_max  = max(float(np.max(p[:, 2])) for p in pts_list)
        height = max(z_max + 15.0, 20.0)
        for scene in self._scenes:
            scene.scene.camera.look_at(
                [0.0, 0.0, 0.0],
                [0.0, 0.0, height],
                [1.0, 0.0, 0.0],
            )

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def _build_stats_text(self, labels: np.ndarray) -> str:
        total  = max(len(labels), 1)
        unique, counts = np.unique(labels, return_counts=True)
        order  = np.argsort(-counts)
        lines  = []
        for cls_id, cnt in zip(unique[order][:12], counts[order][:12]):
            name = self.semantic_map.get(int(cls_id), str(cls_id))
            pct  = 100.0 * cnt / total
            bar  = "█" * int(pct / 5)
            lines.append(f"{name[:14]:<14} {pct:5.1f}% {bar}")
        return "\n".join(lines)

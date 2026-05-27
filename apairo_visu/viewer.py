"""Interactive 3-D LiDAR viewer for apairo datasets.

Usage (programmatic):
    from apairo_visu import LidarViewer, ViewConfig, load_label_config
    cfg = load_label_config("goose")
    LidarViewer.launch(dataset, label_cfg=cfg)

Usage (CLI):
    python -m apairo_visu --dataset goose --root /path/to/goose --split train

Keyboard shortcuts:
    Right / L   next frame
    Left  / H   previous frame
    R           reset camera
    B           bird's-eye (top-down) view
    T           cycle colour mode  (Semantic → Intensity → Height)
    J           toggle trajectory overlay
"""

from __future__ import annotations

from dataclasses import dataclass, field

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

C_TRAJ_PAST = [0.20, 0.60, 1.00]   # blue
C_TRAJ_FUTURE = [1.00, 0.60, 0.10]  # orange


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class ViewConfig:
    """Specifies which keys in sample.data to use for visualisation."""

    point_key: str = "lidar"
    label_key: str | None = "labels"
    intensity_channel: int = 3


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _to_numpy(t) -> np.ndarray:
    if hasattr(t, "numpy"):
        return t.numpy()
    return np.asarray(t)


def _make_lineset(pts: np.ndarray, edges: list, color: list) -> o3d.geometry.LineSet:
    ls = o3d.geometry.LineSet()
    ls.points = o3d.utility.Vector3dVector(pts.astype(np.float64))
    ls.lines = o3d.utility.Vector2iVector(edges)
    ls.colors = o3d.utility.Vector3dVector(np.tile(color, (len(edges), 1)))
    return ls


# ---------------------------------------------------------------------------
# Viewer
# ---------------------------------------------------------------------------


class LidarViewer:
    """Interactive 3-D LiDAR viewer for any apairo AbstractDataset.

    Supports semantic label colouring (with a label config dict), intensity
    grayscale, height viridis, per-class filtering, and optional trajectory
    overlay from a list of 4×4 pose matrices.

    Args:
        dataset:   Any apairo synchronous dataset (SynchronousDataset subclass).
        view_cfg:  Which keys to extract from each Sample.  Defaults to
                   ``point_key="lidar"`` and ``label_key="labels"``.
        label_cfg: Dict with ``color_map``, ``semantic_map``, and optionally
                   ``traversable_map``.  Use ``load_label_config(name)`` for
                   built-in configs.  Pass ``None`` to skip label colouring.
        poses:     Optional list of 4×4 numpy pose matrices (T_world_sensor),
                   one per frame, enabling a trajectory overlay.
        start_idx: First frame to display.
    """

    def __init__(
        self,
        dataset,
        view_cfg: ViewConfig | None = None,
        label_cfg: dict | None = None,
        poses: list[np.ndarray] | None = None,
        start_idx: int = 0,
    ) -> None:
        self.dataset = dataset
        self.cfg = view_cfg or ViewConfig()
        self.current_idx = start_idx
        self._poses = poses

        # Colour / label setup
        if label_cfg is not None:
            self.color_map = normalize_color_map(label_cfg["color_map"])
            self.semantic_map = {
                int(k): v for k, v in label_cfg.get("semantic_map", {}).items()
            }
        else:
            n = 32
            self.color_map = auto_color_map(n)
            self.semantic_map = {i: str(i) for i in range(n)}

        self._trav_ids: set[int] = set()
        if label_cfg:
            self._trav_ids = {int(i) for i in label_cfg.get("traversable_map", [])}

        self._class_ids = sorted(self.semantic_map.keys())
        self.active_classes: set[int] = set(self._class_ids)
        self._has_labels = self.cfg.label_key is not None
        # Default mode: Semantic if labels expected, Height otherwise
        self._display_mode: int = 0 if self._has_labels else 2

        # Cache
        self._cached_pts: np.ndarray | None = None
        self._cached_labels: np.ndarray | None = None

        # GUI refs
        self._window = None
        self._scene: gui.SceneWidget | None = None
        self._panel = None
        self._lbl_frame: gui.Label | None = None
        self._lbl_npts: gui.Label | None = None
        self._lbl_stats: gui.Label | None = None
        self._checkboxes: dict[int, gui.Checkbox] = {}
        self._mat: rendering.MaterialRecord | None = None
        self._mode_combo: gui.Combobox | None = None
        self._cb_traj: gui.Checkbox | None = None
        self._camera_initialized: bool = False
        self._show_traj: bool = False

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
    ) -> None:
        """Create the application and block until the window is closed."""
        app = gui.Application.instance
        app.initialize()
        viewer = LidarViewer(dataset, view_cfg, label_cfg, poses, start_idx)
        viewer._build_window()
        app.run()

    # ------------------------------------------------------------------
    # Window construction
    # ------------------------------------------------------------------

    def _build_window(self) -> None:
        app = gui.Application.instance
        w = app.create_window("apairo — LiDAR Viewer", 1500, 900)
        self._window = w
        em = w.theme.font_size

        # Material
        mat = rendering.MaterialRecord()
        mat.shader = "defaultUnlit"
        mat.point_size = POINT_SIZE
        self._mat = mat

        # 3-D scene
        self._scene = gui.SceneWidget()
        self._scene.scene = rendering.Open3DScene(w.renderer)
        self._scene.scene.set_background([0.08, 0.08, 0.08, 1.0])
        self._scene.set_on_key(self._on_key)

        # Left panel
        panel = gui.Vert(int(0.4 * em), gui.Margins(int(0.6 * em)))

        # — Navigation —
        panel.add_child(self._section_label("Navigation", em))

        self._lbl_frame = gui.Label("— / —")
        self._lbl_frame.text_color = gui.Color(0.85, 0.85, 0.85)
        panel.add_child(self._lbl_frame)

        self._lbl_npts = gui.Label("Points: —")
        self._lbl_npts.text_color = gui.Color(0.6, 0.6, 0.6)
        panel.add_child(self._lbl_npts)

        nav_row = gui.Horiz(int(0.3 * em))
        btn_prev = gui.Button("< Prev")
        btn_prev.set_on_clicked(self._on_prev)
        btn_next = gui.Button("Next >")
        btn_next.set_on_clicked(self._on_next)
        nav_row.add_stretch()
        nav_row.add_child(btn_prev)
        nav_row.add_child(btn_next)
        nav_row.add_stretch()
        panel.add_child(nav_row)

        cam_row = gui.Horiz(int(0.3 * em))
        btn_bev = gui.Button("BEV  [B]")
        btn_bev.set_on_clicked(self._look_bev)
        btn_reset = gui.Button("Reset  [R]")
        btn_reset.set_on_clicked(self._reset_camera)
        cam_row.add_stretch()
        cam_row.add_child(btn_bev)
        cam_row.add_child(btn_reset)
        cam_row.add_stretch()
        panel.add_child(cam_row)

        panel.add_child(gui.Label(""))

        # — Colour mode —
        panel.add_child(self._section_label("Colour mode  [T]", em))
        combo = gui.Combobox()
        for m in DISPLAY_MODES:
            combo.add_item(m)
        combo.selected_index = self._display_mode
        combo.set_on_selection_changed(self._on_mode_changed)
        self._mode_combo = combo
        panel.add_child(combo)

        panel.add_child(gui.Label(""))

        # — Trajectory (only when poses provided) —
        if self._poses is not None:
            panel.add_child(self._section_label("Overlays", em))
            cb_traj = gui.Checkbox("Trajectory  [J]")
            cb_traj.checked = False
            cb_traj.set_on_checked(self._on_traj_toggle)
            self._cb_traj = cb_traj
            panel.add_child(cb_traj)
            panel.add_child(gui.Label(""))

        # — Stats —
        if self._has_labels:
            panel.add_child(self._section_label("Class distribution", em))
            self._lbl_stats = gui.Label("—")
            self._lbl_stats.text_color = gui.Color(0.75, 0.75, 0.75)
            panel.add_child(self._lbl_stats)
            panel.add_child(gui.Label(""))

            # — Class filter —
            panel.add_child(self._section_label("Filter classes", em))

            toggle_row = gui.Horiz(int(0.3 * em))
            btn_show = gui.Button("Show all")
            btn_show.set_on_clicked(self._on_show_all)
            btn_hide = gui.Button("Hide all")
            btn_hide.set_on_clicked(self._on_hide_all)
            toggle_row.add_child(btn_show)
            toggle_row.add_child(btn_hide)
            panel.add_child(toggle_row)

            scroll = gui.ScrollableVert(
                int(0.3 * em), gui.Margins(0, 0, int(0.3 * em), 0)
            )
            for cls_id in self._class_ids:
                name = self.semantic_map.get(cls_id, str(cls_id))
                rgb = self.color_map.get(cls_id, [128, 128, 128])
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

        # Layout
        w.add_child(self._scene)
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
        r = self._window.content_rect
        self._panel.frame = gui.Rect(r.x, r.y, PANEL_W, r.height)
        self._scene.frame = gui.Rect(
            r.x + PANEL_W, r.y, r.width - PANEL_W, r.height
        )

    # ------------------------------------------------------------------
    # Event handlers
    # ------------------------------------------------------------------

    def _on_next(self) -> None:
        self.current_idx = (self.current_idx + 1) % len(self.dataset)
        self._refresh()

    def _on_prev(self) -> None:
        self.current_idx = (self.current_idx - 1) % len(self.dataset)
        self._refresh()

    def _on_mode_changed(self, _text: str, idx: int) -> None:
        self._display_mode = idx
        if self._cached_pts is not None:
            self._update_cloud(self._cached_pts, self._cached_labels)

    def _on_traj_toggle(self, checked: bool) -> None:
        self._show_traj = checked
        if checked:
            self._update_trajectory()
        else:
            self._scene.scene.remove_geometry("traj_past")
            self._scene.scene.remove_geometry("traj_future")

    def _on_class_toggle(self, cls_id: int, checked: bool) -> None:
        if checked:
            self.active_classes.add(cls_id)
        else:
            self.active_classes.discard(cls_id)
        if self._cached_pts is not None:
            self._update_cloud(self._cached_pts, self._cached_labels)

    def _on_show_all(self) -> None:
        self.active_classes = set(self._class_ids)
        for cb in self._checkboxes.values():
            cb.checked = True
        if self._cached_pts is not None:
            self._update_cloud(self._cached_pts, self._cached_labels)

    def _on_hide_all(self) -> None:
        self.active_classes = set()
        for cb in self._checkboxes.values():
            cb.checked = False
        if self._cached_pts is not None:
            self._update_cloud(self._cached_pts, self._cached_labels)

    # Resolve key-down enum once, handling Open3D API differences across versions
    _KEY_DOWN = getattr(gui.KeyEvent, "DOWN", None) or gui.KeyEvent.Type.DOWN

    def _on_key(self, event) -> int:
        if event.type == self._KEY_DOWN:
            k = event.key
            if k in (gui.KeyName.RIGHT, ord("l"), ord("L")):
                self._on_next()
                return gui.Widget.EventCallbackResult.HANDLED
            if k in (gui.KeyName.LEFT, ord("h"), ord("H")):
                self._on_prev()
                return gui.Widget.EventCallbackResult.HANDLED
            if k in (ord("r"), ord("R")):
                self._reset_camera()
                return gui.Widget.EventCallbackResult.HANDLED
            if k in (ord("b"), ord("B")):
                self._look_bev()
                return gui.Widget.EventCallbackResult.HANDLED
            if k in (ord("t"), ord("T")):
                self._cycle_mode()
                return gui.Widget.EventCallbackResult.HANDLED
            if k in (ord("j"), ord("J")) and self._cb_traj is not None:
                self._cb_traj.checked = not self._cb_traj.checked
                self._on_traj_toggle(self._cb_traj.checked)
                return gui.Widget.EventCallbackResult.HANDLED
        return gui.Widget.EventCallbackResult.IGNORED

    def _cycle_mode(self) -> None:
        self._display_mode = (self._display_mode + 1) % len(DISPLAY_MODES)
        if self._mode_combo is not None:
            self._mode_combo.selected_index = self._display_mode
        if self._cached_pts is not None:
            self._update_cloud(self._cached_pts, self._cached_labels)

    # ------------------------------------------------------------------
    # Data loading & rendering
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

        self._cached_pts = pts
        self._cached_labels = labels

        self._lbl_frame.text = f"{self.current_idx + 1} / {len(self.dataset)}"
        self._lbl_npts.text = f"Points: {len(pts):,}"
        if labels is not None and self._lbl_stats is not None:
            self._lbl_stats.text = self._build_stats_text(labels)

        self._update_cloud(pts, labels)
        if self._show_traj and self._poses is not None:
            self._update_trajectory()

    def _update_cloud(
        self, pts: np.ndarray, labels: np.ndarray | None
    ) -> None:
        scene = self._scene.scene

        if self._camera_initialized:
            scene.remove_geometry("cloud")

        xyz = pts[:, :3]
        mask = np.ones(len(xyz), dtype=bool)

        # Apply class filter when in semantic mode
        if self._display_mode == 0 and labels is not None:
            mask = np.isin(labels, list(self.active_classes))

        xyz_f = xyz[mask].astype(np.float64)
        if len(xyz_f) == 0:
            return

        colors = self._compute_colors(pts[mask], labels[mask] if labels is not None else None)

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(xyz_f)
        pcd.colors = o3d.utility.Vector3dVector(colors)
        scene.add_geometry("cloud", pcd, self._mat)

        if not self._camera_initialized:
            self._reset_camera()
            self._camera_initialized = True

    def _compute_colors(
        self, pts: np.ndarray, labels: np.ndarray | None
    ) -> np.ndarray:
        mode = self._display_mode

        if mode == 0 and labels is not None:
            return labels_to_colors(labels, self.color_map)

        if mode == 1 and pts.shape[1] > self.cfg.intensity_channel:
            return intensity_colors(pts[:, self.cfg.intensity_channel])

        # mode == 2 (Height) or fallback
        return height_colors(pts[:, 2])

    def _update_trajectory(self) -> None:
        if self._poses is None:
            return
        scene = self._scene.scene
        scene.remove_geometry("traj_past")
        scene.remove_geometry("traj_future")

        pose_cur = self._poses[self.current_idx]
        T_inv = np.linalg.inv(pose_cur)

        def _world_to_local(poses_slice):
            origins = np.array([p[:3, 3] for p in poses_slice])
            ones = np.ones((len(origins), 1))
            h = np.hstack([origins, ones])
            return (T_inv @ h.T).T[:, :3]

        mat_line = rendering.MaterialRecord()
        mat_line.shader = "unlitLine"
        mat_line.line_width = 2.0

        if self.current_idx > 0:
            past_pts = _world_to_local(self._poses[: self.current_idx + 1])
            edges = [[i, i + 1] for i in range(len(past_pts) - 1)]
            if edges:
                scene.add_geometry(
                    "traj_past",
                    _make_lineset(past_pts, edges, C_TRAJ_PAST),
                    mat_line,
                )

        if self.current_idx < len(self._poses) - 1:
            future_pts = _world_to_local(self._poses[self.current_idx :])
            edges = [[i, i + 1] for i in range(len(future_pts) - 1)]
            if edges:
                scene.add_geometry(
                    "traj_future",
                    _make_lineset(future_pts, edges, C_TRAJ_FUTURE),
                    mat_line,
                )

    # ------------------------------------------------------------------
    # Camera
    # ------------------------------------------------------------------

    def _reset_camera(self) -> None:
        if self._cached_pts is None:
            return
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(
            self._cached_pts[:, :3].astype(np.float64)
        )
        bounds = pcd.get_axis_aligned_bounding_box()
        self._scene.setup_camera(60, bounds, bounds.get_center())

    def _look_bev(self) -> None:
        """Top-down (bird's-eye) view. LiDAR convention: +X forward, +Z up."""
        if self._cached_pts is None:
            return
        z_max = float(np.max(self._cached_pts[:, 2]))
        height = max(z_max + 15.0, 20.0)
        self._scene.scene.camera.look_at(
            [0.0, 0.0, 0.0],
            [0.0, 0.0, height],
            [1.0, 0.0, 0.0],
        )

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def _build_stats_text(self, labels: np.ndarray) -> str:
        total = max(len(labels), 1)
        unique, counts = np.unique(labels, return_counts=True)
        order = np.argsort(-counts)
        lines = []
        for cls_id, cnt in zip(unique[order][:12], counts[order][:12]):
            name = self.semantic_map.get(int(cls_id), str(cls_id))
            pct = 100.0 * cnt / total
            bar = "█" * int(pct / 5)
            lines.append(f"{name[:14]:<14} {pct:5.1f}% {bar}")
        return "\n".join(lines)

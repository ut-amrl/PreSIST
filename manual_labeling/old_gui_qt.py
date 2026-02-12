# manual_labeling/gui_qt.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtWidgets import (
    QApplication,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSplitter,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from .io import index_by_id, load_json_labels, merge_defaults_with_existing, save_json_labels
from .schema import build_default_labels_for_segment_ids, schema_fields_for_gui

from instance_segmentation.abs_instance_segmenter import SegmentationResults


# ----------------------- Utilities -----------------------

def pil_to_qpixmap(pil_img: Image.Image) -> QPixmap:
    rgb = pil_img.convert("RGB")
    arr = np.array(rgb)
    h, w, _ = arr.shape
    qimg = QImage(arr.data, w, h, 3 * w, QImage.Format.Format_RGB888)
    return QPixmap.fromImage(qimg.copy())


def seg_map_from_results(seg_results: SegmentationResults) -> np.ndarray:
    """
    Build HxW int64 map: -1 background, else instance-id at pixel.
    Overlaps resolved by "last wins" (later masks overwrite earlier).
    """
    ids = seg_results.ids.detach().cpu().numpy().astype(np.int64)      # (N,)
    masks = seg_results.masks.detach().cpu().numpy().astype(bool)      # (N,H,W)

    if masks.ndim != 3:
        raise ValueError(f"Expected masks shape (N,H,W), got {masks.shape}")

    n, h, w = masks.shape
    seg_map = np.full((h, w), -1, dtype=np.int64)
    for i in range(n):
        seg_map[masks[i]] = int(ids[i])
    return seg_map


def seg_id_at(seg_map: np.ndarray, x: int, y: int) -> int:
    h, w = seg_map.shape[:2]
    if 0 <= x < w and 0 <= y < h:
        return int(seg_map[y, x])
    return -1


def make_single_mask_overlay(seg_map: np.ndarray, seg_id: int, alpha: float = 0.55) -> Image.Image:
    """
    RGBA overlay showing ONLY seg_id. Everything else transparent.
    Color deterministically derived from seg_id for consistency.
    """
    h, w = seg_map.shape
    mask = (seg_map == int(seg_id))

    r = int((seg_id * 37) % 255)
    g = int((seg_id * 57) % 255)
    b = int((seg_id * 97) % 255)

    overlay = np.zeros((h, w, 4), dtype=np.uint8)
    overlay[..., 0] = r
    overlay[..., 1] = g
    overlay[..., 2] = b
    overlay[..., 3] = 0
    overlay[mask, 3] = int(np.clip(alpha, 0.0, 1.0) * 255)

    return Image.fromarray(overlay, mode="RGBA")


def mask_bbox_xyxy(seg_map: np.ndarray, seg_id: int) -> Optional[Tuple[int, int, int, int]]:
    """
    Returns bounding box (x0, y0, x1, y1) inclusive pixel coords for seg_id, or None if empty.
    """
    ys, xs = np.where(seg_map == int(seg_id))
    if xs.size == 0 or ys.size == 0:
        return None
    x0 = int(xs.min())
    x1 = int(xs.max())
    y0 = int(ys.min())
    y1 = int(ys.max())
    return x0, y0, x1, y1


def draw_bounding_box(
    pil_img: Image.Image,
    bbox_xyxy: Tuple[int, int, int, int],
    color: Tuple[int, int, int] = (255, 255, 0),  # bright yellow
    width: int = 4,
    pad: int = 2,
    with_black_outline: bool = True,
) -> Image.Image:
    """
    Draw a bright rectangle on top of pil_img and return it.
    Optionally draws a black outer outline to increase contrast.
    """
    x0, y0, x1, y1 = bbox_xyxy
    x0 = max(0, x0 - pad)
    y0 = max(0, y0 - pad)
    x1 = min(pil_img.width - 1, x1 + pad)
    y1 = min(pil_img.height - 1, y1 + pad)

    out = pil_img.copy()
    draw = ImageDraw.Draw(out)

    if with_black_outline:
        for i in range(width + 2):
            draw.rectangle([x0 - i, y0 - i, x1 + i, y1 + i], outline=(0, 0, 0))

    for i in range(width):
        draw.rectangle([x0 - i, y0 - i, x1 + i, y1 + i], outline=color)

    return out


# ----------------------- Paths -----------------------

@dataclass
class LabelingPaths:
    output_root: Path
    segmentation_config_name: str
    labeling_tag: str

    def seg_npz_path(self, image_id: str) -> Path:
        return (
            self.output_root
            / "segmentation_results"
            / self.segmentation_config_name
            / image_id
            / "segmentation_results.npz"
        )

    def label_json_path(self, image_id: str) -> Path:
        return self.output_root / "full_results" / self.labeling_tag / f"{image_id}.json"


# ----------------------- Widgets -----------------------

class ClickableImageLabel(QLabel):
    """
    QLabel that forwards left-clicks to a callback as image-space (x,y).
    Note: We set the label size to the pixmap size to avoid scaling issues.
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self._click_cb = None
        self._pixmap: Optional[QPixmap] = None

    def setPixmap(self, pm: QPixmap) -> None:  # type: ignore[override]
        self._pixmap = pm
        super().setPixmap(pm)

    def set_click_callback(self, cb) -> None:
        self._click_cb = cb

    def mousePressEvent(self, event):  # noqa: N802
        if self._pixmap is None or self._click_cb is None:
            return
        if event.button() != Qt.MouseButton.LeftButton:
            return
        x = int(event.position().x())
        y = int(event.position().y())
        self._click_cb(x, y)


# ----------------------- Main Window -----------------------

class LabelingWindow(QMainWindow):
    """
    Shortcuts:
      - Next mask:    J / Down / ] / N
      - Prev mask:    K / Up   / [ / P
      - Next image:   D / Right
      - Prev image:   A / Left
      - Save:         S
      - Toggle overlay: O
    """

    def __init__(
        self,
        image_source,
        paths: LabelingPaths,
        key_map: Optional[Dict[str, str]] = None,
        id_key: str = "id",
        autosave_on_nav: bool = True,
        overlay_alpha: float = 0.55,
    ):
        super().__init__()
        self.setWindowTitle("Segmentation Labeling GUI")

        self.image_source = image_source
        self.paths = paths
        self.key_map = key_map or {}
        self.id_key = id_key
        self.autosave_on_nav = autosave_on_nav
        self.overlay_alpha = overlay_alpha

        self.idx = 0

        # Current image state
        self.current_image_id: Optional[str] = None
        self.current_image_pil: Optional[Image.Image] = None

        # Segmentation state
        self.seg_results: Optional[SegmentationResults] = None
        self._seg_map_cache: Optional[np.ndarray] = None
        self.instance_ids: List[int] = []

        # Labels
        self.current_labels: List[Dict[str, Any]] = []
        self.current_label_map: Dict[str, Dict[str, Any]] = {}
        self.selected_seg_id: Optional[int] = None

        self.show_overlay: bool = True

        # Schema-driven form
        self._form_specs: List[Dict[str, Any]] = schema_fields_for_gui()
        self._widgets_by_field: Dict[str, Any] = {}

        self._required_label_fields: List[str] = self._compute_required_fields()

        self._build_ui()

        # Load first valid image
        next_index = 0
        loaded = False
        while next_index < len(self.image_source):
            try:
                self._load_index(next_index)
                loaded = True
                break
            except Exception as e:
                print(f"Error loading index {next_index}: {e}")
                next_index += 1

        if (not loaded) and len(self.image_source) > 0:
            QMessageBox.critical(self, "Error", "Failed to load any images. Check console for details.")

    # ---------- Schema helpers ----------

    def _compute_required_fields(self) -> List[str]:
        """
        What counts as "labeled":
          - All schema fields with kind in ('float', 'choice'), excluding id.
          - 'text' fields (notes) do not count.
        """
        req: List[str] = []
        for spec in self._form_specs:
            name = spec.get("name")
            kind = spec.get("kind")
            if name == self.id_key:
                continue
            if kind in ("float", "choice"):
                req.append(name)
        return req

    def _is_record_labeled(self, rec: Dict[str, Any]) -> bool:
        for k in self._required_label_fields:
            if k not in rec or rec[k] is None:
                return False
        return True

    def _missing_label_seg_ids(self) -> List[int]:
        missing: List[int] = []
        for sid in self.instance_ids:
            rec = self.current_label_map.get(str(sid))
            if rec is None or (not self._is_record_labeled(rec)):
                missing.append(int(sid))
        return missing

    def _filtered_labeled_records(self) -> List[Dict[str, Any]]:
        return [rec for rec in self.current_labels if self._is_record_labeled(rec)]

    # ---------- UI ----------

    def _build_ui(self) -> None:
        root = QWidget()
        outer = QHBoxLayout(root)

        splitter = QSplitter(Qt.Orientation.Horizontal)

        # Left: info + list + image
        left = QWidget()
        left_layout = QVBoxLayout(left)

        self.info = QLabel("")
        left_layout.addWidget(self.info)

        self.segment_list = QListWidget()
        self.segment_list.currentItemChanged.connect(self._on_segment_list_changed)
        left_layout.addWidget(self.segment_list, stretch=1)

        self.image_label = ClickableImageLabel()
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)
        self.image_label.set_click_callback(self._on_image_click)
        left_layout.addWidget(self.image_label, stretch=3)

        # Right: form + mask nav + image nav + save
        right = QWidget()
        right_layout = QVBoxLayout(right)

        form = QFormLayout()
        self._build_schema_form(form)
        right_layout.addLayout(form)

        mask_row = QHBoxLayout()
        self.prev_mask_btn = QPushButton("Prev Mask (K / [)")
        self.next_mask_btn = QPushButton("Next Mask (J / ])")
        mask_row.addWidget(self.prev_mask_btn)
        mask_row.addWidget(self.next_mask_btn)
        right_layout.addLayout(mask_row)
        self.prev_mask_btn.clicked.connect(self.prev_mask)
        self.next_mask_btn.clicked.connect(self.next_mask)

        btn_row = QHBoxLayout()
        self.prev_btn = QPushButton("Prev Image (A)")
        self.next_btn = QPushButton("Next Image (D)")
        self.save_btn = QPushButton("Save (S)")
        btn_row.addWidget(self.prev_btn)
        btn_row.addWidget(self.next_btn)
        btn_row.addWidget(self.save_btn)
        right_layout.addLayout(btn_row)

        self.prev_btn.clicked.connect(self.prev_image)
        self.next_btn.clicked.connect(self.next_image)
        self.save_btn.clicked.connect(self.save_current)

        splitter.addWidget(left)
        splitter.addWidget(right)
        splitter.setStretchFactor(0, 4)
        splitter.setStretchFactor(1, 3)

        outer.addWidget(splitter)
        self.setCentralWidget(root)

    def _build_schema_form(self, form: QFormLayout) -> None:
        if not self._form_specs:
            raise RuntimeError("schema_fields_for_gui() returned no fields.")

        for spec in self._form_specs:
            name = spec["name"]
            kind = spec.get("kind")
            label = spec.get("label", name)

            if kind == "readonly":
                w = QLineEdit()
                w.setReadOnly(True)

            elif kind == "float":
                w = QDoubleSpinBox()
                w.setRange(float(spec.get("min", 0.0)), float(spec.get("max", 1e9)))
                w.setSingleStep(float(spec.get("step", 1.0)))
                w.setDecimals(int(spec.get("decimals", 2)))
                w.valueChanged.connect(self._write_back_selected)

            elif kind == "choice":
                w = QComboBox()
                choices = spec.get("choices")
                if not choices:
                    raise ValueError(f"choice field '{name}' missing 'choices'")
                w.addItems(list(choices))
                w.currentTextChanged.connect(self._write_back_selected)

            elif kind == "text":
                w = QTextEdit()
                w.textChanged.connect(self._write_back_selected)

            else:
                raise ValueError(f"Unknown ui kind '{kind}' for field '{name}'")

            self._widgets_by_field[name] = w
            form.addRow(label, w)

    # ---------- Loading ----------

    def _status_text(self) -> str:
        total = len(self.image_source)
        nseg = len(self.instance_ids)
        sid = self.selected_seg_id
        missing = len(self._missing_label_seg_ids()) if self.instance_ids else 0
        return f"Image {self.idx+1}/{total} | image_id={self.current_image_id} | masks={nseg} | selected={sid} | missing_labels={missing}"

    def _load_index(self, idx: int) -> None:
        if idx < 0 or idx >= len(self.image_source):
            return

        self.idx = idx
        item = self.image_source.get(idx)

        if not getattr(item, "image_id", None):
            raise ValueError("image_source.get(idx) must return an item with image_id")

        self.current_image_id = str(item.image_id)
        self.current_image_pil = item.image_pil.convert("RGB")
        self.selected_seg_id = None

        seg_path = self.paths.seg_npz_path(self.current_image_id)
        self.seg_results, _ = SegmentationResults.loadFromFile(str(seg_path))
        self._seg_map_cache = seg_map_from_results(self.seg_results)

        ids_np = self.seg_results.ids.detach().cpu().numpy().astype(np.int64)
        self.instance_ids = [int(x) for x in ids_np.tolist()]

        defaults = build_default_labels_for_segment_ids(self.instance_ids, key_map=self.key_map)
        json_path = self.paths.label_json_path(self.current_image_id)
        existing = load_json_labels(json_path)

        self.current_labels = merge_defaults_with_existing(defaults, existing, id_key=self.id_key)
        self.current_label_map = index_by_id(self.current_labels, id_key=self.id_key)

        self._populate_segment_list()

        if self.instance_ids:
            self._select_seg_id(self.instance_ids[0])
        else:
            self._render_image()
            self.info.setText(self._status_text())

    # ---------- Render (ONE mask + bbox) ----------

    def _render_image(self) -> None:
        if self.current_image_pil is None:
            return

        base = self.current_image_pil.convert("RGBA")
        composed = base

        if self.show_overlay and self._seg_map_cache is not None and self.selected_seg_id is not None:
            overlay = make_single_mask_overlay(self._seg_map_cache, int(self.selected_seg_id), alpha=self.overlay_alpha)
            composed = Image.alpha_composite(base, overlay)

            bbox = mask_bbox_xyxy(self._seg_map_cache, int(self.selected_seg_id))
            if bbox is not None:
                composed_rgb = composed.convert("RGB")
                composed_rgb = draw_bounding_box(
                    composed_rgb,
                    bbox,
                    color=(255, 255, 0),
                    width=4,
                    pad=2,
                    with_black_outline=True,
                )
                composed = composed_rgb.convert("RGBA")

        pm = pil_to_qpixmap(composed.convert("RGB"))
        self.image_label.setFixedSize(pm.width(), pm.height())
        self.image_label.setPixmap(pm)

    # ---------- Segment list ----------

    def _populate_segment_list(self) -> None:
        self.segment_list.blockSignals(True)
        self.segment_list.clear()
        for sid in self.instance_ids:
            self.segment_list.addItem(QListWidgetItem(str(sid)))
        self.segment_list.blockSignals(False)

    def _on_segment_list_changed(self, current: QListWidgetItem, previous: QListWidgetItem) -> None:
        if current is None:
            return
        try:
            sid = int(current.text())
        except Exception:
            return
        self._select_seg_id(sid)

    def _sync_list_selection(self, sid: int) -> None:
        self.segment_list.blockSignals(True)
        for i in range(self.segment_list.count()):
            if self.segment_list.item(i).text() == str(sid):
                self.segment_list.setCurrentRow(i)
                break
        self.segment_list.blockSignals(False)

    # ---------- Click select ----------

    def _on_image_click(self, x: int, y: int) -> None:
        if self._seg_map_cache is None:
            return
        sid = seg_id_at(self._seg_map_cache, x, y)
        if sid >= 0:
            self._select_seg_id(sid)

    # ---------- Form (schema-driven) ----------

    def _set_widget_value_blocked(self, widget: Any, kind: str, value: Any, spec: Dict[str, Any]) -> None:
        widget.blockSignals(True)
        try:
            if kind == "readonly":
                widget.setText("" if value is None else str(value))
            elif kind == "float":
                widget.setValue(0.0 if value is None else float(value))
            elif kind == "choice":
                choices = list(spec.get("choices", []))
                v = "" if value is None else str(value)
                widget.setCurrentText(v if (v in choices) else (choices[0] if choices else v))
            elif kind == "text":
                widget.setPlainText("" if value is None else str(value))
        finally:
            widget.blockSignals(False)

    def _get_widget_value(self, widget: Any, kind: str) -> Any:
        if kind == "readonly":
            return str(widget.text())
        if kind == "float":
            return float(widget.value())
        if kind == "choice":
            return str(widget.currentText())
        if kind == "text":
            return str(widget.toPlainText())
        raise ValueError(f"Unknown kind: {kind}")

    def _select_seg_id(self, sid: int) -> None:
        self.selected_seg_id = int(sid)
        self._sync_list_selection(int(sid))

        rec = self.current_label_map.get(str(sid))  # may be None (unlabeled)

        for spec in self._form_specs:
            name = spec["name"]
            kind = spec.get("kind")
            w = self._widgets_by_field[name]

            if name == self.id_key:
                self._set_widget_value_blocked(w, kind, sid, spec)
                continue

            value = None if rec is None else rec.get(name, None)
            self._set_widget_value_blocked(w, kind, value, spec)

        self._render_image()
        self.info.setText(self._status_text())

    def _write_back_selected(self) -> None:
        if self.selected_seg_id is None:
            return

        key = str(self.selected_seg_id)
        rec = self.current_label_map.get(key)
        if rec is None:
            rec = {self.id_key: int(self.selected_seg_id)}
            self.current_labels.append(rec)
            self.current_label_map[key] = rec

        for spec in self._form_specs:
            name = spec["name"]
            kind = spec.get("kind")

            if name == self.id_key:
                rec[self.id_key] = int(self.selected_seg_id)
                continue
            if kind == "readonly":
                continue

            w = self._widgets_by_field[name]
            rec[name] = self._get_widget_value(w, kind)

        self.info.setText(self._status_text())

    # ---------- Save ----------

    def _warn_if_missing_labels(self) -> bool:
        missing = self._missing_label_seg_ids()
        if not missing:
            return True

        preview = ", ".join(str(x) for x in missing[:30])
        suffix = "" if len(missing) <= 30 else f" ... (+{len(missing)-30} more)"
        msg = (
            f"This image has {len(missing)} mask(s) missing labels.\n\n"
            f"Missing mask ids:\n{preview}{suffix}\n\n"
            "Save anyway (only labeled entries will be saved)?"
        )

        res = QMessageBox.warning(
            self,
            "Missing labels",
            msg,
            QMessageBox.StandardButton.Ok | QMessageBox.StandardButton.Cancel,
            QMessageBox.StandardButton.Cancel,
        )
        return res == QMessageBox.StandardButton.Ok

    def save_current(self) -> None:
        if self.current_image_id is None:
            return

        self._write_back_selected()

        if not self._warn_if_missing_labels():
            return

        json_path = self.paths.label_json_path(self.current_image_id)
        to_save = self._filtered_labeled_records()
        save_json_labels(json_path, to_save)

        QMessageBox.information(self, "Saved", f"Saved {len(to_save)} labeled entry(ies):\n{json_path}")

    def _silent_autosave(self) -> None:
        if self.current_image_id is None:
            return
        self._write_back_selected()
        json_path = self.paths.label_json_path(self.current_image_id)
        to_save = self._filtered_labeled_records()
        save_json_labels(json_path, to_save)

    # ---------- Navigation: images ----------

    def next_image(self) -> None:
        if self.autosave_on_nav:
            self._silent_autosave()
        if self.idx + 1 < len(self.image_source):
            self._load_index(self.idx + 1)

    def prev_image(self) -> None:
        if self.autosave_on_nav:
            self._silent_autosave()
        if self.idx - 1 >= 0:
            self._load_index(self.idx - 1)

    # ---------- Navigation: masks ----------

    def next_mask(self) -> None:
        if not self.instance_ids:
            return
        if self.selected_seg_id is None:
            self._select_seg_id(self.instance_ids[0])
            return

        try:
            i = self.instance_ids.index(self.selected_seg_id)
        except ValueError:
            i = 0

        if i + 1 < len(self.instance_ids):
            self._select_seg_id(self.instance_ids[i + 1])

    def prev_mask(self) -> None:
        if not self.instance_ids:
            return
        if self.selected_seg_id is None:
            self._select_seg_id(self.instance_ids[0])
            return

        try:
            i = self.instance_ids.index(self.selected_seg_id)
        except ValueError:
            i = 0

        if i - 1 >= 0:
            self._select_seg_id(self.instance_ids[i - 1])

    # ---------- Keyboard shortcuts ----------

    def keyPressEvent(self, event):  # noqa: N802
        key = event.key()

        # mask nav
        if key in (Qt.Key.Key_J, Qt.Key.Key_Down, Qt.Key.Key_BracketRight, Qt.Key.Key_N):
            self.next_mask()
            return
        if key in (Qt.Key.Key_K, Qt.Key.Key_Up, Qt.Key.Key_BracketLeft, Qt.Key.Key_P):
            self.prev_mask()
            return

        # image nav
        if key in (Qt.Key.Key_D, Qt.Key.Key_Right):
            self.next_image()
            return
        if key in (Qt.Key.Key_A, Qt.Key.Key_Left):
            self.prev_image()
            return

        # save
        if key == Qt.Key.Key_S:
            self.save_current()
            return

        # toggle overlay
        if key == Qt.Key.Key_O:
            self.show_overlay = not self.show_overlay
            self._render_image()
            return

        super().keyPressEvent(event)


# ----------------------- Entrypoint helper -----------------------

def run_gui(image_source, paths: LabelingPaths, key_map: Optional[Dict[str, str]] = None) -> None:
    app = QApplication([])
    win = LabelingWindow(image_source=image_source, paths=paths, key_map=key_map)
    win.show()
    app.exec()


# # labeling/gui_qt.py
# #
# # PyQt6 GUI for labeling per-instance segmentation outputs.
# #
# # Updates in this version:
# # 1) Shows ONE mask at a time (only the currently-selected instance mask is overlaid).
# # 2) Adds explicit mask navigation buttons + shortcuts (Prev/Next Mask).
# # 3) Save behavior:
# #    - Saves ONLY entries that are "labeled" (i.e., all required non-readonly/non-text fields present).
# #    - If any masks are missing labels, shows an OK/Cancel warning before saving.
# #    - Silent autosave (on navigation) will NOT pop warnings; it saves labeled entries only.
# #
# # Output JSON:
# #   output_root/full_results/<labeling_tag>/<image_id>.json
# # Segmentation npz path:
# #   output_root/segmentation_results/<segmentation_config_name>/<image_id>/segmentation_results.npz
# #
# # NOTE: You MUST edit the import path for SegmentationResults below.

# from __future__ import annotations

# from dataclasses import dataclass
# from pathlib import Path
# from typing import Any, Dict, List, Optional, Tuple

# import numpy as np
# from PIL import Image

# from PyQt6.QtCore import Qt
# from PyQt6.QtGui import QImage, QPixmap
# from PyQt6.QtWidgets import (
#     QApplication,
#     QComboBox,
#     QDoubleSpinBox,
#     QFormLayout,
#     QHBoxLayout,
#     QLabel,
#     QLineEdit,
#     QListWidget,
#     QListWidgetItem,
#     QMainWindow,
#     QMessageBox,
#     QPushButton,
#     QSplitter,
#     QTextEdit,
#     QVBoxLayout,
#     QWidget,
# )

# from .io import index_by_id, load_json_labels, merge_defaults_with_existing, save_json_labels
# from .schema import build_default_labels_for_segment_ids, schema_fields_for_gui

# from instance_segmentation.abs_instance_segmenter import SegmentationResults

# from PIL import Image, ImageDraw


# # ----------------------- Utilities -----------------------

# def mask_bbox_xyxy(seg_map: np.ndarray, seg_id: int) -> Optional[Tuple[int, int, int, int]]:
#     """
#     Returns bounding box (x0, y0, x1, y1) inclusive pixel coords for seg_id, or None if empty.
#     """
#     ys, xs = np.where(seg_map == int(seg_id))
#     if xs.size == 0 or ys.size == 0:
#         return None
#     x0 = int(xs.min())
#     x1 = int(xs.max())
#     y0 = int(ys.min())
#     y1 = int(ys.max())
#     return x0, y0, x1, y1


# def draw_bounding_box(
#     pil_img: Image.Image,
#     bbox_xyxy: Tuple[int, int, int, int],
#     color: Tuple[int, int, int] = (255, 255, 0),  # bright yellow
#     width: int = 4,
#     pad: int = 2,
# ) -> Image.Image:
#     """
#     Draw a bright rectangle on top of pil_img and return it.
#     """
#     x0, y0, x1, y1 = bbox_xyxy
#     x0 = max(0, x0 - pad)
#     y0 = max(0, y0 - pad)
#     x1 = min(pil_img.width - 1, x1 + pad)
#     y1 = min(pil_img.height - 1, y1 + pad)

#     out = pil_img.copy()
#     draw = ImageDraw.Draw(out)

#     # Pillow's rectangle uses inclusive bounds; draw multiple rectangles to simulate thickness
#     for i in range(width):
#         draw.rectangle([x0 - i, y0 - i, x1 + i, y1 + i], outline=color)
#     return out


# def pil_to_qpixmap(pil_img: Image.Image) -> QPixmap:
#     rgb = pil_img.convert("RGB")
#     arr = np.array(rgb)
#     h, w, _ = arr.shape
#     qimg = QImage(arr.data, w, h, 3 * w, QImage.Format.Format_RGB888)
#     return QPixmap.fromImage(qimg.copy())


# def seg_map_from_results(seg_results: SegmentationResults) -> np.ndarray:
#     """
#     Build HxW int64 map: -1 background, else instance-id at pixel.
#     Overlaps resolved by "last wins" (later masks overwrite earlier).
#     """
#     ids = seg_results.ids.detach().cpu().numpy().astype(np.int64)      # (N,)
#     masks = seg_results.masks.detach().cpu().numpy().astype(bool)      # (N,H,W)

#     if masks.ndim != 3:
#         raise ValueError(f"Expected masks shape (N,H,W), got {masks.shape}")

#     n, h, w = masks.shape
#     seg_map = np.full((h, w), -1, dtype=np.int64)
#     for i in range(n):
#         seg_map[masks[i]] = int(ids[i])
#     return seg_map


# def seg_id_at(seg_map: np.ndarray, x: int, y: int) -> int:
#     h, w = seg_map.shape[:2]
#     if 0 <= x < w and 0 <= y < h:
#         return int(seg_map[y, x])
#     return -1


# def make_single_mask_overlay(seg_map: np.ndarray, seg_id: int, alpha: float = 0.55) -> Image.Image:
#     """
#     RGBA overlay showing ONLY seg_id. Everything else transparent.

#     Color is deterministically derived from seg_id for consistency.
#     """
#     h, w = seg_map.shape
#     mask = (seg_map == int(seg_id))

#     # deterministic color from id
#     r = int((seg_id * 37) % 255)
#     g = int((seg_id * 57) % 255)
#     b = int((seg_id * 97) % 255)

#     overlay = np.zeros((h, w, 4), dtype=np.uint8)
#     overlay[..., 0] = r
#     overlay[..., 1] = g
#     overlay[..., 2] = b
#     overlay[..., 3] = 0

#     overlay[mask, 3] = int(np.clip(alpha, 0.0, 1.0) * 255)
#     return Image.fromarray(overlay, mode="RGBA")


# # ----------------------- Paths -----------------------

# @dataclass
# class LabelingPaths:
#     output_root: Path
#     segmentation_config_name: str
#     labeling_tag: str

#     def seg_npz_path(self, image_id: str) -> Path:
#         return (
#             self.output_root
#             / "segmentation_results"
#             / self.segmentation_config_name
#             / image_id
#             / "segmentation_results.npz"
#         )

#     def label_json_path(self, image_id: str) -> Path:
#         return self.output_root / "full_results" / self.labeling_tag / f"{image_id}.json"


# # ----------------------- Widgets -----------------------

# class ClickableImageLabel(QLabel):
#     """
#     QLabel that forwards left-clicks to a callback as image-space (x,y).
#     Note: We set the label size to the pixmap size to avoid scaling issues.
#     """
#     def __init__(self, parent=None):
#         super().__init__(parent)
#         self._click_cb = None
#         self._pixmap: Optional[QPixmap] = None

#     def setPixmap(self, pm: QPixmap) -> None:  # type: ignore[override]
#         self._pixmap = pm
#         super().setPixmap(pm)

#     def set_click_callback(self, cb) -> None:
#         self._click_cb = cb

#     def mousePressEvent(self, event):  # noqa: N802
#         if self._pixmap is None or self._click_cb is None:
#             return
#         if event.button() != Qt.MouseButton.LeftButton:
#             return
#         x = int(event.position().x())
#         y = int(event.position().y())
#         self._click_cb(x, y)


# # ----------------------- Main Window -----------------------

# class LabelingWindow(QMainWindow):
#     """
#     image_source: object supporting:
#       - __len__() -> int
#       - get(idx) -> item with fields:
#           item.image_id: str
#           item.image_pil: PIL.Image.Image (RGB)
#           (additional fields are ignored by this GUI)

#     Shortcuts:
#       - Next mask:    J / Down  / N / ]
#       - Prev mask:    K / Up    / P / [
#       - Next image:   D / Right
#       - Prev image:   A / Left
#       - Save:         S
#       - Toggle overlay: O (show/hide selected-mask overlay)
#     """

#     def __init__(
#         self,
#         image_source,
#         paths: LabelingPaths,
#         key_map: Optional[Dict[str, str]] = None,
#         id_key: str = "id",
#         autosave_on_nav: bool = True,
#         overlay_alpha: float = 0.55,
#     ):
#         super().__init__()
#         self.setWindowTitle("Segmentation Labeling GUI")

#         self.image_source = image_source
#         self.paths = paths
#         self.key_map = key_map or {}
#         self.id_key = id_key
#         self.autosave_on_nav = autosave_on_nav
#         self.overlay_alpha = overlay_alpha

#         self.idx = 0

#         # Current image state (only one in memory)
#         self.current_image_id: Optional[str] = None
#         self.current_image_pil: Optional[Image.Image] = None

#         # SegmentationResults stored directly
#         self.seg_results: Optional[SegmentationResults] = None
#         self._seg_map_cache: Optional[np.ndarray] = None  # derived for click+overlay
#         self.instance_ids: List[int] = []

#         # Label records for current image
#         self.current_labels: List[Dict[str, Any]] = []
#         self.current_label_map: Dict[str, Dict[str, Any]] = {}
#         self.selected_seg_id: Optional[int] = None

#         self.show_overlay: bool = True  # now means show selected mask overlay

#         # Schema-driven form specs + widgets
#         self._form_specs: List[Dict[str, Any]] = schema_fields_for_gui()
#         self._widgets_by_field: Dict[str, Any] = {}

#         # Precompute which fields are required to consider a record "labeled"
#         self._required_label_fields: List[str] = self._compute_required_fields()

#         self._build_ui()

#         # Load next image until there's one loaded without error
#         next_index = 0
#         loaded = False
#         while next_index < len(self.image_source):
#             try:
#                 self._load_index(next_index)
#                 loaded = True
#                 break
#             except Exception as e:
#                 print(f"Error loading index {next_index}: {e}")
#                 next_index += 1

#         print("Initialized okay?", loaded)
#         if (not loaded) and len(self.image_source) > 0:
#             QMessageBox.critical(self, "Error", "Failed to load any images. Check console for details.")

#     # ---------- Schema helpers ----------

#     def _compute_required_fields(self) -> List[str]:
#         """
#         Define what "has labels" means:
#           - Must include all schema fields that are editable + meaningful,
#             i.e. kind in ('float', 'choice') and not the id_key.
#           - Text fields (notes) do NOT count as "labeled" by themselves.
#         """
#         req: List[str] = []
#         for spec in self._form_specs:
#             name = spec.get("name")
#             kind = spec.get("kind")
#             if name == self.id_key:
#                 continue
#             if kind in ("float", "choice"):
#                 req.append(name)
#         return req

#     def _is_record_labeled(self, rec: Dict[str, Any]) -> bool:
#         """
#         True if rec contains all required label fields.
#         """
#         for k in self._required_label_fields:
#             if k not in rec:
#                 return False
#             if rec[k] is None:
#                 return False
#         return True

#     def _missing_label_seg_ids(self) -> List[int]:
#         """
#         Returns seg_ids that do not have a labeled record yet.
#         """
#         missing: List[int] = []
#         for sid in self.instance_ids:
#             rec = self.current_label_map.get(str(sid))
#             if rec is None or (not self._is_record_labeled(rec)):
#                 missing.append(int(sid))
#         return missing

#     def _filtered_labeled_records(self) -> List[Dict[str, Any]]:
#         """
#         Records to save: only those that are labeled.
#         """
#         out: List[Dict[str, Any]] = []
#         for rec in self.current_labels:
#             if self._is_record_labeled(rec):
#                 out.append(rec)
#         return out

#     # ---------- UI ----------

#     def _build_ui(self) -> None:
#         root = QWidget()
#         outer = QHBoxLayout(root)

#         splitter = QSplitter(Qt.Orientation.Horizontal)

#         # Left pane: status + segment list + image
#         left = QWidget()
#         left_layout = QVBoxLayout(left)

#         self.info = QLabel("")
#         left_layout.addWidget(self.info)

#         self.segment_list = QListWidget()
#         self.segment_list.currentItemChanged.connect(self._on_segment_list_changed)
#         left_layout.addWidget(self.segment_list, stretch=1)

#         self.image_label = ClickableImageLabel()
#         self.image_label.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)
#         self.image_label.set_click_callback(self._on_image_click)
#         left_layout.addWidget(self.image_label, stretch=3)

#         # Right pane: schema-driven form + buttons
#         right = QWidget()
#         right_layout = QVBoxLayout(right)

#         form = QFormLayout()
#         self._build_schema_form(form)
#         right_layout.addLayout(form)

#         # Mask navigation row (NEW)
#         mask_row = QHBoxLayout()
#         self.prev_mask_btn = QPushButton("Prev Mask (K / [)")
#         self.next_mask_btn = QPushButton("Next Mask (J / ])")
#         mask_row.addWidget(self.prev_mask_btn)
#         mask_row.addWidget(self.next_mask_btn)
#         right_layout.addLayout(mask_row)

#         self.prev_mask_btn.clicked.connect(self.prev_mask)
#         self.next_mask_btn.clicked.connect(self.next_mask)

#         # Image navigation + save row (existing)
#         btn_row = QHBoxLayout()
#         self.prev_btn = QPushButton("Prev Image (A)")
#         self.next_btn = QPushButton("Next Image (D)")
#         self.save_btn = QPushButton("Save (S)")
#         btn_row.addWidget(self.prev_btn)
#         btn_row.addWidget(self.next_btn)
#         btn_row.addWidget(self.save_btn)
#         right_layout.addLayout(btn_row)

#         self.prev_btn.clicked.connect(self.prev_image)
#         self.next_btn.clicked.connect(self.next_image)
#         self.save_btn.clicked.connect(self.save_current)

#         splitter.addWidget(left)
#         splitter.addWidget(right)
#         splitter.setStretchFactor(0, 4)
#         splitter.setStretchFactor(1, 3)

#         outer.addWidget(splitter)
#         self.setCentralWidget(root)

#     def _build_schema_form(self, form: QFormLayout) -> None:
#         """
#         Build a form based on schema_fields_for_gui() specs.
#         Supported kinds:
#           - readonly: QLineEdit read-only
#           - float:    QDoubleSpinBox
#           - choice:   QComboBox (requires 'choices' in spec)
#           - text:     QTextEdit
#         """
#         if not self._form_specs:
#             raise RuntimeError("schema_fields_for_gui() returned no fields. Add ui metadata in schema.")

#         for spec in self._form_specs:
#             name = spec["name"]
#             kind = spec.get("kind")
#             label = spec.get("label", name)

#             if kind == "readonly":
#                 w = QLineEdit()
#                 w.setReadOnly(True)

#             elif kind == "float":
#                 w = QDoubleSpinBox()
#                 w.setRange(float(spec.get("min", 0.0)), float(spec.get("max", 1e9)))
#                 w.setSingleStep(float(spec.get("step", 1.0)))
#                 w.setDecimals(int(spec.get("decimals", 2)))
#                 w.valueChanged.connect(self._write_back_selected)

#             elif kind == "choice":
#                 w = QComboBox()
#                 choices = spec.get("choices")
#                 if not choices:
#                     raise ValueError(f"choice field '{name}' missing 'choices' in schema metadata")
#                 w.addItems(list(choices))
#                 w.currentTextChanged.connect(self._write_back_selected)

#             elif kind == "text":
#                 w = QTextEdit()
#                 w.textChanged.connect(self._write_back_selected)

#             else:
#                 raise ValueError(f"Unknown ui kind '{kind}' for field '{name}'")

#             self._widgets_by_field[name] = w
#             form.addRow(label, w)

#     # ---------- Loading ----------

#     def _status_text(self) -> str:
#         total = len(self.image_source)
#         nseg = len(self.instance_ids)
#         sid = self.selected_seg_id
#         missing = len(self._missing_label_seg_ids()) if self.instance_ids else 0
#         return f"Image {self.idx+1}/{total} | image_id={self.current_image_id} | masks={nseg} | selected={sid} | missing_labels={missing}"

#     def _load_index(self, idx: int) -> None:
#         if idx < 0 or idx >= len(self.image_source):
#             return

#         self.idx = idx
#         item = self.image_source.get(idx)

#         if not getattr(item, "image_id", None):
#             raise ValueError(
#                 f"image_source.get({idx}) returned image_id=None/empty. "
#                 "Labeling requires stable image_id."
#             )

#         self.current_image_id = str(item.image_id)
#         self.current_image_pil = item.image_pil.convert("RGB")
#         self.selected_seg_id = None

#         # Load SegmentationResults directly
#         seg_path = self.paths.seg_npz_path(self.current_image_id)
#         self.seg_results, _ = SegmentationResults.loadFromFile(str(seg_path))

#         # Cache seg_map for O(1) picking + overlay
#         self._seg_map_cache = seg_map_from_results(self.seg_results)

#         # Instance ids in the same order as saved
#         ids_np = self.seg_results.ids.detach().cpu().numpy().astype(np.int64)
#         self.instance_ids = [int(x) for x in ids_np.tolist()]

#         # Load / merge labels JSON
#         defaults = build_default_labels_for_segment_ids(self.instance_ids, key_map=self.key_map)
#         json_path = self.paths.label_json_path(self.current_image_id)
#         existing = load_json_labels(json_path)

#         self.current_labels = merge_defaults_with_existing(defaults, existing, id_key=self.id_key)
#         self.current_label_map = index_by_id(self.current_labels, id_key=self.id_key)

#         self._populate_segment_list()

#         # auto-select first mask
#         if self.instance_ids:
#             self._select_seg_id(self.instance_ids[0])
#         else:
#             self._render_image()

#         self.info.setText(self._status_text())

#     # ---------- Render (ONE mask at a time) ----------

# def _render_image(self) -> None:
#     if self.current_image_pil is None:
#         return
#     base = self.current_image_pil.convert("RGBA")

#     composed = base

#     # overlay only selected mask
#     if (
#         self.show_overlay
#         and self._seg_map_cache is not None
#         and self.selected_seg_id is not None
#     ):
#         overlay = make_single_mask_overlay(
#             self._seg_map_cache,
#             seg_id=int(self.selected_seg_id),
#             alpha=self.overlay_alpha,
#         )
#         composed = Image.alpha_composite(base, overlay)

#         # NEW: bounding box around selected mask
#         bbox = mask_bbox_xyxy(self._seg_map_cache, int(self.selected_seg_id))
#         if bbox is not None:
#             # Draw box on top (convert to RGB for crisp lines, then back if you want)
#             composed_rgb = composed.convert("RGB")
#             composed_rgb = draw_bounding_box(
#                 composed_rgb,
#                 bbox,
#                 color=(255, 255, 0),  # bright yellow
#                 width=4,
#                 pad=2,
#             )
#             composed = composed_rgb.convert("RGBA")

#     pm = pil_to_qpixmap(composed.convert("RGB"))

#     # avoid scaling confusion: make label exactly pixmap size
#     self.image_label.setFixedSize(pm.width(), pm.height())
#     self.image_label.setPixmap(pm)


#     # ---------- Segment list ----------

#     def _populate_segment_list(self) -> None:
#         self.segment_list.blockSignals(True)
#         self.segment_list.clear()
#         for sid in self.instance_ids:
#             self.segment_list.addItem(QListWidgetItem(str(sid)))
#         self.segment_list.blockSignals(False)

#     def _on_segment_list_changed(self, current: QListWidgetItem, previous: QListWidgetItem) -> None:
#         if current is None:
#             return
#         try:
#             sid = int(current.text())
#         except Exception:
#             return
#         self._select_seg_id(sid)

#     def _sync_list_selection(self, sid: int) -> None:
#         self.segment_list.blockSignals(True)
#         for i in range(self.segment_list.count()):
#             if self.segment_list.item(i).text() == str(sid):
#                 self.segment_list.setCurrentRow(i)
#                 break
#         self.segment_list.blockSignals(False)

#     # ---------- Click select ----------

#     def _on_image_click(self, x: int, y: int) -> None:
#         if self._seg_map_cache is None:
#             return
#         sid = seg_id_at(self._seg_map_cache, x, y)
#         if sid >= 0:
#             self._select_seg_id(sid)

#     # ---------- Form (schema-driven) ----------

#     def _set_widget_value_blocked(self, widget: Any, kind: str, value: Any, spec: Dict[str, Any]) -> None:
#         """
#         Set widget value with signals blocked.
#         "Blank policy" for unlabeled/missing:
#           - float: 0.0
#           - choice: first option
#           - text: ""
#         """
#         widget.blockSignals(True)
#         try:
#             if kind == "readonly":
#                 widget.setText("" if value is None else str(value))

#             elif kind == "float":
#                 widget.setValue(0.0 if value is None else float(value))

#             elif kind == "choice":
#                 choices = list(spec.get("choices", []))
#                 if not choices:
#                     widget.setCurrentText("" if value is None else str(value))
#                 else:
#                     v = "" if value is None else str(value)
#                     widget.setCurrentText(v if v in choices else choices[0])

#             elif kind == "text":
#                 widget.setPlainText("" if value is None else str(value))

#         finally:
#             widget.blockSignals(False)

#     def _get_widget_value(self, widget: Any, kind: str) -> Any:
#         if kind == "readonly":
#             return str(widget.text())
#         if kind == "float":
#             return float(widget.value())
#         if kind == "choice":
#             return str(widget.currentText())
#         if kind == "text":
#             return str(widget.toPlainText())
#         raise ValueError(f"Unknown kind: {kind}")

#     def _select_seg_id(self, sid: int) -> None:
#         self.selected_seg_id = int(sid)
#         self._sync_list_selection(int(sid))

#         # Do NOT create a label record on selection.
#         # This allows "unlabeled" masks to remain absent until the user edits.
#         rec = self.current_label_map.get(str(sid))

#         # Populate widgets based on schema
#         for spec in self._form_specs:
#             name = spec["name"]
#             kind = spec.get("kind")
#             w = self._widgets_by_field[name]

#             if name == self.id_key:
#                 self._set_widget_value_blocked(w, kind, sid, spec)
#                 continue

#             value = None if rec is None else rec.get(name, None)
#             self._set_widget_value_blocked(w, kind, value, spec)

#         self._render_image()
#         self.info.setText(self._status_text())

#     def _write_back_selected(self) -> None:
#         if self.selected_seg_id is None:
#             return

#         key = str(self.selected_seg_id)
#         rec = self.current_label_map.get(key)
#         if rec is None:
#             # Create record on first edit
#             rec = {self.id_key: int(self.selected_seg_id)}
#             self.current_labels.append(rec)
#             self.current_label_map[key] = rec

#         # Write back all non-readonly fields
#         for spec in self._form_specs:
#             name = spec["name"]
#             kind = spec.get("kind")
#             if name == self.id_key:
#                 rec[self.id_key] = int(self.selected_seg_id)
#                 continue
#             if kind == "readonly":
#                 continue
#             w = self._widgets_by_field[name]
#             rec[name] = self._get_widget_value(w, kind)

#         self.info.setText(self._status_text())

#     # ---------- Save ----------

#     def _warn_if_missing_labels(self) -> bool:
#         """
#         Returns True if caller should proceed with save, False if cancelled.
#         """
#         missing = self._missing_label_seg_ids()
#         if not missing:
#             return True

#         preview = ", ".join(str(x) for x in missing[:30])
#         suffix = "" if len(missing) <= 30 else f" ... (+{len(missing)-30} more)"

#         msg = (
#             f"This image has {len(missing)} mask(s) missing labels.\n\n"
#             f"Missing mask ids:\n{preview}{suffix}\n\n"
#             "Save anyway (only labeled entries will be saved)?"
#         )

#         res = QMessageBox.warning(
#             self,
#             "Missing labels",
#             msg,
#             QMessageBox.StandardButton.Ok | QMessageBox.StandardButton.Cancel,
#             QMessageBox.StandardButton.Cancel,
#         )
#         return res == QMessageBox.StandardButton.Ok

#     def save_current(self) -> None:
#         if self.current_image_id is None:
#             return

#         # Ensure latest widget edits are written
#         self._write_back_selected()

#         if not self._warn_if_missing_labels():
#             return

#         json_path = self.paths.label_json_path(self.current_image_id)
#         to_save = self._filtered_labeled_records()
#         save_json_labels(json_path, to_save)
#         QMessageBox.information(self, "Saved", f"Saved {len(to_save)} labeled entry(ies):\n{json_path}")

#     def _silent_autosave(self) -> None:
#         """
#         Silent save used for navigation. Does NOT warn.
#         Saves labeled entries only.
#         """
#         if self.current_image_id is None:
#             return
#         self._write_back_selected()
#         json_path = self.paths.label_json_path(self.current_image_id)
#         to_save = self._filtered_labeled_records()
#         save_json_labels(json_path, to_save)

#     # ---------- Navigation: images ----------

#     def next_image(self) -> None:
#         if self.autosave_on_nav:
#             self._silent_autosave()
#         if self.idx + 1 < len(self.image_source):
#             self._load_index(self.idx + 1)

#     def prev_image(self) -> None:
#         if self.autosave_on_nav:
#             self._silent_autosave()
#         if self.idx - 1 >= 0:
#             self._load_index(self.idx - 1)

#     # ---------- Navigation: masks (within image) ----------

#     def next_mask(self) -> None:
#         if not self.instance_ids:
#             return
#         if self.selected_seg_id is None:
#             self._select_seg_id(self.instance_ids[0])
#             return

#         try:
#             i = self.instance_ids.index(self.selected_seg_id)
#         except ValueError:
#             i = 0

#         if i + 1 < len(self.instance_ids):
#             self._select_seg_id(self.instance_ids[i + 1])

#     def prev_mask(self) -> None:
#         if not self.instance_ids:
#             return
#         if self.selected_seg_id is None:
#             self._select_seg_id(self.instance_ids[0])
#             return

#         try:
#             i = self.instance_ids.index(self.selected_seg_id)
#         except ValueError:
#             i = 0

#         if i - 1 >= 0:
#             self._select_seg_id(self.instance_ids[i - 1])

#     # ---------- Keyboard shortcuts ----------

#     def keyPressEvent(self, event):  # noqa: N802
#         key = event.key()

#         # mask nav (within image)
#         if key in (Qt.Key.Key_J, Qt.Key.Key_Down, Qt.Key.Key_N, Qt.Key.Key_BracketRight):
#             self.next_mask()
#             return
#         if key in (Qt.Key.Key_K, Qt.Key.Key_Up, Qt.Key.Key_P, Qt.Key.Key_BracketLeft):
#             self.prev_mask()
#             return

#         # image nav
#         if key in (Qt.Key.Key_D, Qt.Key.Key_Right):
#             self.next_image()
#             return
#         if key in (Qt.Key.Key_A, Qt.Key.Key_Left):
#             self.prev_image()
#             return

#         # save
#         if key == Qt.Key.Key_S:
#             self.save_current()
#             return

#         # toggle overlay
#         if key == Qt.Key.Key_O:
#             self.show_overlay = not self.show_overlay
#             self._render_image()
#             return

#         super().keyPressEvent(event)


# # ----------------------- Entrypoint helper -----------------------

# def run_gui(image_source, paths: LabelingPaths, key_map: Optional[Dict[str, str]] = None) -> None:
#     """
#     image_source: random-access source with __len__ and get(idx)-> {image_id, image_pil}
#     """
#     app = QApplication([])
#     win = LabelingWindow(image_source=image_source, paths=paths, key_map=key_map)
#     win.show()
#     app.exec()





# # labeling/gui_qt.py
# #
# # PyQt6 GUI for labeling per-instance segmentation outputs.
# #
# # Updated to be SCHEMA-DRIVEN:
# # - Builds the right-side form from schema_fields_for_gui() (in labeling/schema.py)
# # - Reads/writes values generically based on field "kind" (float/choice/text/readonly)
# # - Supports unlabeled segments: records are ONLY created when the user edits (write-back).
# #
# # Output JSON:
# #   output_root/full_results/<labeling_tag>/<image_id>.json
# # Segmentation npz path:
# #   output_root/segmentation_results/<segmentation_config_name>/<image_id>/segmentation_results.npz
# #
# # NOTE: You MUST edit the import path for SegmentationResults below.

# from __future__ import annotations

# from dataclasses import dataclass
# from pathlib import Path
# from typing import Any, Dict, List, Optional, Tuple

# import numpy as np
# from PIL import Image

# from PyQt6.QtCore import Qt
# from PyQt6.QtGui import QImage, QPixmap
# from PyQt6.QtWidgets import (
#     QApplication,
#     QComboBox,
#     QDoubleSpinBox,
#     QFormLayout,
#     QHBoxLayout,
#     QLabel,
#     QLineEdit,
#     QListWidget,
#     QListWidgetItem,
#     QMainWindow,
#     QMessageBox,
#     QPushButton,
#     QSplitter,
#     QTextEdit,
#     QVBoxLayout,
#     QWidget,
# )

# from .io import index_by_id, load_json_labels, merge_defaults_with_existing, save_json_labels
# from .schema import build_default_labels_for_segment_ids, schema_fields_for_gui

# from instance_segmentation.abs_instance_segmenter import SegmentationResults


# # ----------------------- Utilities -----------------------

# def pil_to_qpixmap(pil_img: Image.Image) -> QPixmap:
#     rgb = pil_img.convert("RGB")
#     arr = np.array(rgb)
#     h, w, _ = arr.shape
#     qimg = QImage(arr.data, w, h, 3 * w, QImage.Format.Format_RGB888)
#     return QPixmap.fromImage(qimg.copy())


# def seg_map_from_results(seg_results: SegmentationResults) -> np.ndarray:
#     """
#     Build HxW int64 map: -1 background, else instance-id at pixel.
#     Overlaps resolved by "last wins" (later masks overwrite earlier).
#     """
#     ids = seg_results.ids.detach().cpu().numpy().astype(np.int64)      # (N,)
#     masks = seg_results.masks.detach().cpu().numpy().astype(bool)      # (N,H,W)

#     if masks.ndim != 3:
#         raise ValueError(f"Expected masks shape (N,H,W), got {masks.shape}")

#     n, h, w = masks.shape
#     seg_map = np.full((h, w), -1, dtype=np.int64)
#     for i in range(n):
#         seg_map[masks[i]] = int(ids[i])
#     return seg_map


# def seg_id_at(seg_map: np.ndarray, x: int, y: int) -> int:
#     h, w = seg_map.shape[:2]
#     if 0 <= x < w and 0 <= y < h:
#         return int(seg_map[y, x])
#     return -1


# def make_overlay(seg_map: np.ndarray, alpha: float = 0.45) -> Image.Image:
#     """
#     Deterministic color overlay per instance id.
#     Background (-1) is fully transparent.
#     """
#     h, w = seg_map.shape
#     ids = seg_map.astype(np.int64)

#     # hash ids -> rgb
#     r = (ids * 37) % 255
#     g = (ids * 57) % 255
#     b = (ids * 97) % 255
#     overlay_rgb = np.stack([r, g, b], axis=-1).astype(np.uint8)

#     # alpha channel: transparent for background
#     alpha_mask = np.full((h, w), int(np.clip(alpha, 0.0, 1.0) * 255), dtype=np.uint8)
#     alpha_mask[ids == -1] = 0

#     pil_overlay = Image.fromarray(overlay_rgb, mode="RGB")
#     pil_overlay.putalpha(Image.fromarray(alpha_mask, mode="L"))
#     return pil_overlay


# # ----------------------- Paths -----------------------

# @dataclass
# class LabelingPaths:
#     output_root: Path
#     segmentation_config_name: str
#     labeling_tag: str

#     def seg_npz_path(self, image_id: str) -> Path:
#         return (
#             self.output_root
#             / "segmentation_results"
#             / self.segmentation_config_name
#             / image_id
#             / "segmentation_results.npz"
#         )

#     def label_json_path(self, image_id: str) -> Path:
#         return self.output_root / "full_results" / self.labeling_tag / f"{image_id}.json"


# # ----------------------- Widgets -----------------------

# class ClickableImageLabel(QLabel):
#     """
#     QLabel that forwards left-clicks to a callback as image-space (x,y).
#     Note: We set the label size to the pixmap size to avoid scaling issues.
#     """
#     def __init__(self, parent=None):
#         super().__init__(parent)
#         self._click_cb = None
#         self._pixmap: Optional[QPixmap] = None

#     def setPixmap(self, pm: QPixmap) -> None:  # type: ignore[override]
#         self._pixmap = pm
#         super().setPixmap(pm)

#     def set_click_callback(self, cb) -> None:
#         self._click_cb = cb

#     def mousePressEvent(self, event):  # noqa: N802
#         if self._pixmap is None or self._click_cb is None:
#             return
#         if event.button() != Qt.MouseButton.LeftButton:
#             return
#         x = int(event.position().x())
#         y = int(event.position().y())
#         self._click_cb(x, y)


# # ----------------------- Main Window -----------------------

# class LabelingWindow(QMainWindow):
#     """
#     image_source: object supporting:
#       - __len__() -> int
#       - get(idx) -> item with fields:
#           item.image_id: str
#           item.image_pil: PIL.Image.Image (RGB)
#           (additional fields are ignored by this GUI)

#     Shortcuts:
#       - Next segment: J / Down
#       - Prev segment: K / Up
#       - Next image:   D / Right
#       - Prev image:   A / Left
#       - Save:         S
#       - Toggle overlay: O
#     """

#     def __init__(
#         self,
#         image_source,
#         paths: LabelingPaths,
#         key_map: Optional[Dict[str, str]] = None,
#         id_key: str = "id",
#         autosave_on_nav: bool = True,
#         overlay_alpha: float = 0.45,
#     ):
#         super().__init__()
#         self.setWindowTitle("Segmentation Labeling GUI")

#         self.image_source = image_source
#         self.paths = paths
#         self.key_map = key_map or {}
#         self.id_key = id_key
#         self.autosave_on_nav = autosave_on_nav
#         self.overlay_alpha = overlay_alpha

#         self.idx = 0

#         # Current image state (only one in memory)
#         self.current_image_id: Optional[str] = None
#         self.current_image_pil: Optional[Image.Image] = None

#         # SegmentationResults stored directly
#         self.seg_results: Optional[SegmentationResults] = None
#         self._seg_map_cache: Optional[np.ndarray] = None  # derived for click+overlay
#         self.instance_ids: List[int] = []

#         # Label records for current image
#         self.current_labels: List[Dict[str, Any]] = []
#         self.current_label_map: Dict[str, Dict[str, Any]] = {}
#         self.selected_seg_id: Optional[int] = None

#         self.show_overlay: bool = True

#         # Schema-driven form specs + widgets
#         self._form_specs: List[Dict[str, Any]] = schema_fields_for_gui()
#         self._widgets_by_field: Dict[str, Any] = {}

#         self._build_ui()

#         # Load next image until there's one loaded without error
#         next_index = 0
#         loaded = False
#         while next_index < len(self.image_source):
#             try:
#                 self._load_index(next_index)
#                 loaded = True
#                 break
#             except Exception as e:
#                 print(f"Error loading index {next_index}: {e}")
#                 next_index += 1

#         print("Initialized okay?", loaded)
#         if (not loaded) and len(self.image_source) > 0:
#             QMessageBox.critical(self, "Error", "Failed to load any images. Check console for details.")

#     # ---------- UI ----------

#     def _build_ui(self) -> None:
#         root = QWidget()
#         outer = QHBoxLayout(root)

#         splitter = QSplitter(Qt.Orientation.Horizontal)

#         # Left pane: status + segment list + image
#         left = QWidget()
#         left_layout = QVBoxLayout(left)

#         self.info = QLabel("")
#         left_layout.addWidget(self.info)

#         self.segment_list = QListWidget()
#         self.segment_list.currentItemChanged.connect(self._on_segment_list_changed)
#         left_layout.addWidget(self.segment_list, stretch=1)

#         self.image_label = ClickableImageLabel()
#         self.image_label.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)
#         self.image_label.set_click_callback(self._on_image_click)
#         left_layout.addWidget(self.image_label, stretch=3)

#         # Right pane: schema-driven form + buttons
#         right = QWidget()
#         right_layout = QVBoxLayout(right)

#         form = QFormLayout()
#         self._build_schema_form(form)
#         right_layout.addLayout(form)

#         btn_row = QHBoxLayout()
#         self.prev_btn = QPushButton("Prev Image (A)")
#         self.next_btn = QPushButton("Next Image (D)")
#         self.save_btn = QPushButton("Save (S)")
#         btn_row.addWidget(self.prev_btn)
#         btn_row.addWidget(self.next_btn)
#         btn_row.addWidget(self.save_btn)
#         right_layout.addLayout(btn_row)

#         self.prev_btn.clicked.connect(self.prev_image)
#         self.next_btn.clicked.connect(self.next_image)
#         self.save_btn.clicked.connect(self.save_current)

#         splitter.addWidget(left)
#         splitter.addWidget(right)
#         splitter.setStretchFactor(0, 4)
#         splitter.setStretchFactor(1, 3)

#         outer.addWidget(splitter)
#         self.setCentralWidget(root)

#     def _build_schema_form(self, form: QFormLayout) -> None:
#         """
#         Build a form based on schema_fields_for_gui() specs.
#         Supported kinds:
#           - readonly: QLineEdit read-only
#           - float:    QDoubleSpinBox
#           - choice:   QComboBox (requires 'choices' in spec)
#           - text:     QTextEdit
#         """
#         if not self._form_specs:
#             # Fail loudly so it's obvious schema_fields_for_gui isn't set up
#             raise RuntimeError("schema_fields_for_gui() returned no fields. Add ui metadata in schema.")

#         for spec in self._form_specs:
#             name = spec["name"]
#             kind = spec.get("kind")

#             label = spec.get("label", name)

#             if kind == "readonly":
#                 w = QLineEdit()
#                 w.setReadOnly(True)

#             elif kind == "float":
#                 w = QDoubleSpinBox()
#                 w.setRange(float(spec.get("min", 0.0)), float(spec.get("max", 1e9)))
#                 w.setSingleStep(float(spec.get("step", 1.0)))
#                 w.setDecimals(int(spec.get("decimals", 2)))
#                 w.valueChanged.connect(self._write_back_selected)

#             elif kind == "choice":
#                 w = QComboBox()
#                 choices = spec.get("choices")
#                 if not choices:
#                     raise ValueError(f"choice field '{name}' missing 'choices' in schema metadata")
#                 w.addItems(list(choices))
#                 w.currentTextChanged.connect(self._write_back_selected)

#             elif kind == "text":
#                 w = QTextEdit()
#                 w.textChanged.connect(self._write_back_selected)

#             else:
#                 raise ValueError(f"Unknown ui kind '{kind}' for field '{name}'")

#             self._widgets_by_field[name] = w
#             form.addRow(label, w)

#     # ---------- Loading ----------

#     def _status_text(self) -> str:
#         total = len(self.image_source)
#         nseg = len(self.instance_ids)
#         return f"Image {self.idx+1}/{total} | image_id={self.current_image_id} | segments={nseg}"

#     def _load_index(self, idx: int) -> None:
#         if idx < 0 or idx >= len(self.image_source):
#             return

#         self.idx = idx
#         item = self.image_source.get(idx)

#         if not getattr(item, "image_id", None):
#             raise ValueError(
#                 f"image_source.get({idx}) returned image_id=None/empty. "
#                 "Labeling requires stable image_id."
#             )

#         self.current_image_id = str(item.image_id)
#         self.current_image_pil = item.image_pil.convert("RGB")
#         self.selected_seg_id = None

#         # Load SegmentationResults directly
#         seg_path = self.paths.seg_npz_path(self.current_image_id)
#         self.seg_results, _ = SegmentationResults.loadFromFile(str(seg_path))

#         # Cache seg_map for O(1) picking + overlay
#         self._seg_map_cache = seg_map_from_results(self.seg_results)

#         # Instance ids in the same order as saved
#         ids_np = self.seg_results.ids.detach().cpu().numpy().astype(np.int64)
#         self.instance_ids = [int(x) for x in ids_np.tolist()]

#         # Load / merge labels JSON
#         defaults = build_default_labels_for_segment_ids(self.instance_ids, key_map=self.key_map)
#         json_path = self.paths.label_json_path(self.current_image_id)
#         existing = load_json_labels(json_path)

#         self.current_labels = merge_defaults_with_existing(defaults, existing, id_key=self.id_key)
#         self.current_label_map = index_by_id(self.current_labels, id_key=self.id_key)

#         self._populate_segment_list()
#         self._render_image()
#         self.info.setText(self._status_text())

#         # auto-select first segment
#         if self.instance_ids:
#             self._select_seg_id(self.instance_ids[0])

#     # ---------- Render ----------

#     def _render_image(self) -> None:
#         if self.current_image_pil is None:
#             return
#         base = self.current_image_pil.convert("RGBA")

#         if self.show_overlay and self._seg_map_cache is not None:
#             overlay = make_overlay(self._seg_map_cache, alpha=self.overlay_alpha)
#             composed = Image.alpha_composite(base, overlay).convert("RGB")
#         else:
#             composed = base.convert("RGB")

#         pm = pil_to_qpixmap(composed)
#         # avoid scaling confusion: make label exactly pixmap size
#         self.image_label.setFixedSize(pm.width(), pm.height())
#         self.image_label.setPixmap(pm)

#     # ---------- Segment list ----------

#     def _populate_segment_list(self) -> None:
#         self.segment_list.blockSignals(True)
#         self.segment_list.clear()
#         for sid in self.instance_ids:
#             self.segment_list.addItem(QListWidgetItem(str(sid)))
#         self.segment_list.blockSignals(False)

#     def _on_segment_list_changed(self, current: QListWidgetItem, previous: QListWidgetItem) -> None:
#         if current is None:
#             return
#         try:
#             sid = int(current.text())
#         except Exception:
#             return
#         self._select_seg_id(sid)

#     def _sync_list_selection(self, sid: int) -> None:
#         self.segment_list.blockSignals(True)
#         for i in range(self.segment_list.count()):
#             if self.segment_list.item(i).text() == str(sid):
#                 self.segment_list.setCurrentRow(i)
#                 break
#         self.segment_list.blockSignals(False)

#     # ---------- Click select ----------

#     def _on_image_click(self, x: int, y: int) -> None:
#         if self._seg_map_cache is None:
#             return
#         sid = seg_id_at(self._seg_map_cache, x, y)
#         if sid >= 0:
#             self._select_seg_id(sid)

#     # ---------- Form (schema-driven) ----------

#     def _set_widget_value_blocked(self, widget: Any, kind: str, value: Any, spec: Dict[str, Any]) -> None:
#         """
#         Set widget value with signals blocked.
#         For missing/unlabeled fields, we show a simple "blank policy":
#           - float: 0.0
#           - choice: first option
#           - text: ""
#         """
#         widget.blockSignals(True)
#         try:
#             if kind == "readonly":
#                 widget.setText("" if value is None else str(value))

#             elif kind == "float":
#                 widget.setValue(0.0 if value is None else float(value))

#             elif kind == "choice":
#                 choices = list(spec.get("choices", []))
#                 if not choices:
#                     widget.setCurrentText("" if value is None else str(value))
#                 else:
#                     v = "" if value is None else str(value)
#                     widget.setCurrentText(v if v in choices else choices[0])

#             elif kind == "text":
#                 widget.setPlainText("" if value is None else str(value))

#         finally:
#             widget.blockSignals(False)

#     def _get_widget_value(self, widget: Any, kind: str) -> Any:
#         if kind == "readonly":
#             return str(widget.text())
#         if kind == "float":
#             return float(widget.value())
#         if kind == "choice":
#             return str(widget.currentText())
#         if kind == "text":
#             return str(widget.toPlainText())
#         raise ValueError(f"Unknown kind: {kind}")

#     def _select_seg_id(self, sid: int) -> None:
#         self.selected_seg_id = sid
#         self._sync_list_selection(sid)

#         # Do NOT create a label record on selection.
#         # This allows "unlabeled" segments to remain absent from JSON until user edits.
#         rec = self.current_label_map.get(str(sid))

#         # Populate widgets based on schema
#         for spec in self._form_specs:
#             name = spec["name"]
#             kind = spec.get("kind")
#             w = self._widgets_by_field[name]

#             if name == self.id_key:
#                 self._set_widget_value_blocked(w, kind, sid, spec)
#                 continue

#             value = None if rec is None else rec.get(name, None)
#             self._set_widget_value_blocked(w, kind, value, spec)

#     def _write_back_selected(self) -> None:
#         if self.selected_seg_id is None:
#             return

#         key = str(self.selected_seg_id)
#         rec = self.current_label_map.get(key)
#         if rec is None:
#             # Create record on first edit
#             rec = {self.id_key: int(self.selected_seg_id)}
#             self.current_labels.append(rec)
#             self.current_label_map[key] = rec

#         # Write back all non-readonly fields
#         for spec in self._form_specs:
#             name = spec["name"]
#             kind = spec.get("kind")
#             if name == self.id_key:
#                 # Keep id stable
#                 rec[self.id_key] = int(self.selected_seg_id)
#                 continue
#             # readonly fields shouldn't be persisted unless you want them
#             if kind == "readonly":
#                 continue
#             w = self._widgets_by_field[name]
#             rec[name] = self._get_widget_value(w, kind)

#     # ---------- Save ----------

#     def save_current(self) -> None:
#         if self.current_image_id is None:
#             return
#         self._write_back_selected()
#         json_path = self.paths.label_json_path(self.current_image_id)
#         save_json_labels(json_path, self.current_labels)
#         QMessageBox.information(self, "Saved", f"Saved labels:\n{json_path}")

#     def _silent_autosave(self) -> None:
#         if self.current_image_id is None:
#             return
#         self._write_back_selected()
#         json_path = self.paths.label_json_path(self.current_image_id)
#         save_json_labels(json_path, self.current_labels)

#     # ---------- Navigation: images ----------

#     def next_image(self) -> None:
#         if self.autosave_on_nav:
#             self._silent_autosave()
#         if self.idx + 1 < len(self.image_source):
#             self._load_index(self.idx + 1)

#     def prev_image(self) -> None:
#         if self.autosave_on_nav:
#             self._silent_autosave()
#         if self.idx - 1 >= 0:
#             self._load_index(self.idx - 1)

#     # ---------- Navigation: segments (wrap across images) ----------

#     def next_segment(self) -> None:
#         if not self.instance_ids:
#             return
#         if self.selected_seg_id is None:
#             self._select_seg_id(self.instance_ids[0])
#             return

#         try:
#             i = self.instance_ids.index(self.selected_seg_id)
#         except ValueError:
#             i = 0

#         if i + 1 < len(self.instance_ids):
#             self._select_seg_id(self.instance_ids[i + 1])
#         else:
#             # wrap to next image
#             if self.idx + 1 < len(self.image_source):
#                 self.next_image()
#                 if self.instance_ids:
#                     self._select_seg_id(self.instance_ids[0])

#     def prev_segment(self) -> None:
#         if not self.instance_ids:
#             return
#         if self.selected_seg_id is None:
#             self._select_seg_id(self.instance_ids[0])
#             return

#         try:
#             i = self.instance_ids.index(self.selected_seg_id)
#         except ValueError:
#             i = 0

#         if i - 1 >= 0:
#             self._select_seg_id(self.instance_ids[i - 1])
#         else:
#             # wrap to prev image
#             if self.idx - 1 >= 0:
#                 self.prev_image()
#                 if self.instance_ids:
#                     self._select_seg_id(self.instance_ids[-1])

#     # ---------- Keyboard shortcuts ----------

#     def keyPressEvent(self, event):  # noqa: N802
#         key = event.key()

#         # segment nav
#         if key in (Qt.Key.Key_J, Qt.Key.Key_Down):
#             self.next_segment()
#             return
#         if key in (Qt.Key.Key_K, Qt.Key.Key_Up):
#             self.prev_segment()
#             return

#         # image nav
#         if key in (Qt.Key.Key_D, Qt.Key.Key_Right):
#             self.next_image()
#             return
#         if key in (Qt.Key.Key_A, Qt.Key.Key_Left):
#             self.prev_image()
#             return

#         # save
#         if key == Qt.Key.Key_S:
#             self.save_current()
#             return

#         # toggle overlay
#         if key == Qt.Key.Key_O:
#             self.show_overlay = not self.show_overlay
#             self._render_image()
#             return

#         super().keyPressEvent(event)


# # ----------------------- Entrypoint helper -----------------------

# def run_gui(image_source, paths: LabelingPaths, key_map: Optional[Dict[str, str]] = None) -> None:
#     """
#     image_source: random-access source with __len__ and get(idx)-> {image_id, image_pil}
#     """
#     app = QApplication([])
#     win = LabelingWindow(image_source=image_source, paths=paths, key_map=key_map)
#     win.show()
#     app.exec()



# # # labeling/gui_qt.py
# # #
# # # PyQt6 GUI for labeling per-instance segmentation outputs.
# # #
# # # Features:
# # # - Uses SegmentationResults.loadFromFile() directly (no wrapper class)
# # # - Lazy loads ONE image + ONE SegmentationResults at a time (random access via image_source)
# # # - Left panel list of instance ids (from npz ids), click to select
# # # - Click on image to select instance id
# # # - Keyboard shortcuts:
# # #     J / Down   : next segment (wrap to next image)
# # #     K / Up     : prev segment (wrap to prev image)
# # #     D / Right  : next image
# # #     A / Left   : prev image
# # #     S          : save
# # #     O          : toggle overlay on/off
# # #
# # # Output JSON:
# # #   output_root/full_results/<labeling_tag>/<image_id>.json
# # # Segmentation npz path:
# # #   output_root/segmentation_results/<segmentation_config_name>/<image_id>/segmentation_results.npz
# # #
# # # NOTE: You MUST edit the import path for SegmentationResults below.

# # from __future__ import annotations

# # from dataclasses import dataclass
# # from pathlib import Path
# # from typing import Any, Dict, List, Optional

# # import numpy as np
# # from PIL import Image

# # from PyQt6.QtCore import Qt
# # from PyQt6.QtGui import QImage, QPixmap
# # from PyQt6.QtWidgets import (
# #     QApplication,
# #     QComboBox,
# #     QDoubleSpinBox,
# #     QFormLayout,
# #     QHBoxLayout,
# #     QLabel,
# #     QLineEdit,
# #     QListWidget,
# #     QListWidgetItem,
# #     QMainWindow,
# #     QMessageBox,
# #     QPushButton,
# #     QSplitter,
# #     QTextEdit,
# #     QVBoxLayout,
# #     QWidget,
# # )

# # from .io import index_by_id, load_json_labels, merge_defaults_with_existing, save_json_labels
# # # from .schema import UNCERTAINTY_CHOICES, build_default_labels_for_segment_ids
# # from dataclasses import fields as dataclass_fields
# # from .schema import schema_fields_for_gui


# # from instance_segmentation.abs_instance_segmenter import SegmentationResults


# # # ----------------------- Utilities -----------------------

# # def pil_to_qpixmap(pil_img: Image.Image) -> QPixmap:
# #     rgb = pil_img.convert("RGB")
# #     arr = np.array(rgb)
# #     h, w, _ = arr.shape
# #     qimg = QImage(arr.data, w, h, 3 * w, QImage.Format.Format_RGB888)
# #     return QPixmap.fromImage(qimg.copy())


# # def seg_map_from_results(seg_results: SegmentationResults) -> np.ndarray:
# #     """
# #     Build HxW int64 map: -1 background, else instance-id at pixel.
# #     Overlaps resolved by "last wins" (later masks overwrite earlier).
# #     """
# #     ids = seg_results.ids.detach().cpu().numpy().astype(np.int64)      # (N,)
# #     masks = seg_results.masks.detach().cpu().numpy().astype(bool)      # (N,H,W)

# #     if masks.ndim != 3:
# #         raise ValueError(f"Expected masks shape (N,H,W), got {masks.shape}")

# #     n, h, w = masks.shape
# #     seg_map = np.full((h, w), -1, dtype=np.int64)
# #     for i in range(n):
# #         seg_map[masks[i]] = int(ids[i])
# #     return seg_map


# # def seg_id_at(seg_map: np.ndarray, x: int, y: int) -> int:
# #     h, w = seg_map.shape[:2]
# #     if 0 <= x < w and 0 <= y < h:
# #         return int(seg_map[y, x])
# #     return -1


# # def make_overlay(seg_map: np.ndarray, alpha: float = 0.45) -> Image.Image:
# #     """
# #     Deterministic color overlay per instance id.
# #     Background (-1) is fully transparent.
# #     """
# #     h, w = seg_map.shape
# #     ids = seg_map.astype(np.int64)

# #     # hash ids -> rgb
# #     r = (ids * 37) % 255
# #     g = (ids * 57) % 255
# #     b = (ids * 97) % 255
# #     overlay_rgb = np.stack([r, g, b], axis=-1).astype(np.uint8)

# #     # alpha channel: transparent for background
# #     alpha_mask = np.full((h, w), int(np.clip(alpha, 0.0, 1.0) * 255), dtype=np.uint8)
# #     alpha_mask[ids == -1] = 0

# #     pil_overlay = Image.fromarray(overlay_rgb, mode="RGB")
# #     pil_overlay.putalpha(Image.fromarray(alpha_mask, mode="L"))
# #     return pil_overlay


# # # ----------------------- Paths -----------------------

# # @dataclass
# # class LabelingPaths:
# #     output_root: Path
# #     segmentation_config_name: str
# #     labeling_tag: str

# #     def seg_npz_path(self, image_id: str) -> Path:
# #         return (
# #             self.output_root
# #             / "segmentation_results"
# #             / self.segmentation_config_name
# #             / image_id
# #             / "segmentation_results.npz"
# #         )

# #     def label_json_path(self, image_id: str) -> Path:
# #         return self.output_root / "full_results" / self.labeling_tag / f"{image_id}.json"


# # # ----------------------- Widgets -----------------------

# # class ClickableImageLabel(QLabel):
# #     """
# #     QLabel that forwards left-clicks to a callback as image-space (x,y).
# #     Note: We set the label size to the pixmap size to avoid scaling issues.
# #     """
# #     def __init__(self, parent=None):
# #         super().__init__(parent)
# #         self._click_cb = None
# #         self._pixmap: Optional[QPixmap] = None

# #     def setPixmap(self, pm: QPixmap) -> None:  # type: ignore[override]
# #         self._pixmap = pm
# #         super().setPixmap(pm)

# #     def set_click_callback(self, cb) -> None:
# #         self._click_cb = cb

# #     def mousePressEvent(self, event):  # noqa: N802
# #         if self._pixmap is None or self._click_cb is None:
# #             return
# #         if event.button() != Qt.MouseButton.LeftButton:
# #             return
# #         x = int(event.position().x())
# #         y = int(event.position().y())
# #         self._click_cb(x, y)


# # # ----------------------- Main Window -----------------------

# # class LabelingWindow(QMainWindow):
# #     """
# #     image_source: object supporting:
# #       - __len__() -> int
# #       - get(idx) -> item with fields:
# #           item.image_id: str
# #           item.image_pil: PIL.Image.Image (RGB)
# #           (additional fields are ignored by this GUI)

# #     Shortcuts:
# #       - Next segment: J / Down
# #       - Prev segment: K / Up
# #       - Next image:   D / Right
# #       - Prev image:   A / Left
# #       - Save:         S
# #       - Toggle overlay: O

# #     Segment navigation wraps across images:
# #       - next segment past last -> next image, first segment
# #       - prev segment past first -> prev image, last segment
# #     """

# #     def __init__(
# #         self,
# #         image_source,
# #         paths: LabelingPaths,
# #         key_map: Optional[Dict[str, str]] = None,
# #         id_key: str = "id",
# #         autosave_on_nav: bool = True,
# #         overlay_alpha: float = 0.45,
# #     ):
# #         super().__init__()
# #         self.setWindowTitle("Segmentation Labeling GUI")

# #         self.image_source = image_source
# #         self.paths = paths
# #         self.key_map = key_map or {}
# #         self.id_key = id_key
# #         self.autosave_on_nav = autosave_on_nav
# #         self.overlay_alpha = overlay_alpha

# #         self.idx = 0

# #         # Current image state (only one in memory)
# #         self.current_image_id: Optional[str] = None
# #         self.current_image_pil: Optional[Image.Image] = None

# #         # SegmentationResults stored directly
# #         self.seg_results: Optional[SegmentationResults] = None
# #         self._seg_map_cache: Optional[np.ndarray] = None  # derived for click+overlay
# #         self.instance_ids: List[int] = []

# #         # Label records for current image
# #         self.current_labels: List[Dict[str, Any]] = []
# #         self.current_label_map: Dict[str, Dict[str, Any]] = {}
# #         self.selected_seg_id: Optional[int] = None

# #         self.show_overlay: bool = True

# #         self.widgets_by_field: Dict[str, Any] = {}


# #         self._build_ui()
# #         # Load next image until there's one loaded without error (e.g. missing segmentation npz or corrupted file)

# #         nextIndex = 0
# #         loaded = False
# #         while nextIndex < len(self.image_source):
# #             try:
# #                 self._load_index(nextIndex)
# #                 loaded = True
# #                 break
# #             except Exception as e:
# #                 print(f"Error loading index {nextIndex}: {e}")
# #                 nextIndex += 1
# #         print("Initialized okay?", loaded)
# #         if (not loaded) and len(self.image_source) > 0:
# #             QMessageBox.critical(self, "Error", "Failed to load any images. Check console for details.")


# #     # ---------- UI ----------

# #     def _build_ui(self) -> None:
# #         root = QWidget()
# #         outer = QHBoxLayout(root)

# #         splitter = QSplitter(Qt.Orientation.Horizontal)

# #         # Left pane: status + segment list + image
# #         left = QWidget()
# #         left_layout = QVBoxLayout(left)

# #         self.info = QLabel("")
# #         left_layout.addWidget(self.info)

# #         self.segment_list = QListWidget()
# #         self.segment_list.currentItemChanged.connect(self._on_segment_list_changed)
# #         left_layout.addWidget(self.segment_list, stretch=1)

# #         self.image_label = ClickableImageLabel()
# #         self.image_label.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)
# #         self.image_label.set_click_callback(self._on_image_click)
# #         left_layout.addWidget(self.image_label, stretch=3)

# #         # Right pane: label form + buttons
# #         right = QWidget()
# #         right_layout = QVBoxLayout(right)

# #         self.seg_id_display = QLineEdit()
# #         self.seg_id_display.setReadOnly(True)

# #         self.persistence_hours = QDoubleSpinBox()
# #         self.persistence_hours.setRange(0.0, 1e9)
# #         self.persistence_hours.setDecimals(2)
# #         self.persistence_hours.setSingleStep(10.0)

# #         self.uncertainty = QComboBox()
# #         self.uncertainty.addItems(UNCERTAINTY_CHOICES)

# #         self.static_likelihood = QDoubleSpinBox()
# #         self.static_likelihood.setRange(0.0, 1.0)
# #         self.static_likelihood.setDecimals(3)
# #         self.static_likelihood.setSingleStep(0.05)

# #         self.notes = QTextEdit()

# #         form = QFormLayout()
# #         form.addRow("Selected segment id", self.seg_id_display)
# #         form.addRow("expected_persistence_hours", self.persistence_hours)
# #         form.addRow("persistence_uncertainty", self.uncertainty)
# #         form.addRow("static_likelihood", self.static_likelihood)
# #         form.addRow("notes", self.notes)
# #         right_layout.addLayout(form)

# #         btn_row = QHBoxLayout()
# #         self.prev_btn = QPushButton("Prev Image (A)")
# #         self.next_btn = QPushButton("Next Image (D)")
# #         self.save_btn = QPushButton("Save (S)")
# #         btn_row.addWidget(self.prev_btn)
# #         btn_row.addWidget(self.next_btn)
# #         btn_row.addWidget(self.save_btn)
# #         right_layout.addLayout(btn_row)

# #         self.prev_btn.clicked.connect(self.prev_image)
# #         self.next_btn.clicked.connect(self.next_image)
# #         self.save_btn.clicked.connect(self.save_current)

# #         # change handlers: write back
# #         self.persistence_hours.valueChanged.connect(self._write_back_selected)
# #         self.uncertainty.currentTextChanged.connect(self._write_back_selected)
# #         self.static_likelihood.valueChanged.connect(self._write_back_selected)
# #         self.notes.textChanged.connect(self._write_back_selected)

# #         splitter.addWidget(left)
# #         splitter.addWidget(right)
# #         splitter.setStretchFactor(0, 4)
# #         splitter.setStretchFactor(1, 3)

# #         outer.addWidget(splitter)
# #         self.setCentralWidget(root)

# #     # ---------- Loading ----------

# #     def _status_text(self) -> str:
# #         total = len(self.image_source)
# #         nseg = len(self.instance_ids)
# #         return f"Image {self.idx+1}/{total} | image_id={self.current_image_id} | segments={nseg}"

# #     def _load_index(self, idx: int) -> None:
# #         if idx < 0 or idx >= len(self.image_source):
# #             return

# #         self.idx = idx
# #         item = self.image_source.get(idx)

# #         if not getattr(item, "image_id", None):
# #             raise ValueError(
# #                 f"image_source.get({idx}) returned image_id=None/empty. "
# #                 "Labeling requires stable image_id."
# #             )

# #         self.current_image_id = str(item.image_id)
# #         self.current_image_pil = item.image_pil.convert("RGB")
# #         self.selected_seg_id = None

# #         # Load SegmentationResults directly
# #         seg_path = self.paths.seg_npz_path(self.current_image_id)
# #         self.seg_results, _ = SegmentationResults.loadFromFile(str(seg_path))

# #         # Cache seg_map for O(1) picking + overlay
# #         self._seg_map_cache = seg_map_from_results(self.seg_results)

# #         # Instance ids in the same order as saved
# #         ids_np = self.seg_results.ids.detach().cpu().numpy().astype(np.int64)
# #         self.instance_ids = [int(x) for x in ids_np.tolist()]

# #         # Load / merge labels JSON
# #         defaults = build_default_labels_for_segment_ids(self.instance_ids, key_map=self.key_map)
# #         json_path = self.paths.label_json_path(self.current_image_id)
# #         existing = load_json_labels(json_path)

# #         self.current_labels = merge_defaults_with_existing(defaults, existing, id_key=self.id_key)
# #         self.current_label_map = index_by_id(self.current_labels, id_key=self.id_key)

# #         self._populate_segment_list()
# #         self._render_image()
# #         self.info.setText(self._status_text())

# #         # auto-select first segment
# #         if self.instance_ids:
# #             self._select_seg_id(self.instance_ids[0])

# #     # ---------- Render ----------

# #     def _render_image(self) -> None:
# #         if self.current_image_pil is None:
# #             return
# #         base = self.current_image_pil.convert("RGBA")

# #         if self.show_overlay and self._seg_map_cache is not None:
# #             overlay = make_overlay(self._seg_map_cache, alpha=self.overlay_alpha)
# #             composed = Image.alpha_composite(base, overlay).convert("RGB")
# #         else:
# #             composed = base.convert("RGB")

# #         pm = pil_to_qpixmap(composed)
# #         # avoid scaling confusion: make label exactly pixmap size
# #         self.image_label.setFixedSize(pm.width(), pm.height())
# #         self.image_label.setPixmap(pm)

# #     # ---------- Segment list ----------

# #     def _populate_segment_list(self) -> None:
# #         self.segment_list.blockSignals(True)
# #         self.segment_list.clear()
# #         for sid in self.instance_ids:
# #             self.segment_list.addItem(QListWidgetItem(str(sid)))
# #         self.segment_list.blockSignals(False)

# #     def _on_segment_list_changed(self, current: QListWidgetItem, previous: QListWidgetItem) -> None:
# #         if current is None:
# #             return
# #         try:
# #             sid = int(current.text())
# #         except Exception:
# #             return
# #         self._select_seg_id(sid)

# #     def _sync_list_selection(self, sid: int) -> None:
# #         self.segment_list.blockSignals(True)
# #         for i in range(self.segment_list.count()):
# #             if self.segment_list.item(i).text() == str(sid):
# #                 self.segment_list.setCurrentRow(i)
# #                 break
# #         self.segment_list.blockSignals(False)

# #     # ---------- Click select ----------

# #     def _on_image_click(self, x: int, y: int) -> None:
# #         if self._seg_map_cache is None:
# #             return
# #         sid = seg_id_at(self._seg_map_cache, x, y)
# #         if sid >= 0:
# #             self._select_seg_id(sid)

# #     # ---------- Form ----------

# #     def _select_seg_id(self, sid: int) -> None:
# #         self.selected_seg_id = sid
# #         self.seg_id_display.setText(str(sid))
# #         self._sync_list_selection(sid)

# #         rec = self.current_label_map.get(str(sid))
# #         if rec is None:
# #             # create if missing
# #             rec = {self.id_key: str(sid)}
# #             self.current_labels.append(rec)
# #             self.current_label_map[str(sid)] = rec

# #         # populate widgets (block signals to avoid write-back loops)
# #         self.persistence_hours.blockSignals(True)
# #         self.static_likelihood.blockSignals(True)
# #         self.uncertainty.blockSignals(True)
# #         self.notes.blockSignals(True)

# #         self.persistence_hours.setValue(float(rec.get("expected_persistence_hours", 1000.0)))
# #         self.static_likelihood.setValue(float(rec.get("static_likelihood", 1.0)))

# #         unc = str(rec.get("persistence_uncertainty", "low"))
# #         if unc not in UNCERTAINTY_CHOICES:
# #             unc = "low"
# #         self.uncertainty.setCurrentText(unc)

# #         self.notes.setPlainText(str(rec.get("notes", "")))

# #         self.persistence_hours.blockSignals(False)
# #         self.static_likelihood.blockSignals(False)
# #         self.uncertainty.blockSignals(False)
# #         self.notes.blockSignals(False)

# #     def _write_back_selected(self) -> None:
# #         if self.selected_seg_id is None:
# #             return
# #         rec = self.current_label_map.get(str(self.selected_seg_id))
# #         if rec is None:
# #             return
# #         rec["expected_persistence_hours"] = float(self.persistence_hours.value())
# #         rec["persistence_uncertainty"] = str(self.uncertainty.currentText())
# #         rec["static_likelihood"] = float(self.static_likelihood.value())
# #         rec["notes"] = str(self.notes.toPlainText())

# #     # ---------- Save ----------

# #     def save_current(self) -> None:
# #         if self.current_image_id is None:
# #             return
# #         self._write_back_selected()
# #         json_path = self.paths.label_json_path(self.current_image_id)
# #         save_json_labels(json_path, self.current_labels)
# #         QMessageBox.information(self, "Saved", f"Saved labels:\n{json_path}")

# #     def _silent_autosave(self) -> None:
# #         if self.current_image_id is None:
# #             return
# #         self._write_back_selected()
# #         json_path = self.paths.label_json_path(self.current_image_id)
# #         save_json_labels(json_path, self.current_labels)

# #     # ---------- Navigation: images ----------

# #     def next_image(self) -> None:
# #         if self.autosave_on_nav:
# #             self._silent_autosave()
# #         if self.idx + 1 < len(self.image_source):
# #             self._load_index(self.idx + 1)

# #     def prev_image(self) -> None:
# #         if self.autosave_on_nav:
# #             self._silent_autosave()
# #         if self.idx - 1 >= 0:
# #             self._load_index(self.idx - 1)

# #     # ---------- Navigation: segments (wrap across images) ----------

# #     def next_segment(self) -> None:
# #         if not self.instance_ids:
# #             return
# #         if self.selected_seg_id is None:
# #             self._select_seg_id(self.instance_ids[0])
# #             return

# #         try:
# #             i = self.instance_ids.index(self.selected_seg_id)
# #         except ValueError:
# #             i = 0

# #         if i + 1 < len(self.instance_ids):
# #             self._select_seg_id(self.instance_ids[i + 1])
# #         else:
# #             # wrap to next image
# #             if self.idx + 1 < len(self.image_source):
# #                 self.next_image()
# #                 if self.instance_ids:
# #                     self._select_seg_id(self.instance_ids[0])

# #     def prev_segment(self) -> None:
# #         if not self.instance_ids:
# #             return
# #         if self.selected_seg_id is None:
# #             self._select_seg_id(self.instance_ids[0])
# #             return

# #         try:
# #             i = self.instance_ids.index(self.selected_seg_id)
# #         except ValueError:
# #             i = 0

# #         if i - 1 >= 0:
# #             self._select_seg_id(self.instance_ids[i - 1])
# #         else:
# #             # wrap to prev image
# #             if self.idx - 1 >= 0:
# #                 self.prev_image()
# #                 if self.instance_ids:
# #                     self._select_seg_id(self.instance_ids[-1])

# #     # ---------- Keyboard shortcuts ----------

# #     def keyPressEvent(self, event):  # noqa: N802
# #         key = event.key()

# #         # segment nav
# #         if key in (Qt.Key.Key_J, Qt.Key.Key_Down):
# #             self.next_segment()
# #             return
# #         if key in (Qt.Key.Key_K, Qt.Key.Key_Up):
# #             self.prev_segment()
# #             return

# #         # image nav
# #         if key in (Qt.Key.Key_D, Qt.Key.Key_Right):
# #             self.next_image()
# #             return
# #         if key in (Qt.Key.Key_A, Qt.Key.Key_Left):
# #             self.prev_image()
# #             return

# #         # save
# #         if key == Qt.Key.Key_S:
# #             self.save_current()
# #             return

# #         # toggle overlay
# #         if key == Qt.Key.Key_O:
# #             self.show_overlay = not self.show_overlay
# #             self._render_image()
# #             return

# #         super().keyPressEvent(event)


# # # ----------------------- Entrypoint helper -----------------------

# # def run_gui(image_source, paths: LabelingPaths, key_map: Optional[Dict[str, str]] = None) -> None:
# #     """
# #     image_source: random-access source with __len__ and get(idx)-> {image_id, image_pil}
# #     """
# #     app = QApplication([])
# #     win = LabelingWindow(image_source=image_source, paths=paths, key_map=key_map)
# #     win.show()
# #     app.exec()

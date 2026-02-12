# manual_labeling/gui_qt.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw

from tqdm import tqdm

from spf_io.persistence_io import readPersistenceData, writePersistenceData, PERSISTENCE_FILE_NAME

import random

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

# from .io import index_by_id, load_json_labels, merge_defaults_with_existing, save_json_labels
from .io import index_by_id, merge_defaults_with_existing
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

def seg_ids_at(masks: np.ndarray, ids: np.ndarray, x: int, y: int) -> List[int]:
    """
    Return all segment ids whose mask contains pixel (x,y).
    masks: (N,H,W) bool
    ids:   (N,) int64
    """
    n, h, w = masks.shape
    if not (0 <= x < w and 0 <= y < h):
        return []
    hit = masks[:, y, x]
    return [int(i) for i in ids[hit].tolist()]


def make_single_mask_overlay_from_mask(mask: np.ndarray, seg_id: int, alpha: float = 0.55) -> Image.Image:
    """
    mask: (H,W) bool for the selected instance
    """
    h, w = mask.shape

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


def mask_bbox_xyxy_from_mask(mask: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    ys, xs = np.where(mask)
    if xs.size == 0 or ys.size == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())



def make_single_mask_overlay(seg_map: np.ndarray, seg_id: int, alpha: float = 0.55) -> Image.Image:
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
    ys, xs = np.where(seg_map == int(seg_id))
    if xs.size == 0 or ys.size == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def draw_bounding_box(
    pil_img: Image.Image,
    bbox_xyxy: Tuple[int, int, int, int],
    color: Tuple[int, int, int] = (255, 255, 0),
    width: int = 4,
    pad: int = 2,
    with_black_outline: bool = True,
) -> Image.Image:
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


def short_rel_name(s: str, *, max_parts: int = 2) -> str:
    """
    Display a "relative-like" short name:
    - If absolute path: try relative_to(CWD), else last N parts
    - If already relative: show last N parts if long, else as-is
    """
    p = Path(str(s))
    try:
        if p.is_absolute():
            try:
                rel = p.relative_to(Path.cwd())
                parts = rel.parts
            except Exception:
                parts = p.parts
        else:
            parts = p.parts
    except Exception:
        return str(s)

    if len(parts) <= max_parts:
        return "/".join(parts)
    return "/".join(parts[-max_parts:])


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

    def labelJsonDir(self, image_id: str) -> Path:
        return self.output_root / "full_results" / self.labeling_tag / image_id

    def label_json_path(self, image_id: str, autosave: bool = False) -> Path:
        jsonDir = self.labelJsonDir(image_id)
        if autosave:
            return jsonDir / "autosave_persistence_parameters.json"
        return jsonDir / PERSISTENCE_FILE_NAME


# ----------------------- Widgets -----------------------

class ClickableImageLabel(QLabel):
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
      - Next file:    D / Right
      - Prev file:    A / Left
      - Save:         S
      - Toggle overlay: O
      - Open pending selection: Enter / Return
      - Double-click file list: open immediately
      - Apply preset: Ctrl+1 (static), Ctrl+2 (dynamic), Ctrl+3 (temporal)
    """

    def __init__(
        self,
        image_source,
        paths: LabelingPaths,
        key_map: Optional[Dict[str, str]] = None,
        id_key: str = "id",
        autosave_on_nav: bool = True,
        overlay_alpha: float = 0.55,
        differentiate_labeled: bool = True,
    ):
        super().__init__()
        self.setWindowTitle("Segmentation Labeling GUI")

        # Confidence presets — update values as you like
        self.confidence_presets = {
            "vlow": 0.2,
            "low": 0.35,
            "med": 0.5,
            "high": 0.7,
            "vhigh": 0.9,
        }

        self.max_display_w = 1100
        self.max_display_h = 600
        self._display_scale = 1.0  # displayed_size / original_size



        self.image_source = image_source
        self.paths = paths
        self.key_map = key_map or {}
        self.id_key = id_key
        self.autosave_on_nav = autosave_on_nav
        self.overlay_alpha = overlay_alpha
        self.differentiate_labeled = bool(differentiate_labeled)
        self._masks_cache: Optional[np.ndarray] = None   # (N,H,W) bool
        self._ids_cache: Optional[np.ndarray] = None     # (N,) int64


        # Schema-driven form
        self._form_specs: List[Dict[str, Any]] = schema_fields_for_gui()
        self._widgets_by_field: Dict[str, Any] = {}
        self._required_label_fields: List[str] = self._compute_required_fields()

        # Ordering / browser mapping
        self._ordered_indices: List[int] = []
        self._index_to_row: Dict[int, int] = {}
        self._row_to_index: Dict[int, int] = {}
        self._image_id_cache: Dict[int, str] = {}

        # "pending selection" (click in file list does NOT load)
        self._pending_idx: Optional[int] = None

        # Current position in ordered list
        self._order_pos: int = 0

        # Current image state
        self.current_image_id: Optional[str] = None
        self.current_full_img_path: Optional[str] = None
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

        self._build_ui()
        self.resize(1600, 900)


        # Build ordering + fill browser
        self._rebuild_file_order_and_list()

        # Load first item (unlabeled-first if differentiate_labeled, else original order)
        if self._ordered_indices:
            self._load_by_order_pos(0)
        else:
            QMessageBox.critical(self, "Error", "No images in image_source.")

    # ---------- Schema helpers ----------

    def _compute_required_fields(self) -> List[str]:
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

    # ---------- File ordering / browser ----------

    # def _get_image_id_for_index(self, idx: int) -> str:
    #     if idx in self._image_id_cache:
    #         return self._image_id_cache[idx]
    #     item = self.image_source.get(idx)
    #     image_id = str(getattr(item, "image_id"))
    #     if not image_id:
    #         raise ValueError(f"image_source.get({idx}) returned empty image_id")
    #     self._image_id_cache[idx] = image_id
    #     return image_id

    def _get_image_id_for_index(self, idx: int) -> str:
        if idx in self._image_id_cache:
            return self._image_id_cache[idx]

        if hasattr(self.image_source, "get_id"):
            image_id = str(self.image_source.get_id(idx))
        else:
            # fallback (slow): loads image
            item = self.image_source.get(idx)
            image_id = str(getattr(item, "image_id"))

        if not image_id:
            raise ValueError(f"empty image_id at idx={idx}")

        self._image_id_cache[idx] = image_id
        return image_id


    def _file_quick_status(self, image_id: str) -> str:
        """
        Used ONLY when differentiate_labeled=True.
        Fast heuristic: labeled iff the label JSON file exists.
        - UNLABELED: json missing
        - LABELED:   json exists
        """
        json_path = self.paths.label_json_path(image_id)
        return "LABELED" if json_path.exists() else "UNLABELED"


    # def _file_quick_status(self, image_id: str) -> str:
    #     """
    #     Used ONLY when differentiate_labeled=True.
    #     'UNLABELED' if JSON missing or has no fully-labeled entries.
    #     """
    #     json_path = self.paths.label_json_path(image_id)
    #     existing = load_json_labels(json_path)
    #     if not existing:
    #         return "UNLABELED"
    #     labeled_any = any(self._is_record_labeled(rec) for rec in existing if isinstance(rec, dict))
    #     return "LABELED" if labeled_any else "UNLABELED"

    def _format_file_row_text(
        self,
        idx: int,
        *,
        status: Optional[str] = None,
        missing: Optional[int] = None,
        total_masks: Optional[int] = None,
    ) -> str:
        image_id = self._get_image_id_for_index(idx)
        # name = short_rel_name(image_id, max_parts=2)
        name = self._display_name_for_index(idx)


        if not self.differentiate_labeled:
            # speed mode: no labeled/unlabeled check
            return name

        if status is None:
            status = self._file_quick_status(image_id)

        suffix = ""
        if missing is not None and total_masks is not None:
            suffix = f"   (missing {missing}/{total_masks})"

        return f"{status:<9}  {name}{suffix}"


    def _chunk_shuffle_indices(self, indices: List[int], chunk_size: int, seed: Optional[int] = None) -> List[int]:
        if chunk_size <= 1:
            return indices[:]
        rng = random.Random(seed)
        chunks = [indices[i:i + chunk_size] for i in range(0, len(indices), chunk_size)]
        rng.shuffle(chunks)
        out: List[int] = []
        for c in chunks:
            out.extend(c)
        return out


    def _rebuild_file_order_and_list(self) -> None:
        n = len(self.image_source)
        indices = list(range(n))

        chunk_size = 10
        seed = 123  # pick any fixed int for reproducibility (or None for “new each run”)


        if not self.differentiate_labeled:
            # Speed mode: preserve original order; do not touch filesystem for labels.
            self._ordered_indices = self._chunk_shuffle_indices(indices=indices, chunk_size=chunk_size, seed=seed)
        else:
            # Non-speed mode: order by existence of label file only (no JSON reads).
            unlabeled: List[int] = []
            labeled: List[int] = []
            for idx in tqdm(indices):
                try:
                    image_id = self._get_image_id_for_index(idx)
                    json_path = self.paths.label_json_path(image_id)
                    (labeled if json_path.exists() else unlabeled).append(idx)
                except Exception:
                    # if anything weird, put it at the end
                    labeled.append(idx)

            unlabeled = self._chunk_shuffle_indices(unlabeled, chunk_size=chunk_size, seed=seed)
            labeled = self._chunk_shuffle_indices(labeled, chunk_size=chunk_size, seed=seed)

            self._ordered_indices = unlabeled + labeled



        # if not self.differentiate_labeled:
        #     # Speed mode: preserve original order; do not read any JSON.
        #     self._ordered_indices = indices
        # else:
        #     unlabeled: List[int] = []
        #     labeled: List[int] = []
        #     for idx in indices:
        #         try:
        #             image_id = self._get_image_id_for_index(idx)
        #             status = self._file_quick_status(image_id)
        #             (unlabeled if status == "UNLABELED" else labeled).append(idx)
        #         except Exception:
        #             labeled.append(idx)
        #     self._ordered_indices = unlabeled + labeled

        self._index_to_row.clear()
        self._row_to_index.clear()

        self.file_list.blockSignals(True)
        self.file_list.clear()

        for row, idx in enumerate(self._ordered_indices):
            item_text = self._format_file_row_text(idx)
            self.file_list.addItem(QListWidgetItem(item_text))
            self._index_to_row[idx] = row
            self._row_to_index[row] = idx

        self.file_list.blockSignals(False)

        self._pending_idx = None
        self.pending_label.setText("Pending: (none)")

    def _update_file_list_row_text(
        self,
        idx: int,
        *,
        status: Optional[str] = None,
        missing: Optional[int] = None,
        total_masks: Optional[int] = None,
    ) -> None:
        row = self._index_to_row.get(idx)
        if row is None:
            return
        item = self.file_list.item(row)
        if item is None:
            return
        item.setText(self._format_file_row_text(idx, status=status, missing=missing, total_masks=total_masks))

    def _sync_file_list_selection(self, idx: int) -> None:
        row = self._index_to_row.get(idx)
        if row is None:
            return
        self.file_list.blockSignals(True)
        self.file_list.setCurrentRow(row)
        self.file_list.blockSignals(False)

    def _apply_preset(self, preset_key: str) -> None:
        """
        Apply a reusable preset to the current segment's form fields.

        - Presets are defined in schema.py via get_label_presets().
        - Only fields present in the current schema form are applied.
        - Readonly fields are not modified.
        - After applying, it writes back into the current label record.
        """
        if self.selected_seg_id is None:
            QMessageBox.information(self, "No mask selected", "Select a mask before applying a preset.")
            return

        # Import here to avoid circular import issues
        from .schema import get_label_presets

        presets = get_label_presets()
        if preset_key not in presets:
            QMessageBox.warning(self, "Unknown preset", f"Preset '{preset_key}' not found in schema presets.")
            return

        preset_values = presets[preset_key]

        # Apply to widgets (block signals during bulk update)
        # We only set keys that exist in the schema form.
        for spec in self._form_specs:
            name = spec.get("name")
            kind = spec.get("kind")

            if not name or name == self.id_key:
                continue
            if kind == "readonly":
                continue
            if name not in preset_values:
                continue
            if name not in self._widgets_by_field:
                continue

            w = self._widgets_by_field[name]
            self._set_widget_value_blocked(w, kind, preset_values[name], spec)

        # Persist changes into the current record
        self._write_back_selected()

        # Re-render + status update
        # self._render_image()
        # self.info.setText(self._status_text())


    # ---------- UI ----------
    def _build_ui(self) -> None:
        root = QWidget()
        outer = QHBoxLayout(root)
        splitter = QSplitter(Qt.Orientation.Horizontal)

        # Left: info + file browser + pending controls + mask list + image
        left = QWidget()
        left_layout = QVBoxLayout(left)

        self.info = QLabel("")
        left_layout.addWidget(self.info)

        # Add a second info box
        self.debug_info = QLabel("")
        left_layout.addWidget(self.debug_info)

        self.file_list = QListWidget()
        self.file_list.currentItemChanged.connect(self._on_file_list_selection_changed)
        self.file_list.itemDoubleClicked.connect(self._open_pending_selection)  # double click opens
        left_layout.addWidget(self.file_list, stretch=1)

        pending_row = QHBoxLayout()
        self.pending_label = QLabel("Pending: (none)")
        self.open_selected_btn = QPushButton("Open Selected (Enter)")
        pending_row.addWidget(self.pending_label, stretch=1)
        pending_row.addWidget(self.open_selected_btn)
        left_layout.addLayout(pending_row)
        self.open_selected_btn.clicked.connect(self._open_pending_selection)

        self.segment_list = QListWidget()
        self.segment_list.currentItemChanged.connect(self._on_segment_list_changed)
        left_layout.addWidget(self.segment_list, stretch=1)

        self.image_label = ClickableImageLabel()
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)
        self.image_label.set_click_callback(self._on_image_click)
        left_layout.addWidget(self.image_label, stretch=3)

        # Right: form + preset buttons + mask nav + file nav + save
        right = QWidget()
        right_layout = QVBoxLayout(right)

        form = QFormLayout()
        self._build_schema_form(form)
        right_layout.addLayout(form)

        # --- NEW: preset buttons row ---
        preset_row = QHBoxLayout()
        self.static_btn = QPushButton("Static (Ctrl+1)")
        self.months_btn = QPushButton("Weeks-to-Months (Ctrl+2)")
        self.weeks_btn = QPushButton("Days-to-Weeks (Ctrl+3)")
        self.days_btn = QPushButton("Multiple Days (Ctrl+4)")
        self.next_day_btn = QPushButton("Next Day (Ctrl+5)")
        self.several_hours_btn = QPushButton("Several Hours (Ctrl+6)")
        self.couple_hours_btn = QPushButton("Couple Hours (Ctrl+7)")
        self.under_hour_btn = QPushButton("Under Hour (Ctrl+8)")
        self.under_30min_btn = QPushButton("Under 30min (Ctrl+9)")
        self.under_10min_btn = QPushButton("Under 10min (Ctrl+0)")
        self.under_2min_btn = QPushButton("Under 2min (Ctrl+-)")
        self.dynamic_btn = QPushButton("Dynamic (Ctrl+=)")
        # self.dynamic_btn = QPushButton("Dynamic (Ctrl+2)")

        preset_layout = QVBoxLayout()


        all_buttons = [
            self.static_btn,
            self.months_btn,
            self.weeks_btn,
            self.days_btn,
            self.next_day_btn,
            self.several_hours_btn,
            self.couple_hours_btn,
            self.under_hour_btn,
            self.under_30min_btn,
            self.under_10min_btn,
            self.under_2min_btn,
            self.dynamic_btn,
        ]

        # Adds rows dynamically: max 3 per row
        self._add_buttons_in_rows(preset_layout, all_buttons, max_per_row=3)

        right_layout.addLayout(preset_layout)

        # preset_row.addWidget(self.static_btn)
        # preset_row.addWidget(self.months_btn)
        # preset_row.addWidget(self.weeks_btn)
        # preset_row.addWidget(self.days_btn)
        # preset_row.addWidget(self.next_day_btn)
        # preset_row.addWidget(self.several_hours_btn)
        # preset_row.addWidget(self.couple_hours_btn)
        # preset_row.addWidget(self.under_hour_btn)
        # preset_row.addWidget(self.under_30min_btn)
        # preset_row.addWidget(self.under_10min_btn)
        # preset_row.addWidget(self.under_2min_btn)
        # preset_row.addWidget(self.static_btn)
        # preset_row.addWidget(self.dynamic_btn)
        # right_layout.addLayout(preset_row)



        # self.static_btn.clicked.connect(lambda: self._apply_preset("static"))
        # self.dynamic_btn.clicked.connect(lambda: self._apply_preset("dynamic"))
        self.static_btn.clicked.connect(lambda: self._apply_preset("static"))
        self.months_btn.clicked.connect(lambda: self._apply_preset("months"))
        self.weeks_btn.clicked.connect(lambda: self._apply_preset("weeks"))
        self.days_btn.clicked.connect(lambda: self._apply_preset("days"))
        self.next_day_btn.clicked.connect(lambda: self._apply_preset("next_day"))
        self.several_hours_btn.clicked.connect(lambda: self._apply_preset("several_hours"))
        self.couple_hours_btn.clicked.connect(lambda: self._apply_preset("couple_hours"))
        self.under_hour_btn.clicked.connect(lambda: self._apply_preset("under_hour"))
        self.under_30min_btn.clicked.connect(lambda: self._apply_preset("under_30min"))
        self.under_10min_btn.clicked.connect(lambda: self._apply_preset("under_10min"))
        self.under_2min_btn.clicked.connect(lambda: self._apply_preset("under_2min"))
        self.dynamic_btn.clicked.connect(lambda: self._apply_preset("dynamic"))

        # --- Confidence buttons row ---
        # conf_row = QHBoxLayout()
        conf_preset_layout = QVBoxLayout()
        self.conf_vlow_btn = QPushButton("V Low (Q)")
        self.conf_low_btn  = QPushButton("Low (W)")
        self.conf_med_btn  = QPushButton("Med (R)")
        self.conf_high_btn = QPushButton("High (T)")
        self.conf_vhigh_btn = QPushButton("V High (Y)")

        conf_buttons = [
            self.conf_vlow_btn,
            self.conf_low_btn,
            self.conf_med_btn,
            self.conf_high_btn,
            self.conf_vhigh_btn,
        ]
        self._add_buttons_in_rows(conf_preset_layout, conf_buttons, max_per_row=3)
        right_layout.addLayout(conf_preset_layout)

        # conf_row.addWidget(self.conf_vlow_btn)
        # conf_row.addWidget(self.conf_low_btn)
        # conf_row.addWidget(self.conf_med_btn)
        # conf_row.addWidget(self.conf_high_btn)
        # conf_row.addWidget(self.conf_vhigh_btn)
        # right_layout.addLayout(conf_row)

        self.conf_vlow_btn.clicked.connect(lambda: self._apply_confidence_preset("vlow"))
        self.conf_low_btn.clicked.connect(lambda: self._apply_confidence_preset("low"))
        self.conf_med_btn.clicked.connect(lambda: self._apply_confidence_preset("med"))
        self.conf_high_btn.clicked.connect(lambda: self._apply_confidence_preset("high"))
        self.conf_vhigh_btn.clicked.connect(lambda: self._apply_confidence_preset("vhigh"))


        mask_row = QHBoxLayout()
        self.prev_mask_btn = QPushButton("Prev Mask (K / [)")
        self.next_mask_btn = QPushButton("Next Mask (J / ])")
        mask_row.addWidget(self.prev_mask_btn)
        mask_row.addWidget(self.next_mask_btn)
        right_layout.addLayout(mask_row)
        self.prev_mask_btn.clicked.connect(self.prev_mask)
        self.next_mask_btn.clicked.connect(self.next_mask)

        btn_row = QHBoxLayout()
        self.prev_btn = QPushButton("Prev File (A)")
        self.next_btn = QPushButton("Next File (D)")
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

    def _add_buttons_in_rows(self, parent_layout, buttons, max_per_row=3):
        """
        Add QPushButtons into a parent layout using multiple rows,
        with at most max_per_row buttons per row.
        """
        row = None
        count = 0

        for btn in buttons:
            if count % max_per_row == 0:
                # start a new row
                row = QHBoxLayout()
                parent_layout.addLayout(row)
            row.addWidget(btn)
            count += 1


    # def _build_ui(self) -> None:
    #     root = QWidget()
    #     outer = QHBoxLayout(root)
    #     splitter = QSplitter(Qt.Orientation.Horizontal)

    #     # Left: info + file browser + pending controls + mask list + image
    #     left = QWidget()
    #     left_layout = QVBoxLayout(left)

    #     self.info = QLabel("")
    #     left_layout.addWidget(self.info)

    #     self.file_list = QListWidget()
    #     self.file_list.currentItemChanged.connect(self._on_file_list_selection_changed)
    #     self.file_list.itemDoubleClicked.connect(self._open_pending_selection)  # double click opens
    #     left_layout.addWidget(self.file_list, stretch=1)

    #     pending_row = QHBoxLayout()
    #     self.pending_label = QLabel("Pending: (none)")
    #     self.open_selected_btn = QPushButton("Open Selected (Enter)")
    #     pending_row.addWidget(self.pending_label, stretch=1)
    #     pending_row.addWidget(self.open_selected_btn)
    #     left_layout.addLayout(pending_row)

    #     self.open_selected_btn.clicked.connect(self._open_pending_selection)

    #     self.segment_list = QListWidget()
    #     self.segment_list.currentItemChanged.connect(self._on_segment_list_changed)
    #     left_layout.addWidget(self.segment_list, stretch=1)

    #     self.image_label = ClickableImageLabel()
    #     self.image_label.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)
    #     self.image_label.set_click_callback(self._on_image_click)
    #     left_layout.addWidget(self.image_label, stretch=3)

    #     # Right: form + mask nav + file nav + save
    #     right = QWidget()
    #     right_layout = QVBoxLayout(right)

    #     form = QFormLayout()
    #     self._build_schema_form(form)
    #     right_layout.addLayout(form)

    #     mask_row = QHBoxLayout()
    #     self.prev_mask_btn = QPushButton("Prev Mask (K / [)")
    #     self.next_mask_btn = QPushButton("Next Mask (J / ])")
    #     mask_row.addWidget(self.prev_mask_btn)
    #     mask_row.addWidget(self.next_mask_btn)
    #     right_layout.addLayout(mask_row)
    #     self.prev_mask_btn.clicked.connect(self.prev_mask)
    #     self.next_mask_btn.clicked.connect(self.next_mask)

    #     btn_row = QHBoxLayout()
    #     self.prev_btn = QPushButton("Prev File (A)")
    #     self.next_btn = QPushButton("Next File (D)")
    #     self.save_btn = QPushButton("Save (S)")
    #     btn_row.addWidget(self.prev_btn)
    #     btn_row.addWidget(self.next_btn)
    #     btn_row.addWidget(self.save_btn)
    #     right_layout.addLayout(btn_row)

    #     self.prev_btn.clicked.connect(self.prev_image)
    #     self.next_btn.clicked.connect(self.next_image)
    #     self.save_btn.clicked.connect(self.save_current)

    #     splitter.addWidget(left)
    #     splitter.addWidget(right)
    #     splitter.setStretchFactor(0, 4)
    #     splitter.setStretchFactor(1, 3)

    #     outer.addWidget(splitter)
    #     self.setCentralWidget(root)

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

    # ---------- Status / loading ----------

    def _apply_confidence_preset(self, key: str) -> None:
        if self.selected_seg_id is None:
            QMessageBox.information(self, "No mask selected", "Select a mask first.")
            return

        if key not in self.confidence_presets:
            QMessageBox.warning(self, "Unknown preset", f"Unknown confidence preset '{key}'")
            return

        value = self.confidence_presets[key]

        # Find the confidence field defined in schema
        for spec in self._form_specs:
            if spec["name"] == "persistence_conf":   # <-- If your field name is different, change here
                w = self._widgets_by_field["persistence_conf"]
                kind = spec.get("kind")
                self._set_widget_value_blocked(w, kind, value, spec)
                break


        self._write_back_selected()
        self._render_image()
        self.info.setText(self._status_text())



    def _status_text(self) -> str:
        total = len(self._ordered_indices)
        pos = self._order_pos + 1 if self._ordered_indices else 0
        nseg = len(self.instance_ids)
        sid = self.selected_seg_id
        missing = len(self._missing_label_seg_ids()) if self.instance_ids else 0
        return f"File {pos}/{total} | image_id={short_rel_name(self.current_image_id or '', max_parts=2)} | masks={nseg} | selected={sid} | missing_labels={missing}"

    def _load_by_order_pos(self, order_pos: int) -> None:
        if not self._ordered_indices:
            return
        order_pos = max(0, min(order_pos, len(self._ordered_indices) - 1))
        self._order_pos = order_pos
        idx = self._ordered_indices[order_pos]
        self._load_index(idx)
        self._sync_file_list_selection(idx)

    def _load_index(self, idx: int) -> None:
        item = self.image_source.get(idx)
        image_id = str(getattr(item, "image_id"))
        if not image_id:
            raise ValueError(f"image_source.get({idx}) returned empty image_id")


        self.current_image_id = image_id
        self.current_full_img_path = item.path
        self.current_image_pil = item.image_pil.convert("RGB")
        self.selected_seg_id = None

        seg_path = self.paths.seg_npz_path(self.current_image_id)
        self.seg_results, _ = SegmentationResults.loadFromFile(str(seg_path))

        self.seg_results, _ = SegmentationResults.loadFromFile(str(seg_path))

        ids_np = self.seg_results.ids.detach().cpu().numpy().astype(np.int64)      # (N,)
        masks_np = self.seg_results.masks.detach().cpu().numpy().astype(bool)      # (N,H,W)

        self._ids_cache = ids_np
        self._masks_cache = masks_np

        # seg_map is optional now; keep if you want debug/compat
        self._seg_map_cache = seg_map_from_results(self.seg_results)

        self.instance_ids = [int(x) for x in ids_np.tolist()]


        # self._seg_map_cache = seg_map_from_results(self.seg_results)

        # ids_np = self.seg_results.ids.detach().cpu().numpy().astype(np.int64)
        # self.instance_ids = [int(x) for x in ids_np.tolist()]

        defaults = build_default_labels_for_segment_ids(self.instance_ids, key_map=self.key_map)
        json_path = self.paths.label_json_path(self.current_image_id)
        autosave_json_path = self.paths.label_json_path(self.current_image_id, autosave=True)

        if (json_path.exists()):
            # existing = load_json_labels(json_path)
            existing = readPersistenceData(json_path)
        elif (autosave_json_path.exists()):
            # existing = load_json_labels(json_path)
            existing = readPersistenceData(autosave_json_path)
        else:
            existing = []

        self.current_labels = merge_defaults_with_existing(defaults, existing, id_key=self.id_key)
        self.current_label_map = index_by_id(self.current_labels, id_key=self.id_key)

        self._populate_segment_list()

        if self.instance_ids:
            self._select_seg_id(self.instance_ids[0])
        else:
            self._render_image()

        self.info.setText(self._status_text())

        # If differentiating, update row with missing count now that masks are loaded
        if self.differentiate_labeled:
            missing = len(self._missing_label_seg_ids())
            status = "LABELED" if (len(self._filtered_labeled_records()) > 0) else "UNLABELED"
            self._update_file_list_row_text(idx, status=status, missing=missing, total_masks=len(self.instance_ids))

    # ---------- Rendering ----------
    def _render_image(self) -> None:
        if self.current_image_pil is None:
            return

        base = self.current_image_pil.convert("RGBA")
        composed = base

        if (
            self.show_overlay
            and self.selected_seg_id is not None
            and self._masks_cache is not None
            and self._ids_cache is not None
        ):
            # find the mask for selected id
            sid = int(self.selected_seg_id)
            idxs = np.where(self._ids_cache == sid)[0]
            if idxs.size > 0:
                m = self._masks_cache[int(idxs[0])]  # (H,W) bool

                overlay = make_single_mask_overlay_from_mask(m, sid, alpha=self.overlay_alpha)
                composed = Image.alpha_composite(base, overlay)

                bbox = mask_bbox_xyxy_from_mask(m)
                if bbox is not None:
                    composed_rgb = draw_bounding_box(
                        composed.convert("RGB"),
                        bbox,
                        color=(255, 255, 0),
                        width=4,
                        pad=2,
                        with_black_outline=True,
                    )
                    composed = composed_rgb.convert("RGBA")

        pm = pil_to_qpixmap(composed.convert("RGB"))

        # --- scale to fit a maximum display size ---
        orig_w, orig_h = pm.width(), pm.height()
        scale = min(self.max_display_w / orig_w, self.max_display_h / orig_h, 1.0)
        self._display_scale = float(scale)

        disp_w = int(orig_w * scale)
        disp_h = int(orig_h * scale)

        pm_scaled = pm.scaled(
            disp_w,
            disp_h,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )

        # IMPORTANT: do NOT set fixed size to original pixels
        self.image_label.setPixmap(pm_scaled)
        self.image_label.setFixedSize(disp_w, disp_h)



    # ---------- File browser selection/confirm ----------

    def _on_file_list_selection_changed(self, current: QListWidgetItem, previous: QListWidgetItem) -> None:
        row = self.file_list.currentRow()
        if row < 0:
            self._pending_idx = None
            self.pending_label.setText("Pending: (none)")
            return
        idx = self._row_to_index.get(row)
        if idx is None:
            self._pending_idx = None
            self.pending_label.setText("Pending: (none)")
            return

        self._pending_idx = idx
        image_id = self._get_image_id_for_index(idx)
        self.pending_label.setText(f"Pending: {short_rel_name(image_id, max_parts=2)}")

    def _open_pending_selection(self) -> None:
        if self._pending_idx is None:
            return

        # Optional silent autosave before switching
        if self.autosave_on_nav:
            self._silent_autosave()

        idx = self._pending_idx
        try:
            self._order_pos = self._ordered_indices.index(idx)
        except ValueError:
            self._order_pos = 0

        self._load_index(idx)

    # ---------- Mask list ----------

    def _populate_segment_list(self) -> None:
        self.segment_list.blockSignals(True)
        self.segment_list.clear()
        for sid in self.instance_ids:
            # Skip all segments that have area less than 0.001 * the image area, to avoid cluttering the segment list with tiny masks that are hard to label and likely not important.
            # if self._seg_map_cache is not None:
            #     mask_size = np.sum(self._seg_map_cache == sid)
            #     img_area = self._seg_map_cache.shape[0] * self._seg_map_cache.shape[1]
            #     if img_area > 0 and mask_size < 0.001 * img_area:
            #         continue
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

    def _sync_mask_list_selection(self, sid: int) -> None:
        self.segment_list.blockSignals(True)
        for i in range(self.segment_list.count()):
            if self.segment_list.item(i).text() == str(sid):
                self.segment_list.setCurrentRow(i)
                break
        self.segment_list.blockSignals(False)

    # ---------- Click select ----------
    def _on_image_click(self, x: int, y: int) -> None:
        # Map from displayed coordinates back to original image coordinates
        if self._display_scale <= 0:
            return
        ox = int(x / self._display_scale)
        oy = int(y / self._display_scale)

        # If you switched to overlap-safe selection earlier:
        if self._masks_cache is not None and self._ids_cache is not None:
            hits = seg_ids_at(self._masks_cache, self._ids_cache, ox, oy)
            if not hits:
                return
            hits = sorted(hits)
            if len(hits) == 1:
                self._select_seg_id(hits[0])
                return
            if self.selected_seg_id in hits:
                i = hits.index(self.selected_seg_id)
                self._select_seg_id(hits[(i + 1) % len(hits)])
            else:
                self._select_seg_id(hits[0])
            return

        # Fallback to seg_map if you still rely on it
        if self._seg_map_cache is None:
            return
        sid = seg_id_at(self._seg_map_cache, ox, oy)
        if sid >= 0:
            self._select_seg_id(sid)


    # ---------- Form read/write ----------

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
        self._sync_mask_list_selection(int(sid))

        # Print the size of the mask for the selected segment in the info bar (for debugging)
        # Also note if any of the bounding box values exceed the image dimensions, which may indicate a bug in the seg map generation

        if self._seg_map_cache is not None:

            # debugText = f"selected mask id: {sid}"
            # Print the absolute path to the segmentation NPZ file being loaded, to help debug issues with incorrect seg maps being loaded
            debugText = f"selected mask id: {sid} "
            mask_size = None
            bbox = None

            if self._masks_cache is not None and self._ids_cache is not None:
                idxs = np.where(self._ids_cache == int(sid))[0]
                if idxs.size > 0:
                    m = self._masks_cache[int(idxs[0])]  # (H,W) bool
                    mask_size = int(m.sum())
                    bbox = mask_bbox_xyxy_from_mask(m)

            if mask_size is None:
                # fallback (non-overlap-safe) if caches aren't available
                if self._seg_map_cache is not None:
                    mask_size = int(np.sum(self._seg_map_cache == sid))
                    bbox = mask_bbox_xyxy(self._seg_map_cache, sid)

            debugText += f" | selected mask size: {mask_size} pixels"

            if bbox is not None and self.current_image_pil is not None:
                img_w, img_h = self.current_image_pil.size
                debugText += f" | Image size: {img_w}x{img_h}"
                x0, y0, x1, y1 = bbox
                if x0 < 0 or y0 < 0 or x1 > img_w or y1 > img_h:
                    debugText += f" | WARNING: bbox {bbox} exceeds image dimensions ({img_w}x{img_h})"

            self.debug_info.setText(debugText)


        rec = self.current_label_map.get(str(sid))  # may be None

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

        # Put cursor in the first editable field
        self._focus_first_mutable_input()


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

    # ---------- Save / missing labels ----------

    def _missing_label_seg_ids(self) -> List[int]:
        missing: List[int] = []
        for sid in self.instance_ids:
            rec = self.current_label_map.get(str(sid))
            if rec is None or (not self._is_record_labeled(rec)):
                missing.append(int(sid))
        return missing

    def _filtered_labeled_records(self) -> List[Dict[str, Any]]:
        return [rec for rec in self.current_labels if self._is_record_labeled(rec)]

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
        # save_json_labels(json_path, to_save)
        # Get the full path for the image

        writePersistenceData(json_path, to_save, imgFileName=self.current_full_img_path if self.current_full_img_path else None)

        # If we successfully saved permanent labels, remove autosaved file if present
        autosave_json_path = self.paths.label_json_path(self.current_image_id, autosave=True)
        try:
            if autosave_json_path.exists():
                autosave_json_path.unlink()
        except Exception:
            # If we fail to remove the autosave file, it's not a big deal, so we silently ignore any errors here.
            pass

        # Update row display if differentiating
        current_idx = self._ordered_indices[self._order_pos]
        if self.differentiate_labeled:
            missing = len(self._missing_label_seg_ids())
            status = "LABELED" if (len(to_save) > 0) else "UNLABELED"
            self._update_file_list_row_text(current_idx, status=status, missing=missing, total_masks=len(self.instance_ids))

        QMessageBox.information(self, "Saved", f"Saved {len(to_save)} labeled entry(ies):\n{json_path}")

        self.next_image()

    def _focus_first_mutable_input(self) -> None:
        """
        Move keyboard focus to the first mutable form input (schema order).
        Skips readonly fields and the id field.
        """
        for spec in self._form_specs:
            name = spec.get("name")
            kind = spec.get("kind")

            if not name or name == self.id_key:
                continue
            if kind in (None, "readonly"):
                continue

            w = self._widgets_by_field.get(name)
            if w is None:
                continue
            if not w.isEnabled():
                continue

            # Give focus + select contents when appropriate
            w.setFocus(Qt.FocusReason.OtherFocusReason)

            # Nice UX: highlight current value so typing replaces it
            try:
                if isinstance(w, QLineEdit):
                    w.selectAll()
                elif isinstance(w, QDoubleSpinBox):
                    w.selectAll()
                elif isinstance(w, QTextEdit):
                    w.selectAll()
            except Exception:
                pass

            return


    def _display_name_for_index(self, idx: int) -> str:
        # Prefer fast metadata name if ImageSource provides it
        if hasattr(self.image_source, "get_relname"):
            try:
                return str(self.image_source.get_relname(idx))
            except Exception:
                pass

        # Fallback: shorten image_id
        image_id = self._get_image_id_for_index(idx)
        return short_rel_name(image_id, max_parts=2)


    def _silent_autosave(self) -> None:
        if self.current_image_id is None:
            return
        self._write_back_selected()
        # json_path = self.paths.label_json_path(self.current_image_id)
        # to_save = self._filtered_labeled_records()
        # save_json_labels(json_path, to_save)
        autosave_json_path = self.paths.label_json_path(self.current_image_id, autosave=True)
        # save_json_labels(autosave_json_path, self.current_labels)
        writePersistenceData(autosave_json_path, self.current_labels, imgFileName=self.current_full_img_path if self.current_full_img_path else None)

        current_idx = self._ordered_indices[self._order_pos]
        if self.differentiate_labeled:
            missing = len(self._missing_label_seg_ids())
            status = self._file_quick_status(self.current_image_id)  # use quick heuristic for status during autosave to avoid JSON read
            self._update_file_list_row_text(current_idx, status=status, missing=missing, total_masks=len(self.instance_ids))

    # ---------- Navigation: files (ordered) ----------

    def next_image(self) -> None:
        if self.autosave_on_nav:
            self._silent_autosave()
        if self._order_pos + 1 < len(self._ordered_indices):
            self._load_by_order_pos(self._order_pos + 1)

    def prev_image(self) -> None:
        if self.autosave_on_nav:
            self._silent_autosave()
        if self._order_pos - 1 >= 0:
            self._load_by_order_pos(self._order_pos - 1)

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
        mods = event.modifiers()

        # open pending selection
        if key in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
            self._open_pending_selection()
            return

        # mask nav
        if key in (Qt.Key.Key_J, Qt.Key.Key_Down, Qt.Key.Key_BracketRight, Qt.Key.Key_N):
            self.next_mask()
            return
        if key in (Qt.Key.Key_K, Qt.Key.Key_Up, Qt.Key.Key_BracketLeft, Qt.Key.Key_P):
            self.prev_mask()
            return

        # file nav
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

        if key == Qt.Key.Key_Q:
            self._apply_confidence_preset("vlow")
            return
        if key == Qt.Key.Key_W:
            self._apply_confidence_preset("low")
            return
        if key == Qt.Key.Key_R:
            self._apply_confidence_preset("med")
            return
        if key == Qt.Key.Key_T:
            self._apply_confidence_preset("high")
            return
        if key == Qt.Key.Key_Y:
            self._apply_confidence_preset("vhigh")
            return

        if key == Qt.Key.Key_1:
            self._apply_preset("static")
            return
        if key == Qt.Key.Key_2:
            self._apply_preset("months")
            return
        if key == Qt.Key.Key_3:
            self._apply_preset("weeks")
            return
        if key == Qt.Key.Key_4:
            self._apply_preset("days")
            return
        if key == Qt.Key.Key_5:
            self._apply_preset("next_day")
            return
        if key == Qt.Key.Key_6:
            self._apply_preset("several_hours")
            return
        if key == Qt.Key.Key_7:
            self._apply_preset("couple_hours")
            return
        if key == Qt.Key.Key_8:
            self._apply_preset("under_hour")
            return
        if key == Qt.Key.Key_9:
            self._apply_preset("under_30min")
            return
        if key == Qt.Key.Key_0:
            self._apply_preset("under_10min")
            return
        if key == Qt.Key.Key_Minus:
            self._apply_preset("under_2min")
            return
        if key == Qt.Key.Key_Equal:
            self._apply_preset("dynamic")
            return

        # if key == Qt.Key.Key_2:
        #     self._apply_preset("dynamic")
        #     return

        super().keyPressEvent(event)


# ----------------------- Entrypoint helper -----------------------

def run_gui(
    image_source,
    paths: LabelingPaths,
    key_map: Optional[Dict[str, str]] = None,
    *,
    differentiate_labeled: bool = True,
) -> None:
    app = QApplication([])
    win = LabelingWindow(
        image_source=image_source,
        paths=paths,
        key_map=key_map,
        differentiate_labeled=differentiate_labeled,
    )
    win.show()
    app.exec()

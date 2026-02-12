from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Literal, Optional, Sequence, Tuple, Union, Any, Mapping

import json
import math
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image, ImageOps

from spf_io.persistence_io import readPersistenceData, PERSISTENCE_FILE_NAME
from instance_segmentation.abs_instance_segmenter import SegmentationResults
from training.dataset.split_management import SplitName, SplitIdKind, SplitGranularity


import torchvision.transforms as T

ReturnMode = Literal["per_image", "per_instance"]



# -------------------------
# DINOv2 default transform (Option 3)
# -------------------------
def make_default_dinov2_transform() -> Callable[[torch.Tensor], torch.Tensor]:
    """
    Returns a torchvision transform that:
      - converts uint8 [0..255] -> float32 [0..1]
      - normalizes with ImageNet mean/std (typical for DINOv2 backbones)
    """
    return T.Compose(
        [
            T.ConvertImageDtype(torch.float32),  # scales uint8 -> float32 in [0,1]
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def _read_lines(path: Union[str, Path]) -> List[str]:
    p = Path(path)
    return [ln.strip() for ln in p.read_text().splitlines() if ln.strip() and not ln.strip().startswith("#")]


SplitEntry = Union[str, Tuple[str, int], Dict[str, object], List[object], Tuple[object, ...]]

def _parse_split_line_to_entry(line: str, *, pair_separators: Tuple[str, ...]) -> Union[str, Tuple[str, int]]:
    """
    Accepts:
      - "id"
      - "id#12" (default)
      - "id 12"
    Returns either "id" or ("id", 12).
    """
    # First try explicit separators like '#'
    for sep in pair_separators:
        if sep and sep in line:
            left, right = line.split(sep, 1)
            left = left.strip()
            right = right.strip()
            if not left:
                break
            if right:
                try:
                    return (left, int(right))
                except ValueError:
                    # If right isn't an int, treat as not-a-pair
                    return line
            return left

    # Then try whitespace form: "id 12"
    parts = line.split()
    if len(parts) == 2:
        a, b = parts[0].strip(), parts[1].strip()
        try:
            return (a, int(b))
        except ValueError:
            return line

    return line

def _load_split_entries(
    split: SplitName,
    *,
    split_dir: Optional[Union[str, Path]] = None,
    train_file: Optional[Union[str, Path]] = None,
    val_file: Optional[Union[str, Path]] = None,
    test_file: Optional[Union[str, Path]] = None,
    split_json: Optional[Union[str, Path]] = None,
    pair_separators: Tuple[str, ...] = ("#",),
) -> List[Union[str, Tuple[str, int], Dict[str, object], List[object], Tuple[object, ...]]]:
    """
    Like _load_split_ids but supports pair entries too.

    For txt files: parses lines into either "id" or ("id", instance_id).
    For json: returns entries as-is; parsing happens later to support dict/list forms.
    """
    if split_json is not None:
        obj = json.loads(Path(split_json).read_text())
        if "splits" in obj:
            obj = obj["splits"]
        if split not in obj:
            raise KeyError(f"Split {split!r} not found in {split_json}")
        return list(obj[split])

    if any([train_file, val_file, test_file]):
        file_map = {"train": train_file, "val": val_file, "test": test_file}
        p = file_map[split]
        if p is None:
            raise ValueError(f"Missing file for split={split!r}")
        lines = _read_lines(p)
        return [_parse_split_line_to_entry(ln, pair_separators=pair_separators) for ln in lines]

    if split_dir is None:
        raise ValueError("Provide either split_json, explicit split files, or split_dir")

    lines = _read_lines(Path(split_dir) / f"{split}.txt")
    return [_parse_split_line_to_entry(ln, pair_separators=pair_separators) for ln in lines]

# def _load_split_ids(
#     split: SplitName,
#     *,
#     split_dir: Optional[Union[str, Path]] = None,
#     train_file: Optional[Union[str, Path]] = None,
#     val_file: Optional[Union[str, Path]] = None,
#     test_file: Optional[Union[str, Path]] = None,
#     split_json: Optional[Union[str, Path]] = None,
# ) -> List[str]:
#     """
#     Loads split entries from either:
#       - split_json: {"train":[...], "val":[...], "test":[...]}  (or nested under "splits")
#       - explicit train/val/test file paths
#       - split_dir/{train,val,test}.txt

#     NOTE: Entries can be *either* relative image paths or image_ids depending on cfg.split_id_kind.
#     """
#     if split_json is not None:
#         obj = json.loads(Path(split_json).read_text())
#         if "splits" in obj:
#             obj = obj["splits"]
#         if split not in obj:
#             raise KeyError(f"Split {split!r} not found in {split_json}")
#         return list(obj[split])

#     if any([train_file, val_file, test_file]):
#         file_map = {"train": train_file, "val": val_file, "test": test_file}
#         p = file_map[split]
#         if p is None:
#             raise ValueError(f"Missing file for split={split!r}")
#         return _read_lines(p)

#     if split_dir is None:
#         raise ValueError("Provide either split_json, explicit split files, or split_dir")

#     return _read_lines(Path(split_dir) / f"{split}.txt")


def label_transform_log_ratio(raw: dict) -> torch.Tensor:
    t50 = float(raw["median_persistence_hours"])
    t90 = float(raw["p90_persistence_hours"])
    c = float(raw["persistence_conf"])

    eps = 1e-8
    y50 = math.log(max(t50, eps))
    ratio = t90 / max(t50, eps) - 1.0
    yr = math.log(max(ratio, eps))
    yc = c

    return torch.tensor([y50, yr, yc], dtype=torch.float32)


@dataclass(frozen=True)
class SegmentationDatasetConfig:
    # Roots
    input_root_dir: Union[str, Path]
    output_root_dir: Union[str, Path]

    # Output config names
    segmentation_config_name: str
    full_results_config_name: str

    # Splits contain relpaths or image_ids
    split_id_kind: SplitIdKind = "relpath"

    split_granularity: SplitGranularity = "image"

    split_pair_separators: Tuple[str, ...] = ("#",)

    # Filenames
    seg_npz_name: str = "segmentation_results.npz"
    labels_json_name: str = PERSISTENCE_FILE_NAME

    # Split specification
    split_dir: Optional[Union[str, Path]] = None
    split_json: Optional[Union[str, Path]] = None
    train_split_file: Optional[Union[str, Path]] = None
    val_split_file: Optional[Union[str, Path]] = None
    test_split_file: Optional[Union[str, Path]] = None

    # Reading behavior
    image_loader: Optional[Callable[[Path], torch.Tensor]] = None

    # Option 3: torchvision transform that maps image->image
    # If None, we'll supply a default DINOv2 transform.
    image_transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = None

    # Labels
    label_transform: Optional[Callable[[Dict[str, float]], object]] = None

    # Storage dtypes
    masks_dtype: torch.dtype = torch.uint8
    boxes_dtype: torch.dtype = torch.float32

    def make_dataset(
        self,
        split: SplitName,
        relpath_to_image_id: Callable[[str], str],
        image_id_to_relpath: Callable[[str], str],
        *,
        mode: ReturnMode = "per_instance",
        strict: bool = True,
    ) -> Dataset:
        return _SegmentationDataset(
            cfg=self,
            split=split,
            mode=mode,
            strict=strict,
            relpath_to_image_id=relpath_to_image_id,
            image_id_to_relpath=image_id_to_relpath,
        )


class _SegmentationDataset(Dataset):
    def __init__(
        self,
        cfg: SegmentationDatasetConfig,
        split: SplitName,
        mode: ReturnMode,
        relpath_to_image_id: Callable[[str], str],
        image_id_to_relpath: Callable[[str], str],
        strict: bool,
    ):
        super().__init__()

        self.cfg = cfg
        self.split = split
        self.mode = mode
        self.strict = strict
        self.relpath_to_image_id = relpath_to_image_id
        self.image_id_to_relpath = image_id_to_relpath
        # ----------------------------
        # Root structure
        # ----------------------------
        self.input_root_dir = Path(cfg.input_root_dir)
        self.output_root_dir = Path(cfg.output_root_dir)

        self._seg_base = (
            self.output_root_dir
            / "segmentation_results"
            / cfg.segmentation_config_name
        )
        self._labels_base = (
            self.output_root_dir
            / "full_results"
            / cfg.full_results_config_name
        )

        # ----------------------------
        # Default transform: DINOv2-style (Option 3)
        # ----------------------------
        # image_transform = torchvision transform (image-only)
        self._image_transform = (
            cfg.image_transform if cfg.image_transform is not None else make_default_dinov2_transform()
        )

        # ----------------------------
        # Label transformation & image loader
        # ----------------------------
        self._label_transform = cfg.label_transform or label_transform_log_ratio
        self._image_loader = cfg.image_loader or self._default_image_loader
        self._label_cache: Dict[Path, dict] = {}

        # ----------------------------
        # Load split entries
        #   These may be relative image paths OR image_ids depending on split_id_kind
        # ----------------------------
        # split_entries = _load_split_ids(
        split_entries = _load_split_entries(
            split,
            split_dir=cfg.split_dir,
            train_file=cfg.train_split_file,
            val_file=cfg.val_split_file,
            test_file=cfg.test_split_file,
            split_json=cfg.split_json,
            pair_separators=cfg.split_pair_separators,
        )

        # ----------------------------
        # Build records + indices
        # ----------------------------
        # Each record corresponds to a single "image sample" (one seg file + one label file + maybe an image path)
        self._records: List[Dict[str, object]] = []
        self._instance_index: List[Tuple[int, int]] = []

        # Map image_id -> rec_idx
        rec_by_image_id: Dict[str, int] = {}
        # Map rec_idx -> None (means ALL) or set(instance_ids)
        allowed_instances: Dict[int, Optional[set[int]]] = {}


        # Map image_id -> rec_idx
        rec_by_image_id: Dict[str, int] = {}
        # Map rec_idx -> None (means ALL) or set(instance_ids)
        allowed_instances: Dict[int, Optional[set[int]]] = {}

        def normalize_entry(e: SplitEntry) -> Tuple[str, Optional[int]]:
            """
            Returns (id_token, maybe_instance_id).
            id_token is relpath or image_id depending on split_id_kind.
            """
            # txt parsing yields "id" or ("id", iid)
            if isinstance(e, tuple) and len(e) == 2 and isinstance(e[0], str):
                return e[0], int(e[1])

            if isinstance(e, str):
                return e, None

            # JSON dict form: {"id": "...", "instance_id": 3}
            if isinstance(e, dict):
                # allow either "id" or explicit "relpath"/"image_id"
                if "id" in e:
                    id_token = str(e["id"])
                elif "relpath" in e:
                    id_token = str(e["relpath"])
                elif "image_id" in e:
                    id_token = str(e["image_id"])
                else:
                    raise KeyError(f"Split dict entry must contain one of: id/relpath/image_id. Got keys={list(e.keys())}")

                iid = e.get("instance_id", None)
                return id_token, (None if iid is None else int(iid))

            # JSON list form: ["id", 3]
            if isinstance(e, (list, tuple)) and len(e) == 2:
                return str(e[0]), (None if e[1] is None else int(e[1]))

            raise TypeError(f"Unsupported split entry type: {type(e)}; entry={e!r}")

        for raw_entry in split_entries:
            id_token, maybe_iid = normalize_entry(raw_entry)

            # Enforce split_granularity if you want strict behavior
            if cfg.split_granularity == "image" and maybe_iid is not None:
                # If someone accidentally put pairs in image mode:
                msg = f"[SegDataset:{self.split}] Found instance id in split entry but split_granularity='image': {raw_entry!r}"
                if self.strict:
                    raise ValueError(msg)
                print(msg)
                maybe_iid = None

            if cfg.split_granularity == "image_instance" and maybe_iid is None:
                msg = f"[SegDataset:{self.split}] Missing instance id in split entry but split_granularity='image_instance': {raw_entry!r}"
                if self.strict:
                    raise ValueError(msg)
                print(msg)
                # skip or treat as "all"? safer to skip
                continue

            # Resolve relpath + image_id deterministically
            if self.cfg.split_id_kind == "relpath":
                relpath = id_token
                image_id = self.relpath_to_image_id(relpath)
            else:
                image_id = id_token
                relpath = self.image_id_to_relpath(image_id)

            # Create record if needed
            if image_id in rec_by_image_id:
                rec_idx = rec_by_image_id[image_id]
            else:
                image_path = self.input_root_dir / relpath
                key_dir = str(image_id)
                seg_path = self._seg_base / key_dir / cfg.seg_npz_name
                labels_path = self._labels_base / key_dir / cfg.labels_json_name

                missing: List[str] = []
                if not image_path.exists():
                    missing.append("missing image")
                if not seg_path.exists():
                    missing.append("missing segmentation")

                if missing:
                    msg = (
                        f"[SegDataset:{self.split}] Skipping entry={raw_entry!r} "
                        f"(image_id={image_id!r}, relpath={relpath!r}) because {', '.join(missing)}. "
                        f"image_path={str(image_path)!r} seg_path={str(seg_path)!r}"
                    )
                    if self.strict:
                        raise FileNotFoundError(msg)
                    print(msg)
                    continue

                # Labels required for per_instance indexing (later)
                if self.mode == "per_instance" and not labels_path.exists():
                    msg = (
                        f"[SegDataset:{self.split}] Labels missing; cannot index instances for image "
                        f"(image_id={image_id!r}, relpath={relpath!r}). labels_path={str(labels_path)!r}"
                    )
                    if self.strict:
                        raise FileNotFoundError(msg)
                    print(msg)

                rec = {
                    "entry": id_token,
                    "relpath": relpath,
                    "image_id": image_id,
                    "key_dir": key_dir,
                    "seg_path": seg_path,
                    "labels_path": labels_path,
                    "image_path": image_path,
                }
                rec_idx = len(self._records)
                self._records.append(rec)
                rec_by_image_id[image_id] = rec_idx
                allowed_instances[rec_idx] = None  # default: ALL

            # Accumulate allowed instance ids if in pair mode
            if cfg.split_granularity == "image_instance":
                s = allowed_instances.get(rec_idx)
                if s is None:
                    s = set()
                    allowed_instances[rec_idx] = s
                assert isinstance(s, set)
                s.add(int(maybe_iid))  # type: ignore[arg-type]

        # ----------------------------
        # Build instance index
        # ----------------------------
        if self.mode == "per_instance":
            for rec_idx, rec in enumerate(self._records):
                labels_path: Path = rec["labels_path"]  # type: ignore
                if not labels_path.exists():
                    continue

                wanted = allowed_instances.get(rec_idx, None)  # None -> ALL
                if wanted is None:
                    # all labels in that image
                    for iid in self._label_instance_ids(labels_path):
                        self._instance_index.append((rec_idx, int(iid)))
                else:
                    # only specified instance ids
                    # validate existence in labels
                    obj = self._load_label_obj(labels_path)
                    for iid in sorted(wanted):
                        if str(int(iid)) not in obj:
                            msg = (
                                f"[SegDataset:{self.split}] Split requested instance_id={iid} "
                                f"but label not found in {labels_path}"
                            )
                            if self.strict:
                                raise KeyError(msg)
                            print(msg)
                            continue
                        self._instance_index.append((rec_idx, int(iid)))

        # for entry in split_entries:

        #     # --- Resolve relpath + image_id deterministically ---
        #     if self.cfg.split_id_kind == "relpath":
        #         relpath = entry
        #         image_id = self.relpath_to_image_id(relpath)
        #     else:  # image_id
        #         image_id = entry
        #         relpath = self.image_id_to_relpath(image_id)

        #     image_path = self.input_root_dir / relpath
        #     key_dir = str(image_id)

        #     seg_path = self._seg_base / key_dir / cfg.seg_npz_name
        #     labels_path = self._labels_base / key_dir / cfg.labels_json_name

        #     # Skip entries missing image and/or segmentation results (log skips).
        #     missing: List[str] = []
        #     if not image_path.exists():
        #         missing.append("missing image")
        #     if not seg_path.exists():
        #         missing.append("missing segmentation")

        #     if missing:
        #         msg = (
        #             f"[SegDataset:{self.split}] Skipping entry={entry!r} "
        #             f"(image_id={image_id!r}, relpath={relpath!r}) because {', '.join(missing)}. "
        #             f"image_path={str(image_path)!r} seg_path={str(seg_path)!r}"
        #         )
        #         if self.strict:
        #             raise FileNotFoundError(msg)
        #         print(msg)
        #         continue

        #     # Labels are required for per_instance indexing. If missing, log and skip indexing.
        #     if self.mode == "per_instance" and not labels_path.exists():
        #         msg = (
        #             f"[SegDataset:{self.split}] Labels missing; will not index instances for entry={entry!r} "
        #             f"(image_id={image_id!r}, relpath={relpath!r}). labels_path={str(labels_path)!r}"
        #         )
        #         if self.strict:
        #             raise FileNotFoundError(msg)
        #         print(msg)

        #     rec = {
        #         "entry": entry,
        #         "relpath": relpath,
        #         "image_id": image_id,
        #         "key_dir": key_dir,
        #         "seg_path": seg_path,
        #         "labels_path": labels_path,
        #         "image_path": image_path,
        #     }

        #     rec_idx = len(self._records)
        #     self._records.append(rec)

        #     if self.mode == "per_instance":
        #         if labels_path.exists():
        #             for iid in self._label_instance_ids(labels_path):
        #                 self._instance_index.append((rec_idx, int(iid)))


    def __len__(self) -> int:
        if self.mode == "per_instance":
            return len(self._instance_index)
        else:
            return len(self._records)


    # -------------------------
    # __getitem__
    # -------------------------
    def __getitem__(self, idx: int):
        if self.mode == "per_image":
            # Not worrying about per_image mode currently
            # Should still implement necessary components in the constructor so it's easier to pivot to later if needed
            raise NotImplementedError("per_image mode is not implemented yet. Use per_instance mode.")


        rec_idx, inst_id = self._instance_index[idx]
        rec = self._records[rec_idx]

        seg_path: Path = rec["seg_path"]  # type: ignore
        labels_path: Path = rec["labels_path"]  # type: ignore
        image_path: Path = rec["image_path"]  # type: ignore

        if self.strict:
            if not seg_path.exists():
                raise FileNotFoundError(f"Missing segmentation file: {seg_path}")
            if not labels_path.exists():
                raise FileNotFoundError(f"Missing labels file: {labels_path}")
            if not image_path.exists():
                raise FileNotFoundError(f"Missing image file: {image_path}")

        seg = self._load_seg(seg_path)

        # Load + transform image
        image_u8 = self._image_loader(image_path)
        image = self._image_transform(image_u8)

        # Load label
        label = self._load_labels(labels_path, [inst_id])[0]

        # Find mask
        ids = seg["ids"]
        masks = seg["masks"]

        pos = (ids == int(inst_id)).nonzero(as_tuple=False).view(-1)
        if pos.numel() == 0:
            if self.strict:
                raise KeyError(
                    f"Mask for labeled instance_id={inst_id} not found in {seg_path}"
                )
            H, W = int(image_u8.shape[1]), int(image_u8.shape[2])
            mask = torch.zeros((H, W), dtype=self.cfg.masks_dtype)
        else:
            j = int(pos[0].item())
            mask = masks[j]

        return self._format_training_sample(
            image=image,
            mask=mask,
            label=label,
            image_id=rec["image_id"],
            instance_id=inst_id,
        )



    def _normalize_persistence_obj(self, obj: Any, *, path: Path) -> Dict[str, dict]:
        """
        Normalizes persistence data into:
            { "<instance_id>": { ...raw fields... }, ... }

        Supports:
          - dict keyed by instance id strings/ints (legacy)
          - list of dict entries with an "id" field (new)
        """
        # Legacy: already a mapping. Ensure keys are strings.
        if isinstance(obj, Mapping):
            out: Dict[str, dict] = {}
            for k, v in obj.items():
                if not isinstance(v, dict):
                    # allow skipping weird metadata keys if present
                    continue
                try:
                    ik = str(int(k))
                except Exception:
                    # ignore non-instance keys
                    continue
                out[ik] = v
            return out

        # New: list of entries, each has "id"
        if isinstance(obj, list):
            out: Dict[str, dict] = {}
            for i, entry in enumerate(obj):
                if not isinstance(entry, dict):
                    raise TypeError(
                        f"Labels list at {path} has non-dict entry at index {i}: {type(entry)}"
                    )
                if "id" not in entry:
                    raise KeyError(
                        f"Labels list at {path} missing 'id' key at index {i}. Keys={list(entry.keys())}"
                    )
                try:
                    ik = str(int(entry["id"]))
                except Exception as e:
                    raise TypeError(
                        f"Labels list at {path} has non-int-convertible 'id' at index {i}: {entry['id']!r}"
                    ) from e
                out[ik] = entry
            return out

        raise TypeError(
            f"Unsupported labels type at {path}: expected dict or list, got {type(obj)}"
        )

    def _load_label_obj(self, path: Path) -> Dict[str, dict]:
        """Loads normalized label dict (cached). Keys are instance_ids as strings."""
        if path not in self._label_cache:
            raw_obj = readPersistenceData(path)  # may be dict OR list now
            self._label_cache[path] = self._normalize_persistence_obj(raw_obj, path=path)

        obj = self._label_cache[path]
        if not isinstance(obj, dict):
            # defensive; after normalization it should always be dict
            raise TypeError(f"Normalized labels at {path} must be a dict, got {type(obj)}")
        return obj  # type: ignore[return-value]


    def _label_instance_ids(self, path: Path) -> List[int]:
        """Returns label instance ids as ints (sorted for determinism)."""
        obj = self._load_label_obj(path)
        ids: List[int] = []
        for k in obj.keys():
            try:
                ids.append(int(k))
            except Exception:
                continue
        ids.sort()
        return ids


    def _format_training_sample(
        self,
        image: torch.Tensor,
        mask: torch.Tensor,
        label: object,
        *,
        image_id: str,
        instance_id: int,
    ) -> Dict[str, object]:
        if not torch.is_tensor(label):
            raise TypeError("label_transform must return a torch.Tensor of shape (3,) for this training format.")
        if label.numel() != 3:
            raise ValueError(f"Expected label tensor with 3 elements [y50, yr, yc], got shape {tuple(label.shape)}")

        label = label.to(torch.float32).view(-1)
        y50, yr, yc = label[0], label[1], label[2]

        if mask.dtype not in (torch.uint8, torch.bool):
            mask = mask.to(torch.uint8)

        return {
            "image": image,  # (C,H,W)
            "mask": mask,    # (H,W)
            "y50": y50,
            "yr": yr,
            "yc": yc,
            "image_id": image_id,
            "instance_id": torch.tensor(instance_id, dtype=torch.int64),
        }


    def _default_image_loader(self, path: Path) -> torch.Tensor:
        # EXIF-safe, CHW uint8
        img = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
        arr = np.asarray(img)  # HWC uint8
        return torch.from_numpy(arr).permute(2, 0, 1).contiguous()

    def _load_seg(self, path: Path) -> Dict[str, torch.Tensor]:
        seg, _img_path = SegmentationResults.loadFromFile(str(path))

        out: Dict[str, torch.Tensor] = {
            "ids": seg.ids.to(torch.int64),
            "masks": seg.masks.to(self.cfg.masks_dtype),
            "boxes": seg.boxes.to(self.cfg.boxes_dtype),
        }

        return out

    def _load_labels(self, path: Path, instance_ids: Sequence[int]) -> List[object]:
        obj = self._load_label_obj(path)

        labels: List[object] = []
        for iid in instance_ids:
            key = str(int(iid))
            if key not in obj:
                raise KeyError(f"Label for instance_id={iid} not found in {path}")
            raw = obj[key]
            base = {
                "median_persistence_hours": float(raw["median_persistence_hours"]),
                "p90_persistence_hours": float(raw["p90_persistence_hours"]),
                "persistence_conf": float(raw["persistence_conf"]),
            }
            labels.append(self._label_transform(base))
        return labels



# cfg = SegmentationDatasetConfig(
#     input_root_dir="/path/to/input_root_dir",
#     output_root_dir="/path/to/output_root_dir",
#     segmentation_config_name="my_seg_cfg",
#     full_results_config_name="my_full_cfg",
#     split_dir="/path/to/splits",         # contains train.txt/val.txt/test.txt
#     split_id_kind="relpath",             # split files contain relative image paths
#     relpath_to_image_id=relpath_to_id,
# )

# ds = cfg.make_dataset("train", mode="per_instance", strict=True)

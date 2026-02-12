from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Literal, Optional, Sequence, Tuple, Union

import json
import math
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image, ImageOps

from spf_io.persistence_io import readPersistenceData


import torchvision.transforms as T

SplitName = Literal["train", "val", "test"]
ReturnMode = Literal["per_image", "per_instance"]
SplitIdKind = Literal["relpath", "image_id"]



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


def _load_split_ids(
    split: SplitName,
    *,
    split_dir: Optional[Union[str, Path]] = None,
    train_file: Optional[Union[str, Path]] = None,
    val_file: Optional[Union[str, Path]] = None,
    test_file: Optional[Union[str, Path]] = None,
    split_json: Optional[Union[str, Path]] = None,
) -> List[str]:
    """
    Loads split entries from either:
      - split_json: {"train":[...], "val":[...], "test":[...]}  (or nested under "splits")
      - explicit train/val/test file paths
      - split_dir/{train,val,test}.txt

    NOTE: Entries can be *either* relative image paths or image_ids depending on cfg.split_id_kind.
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
        return _read_lines(p)

    if split_dir is None:
        raise ValueError("Provide either split_json, explicit split files, or split_dir")

    return _read_lines(Path(split_dir) / f"{split}.txt")


# def label_transform_log_ratio(raw: dict) -> torch.Tensor:
#     t50 = float(raw["median_persistence_hours"])
#     t90 = float(raw["p90_persistence_hours"])
#     c = float(raw["persistence_conf"])

#     eps = 1e-8
#     y50 = math.log(max(t50, eps))
#     ratio = t90 / max(t50, eps) - 1.0
#     yr = math.log(max(ratio, eps))
#     yc = c

#     return torch.tensor([y50, yr, yc], dtype=torch.float32)

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

    # Filenames
    seg_npz_name: str = "segmentation_results.npz"
    labels_json_name: str = "persistence_parameters.json"

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

    # Pair transforms hook (kept for backwards compatibility)
    transforms: Optional[Callable[[torch.Tensor, Dict], Tuple[torch.Tensor, Dict]]] = None

    # Labels
    label_transform: Optional[Callable[[Dict[str, float]], object]] = None

    # Storage dtypes
    masks_dtype: torch.dtype = torch.uint8
    boxes_dtype: torch.dtype = torch.float32

    def make_dataset(
        self,
        split: SplitName,
        relpath_to_image_id: Callable[[str], str],
        *,
        mode: ReturnMode = "per_instance",
        include_scores: bool = False,
        strict: bool = True,
    ) -> Dataset:
        return _SegmentationDataset(
            cfg=self,
            split=split,
            mode=mode,
            include_scores=include_scores,
            strict=strict,
            relpath_to_image_id=relpath_to_image_id,
        )

# @dataclass(frozen=True)
# class SegmentationDatasetConfig:
#     # Roots
#     input_root_dir: Union[str, Path]
#     output_root_dir: Union[str, Path]

#     # Output config names
#     segmentation_config_name: str
#     full_results_config_name: str

#     # How to interpret split entries
#     # - "relpath": split files contain paths relative to input_root_dir (recommended)
#     # - "image_id": split files contain image_ids directly
#     split_id_kind: SplitIdKind = "relpath"

#     # Function: relative image path -> image_id
#     # relpath_to_image_id: Callable[[str], str] = None  # must be provided

#     # Filenames (match your structure)
#     seg_npz_name: str = "segmentation_results.npz"
#     labels_json_name: str = "persistence_parameters.json"

#     # Split specification (choose ONE style)
#     split_dir: Optional[Union[str, Path]] = None
#     split_json: Optional[Union[str, Path]] = None
#     train_split_file: Optional[Union[str, Path]] = None
#     val_split_file: Optional[Union[str, Path]] = None
#     test_split_file: Optional[Union[str, Path]] = None

#     # Reading behavior
#     image_loader: Optional[Callable[[Path], torch.Tensor]] = None
#     transforms: Optional[Callable[[torch.Tensor, Dict], Tuple[torch.Tensor, Dict]]] = None
#     label_transform: Optional[Callable[[Dict[str, float]], object]] = None

#     # Storage dtypes
#     masks_dtype: torch.dtype = torch.uint8
#     boxes_dtype: torch.dtype = torch.float32

#     # def make_dataset(
#     #     self,
#     #     split: SplitName,
#     #     *,
#     #     mode: ReturnMode = "per_instance",
#     #     include_scores: bool = False,
#     #     strict: bool = True,
#     # ) -> Dataset:
#     #     return _SegmentationDataset(
#     #         cfg=self,
#     #         split=split,
#     #         mode=mode,
#     #         include_scores=include_scores,
#     #         strict=strict,
#     #     )




# class _SegmentationDataset(Dataset):
#     def __init__(
#         self,
#         cfg: SegmentationDatasetConfig,
#         split: SplitName,
#         mode: ReturnMode,
#         include_scores: bool,
#         strict: bool,
#         rel_path_to_image_id: Callable[[str], str],
#     ):
#         super().__init__()

#         self.cfg = cfg
#         self.split = split
#         self.mode = mode
#         self.include_scores = include_scores
#         self.strict = strict

#         self.input_root_dir = Path(cfg.input_root_dir)
#         self.output_root_dir = Path(cfg.output_root_dir)

#         # Base dirs from your structure
#         self._seg_base = (
#             self.output_root_dir
#             / "segmentation_results"
#             / cfg.segmentation_config_name
#         )
#         self._labels_base = (
#             self.output_root_dir
#             / "full_results"
#             / cfg.full_results_config_name
#         )

#         split_entries = _load_split_ids(
#             split,
#             split_dir=cfg.split_dir,
#             train_file=cfg.train_split_file,
#             val_file=cfg.val_split_file,
#             test_file=cfg.test_split_file,
#             split_json=cfg.split_json,
#         )

#         # Build per-image index
#         self._image_index: List[Dict[str, object]] = []
#         for entry in split_entries:
#             if cfg.split_id_kind == "relpath":
#                 relpath = entry
#                 image_id = str(rel_path_to_image_id(relpath))
#             else:
#                 image_id = entry
#                 relpath = None  # unknown / not needed

#             img_path = (self.input_root_dir / relpath) if relpath is not None else None
#             seg_path = self._seg_base / image_id / cfg.seg_npz_name
#             lab_path = self._labels_base / image_id / cfg.labels_json_name

#             missing: List[Path] = []
#             if img_path is not None and not img_path.exists():
#                 missing.append(img_path)
#             if not seg_path.exists():
#                 missing.append(seg_path)
#             if not lab_path.exists():
#                 missing.append(lab_path)

#             if missing:
#                 if strict:
#                     raise FileNotFoundError(
#                         f"Missing files for entry={entry!r} (image_id={image_id}): {missing}"
#                     )
#                 continue

#             self._image_index.append(
#                 {
#                     "image_id": image_id,
#                     "relpath": relpath,
#                     "img": img_path,   # may be None if split_id_kind="image_id"
#                     "seg": seg_path,
#                     "lab": lab_path,
#                 }
#             )

#         # Per-instance index
#         self._instance_index: Optional[List[Tuple[int, int]]] = None
#         if self.mode == "per_instance":
#             pairs: List[Tuple[int, int]] = []
#             for img_idx, rec in enumerate(self._image_index):
#                 seg = np.load(rec["seg"], allow_pickle=True)
#                 n = int(len(seg["ids"]))
#                 for inst_pos in range(n):
#                     pairs.append((img_idx, inst_pos))
#             self._instance_index = pairs

#         self._label_transform = cfg.label_transform or label_transform_log_ratio
#         self._image_loader = cfg.image_loader or self._default_image_loader
#         self._label_cache: Dict[Path, dict] = {}

class _SegmentationDataset(Dataset):
    def __init__(
        self,
        cfg: SegmentationDatasetConfig,
        split: SplitName,
        mode: ReturnMode,
        include_scores: bool,
        relpath_to_image_id: Callable[[str], str],
        strict: bool,
    ):
        super().__init__()

        self.cfg = cfg
        self.split = split
        self.mode = mode
        self.include_scores = include_scores
        self.strict = strict
        self.relpath_to_image_id = relpath_to_image_id
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
        # transforms       = optional pair-transform (image, target) -> (image, target)
        self._image_transform = (
            cfg.image_transform if cfg.image_transform is not None else make_default_dinov2_transform()
        )
        self._pair_transforms = cfg.transforms  # takes precedence if provided

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
        split_entries = _load_split_ids(
            split,
            split_dir=cfg.split_dir,
            train_file=cfg.train_split_file,
            val_file=cfg.val_split_file,
            test_file=cfg.test_split_file,
            split_json=cfg.split_json,
        )

        # ----------------------------
        # Build index
        # ----------------------------
        self._image_index: List[Dict[str, object]] = []

        for entry in split_entries:

            # Determine relpath and image_id depending on split_id_kind
            if cfg.split_id_kind == "relpath":
                relpath = entry                      # e.g. "folder/img.jpg"
                image_id = str(relpath_to_image_id(relpath))
            else:
                image_id = entry                     # directly the id
                relpath = None                       # unknown unless user provided mapping

            # Image file path (only if relpath is available)
            img_path = (self.input_root_dir / relpath) if relpath is not None else None

            # Structured output paths
            seg_path = self._seg_base / image_id / cfg.seg_npz_name
            lab_path = self._labels_base / image_id / cfg.labels_json_name

            # Missing-file check
            missing = []
            if img_path is not None and not img_path.exists():
                missing.append(img_path)
            if not seg_path.exists():
                missing.append(seg_path)
            if not lab_path.exists():
                missing.append(lab_path)

            if missing:
                if strict:
                    raise FileNotFoundError(
                        f"Missing files for entry={entry!r} (image_id={image_id}): {missing}"
                    )
                continue

            self._image_index.append(
                {
                    "image_id": image_id,
                    "relpath": relpath,
                    "img": img_path,
                    "seg": seg_path,
                    "lab": lab_path,
                }
            )

        # ----------------------------
        # Build per-instance index (mode = per_instance)
        # ----------------------------
        self._instance_index: Optional[List[Tuple[int, int]]] = None
        if self.mode == "per_instance":
            # Use the number of instances in the label file to build the index, since some images may have zero instances and we want to skip them and we may not have labels for all masks
            for img_idx, rec in enumerate(self._image_index):
                labelPath = rec["lab"]



            # inst_list: List[Tuple[int, int]] = []
            # for img_idx, rec in enumerate(self._image_index):
            #     with np.load(rec["seg"], allow_pickle=True) as seg:
            #         N = int(len(seg["ids"]))
            #     for inst_pos in range(N):
            #         inst_list.append((img_idx, inst_pos))
            # self._instance_index = inst_list

    def _load_label_obj(self, path: Path) -> dict:
        """Loads the raw label dict (cached). Keys are instance_ids as strings."""
        if path not in self._label_cache:
            self._label_cache[path] = readPersistenceData(path)  # or json.loads(path.read_text())
        obj = self._label_cache[path]
        if not isinstance(obj, dict):
            raise TypeError(f"Labels at {path} must be a dict, got {type(obj)}")
        return obj

    def _label_instance_ids(self, path: Path) -> List[int]:
        """Returns label instance ids as ints (sorted for determinism)."""
        obj = self._load_label_obj(path)
        ids: List[int] = []
        for k in obj.keys():
            try:
                ids.append(int(k))
            except Exception:
                # ignore non-instance keys if they exist
                continue
        ids.sort()
        return ids

    # def _load_seg_ids_only(self, path: Path) -> torch.Tensor:
    #     """Loads only seg ids (fast, avoids loading masks/boxes during indexing)."""
    #     with np.load(path, allow_pickle=True) as z:
    #         ids = torch.from_numpy(z["ids"].astype(np.int64))
    #     return ids


    # def __len__(self) -> int:
    #     if self.mode == "per_instance":
    #         assert self._instance_index is not None
    #         return len(self._instance_index)
    #     return len(self._image_index)

    def _format_training_sample(
        self,
        image: torch.Tensor,
        mask: torch.Tensor,
        label: object,
        *,
        image_id: str,
        instance_id: int,
    ) -> Dict[str, torch.Tensor]:
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

    # -------------------------
    # __getitem__ (showing the transform change in per_instance)
    # -------------------------
    def __getitem__(self, idx: int):
        if self.mode == "per_image":
            raise NotImplementedError("per_image mode is not implemented yet. Use per_instance mode.")

        assert self._instance_index is not None
        img_idx, inst_pos = self._instance_index[idx]
        rec = self._image_index[img_idx]

        if rec["img"] is None:
            raise ValueError(
                "This dataset was built with split_id_kind='image_id', so image paths are unknown.\n"
                "Either: (a) use split_id_kind='relpath', or (b) modify the dataset to resolve image_id -> relpath."
            )

        image = self._image_loader(rec["img"])     # CHW uint8
        seg = self._load_seg(rec["seg"])
        inst_id = int(seg["ids"][inst_pos].item())

        label = self._load_labels(rec["lab"], [inst_id])[0]
        mask = seg["masks"][inst_pos]

        # ---- NEW: apply torchvision image_transform by default ----
        # If user has a pair-transform (image,target)->(image,target), let it override.
        if self._pair_transforms is not None:
            target = {
                "image_id": rec["image_id"],
                "instance_id": inst_id,
                "mask": mask,
                "label": label,
                "box": seg["boxes"][inst_pos],
            }
            image, target = self._pair_transforms(image, target)
            mask = target["mask"]
            label = target["label"]
        else:
            # Default path: image-only torchvision transform
            image = self._image_transform(image)

        return self._format_training_sample(
            image=image,
            mask=mask,
            label=label,
            image_id=rec["image_id"],
            instance_id=inst_id,
        )


    # def __getitem__(self, idx: int):
    #     if self.mode == "per_image":
    #         raise NotImplementedError("per_image mode is not implemented yet. Use per_instance mode.")

    #     assert self._instance_index is not None
    #     img_idx, inst_pos = self._instance_index[idx]
    #     rec = self._image_index[img_idx]

    #     if rec["img"] is None:
    #         raise ValueError(
    #             "This dataset was built with split_id_kind='image_id', so image paths are unknown.\n"
    #             "Either: (a) use split_id_kind='relpath', or (b) modify the dataset to resolve image_id -> relpath."
    #         )

    #     image = self._image_loader(rec["img"])
    #     seg = self._load_seg(rec["seg"])
    #     inst_id = int(seg["ids"][inst_pos].item())

    #     label = self._load_labels(rec["lab"], [inst_id])[0]
    #     mask = seg["masks"][inst_pos]

    #     if self.cfg.transforms:
    #         target = {
    #             "image_id": rec["image_id"],
    #             "instance_id": inst_id,
    #             "mask": mask,
    #             "label": label,
    #             "box": seg["boxes"][inst_pos],
    #         }
    #         image, target = self.cfg.transforms(image, target)
    #         mask = target["mask"]
    #         label = target["label"]

    #     return self._format_training_sample(
    #         image=image,
    #         mask=mask,
    #         label=label,
    #         image_id=rec["image_id"],
    #         instance_id=inst_id,
    #     )


    # -------------------------
    # Loaders
    # -------------------------
    # def _default_image_loader(self, path: Path) -> torch.Tensor:
    #     img = Image.open(path).convert("RGB")
    #     arr = np.asarray(img)  # HWC uint8
    #     return torch.from_numpy(arr).permute(2, 0, 1).contiguous()

    def _default_image_loader(self, path: Path) -> torch.Tensor:
        # EXIF-safe, CHW uint8
        img = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
        arr = np.asarray(img)  # HWC uint8
        return torch.from_numpy(arr).permute(2, 0, 1).contiguous()

    def _load_seg(self, path: Path) -> Dict[str, torch.Tensor]:
        with np.load(path, allow_pickle=True) as z:
            ids = torch.from_numpy(z["ids"].astype(np.int64))
            masks_np = z["masks"]  # (N,H,W)
        masks = torch.from_numpy(masks_np).to(self.cfg.masks_dtype)

        boxes = torch.from_numpy(z["boxes"]).to(self.cfg.boxes_dtype)

        out: Dict[str, torch.Tensor] = {"ids": ids, "masks": masks, "boxes": boxes}

        if self.include_scores and "scores" in z and z["scores"] is not None:
            out["scores"] = torch.from_numpy(z["scores"]).to(torch.float32)

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


    # def _load_labels(self, path: Path, instance_ids: Sequence[int]=None) -> List[object]:
    #     if path not in self._label_cache:
    #         # self._label_cache[path] = json.loads(path.read_text())
    #         self._label_cache[path] = readPersistenceData(path)
    #     obj = self._label_cache[path]

    #     labels: List[object] = []
    #     for iid in instance_ids:
    #         raw = obj[str(int(iid))]
    #         base = {
    #             "median_persistence_hours": float(raw["median_persistence_hours"]),
    #             "p90_persistence_hours": float(raw["p90_persistence_hours"]),
    #             "persistence_conf": float(raw["persistence_conf"]),
    #         }
    #         labels.append(self._label_transform(base))
    #     return labels


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

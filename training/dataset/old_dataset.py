

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Literal, Optional, Sequence, Tuple, Union

import json
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image


import math


SplitName = Literal["train", "val", "test"]
ReturnMode = Literal["per_image", "per_instance"]


def _read_lines(path: Union[str, Path]) -> List[str]:
    p = Path(path)
    return [ln.strip() for ln in p.read_text().splitlines() if ln.strip() and not ln.strip().startswith("#")]


def _load_image_id_map(path: Union[str, Path]) -> Dict[str, str]:
    """
    Loads mapping: image_relpath -> image_id from a text file.

    Supported formats per line:
      - "<relpath>\t<image_id>"
      - "<relpath>,<image_id>"
      - "<relpath> <image_id>" (last token is id)

    Lines starting with # are ignored.
    """
    mapping: Dict[str, str] = {}
    for ln in _read_lines(path):
        if "\t" in ln:
            rel, iid = ln.split("\t", 1)
        elif "," in ln:
            rel, iid = ln.split(",", 1)
        else:
            parts = ln.split()
            if len(parts) < 2:
                raise ValueError(f"Bad mapping line (need relpath + id): {ln!r}")
            rel, iid = " ".join(parts[:-1]), parts[-1]
        mapping[rel.strip()] = iid.strip()
    return mapping


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
    Loads split IDs (image_id) from either:
      - split_json: {"train":[...], "val":[...], "test":[...]}  (or nested under "splits")
      - explicit train/val/test file paths
      - split_dir/{train,val,test}.txt
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


def label_transform_log_ratio(raw: dict) -> torch.Tensor:
    """
    raw dict contains:
        raw["median_persistence_hours"] = t50_n
        raw["p90_persistence_hours"] = t90_n
        raw["persistence_conf"] = c_n

    Returns tensor([y50, yr, yc]).
    """
    t50 = float(raw["median_persistence_hours"])
    t90 = float(raw["p90_persistence_hours"])
    c   = float(raw["persistence_conf"])

    # Small epsilon for numerical safety
    eps = 1e-8

    # y^50 = log(t50)
    y50 = math.log(max(t50, eps))

    # y^r = log( t90/t50 - 1 )
    ratio = t90 / max(t50, eps) - 1.0
    yr = math.log(max(ratio, eps))

    # y^c = c
    yc = c

    return torch.tensor([y50, yr, yc], dtype=torch.float32)



@dataclass(frozen=True)
class SegmentationDatasetConfig:
    # Roots
    image_root: Union[str, Path]
    seg_root: Union[str, Path]
    labels_root: Union[str, Path]

    # mapping relpath -> image_id
    image_id_map_file: Union[str, Path]

    # Where to find per-image segmentation/labels within <root>/<image_id>/...
    # You can set these to match your actual filenames.
    seg_npz_name: str = "segmentation_results.npz"
    labels_json_name: str = "persistence_parameters.json"

    # Split specification (choose ONE style)
    split_dir: Optional[Union[str, Path]] = None                # contains train.txt/val.txt/test.txt (image_id per line)
    split_json: Optional[Union[str, Path]] = None               # JSON with {"train":[...],...} or {"splits":{...}}
    train_split_file: Optional[Union[str, Path]] = None
    val_split_file: Optional[Union[str, Path]] = None
    test_split_file: Optional[Union[str, Path]] = None

    # Reading behavior
    image_loader: Optional[Callable[[Path], torch.Tensor]] = None
    # Optional transforms on (image, target). Keep it torchvision-style if you like.
    transforms: Optional[Callable[[torch.Tensor, Dict], Tuple[torch.Tensor, Dict]]] = None

    # Optional transform for labels: takes {"median_persistence_hours":..., "p90_persistence_hours":..., "persistence_conf":...} -> anything
    label_transform: Optional[Callable[[Dict[str, float]], object]] = None

    # Masks storage: if your npz stores masks as float/bool/uint8, we’ll cast.
    masks_dtype: torch.dtype = torch.uint8
    boxes_dtype: torch.dtype = torch.float32

    def make_dataset(
        self,
        split: SplitName,
        *,
        mode: ReturnMode = "per_image",
        include_scores: bool = False,
        strict: bool = True,
    ) -> Dataset:
        """
        mode:
          - "per_image": each __getitem__ returns all instances for an image
          - "per_instance": each __getitem__ returns a single instance (increases length)

        strict:
          - True: raise if a split ID can’t be resolved to an image path or missing files
          - False: silently skip missing entries when building the index
        """
        return _SegmentationDataset(
            cfg=self,
            split=split,
            mode=mode,
            include_scores=include_scores,
            strict=strict,
        )


class _SegmentationDataset(Dataset):
    def __init__(
        self,
        cfg: SegmentationDatasetConfig,
        split: SplitName,
        mode: ReturnMode,
        include_scores: bool,
        strict: bool,
    ):
        super().__init__()
        self.cfg = cfg
        self.split = split
        self.mode = mode
        self.include_scores = include_scores
        self.strict = strict

        self.image_root = Path(cfg.image_root)
        self.seg_root = Path(cfg.seg_root)
        self.labels_root = Path(cfg.labels_root)

        self._relpath_to_id = _load_image_id_map(cfg.image_id_map_file)

        split_ids = _load_split_ids(
            split,
            split_dir=cfg.split_dir,
            train_file=cfg.train_split_file,
            val_file=cfg.val_split_file,
            test_file=cfg.test_split_file,
            split_json=cfg.split_json,
        )

        # TODO update to use image ids directly instead of relpaths

        # Invert mapping to get an image relpath for each image_id
        # (If multiple relpaths share an id, pick the first deterministically.)
        id_to_relpath: Dict[str, str] = {}
        for rel, iid in self._relpath_to_id.items():
            id_to_relpath.setdefault(iid, rel)

        # Build base per-image index: list of dicts with resolved paths
        self._image_index: List[Dict[str, Path]] = []
        for iid in split_ids:
            rel = id_to_relpath.get(iid)
            if rel is None:
                if strict:
                    raise KeyError(f"image_id {iid!r} not found in mapping file")
                continue

            img_path = self.image_root / rel
            seg_path = self.seg_root / iid / cfg.seg_npz_name
            lab_path = self.labels_root / iid / cfg.labels_json_name

            missing = [p for p in [img_path, seg_path, lab_path] if not p.exists()]
            if missing:
                if strict:
                    raise FileNotFoundError(f"Missing files for image_id={iid}: {missing}")
                continue

            self._image_index.append(
                {"image_id": Path(iid), "img": img_path, "seg": seg_path, "lab": lab_path}
            )

        # If per-instance, precompute (image_idx, instance_pos) pairs.
        # This costs one pass loading seg files, but gives a clean constant-time __getitem__.
        self._instance_index: Optional[List[Tuple[int, int]]] = None
        if self.mode == "per_instance":
            pairs: List[Tuple[int, int]] = []
            for img_idx, rec in enumerate(self._image_index):
                seg = np.load(rec["seg"], allow_pickle=True)
                ids = seg["ids"]
                n = int(len(ids))
                for inst_pos in range(n):
                    pairs.append((img_idx, inst_pos))
            self._instance_index = pairs

        self._label_transform = cfg.label_transform or label_transform_log_ratio
        self._image_loader = cfg.image_loader or self._default_image_loader
        self._label_cache = {}

    def __len__(self) -> int:
        if self.mode == "per_instance":
            assert self._instance_index is not None
            return len(self._instance_index)
        return len(self._image_index)

    def _format_training_sample(
        self,
        image: torch.Tensor,
        mask: torch.Tensor,
        label: object,
        *,
        image_id: str,
        instance_id: int,
    ) -> Dict[str, torch.Tensor]:
        """
        Produces the exact keys your training code expects.
        Assumes label_transform returns torch.Tensor([y50, yr, yc]).
        """
        if not torch.is_tensor(label):
            raise TypeError(
                "label_transform must return a torch.Tensor of shape (3,) for this training format."
            )
        if label.numel() != 3:
            raise ValueError(f"Expected label tensor with 3 elements [y50, yr, yc], got shape {tuple(label.shape)}")

        label = label.to(torch.float32).view(-1)
        y50, yr, yc = label[0], label[1], label[2]

        # Ensure mask is something like uint8/bool; training often expects 0/1 mask.
        # If yours is uint8 already, this is a no-op.
        if mask.dtype not in (torch.uint8, torch.bool):
            mask = mask.to(torch.uint8)

        return {
            "image": image,          # (C,H,W)
            "mask": mask,            # (H,W)
            "y50": y50,              # ()
            "yr": yr,                # ()
            "yc": yc,                # ()
            # Optional metadata (won't hurt unless your _unpack_batch is strict)
            "image_id": image_id,
            "instance_id": torch.tensor(instance_id, dtype=torch.int64),
        }


    def __getitem__(self, idx: int):
        if self.mode == "per_image":
            # Throw error -- currently not supported
            raise NotImplementedError("per_image mode is not implemented yet. Use per_instance mode.")
            rec = self._image_index[idx]
            image = self._image_loader(rec["img"])
            seg = self._load_seg(rec["seg"])
            labels = self._load_labels(rec["lab"], seg["ids"].tolist())
            target = self._build_target_all(rec, seg, labels)

            if self.cfg.transforms:
                image, target = self.cfg.transforms(image, target)
            return image, target

        # per_instance
        assert self._instance_index is not None
        img_idx, inst_pos = self._instance_index[idx]
        rec = self._image_index[img_idx]

        image = self._image_loader(rec["img"])             # (C,H,W)
        seg = self._load_seg(rec["seg"])
        inst_id = int(seg["ids"][inst_pos].item())

        labels = self._load_labels(rec["lab"], [inst_id])  # list length 1
        label = labels[0]                                  # expected tensor([y50, yr, yc])

        mask = seg["masks"][inst_pos]                      # (H,W)

        # If you have transforms that expect the old (image, target) format, keep using them:
        if self.cfg.transforms:
            # Build a small target for transforms
            target = {
                "image_id": rec["image_id"].name,
                "instance_id": inst_id,
                "mask": mask,
                "label": label,
                "box": seg["boxes"][inst_pos],
            }
            image, target = self.cfg.transforms(image, target)
            mask = target["mask"]
            label = target["label"]

        return self._format_training_sample(
            image=image,
            mask=mask,
            label=label,
            image_id=rec["image_id"].name,
            instance_id=inst_id,
        )


    # -------------------------
    # Loaders
    # -------------------------
    def _default_image_loader(self, path: Path) -> torch.Tensor:
        # CHW uint8
        img = Image.open(path).convert("RGB")
        arr = np.asarray(img)  # HWC uint8
        ten = torch.from_numpy(arr).permute(2, 0, 1).contiguous()
        return ten

    def _load_seg(self, path: Path) -> Dict[str, torch.Tensor]:
        z = np.load(path, allow_pickle=True)

        ids = torch.from_numpy(z["ids"].astype(np.int64))

        masks_np = z["masks"]
        # Expect (N,H,W); cast to cfg.masks_dtype
        masks = torch.from_numpy(masks_np).to(self.cfg.masks_dtype)

        boxes = torch.from_numpy(z["boxes"]).to(self.cfg.boxes_dtype)

        out = {"ids": ids, "masks": masks, "boxes": boxes}

        if self.include_scores and "scores" in z and z["scores"] is not None:
            out["scores"] = torch.from_numpy(z["scores"]).to(torch.float32)

        return out

    def _load_labels(self, path: Path, instance_ids: Sequence[int]) -> List[object]:
        if path not in self._label_cache:
            obj = json.loads(path.read_text())
            self._label_cache[path] = obj
        else:
            obj = self._label_cache[path]

        labels = []
        for iid in instance_ids:
            raw = obj[str(int(iid))]
            base = {k: float(raw[k]) for k in ("median_persistence_hours", "p90_persistence_hours", "persistence_conf")}
            labels.append(self._label_transform(base))
        return labels


    # def _load_labels(self, path: Path, instance_ids: Sequence[int]) -> List[object]:
    #     """
    #     Assumes labels.json is keyed by instance id (string or int), like:
    #     {
    #       "12": {"median_persistence_hours": 0.3, "p90_persistence_hours": 0.8, "persistence_conf": 0.95},
    #       "99": {"median_persistence_hours": 0.1, "p90_persistence_hours": 0.2, "persistence_conf": 0.70}
    #     }

    #     Returns labels aligned to instance_ids.
    #     """
    #     obj = json.loads(path.read_text())

    #     labels: List[object] = []
    #     for iid in instance_ids:
    #         key1 = str(int(iid))
    #         key2 = int(iid)
    #         if key1 in obj:
    #             raw = obj[key1]
    #         elif key2 in obj:
    #             raw = obj[key2]
    #         else:
    #             if self.strict:
    #                 raise KeyError(f"Instance id {iid} missing in labels file {path}")
    #             raw = {"median_persistence_hours": float("nan"), "p90_persistence_hours": float("nan"), "persistence_conf": float("nan")}

    #         # enforce keys exist
    #         base = {
    #             "median_persistence_hours": float(raw["median_persistence_hours"]),
    #             "p90_persistence_hours": float(raw["p90_persistence_hours"]),
    #             "persistence_conf": float(raw["persistence_conf"]),
    #         }
    #         labels.append(self._label_transform(base))
    #     return labels

    # -------------------------
    # Target builders
    # -------------------------
    def _build_target_all(
        self,
        rec: Dict[str, Path],
        seg: Dict[str, torch.Tensor],
        labels: List[object],
    ) -> Dict:
        # For per-image mode, return all masks and aligned labels.
        target: Dict = {
            "image_id": rec["image_id"].name,
            "instance_ids": seg["ids"],  # (N,)
            "masks": seg["masks"],       # (N,H,W)
            "boxes": seg["boxes"],       # (N,4)
            "labels": labels,            # list aligned with instance_ids
        }
        if "scores" in seg:
            target["scores"] = seg["scores"]
        return target

    def _build_target_one(
        self,
        rec: Dict[str, Path],
        seg: Dict[str, torch.Tensor],
        inst_pos: int,
        label: object,
    ) -> Dict:
        # For per-instance mode, slice one instance.
        target: Dict = {
            "image_id": rec["image_id"].name,
            "instance_id": int(seg["ids"][inst_pos].item()),
            "mask": seg["masks"][inst_pos],   # (H,W)
            "box": seg["boxes"][inst_pos],    # (4,)
            "label": label,
        }
        if "scores" in seg:
            target["score"] = float(seg["scores"][inst_pos].item())
        return target



# cfg = SegmentationDatasetConfig(
#     image_root=".../images",
#     seg_root=".../segmentations",
#     labels_root=".../labels",
#     image_id_map_file=".../image_relpath_to_id.txt",
#     split_dir=".../splits/v1",  # contains train.txt/val.txt/test.txt (image_id per line)
#     seg_npz_name="segmentation.npz",
#     labels_json_name="labels.json",
#     # Optional label transform: e.g. turn dict -> tensor
#     label_transform=lambda d: torch.tensor([d["med"], d["p90"], d["conf"]], dtype=torch.float32),
# )

# train_ds = cfg.make_dataset("train", mode="per_image")
# train_instances = cfg.make_dataset("train", mode="per_instance")

# make_split_config.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Literal, Sequence, Tuple, Union, Optional
import json
import random

import hydra
from omegaconf import OmegaConf

from spf_io.persistence_io import readPersistenceData, PERSISTENCE_FILE_NAME


SplitName = Literal["train", "val", "test"]
SplitIdKind = Literal["relpath", "image_id"]
SplitGranularity = Literal["image", "image_instance"]


# @dataclass(frozen=True)
@dataclass
class SplitGenConfig:
    # Required (top-level)
    search_path_config: str
    split_json_out: str

    # Split behavior (top-level)
    output_root_dir: Optional[str] = None
    train_ratio: float = 0.8
    val_ratio: float = 0.1
    test_ratio: float = 0.1
    split_id_kind: SplitIdKind = "image_id"
    split_granularity: SplitGranularity = "image"
    seed: int = 42

    # Instance mode extras
    drop_empty_instances: bool = True

    def get_search_subdir(self) -> Path:
        print(f"Using search_path_config: {self.search_path_config}")
        print(f"Using output_root_dir: {self.output_root_dir}")
        return Path(self.output_root_dir) / "full_results" / self.search_path_config



# -----------------------------
# Utilities
# -----------------------------
def normalize_ratios(train: float, val: float, test: float) -> Tuple[float, float, float]:
    s = float(train) + float(val) + float(test)
    if s <= 0:
        raise ValueError("Split ratios must sum to a positive number.")
    return train / s, val / s, test / s


def compute_split_counts(n: int, ratios: Tuple[float, float, float]) -> Tuple[int, int, int]:
    tr, vr, _ = ratios
    n_train = int(round(n * tr))
    n_val = int(round(n * vr))
    if n_train + n_val > n:
        n_val = max(0, n - n_train)
    n_test = n - n_train - n_val
    return n_train, n_val, n_test


def shuffled_split(
    items: Sequence[object],
    *,
    ratios: Tuple[float, float, float],
    seed: int,
) -> Tuple[List[object], List[object], List[object]]:
    rng = random.Random(seed)
    arr = list(items)
    rng.shuffle(arr)
    n_train, n_val, _ = compute_split_counts(len(arr), ratios)
    train = arr[:n_train]
    val = arr[n_train : n_train + n_val]
    test = arr[n_train + n_val :]
    return train, val, test


def find_persistence_files(
    search_root: Union[str, Path],
    *,
    persistence_file_name: str = PERSISTENCE_FILE_NAME,
) -> Dict[str, Path]:
    if not search_root.exists():
        raise FileNotFoundError(f"search root does not exist: {search_root}")

    out: Dict[str, Path] = {}
    for p in search_root.rglob(persistence_file_name):
        image_id = p.parent.name
        out[image_id] = p
    return out


def extract_instance_ids(persistence_path: Union[str, Path]) -> List[int]:
    """
    Supports either:
      - dict with keys being instance-id strings (what your dataset expects)
      - list of dicts with an 'id' field (what you mentioned)
    """
    obj = readPersistenceData(str(persistence_path))
    print("Type of persistence data object:", type(obj))

    ids: List[int] = []
    if isinstance(obj, dict):
        for k in obj.keys():
            try:
                ids.append(int(k))
            except Exception:
                pass
    elif isinstance(obj, list):
        for entry in obj:
            if isinstance(entry, dict) and "id" in entry:
                try:
                    ids.append(int(entry["id"]))
                except Exception:
                    pass
    else:
        raise TypeError(f"Unsupported persistence object type: {type(obj)} at {persistence_path}")

    ids.sort()
    return ids


def id_token_for_image(
    image_id: str,
    *,
    split_id_kind: SplitIdKind,
    image_id_to_relpath: Callable[[str], str],
) -> str:
    return image_id if split_id_kind == "image_id" else image_id_to_relpath(image_id)


# -----------------------------
# Formatting helpers (reusable for non-random splits too)
# -----------------------------
def format_image_splits(
    train: Sequence[str],
    val: Sequence[str],
    test: Sequence[str],
) -> Dict[str, List[str]]:
    return {"train": list(train), "val": list(val), "test": list(test)}


def format_instance_splits(
    train: Sequence[Tuple[str, int]],
    val: Sequence[Tuple[str, int]],
    test: Sequence[Tuple[str, int]],
) -> Dict[str, List[List[object]]]:
    # Emit JSON list form: ["id_token", instance_id]
    return {
        "train": [[a, int(b)] for a, b in train],
        "val": [[a, int(b)] for a, b in val],
        "test": [[a, int(b)] for a, b in test],
    }


def write_split_json(obj: Dict[str, object], out_path: Union[str, Path]) -> None:
    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=2, sort_keys=True))


# -----------------------------
# Generator
# -----------------------------
def generate_random_splits(
    cfg: SplitGenConfig,
    *,
    image_id_to_relpath: Callable[[str], str],
) -> Dict[str, object]:
    ratios = normalize_ratios(cfg.train_ratio, cfg.val_ratio, cfg.test_ratio)

    image_id_to_persistence = find_persistence_files(cfg.get_search_subdir())
    image_ids = sorted(image_id_to_persistence.keys())

    if cfg.split_granularity == "image":
        train_ids, val_ids, test_ids = shuffled_split(image_ids, ratios=ratios, seed=cfg.seed)

        # Map to id tokens (image_id or relpath)
        train_tok = [id_token_for_image(i, split_id_kind=cfg.split_id_kind, image_id_to_relpath=image_id_to_relpath) for i in train_ids]
        val_tok   = [id_token_for_image(i, split_id_kind=cfg.split_id_kind, image_id_to_relpath=image_id_to_relpath) for i in val_ids]
        test_tok  = [id_token_for_image(i, split_id_kind=cfg.split_id_kind, image_id_to_relpath=image_id_to_relpath) for i in test_ids]

        splits = format_image_splits(train_tok, val_tok, test_tok)

    elif cfg.split_granularity == "image_instance":
        # Build all (image_id, iid) items
        items: List[Tuple[str, int]] = []
        for image_id, p in image_id_to_persistence.items():
            iids = extract_instance_ids(p)
            if not iids and cfg.drop_empty_instances:
                continue
            for iid in iids:
                items.append((image_id, iid))

        train_pairs, val_pairs, test_pairs = shuffled_split(items, ratios=ratios, seed=cfg.seed)

        # Map image_id -> token (image_id or relpath)
        def map_pair(pair: Tuple[str, int]) -> Tuple[str, int]:
            img_id, iid = pair
            tok = id_token_for_image(img_id, split_id_kind=cfg.split_id_kind, image_id_to_relpath=image_id_to_relpath)
            return tok, int(iid)

        train_tok = [map_pair(p) for p in train_pairs]
        val_tok   = [map_pair(p) for p in val_pairs]
        test_tok  = [map_pair(p) for p in test_pairs]

        splits = format_instance_splits(train_tok, val_tok, test_tok)

    else:
        raise ValueError(f"Unknown split_granularity: {cfg.split_granularity!r}")

    return {
        "meta": {
            "output_root_dir": cfg.output_root_dir,
            "search_path_config": cfg.search_path_config,
            "split_id_kind": cfg.split_id_kind,
            "split_granularity": cfg.split_granularity,
            "ratios": {"train": cfg.train_ratio, "val": cfg.val_ratio, "test": cfg.test_ratio},
            "seed": cfg.seed,
        },
        "splits": splits,
    }



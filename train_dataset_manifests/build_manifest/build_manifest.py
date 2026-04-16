#!/usr/bin/env python3
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf
from tqdm import tqdm


IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".gif",
    ".webp",
    ".tif",
    ".tiff",
    ".jp2",
}


@dataclass
class SourceConfig:
    # Path relative to dataset_root by default.
    # Can also be absolute if allow_absolute_source_paths=true.
    path: str
    weight: float


@dataclass
class ManifestConfig:
    dataset_root: str
    sources: List[SourceConfig] = field(default_factory=list)

    max_data_slice_size: int = 100000
    d_slice_indices: List[int] = field(default_factory=lambda: [0])  # zero-based slice indices
    manifest_output_directory: str = "./manifests"
    data_mix_prefix: str = "data_mix"

    random_seed: Optional[int] = 1234
    recursive: bool = True
    allow_absolute_source_paths: bool = False

    # If true, sort discovered files before shuffling so results are stable
    # across machines/filesystems for the same seed.
    sort_before_shuffle: bool = True


cs = ConfigStore.instance()
cs.store(name="manifest_config", node=ManifestConfig)


def is_image_file(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS


def resolve_source_path(
    dataset_root: Path,
    source_path_str: str,
    allow_absolute_source_paths: bool,
) -> Path:
    source_path = Path(source_path_str)
    if source_path.is_absolute():
        if not allow_absolute_source_paths:
            raise ValueError(
                f"Source path '{source_path}' is absolute, but "
                "allow_absolute_source_paths=false."
            )
        resolved = source_path.resolve()
    else:
        resolved = (dataset_root / source_path).resolve()

    return resolved


def validate_source_under_root(dataset_root: Path, source_path: Path) -> None:
    try:
        source_path.relative_to(dataset_root)
    except ValueError as exc:
        raise ValueError(
            f"Source path '{source_path}' is not under dataset_root '{dataset_root}'."
        ) from exc


def discover_images(
    source_dir: Path,
    recursive: bool,
    sort_before_shuffle: bool,
) -> List[Path]:
    if not source_dir.exists():
        raise FileNotFoundError(f"Source directory does not exist: {source_dir}")
    if not source_dir.is_dir():
        raise NotADirectoryError(f"Source path is not a directory: {source_dir}")

    iterator = source_dir.rglob("*") if recursive else source_dir.glob("*")
    files = [p for p in tqdm(iterator, desc=f"Scanning {source_dir.name}", unit="file", leave=False) if is_image_file(p)]

    if sort_before_shuffle:
        files.sort()

    return files


def weighted_fair_mix(
    per_source_files: List[List[Path]],
    weights: List[float],
) -> List[Path]:
    """
    Build a mixed list using a deficit-based weighted fair scheduler.

    Properties:
    - Early prefixes track requested weights reasonably well.
    - Each source is internally pre-shuffled.
    - If a source runs out, it is simply skipped thereafter.
    """
    if len(per_source_files) != len(weights):
        raise ValueError("per_source_files and weights must have same length")

    if not per_source_files:
        return []

    for w in weights:
        if w < 0:
            raise ValueError(f"Weights must be non-negative, got {w}")

    if all(w == 0 for w in weights):
        raise ValueError("At least one source must have a positive weight")

    n = len(per_source_files)
    indices = [0] * n
    deficits = [0.0] * n
    active = [len(files) > 0 for files in per_source_files]
    mixed: List[Path] = []

    total_remaining = sum(len(files) for files in per_source_files)

    pbar = tqdm(total=total_remaining, desc="Mixing sources", unit="file")
    while len(mixed) < total_remaining:
        any_active = False

        # Accumulate deficit for still-active sources.
        for i in range(n):
            if active[i]:
                deficits[i] += weights[i]
                any_active = True

        if not any_active:
            break

        # Emit as many items as possible from sources whose deficit permits it.
        # A source can emit floor(deficit) items, but we interleave by repeatedly
        # choosing the currently largest deficit >= 1 to avoid long bursts.
        while True:
            eligible = [
                i for i in range(n)
                if active[i] and deficits[i] >= 1.0 and indices[i] < len(per_source_files[i])
            ]
            if not eligible:
                break

            # Pick the source with the highest current deficit.
            best_i = max(eligible, key=lambda i: deficits[i])

            mixed.append(per_source_files[best_i][indices[best_i]])
            pbar.update(1)
            indices[best_i] += 1
            deficits[best_i] -= 1.0

            if indices[best_i] >= len(per_source_files[best_i]):
                active[best_i] = False
                deficits[best_i] = 0.0

        # Handle tiny weights / stalled cases:
        # if no source has deficit >= 1 but some are active, force one emission
        # from the source with the largest deficit.
        if len(mixed) < total_remaining:
            eligible_active = [
                i for i in range(n)
                if active[i] and indices[i] < len(per_source_files[i])
            ]
            if not eligible_active:
                break

            if not any(deficits[i] >= 1.0 for i in eligible_active):
                best_i = max(eligible_active, key=lambda i: deficits[i])
                mixed.append(per_source_files[best_i][indices[best_i]])
                pbar.update(1)
                indices[best_i] += 1
                deficits[best_i] = max(0.0, deficits[best_i] - 1.0)

                if indices[best_i] >= len(per_source_files[best_i]):
                    active[best_i] = False
                    deficits[best_i] = 0.0

    pbar.close()
    return mixed


def to_relative_paths(dataset_root: Path, files: List[Path]) -> List[str]:
    rels = []
    for path in tqdm(files, desc="Resolving paths", unit="file", leave=False):
        rels.append(path.resolve().relative_to(dataset_root).as_posix())
    return rels


@hydra.main(version_base=None, config_name="manifest_config")
def main(cfg: ManifestConfig) -> None:
    print("Configuration:")
    print(OmegaConf.to_yaml(OmegaConf.structured(cfg)))

    if cfg.max_data_slice_size <= 0:
        raise ValueError("max_data_slice_size must be > 0")
    if not cfg.d_slice_indices:
        raise ValueError("d_slice_indices must not be empty")
    if any(idx < 0 for idx in cfg.d_slice_indices):
        raise ValueError("All d_slice_indices must be >= 0")
    if not cfg.sources:
        raise ValueError("No sources provided")

    rng = random.Random(cfg.random_seed)

    dataset_root = Path(cfg.dataset_root).resolve()
    if not dataset_root.exists():
        raise FileNotFoundError(f"dataset_root does not exist: {dataset_root}")
    if not dataset_root.is_dir():
        raise NotADirectoryError(f"dataset_root is not a directory: {dataset_root}")

    resolved_sources: List[Path] = []
    weights: List[float] = []

    for src in cfg.sources:
        if src.weight < 0:
            raise ValueError(f"Weight must be non-negative, got {src.weight} for {src.path}")

        source_dir = resolve_source_path(
            dataset_root=dataset_root,
            source_path_str=src.path,
            allow_absolute_source_paths=cfg.allow_absolute_source_paths,
        )
        validate_source_under_root(dataset_root, source_dir)

        resolved_sources.append(source_dir)
        weights.append(src.weight)

    per_source_files: List[List[Path]] = []
    for src_cfg, source_dir in zip(cfg.sources, resolved_sources):
        files = discover_images(
            source_dir=source_dir,
            recursive=cfg.recursive,
            sort_before_shuffle=cfg.sort_before_shuffle,
        )
        rng.shuffle(files)
        per_source_files.append(files)
        print(f"Found {len(files)} images under source='{src_cfg.path}' weight={src_cfg.weight}")

    mixed_files = weighted_fair_mix(per_source_files, weights)
    total = len(mixed_files)
    print(f"Total mixed images: {total}")

    output_dir = Path(cfg.manifest_output_directory).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    for d_slice_idx in cfg.d_slice_indices:
        start = d_slice_idx * cfg.max_data_slice_size
        end = min(start + cfg.max_data_slice_size, total)

        if start >= total:
            raise IndexError(
                f"d_slice_idx={d_slice_idx} is out of range. "
                f"Requested slice starts at {start}, but total mixed images is {total}."
            )

        selected = mixed_files[start:end]
        selected_rel = to_relative_paths(dataset_root, selected)

        output_path = output_dir / f"{cfg.data_mix_prefix}_{d_slice_idx}.txt"
        with output_path.open("w", encoding="utf-8") as f:
            for rel_path in selected_rel:
                f.write(rel_path + "\n")

        print(f"Wrote {len(selected_rel)} paths to: {output_path}")
        print(f"Slice range: [{start}, {end})")


if __name__ == "__main__":
    main()


"""

python build_manifest.py --config-path conf --config-name config

python build_manifest.py \
  --config-path conf \
  --config-name config \
  'd_slice_indices=[3]' \
  max_data_slice_size=50000 \
  data_mix_prefix=my_mix \
  manifest_output_directory=./out_manifests

python build_manifest.py \
  --config-path conf \
  --config-name config \
  'd_slice_indices=[0,1,2,3]' \
  max_data_slice_size=50000 \
  data_mix_prefix=my_mix \
  manifest_output_directory=./out_manifests

python build_manifest.py \
  --config-path conf \
  --config-name config \
  'sources=[{path:coco/train2017,weight:0.7},{path:lvis/train2017,weight:0.3}]'

"""
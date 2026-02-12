# image_id_pipeline/exporter.py
from __future__ import annotations


import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

from .registry import ImageRegistry


class RegistryExporter:
    def __init__(self, registry: ImageRegistry):
        self.registry = registry

    def export(self, out_dir: Path) -> None:
        out_dir.mkdir(parents=True, exist_ok=True)
        self._export_path_to_id(out_dir / "path_to_id.csv")
        self._export_id_to_paths(out_dir / "id_to_paths.jsonl")

    def _export_path_to_id(self, csv_path: Path) -> None:
        rows = self.registry.all_rows()
        with csv_path.open("w", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            w.writerow(["root", "relpath", "abspath", "image_id", "file_size", "mtime_ns", "indexed_at_ns"])
            for r in rows:
                w.writerow([r["root"], r["relpath"], r["abspath"], r["image_id"], r["file_size"], r["mtime_ns"], r["indexed_at_ns"]])

    def _export_id_to_paths(self, jsonl_path: Path) -> None:
        rows = self.registry.all_rows()
        id_map: Dict[str, List[dict]] = defaultdict(list)
        for r in rows:
            id_map[r["image_id"]].append({"root": r["root"], "relpath": r["relpath"], "abspath": r["abspath"]})

        with jsonl_path.open("w", encoding="utf-8") as f:
            for image_id in sorted(id_map.keys()):
                payload = {"image_id": image_id, "paths": sorted(id_map[image_id], key=lambda x: (x["root"], x["relpath"]))}
                f.write(json.dumps(payload, ensure_ascii=False) + "\n")

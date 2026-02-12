# labeling/io.py
from __future__ import annotations
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

# def load_json_labels(json_path: Path) -> Optional[List[Dict[str, Any]]]:
#     if not json_path.exists():
#         return None
#     with json_path.open("r", encoding="utf-8") as f:
#         return json.load(f)

# def save_json_labels(json_path: Path, labels: List[Dict[str, Any]]) -> None:
#     json_path.parent.mkdir(parents=True, exist_ok=True)
#     tmp_path = json_path.with_suffix(".json.tmp")
#     with tmp_path.open("w", encoding="utf-8") as f:
#         json.dump(labels, f, indent=2, ensure_ascii=False)
#     tmp_path.replace(json_path)

def index_by_id(labels: List[Dict[str, Any]], id_key: str = "id") -> Dict[str, Dict[str, Any]]:
    out = {}
    for rec in labels:
        if id_key in rec:
            out[str(rec[id_key])] = rec
    return out

def merge_defaults_with_existing(
    defaults: List[Dict[str, Any]],
    existing: Optional[List[Dict[str, Any]]],
    id_key: str = "id",
) -> List[Dict[str, Any]]:
    """
    - Keep any extra keys in existing records
    - Ensure every default id exists
    - Preserve existing values when present
    """
    if existing is None:
        return defaults

    existing_map = index_by_id(existing, id_key=id_key)

    merged = []
    for d in defaults:
        sid = str(d.get(id_key))
        if sid in existing_map:
            rec = dict(d)          # start from defaults
            rec.update(existing_map[sid])  # overwrite with existing values
            merged.append(rec)
        else:
            merged.append(d)

    # If existing has ids not in defaults, append them (so nothing is lost)
    default_ids = {str(d.get(id_key)) for d in defaults}
    for sid, rec in existing_map.items():
        if sid not in default_ids:
            merged.append(rec)

    return merged

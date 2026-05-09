"""One-shot preprocessor: manifest -> track_id -> location_key map.

Reads an annotation-tool manifest (the same format consumed by
AnnotationToolBackingDataset and generate_temporal_split.py), classifies each
track into a location key (space_id when present, else a discretized bbox-cell),
and emits a JSON sidecar consumed at eval time by LocationKeyedLookupInferencer.

Output schema:
    {
      "<track_id>": "<location_key>",
      ...
    }

The output path defaults to
    <manifest_dir>/<manifest_stem>.track_id_to_location_cell<N>.json
so train/test manifests sharing a directory don't collide and multiple cell
sizes can coexist for the same manifest.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from persistence_eval.methods.training.location_keying import location_key_for_track


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", type=Path, required=True,
                   help="Path to annotation-tool manifest JSON.")
    p.add_argument("--cell-size-px", type=int, default=64,
                   help="Cell size for transient tracks without space_id.")
    p.add_argument("--output", type=Path, default=None,
                   help="Defaults to <manifest_dir>/track_id_to_location_cell<N>.json")
    p.add_argument("--force", action="store_true", help="Overwrite existing output.")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    manifest_path = args.manifest.resolve()
    if not manifest_path.is_file():
        print(f"ERROR: manifest not found: {manifest_path}", file=sys.stderr)
        return 1

    manifest = json.loads(manifest_path.read_text())
    manifest_dir = manifest_path.parent

    mapping: dict[str, str] = {}
    counts_by_kind = {"space_id": 0, "cell": 0, "unlocated": 0}
    for rel in manifest["track_files"]:
        track = json.loads((manifest_dir / rel).read_text())
        key = location_key_for_track(track, args.cell_size_px)
        mapping[track["track_id"]] = key
        if key.startswith("cell_"):
            counts_by_kind["cell"] += 1
        elif key.startswith("transient_unlocated_"):
            counts_by_kind["unlocated"] += 1
        else:
            counts_by_kind["space_id"] += 1

    output = args.output.resolve() if args.output else (
        manifest_dir
        / f"{manifest_path.stem}.track_id_to_location_cell{args.cell_size_px}.json"
    )
    if output.exists() and not args.force:
        print(f"ERROR: {output} exists; use --force to overwrite.", file=sys.stderr)
        return 1
    output.write_text(json.dumps(mapping, indent=2, sort_keys=True))

    print(f"Wrote {output}")
    print(f"  total tracks:        {len(mapping)}")
    print(f"  space_id-keyed:      {counts_by_kind['space_id']}")
    print(f"  cell-keyed:          {counts_by_kind['cell']}  (cell_size_px={args.cell_size_px})")
    print(f"  unlocated transient: {counts_by_kind['unlocated']}")
    print(f"  unique location keys: {len(set(mapping.values()))}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

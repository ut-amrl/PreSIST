"""
Single source of truth for the openloris_location `*_correct_rot` rewrite mapping.

Mirrors the structure of exmaps_rewrites.py.

Imported by:
  - rewrite_correct_rot.py (rewrites existing manifests)
  - build_manifest/build_manifest.py (filters fresh discoveries)
"""

from __future__ import annotations

from typing import Dict


PREFIX = "openloris_location/"


# old_subdir (relative to openloris_location/) -> new_subdir (relative to openloris_location/)
SUBDIR_REWRITES: Dict[str, str] = {
    "home/002": "home/002_correct_rot",
    "home/003": "home/003_correct_rot",
}


# Reverse map.
NEW_TO_OLD: Dict[str, str] = {new: old for old, new in SUBDIR_REWRITES.items()}

"""
Single source of truth for the exmaps `*_correct_rot` rewrite mapping.

Imported by both:
  - rewrite_exmaps_correct_rot.py (rewrites existing manifests)
  - build_manifest/build_manifest.py (filters fresh discoveries)
"""

from __future__ import annotations

from typing import Dict


EXMAPS_PREFIX = "exmaps/"


# old_subdir (relative to exmaps/) -> new_subdir (relative to exmaps/)
SUBDIR_REWRITES: Dict[str, str] = {
    "2020-03-22/run_1/ar_core_data": "2020-03-22/run_1/ar_core_data_correct_rot",
    "2020-03-22/run_2/ar_core_data": "2020-03-22/run_2/ar_core_data_correct_rot",
    "2020-03-22/run_3/ar_core_data": "2020-03-22/run_3/ar_core_data_correct_rot",
    "2020-03-22/run_4/ar_core_data": "2020-03-22/run_4/ar_core_data_correct_rot",
    "2020-03-22/run_5/ar_core_data": "2020-03-22/run_5/ar_core_data_correct_rot",
    "2020-03-23/coop_local/run_1/ar_core_data": "2020-03-23/coop_local/run_1/ar_core_data_correct_rot",
    "2020-03-23/coop_local/run_2/ar_core_data": "2020-03-23/coop_local/run_2/ar_core_data_correct_rot",
    "2020-03-23/coop_local/run_3/ar_core_data": "2020-03-23/coop_local/run_3/ar_core_data_correct_rot",
    "2020-03-29/run_1": "2020-03-29/run_1_correct_rot",
    "2020-04-04/run_1": "2020-04-04/run_1_correct_rot",
    "2020-04-23/run_1": "2020-04-23/run_1_correct_rot",
    "2020-04-25/run_1": "2020-04-25/run_1_correct_rot",
    "2020-04-26/run_1": "2020-04-26/run_1_correct_rot",
    "2020-04-27/run_1": "2020-04-27/run_1_correct_rot",
    "2020-05-02/run_1": "2020-05-02/run_1_correct_rot",
    "2020-05-06/run_1": "2020-05-06/run_1_correct_rot",
    "2020-05-08/run_1": "2020-05-08/run_1_correct_rot",
    "2020-05-16/run_1": "2020-05-16/run_1_correct_rot",
    "2020-05-25/run_1": "2020-05-25/run_1_correct_rot",
    "2020-06-13/run_1/data_all": "2020-06-13/run_1/data_all_correct_rot",
    "2020-06-13/run_2/data_all": "2020-06-13/run_2/data_all_correct_rot",
    "2020-06-17/evening/run_1/data_all": "2020-06-17/evening/run_1/data_all_correct_rot",
}


# Reverse map: new_subdir -> old_subdir. Built once.
NEW_TO_OLD: Dict[str, str] = {new: old for old, new in SUBDIR_REWRITES.items()}

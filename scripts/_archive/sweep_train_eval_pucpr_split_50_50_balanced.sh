#!/bin/bash
# Train + eval the Perpetua family against the
# PUCPR_repaired_split_flanks_fixed 50/50 split, with the per-track-balanced
# PUCPR test benchmark on the eval side. Replaces the archived
# sweep_train_eval_pucpr_repaired_split_50_50_balanced.sh.
#
# Wires the per-(dist, grouping) train/eval configs created for this dataset
# via TRAIN_CFG_TEMPLATE / EVAL_CFG_TEMPLATE — sweep_train_eval.sh substitutes
# {DIST} and {GROUPING} into those templates per variant. Because the new
# train cfgs hardcode `grouping:` per file and `experiment.name` already
# encodes both axes plus the dataset, no EXP_SUFFIX is needed.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

SPLIT_DIR=/workspace/data/input/new_pklot/postprocessed/annotations/PUCPR_repaired_split_flanks_fixed/splits/50_50

export TRAIN_PRESET="${TRAIN_PRESET:-annotation_tool_pklot_split_train}"
export BENCH_PRESET="${BENCH_PRESET:-annotation_tool_pucpr_split_test_balanced}"

export TEST_MANIFEST="${TEST_MANIFEST:-$SPLIT_DIR/test_from_2012-10-16.json}"
export TEST_SIDECAR="${TEST_SIDECAR:-$SPLIT_DIR/test_from_2012-10-16.track_id_to_location_cell${CELL_SIZE:-128}.json}"

export OUT_CSV="${OUT_CSV:-/workspace/data/output/persistence_eval_outputs/sweep_metrics_pucpr_split_50_50_balanced.csv}"

# Per-experiment cfg templates. {DIST} and {GROUPING} are substituted by
# sweep_train_eval.sh. The new train cfgs hardcode grouping in-file; we still
# pass `grouping=$GROUPING` from sweep_train_eval.sh (matches the in-file
# value, so Hydra accepts it as a redundant no-op).
export TRAIN_CFG_TEMPLATE="train_config_perpetua_{DIST}_{GROUPING}_pucpr_split_50_50"
export EVAL_CFG_TEMPLATE="eval_config_perpetua_{DIST}_{GROUPING}_pucpr_split_50_50_balanced"

# experiment.name in the new train cfgs is already
# `perpetua_<dist>_<grouping>_pucpr_split_50_50`, so EXP_SUFFIX is empty.
export EXP_SUFFIX="${EXP_SUFFIX:-_pucpr_split_50_50}"

exec bash "$SCRIPT_DIR/sweep_train_eval.sh" "$@"

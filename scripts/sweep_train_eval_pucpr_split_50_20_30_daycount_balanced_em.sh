#!/bin/bash
# Day-count train+eval for all Perpetua variants in EM mode on the 50/20/30
# split. Difference vs the non-_em wrapper:
#
#   - single_* variants run with `+trainer.fit_method=em_k1`, routing fits
#     through the upstream-faithful K=1 EM (censoring-aware) instead of the
#     naive closed-form MLE. Their experiment.name gets an additional
#     `_em_k1` suffix so checkpoints land at a parallel model_dir and the
#     existing naive-MLE results stay intact.
#   - mixture_* variants are unchanged from the non-_em wrapper — they
#     already use upstream EM via the mixture trainer. The shared
#     experiment.name means existing mixture checkpoints are reused
#     (skip-train kicks in).
#
# Test benchmark: per-track-balanced PUCPR test split
# (`annotation_tool_pucpr_split_50_20_30_test_balanced`).
#
# REQUIRES: spf_perpetua_container (jax import path for em_step).
# Day-count subsetting must be materialized first via
# generate_daycount_train_manifests.py.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

SPLIT_DIR=/workspace/data/input/new_pklot/postprocessed/annotations/PUCPR_repaired_split_flanks_fixed/splits/50_20_30

export DAYCOUNT_ROOT="${DAYCOUNT_ROOT:-$SPLIT_DIR/daycount}"

export TRAIN_PRESET="${TRAIN_PRESET:-annotation_tool_pklot_split_50_20_30_train}"
export BENCH_PRESET="${BENCH_PRESET:-annotation_tool_pucpr_split_50_20_30_test_balanced}"

export TEST_MANIFEST="${TEST_MANIFEST:-$SPLIT_DIR/test_from_2012-10-30.json}"
export TEST_SIDECAR="${TEST_SIDECAR:-$SPLIT_DIR/test_from_2012-10-30.track_id_to_location_cell${CELL_SIZE:-128}.json}"

# Single + mixture rows land in the same EM CSV so the writeup can plot
# the full 8-variant family side-by-side.
export OUT_CSV="${OUT_CSV:-/workspace/data/output/persistence_eval_outputs/sweep_metrics_pucpr_split_50_20_30_daycount_balanced_em.csv}"

export TRAIN_CFG_TEMPLATE="train_config_perpetua_{DIST}_{GROUPING}_pucpr_split_50_20_30_daycount"
export EVAL_CFG_TEMPLATE="eval_config_perpetua_{DIST}_{GROUPING}_pucpr_split_50_20_30_daycount_balanced"

# Shared suffix applied to all 8 variants. Singles get an *additional* _em_k1
# suffix layered on top (see SINGLE_EXP_SUFFIX below), so:
#   single_*  -> perpetua_<dist>_<grouping>_pucpr_split_50_20_30_daycount_em_k1
#   mixture_* -> perpetua_<dist>_<grouping>_pucpr_split_50_20_30_daycount
# Identical mixture exp.name means existing mixture checkpoints are reused.
export EXP_SUFFIX="${EXP_SUFFIX:-_pucpr_split_50_20_30_daycount}"

# Singles-only: extra suffix + Hydra train override. The daycount driver
# applies these only when DIST starts with "single_".
export SINGLE_EXP_SUFFIX="${SINGLE_EXP_SUFFIX:-_em_k1}"
export SINGLE_TRAIN_EXTRA="${SINGLE_TRAIN_EXTRA:-+trainer.fit_method=em_k1}"

exec bash "$SCRIPT_DIR/sweep_train_eval_daycount.sh" "$@"

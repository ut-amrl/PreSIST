#!/bin/bash
# Diagnostic wrapper: same trained models as the 50/50 PUCPR_repaired_split
# sweep, but EVALUATED ON THE TRAIN MANIFEST instead of the held-out test
# manifest. Used to expose train-vs-test gap as a function of subsample
# fraction (suspected mixture overfitting under temporal distribution shift).
#
# Reuses the existing trained-model directories (same EXP_SUFFIX as the test
# sweep), so trainings are skipped. FORCE_EVAL=1 is required because the
# upstream eval-skip glob in sweep_train_eval.sh matches across benchmark
# datasets and would otherwise reuse the test-side metric_results.json.
#
# Output CSV lives alongside the test CSV but with a _TRAIN suffix so the
# existing plot_train_pct_metrics.py pipeline can produce a parallel set of
# train-only plots without colliding.
#
# Variants/fractions/etc. are inherited from sweep_train_eval.sh — override
# the usual env vars (VARIANTS, FRACTIONS, SEED, CELL_SIZE, MIXTURE_FAST,
# etc.) if needed.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

SPLIT_DIR=/workspace/data/input/new_pklot/postprocessed/annotations/PUCPR_repaired_split/splits/50_50

# Same training preset as the test wrapper — the trainer side has no effect
# (training is skipped via persistence_params.json check), but the override
# still has to be valid Hydra.
export TRAIN_PRESET="${TRAIN_PRESET:-annotation_tool_pklot_repaired_split_train}"

# Point the benchmark at the train manifest's preset. Its `name:` differs
# from the test preset, so eval outputs land in their own dataset subdir
# (no collision with the test-side results).
export BENCH_PRESET="${BENCH_PRESET:-annotation_tool_pucpr_repaired_split_train}"

# Tell the upstream existence check + manifest_path override to use the
# train manifest, and feed the correct sidecar to the location-keyed
# inferencer.
export TEST_MANIFEST="${TEST_MANIFEST:-$SPLIT_DIR/train_through_2012-10-15.json}"
export TEST_SIDECAR="${TEST_SIDECAR:-$SPLIT_DIR/train_through_2012-10-15.track_id_to_location_cell${CELL_SIZE:-128}.json}"

# Parallel CSV — keeps the test-side sweep CSV intact.
export OUT_CSV="${OUT_CSV:-/workspace/data/output/persistence_eval_outputs/sweep_metrics_pucpr_repaired_split_50_50_TRAIN.csv}"

# Match the test wrapper's EXP_SUFFIX so MODEL_DIR points at the SAME trained
# models. Combined with the persistence_params.json skip check, this means
# the trainer is a no-op for every cell.
export EXP_SUFFIX="${EXP_SUFFIX:-_pucpr_repaired_split_50_50}"

# REQUIRED: the eval-skip find in sweep_train_eval.sh isn't scoped by
# benchmark dataset, so without FORCE_EVAL it would match the existing
# test-side metric_results.json and skip the train-side eval entirely.
export FORCE_EVAL="${FORCE_EVAL:-1}"

exec bash "$SCRIPT_DIR/sweep_train_eval.sh" "$@"

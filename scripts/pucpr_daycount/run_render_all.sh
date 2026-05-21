#!/bin/bash
# Driver: render PDF sweep plots + LaTeX tables for each (scenario, grouping)
# combination. Output: 6 scenarios × 3 groupings × 4 metrics = 72 PDFs + 72 .tex.
#
# Run inside spf_perpetua_container (matplotlib + container paths). Idempotent
# — re-running just overwrites the output files.
set -euo pipefail

CSV_ROOT="/workspace/data/output/persistence_eval_outputs"
PLOT_ROOT="/workspace/data/output/persistence_eval_plots"
SCRIPT="$(dirname "$0")/render_sweep_plots.py"

SCENARIOS=(
  "percentile_horizons"
  "percentile_scaled_horizons_5min"
  "percentile_scaled_horizons_15min"
  "percentile_anchor_window_horizons"
  "percentile_anchor_window_horizons_5min"
  "percentile_anchor_window_horizons_15min"
)
GROUPINGS=("both" "class" "location")

for scenario in "${SCENARIOS[@]}"; do
  PERP_CSV="$CSV_ROOT/sweep_metrics_pucpr_split_50_20_30_daycount_p0p01_mins5_all8_${scenario}_pm0p01_pf0p01.csv"
  VL_CSV="$CSV_ROOT/sweep_metrics_pucpr_split_50_20_30_daycount_p0p01_mins5_vision_llm_${scenario}_pm0p01_pf0p01.csv"
  for grouping in "${GROUPINGS[@]}"; do
    OUT_DIR="$PLOT_ROOT/daycount_${scenario}_matched_noise_${grouping}"
    echo "=== $scenario | $grouping -> $OUT_DIR"
    python "$SCRIPT" \
      --perp-csv "$PERP_CSV" \
      --vl-csv "$VL_CSV" \
      --grouping "$grouping" \
      --out-dir "$OUT_DIR"
  done
done

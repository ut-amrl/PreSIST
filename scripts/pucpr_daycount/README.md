# scripts/pucpr_daycount/

Train / evaluate / aggregate scripts that reproduce the PUCPR-PKLot daycount
sweep figures. All commands run inside `spf_perpetua_container`, with cwd at
the repo root.

```bash
# 0. one-time per split — materialize daycount subsets
python generate_daycount_train_manifests.py \
  --source-train-manifest /workspace/data/input/new_pklot/postprocessed/annotations/PUCPR_repaired_split_flanks_fixed/splits/50_20_30/train_through_2012-10-15.json \
  --days 0.5 1 2 3 4 5 7 10 14 21 28 35 --force

# 1. train + default-scenario eval (percentile_horizons, matched noise = 0.01)
bash scripts/pucpr_daycount/train_eval_perpetua.sh                   # 8 perpetua variants, no GPU
bash scripts/pucpr_daycount/train_vision.sh <GPU_IDS>                # vision train
bash scripts/pucpr_daycount/eval_vision.sh <GPU_IDS>                 # vision inference + eval

# 2. alternate-scenario eval over existing artifacts
EVAL_SCENARIO=percentile_scaled_horizons_5min  bash scripts/pucpr_daycount/eval_perpetua_scenarios.sh
EVAL_SCENARIO=percentile_scaled_horizons_15min bash scripts/pucpr_daycount/eval_perpetua_scenarios.sh
EVAL_SCENARIO=percentile_scaled_horizons_5min  bash scripts/pucpr_daycount/eval_vision_llm_scenarios.sh
EVAL_SCENARIO=percentile_scaled_horizons_15min bash scripts/pucpr_daycount/eval_vision_llm_scenarios.sh

# 3. plots + LaTeX tables — 6 scenarios × 3 groupings × 4 metrics
bash scripts/pucpr_daycount/run_render_all.sh
```

See [../../PKLot.md](../../PKLot.md) for split prep, the full knob reference
(`DAYS`, `KINDS`, `JOBS`, `EVAL_NOISE_PM/PF`, `FORCE_TRAIN`, `EXP_TAG`, …),
recipes for adding new variants, output paths, debug utilities, and
archive layout.

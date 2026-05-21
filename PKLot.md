# PKLot evaluation guide

End-to-end recipe for training and evaluating persistence models against the
PUCPR (PKLot) dataset using the **flanks-fixed** annotations
(`PUCPR_repaired_split_flanks_fixed/export_augmented.json`). The previous
annotations had a transient-track labelling defect (missing bounding absents);
every config/script described here targets the repaired manifest.

All commands run **inside the perpetua docker container** unless noted. Start
the container with `bash docker_start_datasets_plus_spf.sh` (the
`new_pklot` volume mount provides the data at `/workspace/data/input/new_pklot/`).

---

## 0. One-time: regenerate the 50/50 temporal split

The `splits/50_50/` directory may contain stale files copied from the old
manifest (check `split_metadata.json` — `source_manifest` must end in
`_flanks_fixed/export_augmented.json`). Regenerate if needed:

```bash
python generate_temporal_split.py \
  --manifest /workspace/data/input/new_pklot/postprocessed/annotations/PUCPR_repaired_split_flanks_fixed/export_augmented.json \
  --train-fraction 0.5 --force
```

This emits `train_through_2012-10-15.json`, `test_from_2012-10-16.json`, the
two `*.track_id_to_location_cell128.json` sidecars, and an updated
`split_metadata.json` under `.../PUCPR_repaired_split_flanks_fixed/splits/50_50/`.

---

## 1. Method families and configs

Three method families are wired up. All share the per-track-balanced PUCPR
test benchmark (`benchmark_dataset/annotation_tool_pucpr_split_test_balanced.yaml`)
and the same train manifest
(`train_dataset/annotation_tool_pklot_split_train{,_vision}.yaml`,
`cell_size_px: 128`).

| Family   | Trains? | Sweeps over (fraction, seed)? | Sweep script                                                                                  |
| -------- | ------- | ----------------------------- | --------------------------------------------------------------------------------------------- |
| Vision   | yes     | yes                           | `scripts/sweep_vision_anchor_midpoint_pucpr_split_50_50_balanced_train.sh`                    |
| Perpetua | yes     | yes                           | `scripts/sweep_train_eval_pucpr_split_50_50_balanced.sh`                                      |
| Gemma    | no      | no — single run               | (no script; one-shot `python eval_persistence.py --config-name=...`)                          |

For each family there is a matching **eval-on-train** (`_ON_TRAIN`) config and
script that points the eval benchmark at the TRAIN manifest
(`annotation_tool_pucpr_split_train_balanced.yaml`) so train-vs-test gap can be
plotted as a function of subsample fraction. Same trained checkpoints are
reused — only the benchmark dataset changes, and outputs land in a distinct
`pucpr_split_new_pklot_PUCPR_train_50_50_balanced/` subdir.

### Vision — `vision_anchor_midpoint`, balanced-train + balanced-eval

- Train cfg: `train_config_vision_anchor_midpoint_pucpr_split_50_50_balanced_train`
- Test eval cfg: `eval_config_vision_anchor_midpoint_pucpr_split_50_50_balanced_train_balanced_eval`
- ON_TRAIN eval cfg: `..._balanced_train_balanced_eval_ON_TRAIN`
- `experiment.name`: `vision_anchor_midpoint_pucpr_split_50_50_balanced_train`

### Perpetua — 8 variants

Cartesian product of `{single_exp, single_lognormal, mixture_exp, mixture_lognormal}` ×
`{location, class}`.

- Train cfg: `train_config_perpetua_<dist>_<grouping>_pucpr_split_50_50`
- Test eval cfg: `eval_config_perpetua_<dist>_<grouping>_pucpr_split_50_50_balanced`
- ON_TRAIN eval cfg: `..._balanced_ON_TRAIN`
- `experiment.name`: `perpetua_<dist>_<grouping>_pucpr_split_50_50`

### Gemma LLM — single run (not learned)

- Test eval cfg: `eval_config_llm_pucpr_split_50_50_gemma_v3iqrBehav4_mO20_percentile_balanced`
- ON_TRAIN eval cfg: `..._balanced_ON_TRAIN`
- `experiment.name`: `llm_pipeline_gemma_v3iqrBehav4_mO20_pucpr_split_50_50`

---

## 2. Run

### 2a. Vision sweep

```bash
# Train (default: fractions 0.1..1.0, seed 0). First positional is GPU IDs.
bash scripts/sweep_vision_anchor_midpoint_pucpr_split_50_50_balanced_train.sh 0,1,2,3 --jobs 4

# Eval on test set:
bash scripts/sweep_eval_vision_anchor_midpoint_pucpr_split_50_50_balanced_train_balanced_eval.sh 0

# Eval on train set (overfitting diagnostic — reuses checkpoints):
bash scripts/sweep_eval_vision_anchor_midpoint_pucpr_split_50_50_balanced_train_balanced_eval_ON_TRAIN.sh 0
```

Overrides (both train and eval sweeps): `--jobs N`, `--fractions "0.1 1.0"`, `--seeds "0 1 2"`, `--force` (retrain or re-eval even if outputs exist), `--config <CFG>` (train sweep only).

Eval CSV: `/workspace/data/output/persistence_eval_outputs/sweep_eval_metrics_vision_anchor_midpoint_pucpr_split_50_50_balanced_train_balanced_eval{,_ON_TRAIN}.csv`.

### 2b. Perpetua sweep

```bash
# Train + eval (test set):
bash scripts/sweep_train_eval_pucpr_split_50_50_balanced.sh

# Eval-on-train (reuses checkpoints; FORCE_EVAL=1 is set automatically):
bash scripts/sweep_train_eval_pucpr_split_50_50_balanced_ON_TRAIN.sh
```

Overrides (env vars, set before invoking):

| Env var          | Default                                                              | Notes                                                                 |
| ---------------- | -------------------------------------------------------------------- | --------------------------------------------------------------------- |
| `VARIANTS`       | all 8 `<dist>:<grouping>` pairs                                      | Space-separated. Restrict to a single variant for fast smoke tests.   |
| `FRACTIONS`      | `0.1 1.0 0.5 0.3 0.7 0.2 0.4 0.6 0.8 0.9`                            | Space-separated.                                                      |
| `SEED`           | `0`                                                                  |                                                                       |
| `CELL_SIZE`      | `128`                                                                | Must match the cell size in the train dataset YAML.                   |
| `MIXTURE_FAST`   | unset                                                                | When set, disables K-sweep / retrain loop for mixture trainers.       |
| `FORCE_TRAIN`    | unset                                                                | Retrain even if `persistence_params.json` exists.                     |
| `FORCE_EVAL`     | unset (test); `1` (ON_TRAIN)                                         | Re-eval even if a fresh `metric_results.json` exists for the cell.    |
| `OUT_CSV`        | `.../sweep_metrics_pucpr_split_50_50_balanced{,_ON_TRAIN}.csv`       | Override to write multiple runs side-by-side.                         |
| `TEST_MANIFEST`  | `.../splits/50_50/test_from_2012-10-16.json` (test) / `train_through_2012-10-15.json` (ON_TRAIN) | Both wrappers also export `TEST_SIDECAR` to match.                    |

Single-variant smoke test:
```bash
VARIANTS="single_exp:location" FRACTIONS="1.0" \
  bash scripts/sweep_train_eval_pucpr_split_50_50_balanced.sh
```

Single training cell (no sweep wrapper):
```bash
CUDA_VISIBLE_DEVICES=0 python train_persistence_models.py \
  --config-name=train_config_perpetua_single_exp_location_pucpr_split_50_50 \
  experiment.subsample_fraction=1.0 experiment.subsample_seed=0
```

### 2c. Gemma — single run

```bash
# Test set:
CUDA_VISIBLE_DEVICES=0 python eval_persistence.py \
  --config-name=eval_config_llm_pucpr_split_50_50_gemma_v3iqrBehav4_mO20_percentile_balanced

# Train set (fairness check — Gemma can't overfit, but exposes prompt drift):
CUDA_VISIBLE_DEVICES=0 python eval_persistence.py \
  --config-name=eval_config_llm_pucpr_split_50_50_gemma_v3iqrBehav4_mO20_percentile_balanced_ON_TRAIN
```

Gemma requires the vLLM server in `start_gemma_vllm_server.sh` to be running.
See that script's comments for GPU/memory recommendations.

---

## 3. Plotting

Three plot scripts share a manifest format (`label,path[,has_train_pct]` CSV
with container paths to evaluation-run dirs). Build a manifest from the sweep
CSV with `scripts/build_plot_manifest.py`, then feed it to the plotter.

### 3a. Train-percentage curves (`plot_train_pct_metrics.py`)

For each metric, plots `metric vs train fraction` with one curve per `label`
in the manifest. Use this for the headline "more data helps how much" plots.

```bash
# Get the total train-track count once (used to scale --x-units tracks):
python -c "import json; d=json.load(open('/workspace/data/input/new_pklot/postprocessed/annotations/PUCPR_repaired_split_flanks_fixed/splits/50_50/train_through_2012-10-15.json')); print(len(d.get('tracks', d)))"

# Build a manifest (e.g. perpetua single_exp at all fractions):
python scripts/build_plot_manifest.py \
  --csv /workspace/data/output/persistence_eval_outputs/sweep_metrics_pucpr_split_50_50_balanced.csv \
  --variant single_exp --grouping location \
  --label-template '{variant}_{grouping}' \
  --out /tmp/perpetua_single_exp_location.csv

# Plot:
python plot_train_pct_metrics.py \
  --manifest /tmp/perpetua_single_exp_location.csv \
  --x-units percent --x-units tracks \
  --n-total-tracks <N_FROM_ABOVE> \
  --debug-csv /workspace/data/output/persistence_eval_plots/perpetua_single_exp_location/debug.csv
```

Notable flags: `--metric accuracy --metric f1` (repeatable; default plots all),
`--ext pdf|png|svg`, `--plot-subdir custom_name`, `--prefix run1_`,
`--title "..."`. Passing multiple `--x-units` appends a `__<unit>` suffix to
each output filename so percent- and tracks-axis plots can sit side-by-side.

Bundled three-way comparison manifest (vision + perpetua + gemma) lives at
`plot_manifests/three_way_train_pct/manifest.csv`. **The bundled file currently
references the archived `pucpr_repaired_split` paths** — rebuild it from the
new sweep CSV (above) before reusing. See
`plot_manifests/three_way_train_pct/README.md` for the historical layout.

### 3b. Per-horizon metric curves (`plot_horizon_metrics.py`)

Same manifest format, but plots `metric vs prediction horizon` (one curve per
label). Default `--x-units hours`; alternatives include `seconds`, `days`.
Flags `--show-base-rate` (overlay positive-class base rate per horizon),
`--threshold 0.5`, `--min-count 1` (drop horizon buckets with too few
observations).

### 3c. Disappearance-time ECDFs (`plot_disappearance_distributions.py`)

Single overlay ECDF of true vs predicted disappearance times per `(label,
fraction)` row. Pulls per-episode times directly from `prediction_records.json`.

```bash
python plot_disappearance_distributions.py \
  --manifest plot_manifests/three_way_train_pct/manifest.csv \
  --fractions "0.1 0.5 1.0" --xlog --time-units hours
```

Warns if any row's truth ECDF diverges from the others — a useful sanity check
that all rows actually evaluated against the same test subset.

### 3d. Qualitative examples (`plot_pklot_examples.py`)

Renders RGB image + masked anchor crop + predicted parameters / horizons /
ground-truth lifetime for sampled track-anchor observations from an inference
run. Useful when a quantitative metric looks off and you want to *see* what
the model is producing.

```bash
python plot_pklot_examples.py \
  --manifest plot_manifests/three_way_train_pct/manifest.csv \
  --n 8 --seed 0 \
  --output-root /workspace/data/output/plot_examples
```

Flags: `--run-dir` (use specific eval dirs instead of a manifest),
`--methods <label> [...]` (filter), `--label custom_name`, `--subdir foo`.

---

## 4. Debug / inspection utilities

| Script                                          | What it does                                                                                                                                                                |
| ----------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `scripts/inspect_trained_model.py <model_dir>`  | Health checks on a trained Perpetua model: degenerate fits, implausible lifetimes, severely skewed bucket sizes. Exit 0 if all sanity checks pass; `--verbose` for details. |
| `scripts/check_truth_seed_variance.py`          | Re-derives the percentile_horizons truth distribution offline and sweeps `shuffle_seed` values. Use when you suspect the test sampling is shifting your truth ECDF.         |
| `scripts/inspect_anchor_class_images.py`        | For closed-set CLIP runs: dumps RGB + masked crop for any anchor whose assigned class fails a filter (e.g., `--exclude-class car`). Diagnoses CLIP misclassifications.      |
| `scripts/debug_clip_alignment.py`               | For a given image + segmentation cache, prints top-5 CLIP-matching classes per mask. Helps diagnose CLIP embedding mismatches.                                              |

Typical workflow when a sweep produces surprising numbers:

1. Spot-check a single trained checkpoint with `inspect_trained_model.py`.
2. Open the offending eval run dir under `persistence_eval_outputs/` and
   render qualitative examples with `plot_pklot_examples.py --run-dir <dir>`.
3. If the truth ECDF in `plot_disappearance_distributions.py` looks shifted,
   run `check_truth_seed_variance.py` to confirm it's not seed-driven.

---

## 5. Where outputs land

| Artifact                    | Path (container)                                                                                                |
| --------------------------- | --------------------------------------------------------------------------------------------------------------- |
| Trained models              | `/workspace/data/output/perpetua/<experiment.name>/frac<F>_seed<S>/`                                            |
| Eval `metric_results.json`  | `/workspace/data/output/persistence_eval_outputs/<dataset_name>/<method_name>/<benchmark>/inference_runs/.../evaluation_runs/...` |
| Sweep CSVs                  | `/workspace/data/output/persistence_eval_outputs/sweep_{metrics,eval_metrics_*}_pucpr_split_50_50_balanced*.csv`  |
| Plots                       | `/workspace/data/output/persistence_eval_plots/<plot-subdir>/`                                                  |

Train and test runs land under distinct `<dataset_name>` subdirs (driven by
the benchmark dataset's `name:` field) so eval-on-train results never collide
with held-out-test results.

---

## 6. Day-count sweep on the 50/20/30 split (new)

Parallel to the fraction sweep above, this mode varies **the number of
contiguous days of training data** anchored to the start of the train window.
It targets the `splits/50_20_30/` train/val/test split (50% train ≈ 35 days,
20% val, 30% test ≈ 22 days, ≥1000 test tracks). The 50/50 split and its
fraction sweep are unchanged.

### 6a. Generate the 50/20/30 split (one-time)

Dry-run viability check first (no files written; aborts if test < 1000 tracks):
```bash
python generate_temporal_split.py \
  --manifest /workspace/data/input/new_pklot/postprocessed/annotations/PUCPR_repaired_split_flanks_fixed/export_augmented.json \
  --train-fraction 0.5 --val-fraction 0.2 --min-test-tracks 1000 --dry-run
```

Then materialize:
```bash
python generate_temporal_split.py \
  --manifest /workspace/data/input/new_pklot/postprocessed/annotations/PUCPR_repaired_split_flanks_fixed/export_augmented.json \
  --train-fraction 0.5 --val-fraction 0.2 --min-test-tracks 1000

# Test sidecar for per-location inferencers:
python build_track_location_index.py \
  --manifest /workspace/data/input/new_pklot/postprocessed/annotations/PUCPR_repaired_split_flanks_fixed/splits/50_20_30/test_from_2012-10-30.json \
  --cell-size-px 128
```

Outputs land at `.../splits/50_20_30/{train_through_2012-10-15.json,
val_from_2012-10-16_through_2012-10-29.json, test_from_2012-10-30.json,
discarded_tracks.json, split_metadata.json}` (manifests reference the
original per-track JSONs unchanged — no track-file duplication).

### 6b. Materialize day-count train manifests

Each N gets a self-contained subdirectory with truncated per-track JSONs
(observations clipped to `[train_start, train_start + N)`; tracks with no
positive observation in window are dropped; boundaries recomputed from
kept observations).

```bash
python generate_daycount_train_manifests.py \
  --source-train-manifest /workspace/data/input/new_pklot/postprocessed/annotations/PUCPR_repaired_split_flanks_fixed/splits/50_20_30/train_through_2012-10-15.json \
  --days 1 2 3 4 5 7 10 14 21 28 35
```

Result layout (`splits/50_20_30/daycount/days_NNNN/`):
- `train_first_<N>d_through_<cutoff_minus_one>.json` — the manifest
- `tracks/<track_id>.json` — truncated per-track JSONs
- `daycount_metadata.json` — provenance (source sha, train_start, kept/dropped counts)

For reference, kept-track counts at the default grid: N=1→342 tracks, N=2→993,
N=3→1565, N=5→2494, N=7→3160, N=14→5576, N=21→6170, N=35→7333 (full).

### 6c. Sweep configs (new files)

| Family   | Train cfg                                                                                       | Eval cfg                                                                                                          |
| -------- | ----------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------- |
| Vision   | `train_config_vision_anchor_midpoint_pucpr_split_50_20_30_daycount_balanced_train`              | `eval_config_vision_anchor_midpoint_pucpr_split_50_20_30_daycount_balanced_train_balanced_eval`                   |
| Perpetua | `train_config_perpetua_<dist>_<grouping>_pucpr_split_50_20_30_daycount`                         | `eval_config_perpetua_<dist>_<grouping>_pucpr_split_50_20_30_daycount_balanced`                                   |
| Gemma    | (none — inference only)                                                                         | `eval_config_llm_pucpr_split_50_20_30_gemma_v3iqrBehav4_mO20_percentile_balanced`                                 |

Each new training top-level cfg cross-references its eval counterpart in a
header comment. `experiment.day_count` is mandatory and is supplied by the
sweep wrapper. `paths.model_dir` resolves to
`<model_root>/<experiment.name>/days<N>_seed<S>` (no zero-padding).

The new train_dataset presets bind `external_val_manifest_path` to the
50/20/30 val manifest, so `TrainingDataset` loads val from disk instead of
slicing it off train (matters at small N where the train pool is too small
to safely split).

### 6d. Sweep wrappers

```bash
# Perpetua train + eval (mixture variants need spf_perpetua_container — they
# import jax). Single variants run fine in pf_plus_datasets_v2.
DAYS="1 2 3 4 5 7 10 14 21 28 35" \
  bash scripts/sweep_train_eval_pucpr_split_50_20_30_daycount_balanced.sh

# Vision train (GPU IDs as first positional arg):
bash scripts/sweep_vision_anchor_midpoint_pucpr_split_50_20_30_daycount_balanced_train.sh 0 \
  --days "1 2 3 4 5 7 10 14 21 28 35"

# Vision eval:
bash scripts/sweep_eval_vision_anchor_midpoint_pucpr_split_50_20_30_daycount_balanced_train_balanced_eval.sh 0
```

Default `DAYS` grid is computed from `split_metadata.json` (`train_days`
field): `1 2 3 4 5 7 10 14 21 28 <train_days>` — denser at low end, last
entry replaced with the actual `train_days` so we always include the full
window.

| Env var       | Default                                                                | Notes                                                                                  |
| ------------- | ---------------------------------------------------------------------- | -------------------------------------------------------------------------------------- |
| `DAYS`        | dense low-end grid capped at `train_days`                              | Space-separated.                                                                       |
| `DAYCOUNT_ROOT` | `<SPLIT_DIR>/daycount`                                                | Holds `days_NNNN/` subdirs from `generate_daycount_train_manifests.py`.                |
| `SEEDS`       | `0`                                                                    | (Perpetua wrapper). Vision wrapper uses `--seeds`.                                     |
| `MIXTURE_FAST`, `FORCE_TRAIN`, `FORCE_EVAL`, `OUT_CSV`, `VARIANTS`, `EVAL_SCENARIO` | same semantics as the fraction sweep | See section 2b. |

Sweep CSVs:
- `/workspace/data/output/persistence_eval_outputs/sweep_metrics_pucpr_split_50_20_30_daycount_balanced.csv`
- `.../sweep_eval_metrics_vision_anchor_midpoint_pucpr_split_50_20_30_daycount_balanced.csv`

### 6e. LLM single run on the new test split

```bash
CUDA_VISIBLE_DEVICES=0 python eval_persistence.py \
  --config-name=eval_config_llm_pucpr_split_50_20_30_gemma_v3iqrBehav4_mO20_percentile_balanced
```

LLM is inference-only (pretrained Gemma) — no day-count axis, no sweep.

### 6f. Granularity gate

Before launching the full N-axis sweep, run all variants at N=1 and inspect
`balanced_accuracy@0.5` in the CSV. **If any variant is already at ceiling
(>= 0.95) at N=1, the day grid is too coarse** — switch the manifest
generator to hour granularity (would require a `--hours` arg in
`generate_daycount_train_manifests.py` and corresponding sweep
plumbing). On the current PUCPR 50/20/30 split, single-component perpetua
at N=1 sits around `balanced_accuracy@0.5 ≈ 0.74`, so day granularity is
fine for the headline experiment.

---

## 7. Archived (pre-flanks-fix) configs and scripts

The previous PKLot/PUCPR configs and sweep scripts (targeting the
`PUCPR_repaired_split` manifest with the transient-track labelling defect)
have been moved to:

- `configs/train_confs/_archive_pklot/`
- `scripts/_archive_pklot/`

Kept for reference / re-running historical experiments only. **Do not use
these for new work** — the labels are wrong (transients without bounding
absents).

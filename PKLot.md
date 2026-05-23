# PKLot evaluation guide

End-to-end recipe for training and evaluating persistence models against the
PUCPR (PKLot) dataset using the **flanks-fixed** annotations
(`PUCPR_repaired_split_flanks_fixed/export_augmented.json`). Every config /
script described here targets the repaired manifest; earlier annotations had
a transient-track labelling defect (missing bounding absents).

All commands run **inside the perpetua docker container** unless noted. Start
the container with `bash docker_start_datasets_plus_spf.sh` (the `new_pklot`
volume mount provides the data at `/workspace/data/input/new_pklot/`).

---

## 0. Quickstart

```bash
# 0. one-time per split — materialize daycount subsets
python generate_daycount_train_manifests.py \
  --source-train-manifest /workspace/data/input/new_pklot/postprocessed/annotations/PUCPR_repaired_split_flanks_fixed/splits/50_20_30/train_through_2012-10-15.json \
  --days 0.5 1 2 3 4 5 7 10 14 21 28 35 --force

# 1. train + default-scenario eval (no GPU)
bash scripts/pucpr_daycount/train_eval_perpetua.sh

# 2. train + run vision inference + eval (GPU)
bash scripts/pucpr_daycount/train_vision.sh <GPU_IDS>
bash scripts/pucpr_daycount/eval_vision.sh <GPU_IDS>

# 3. alternate-scenario eval over existing artifacts (perpetua + vision + LLM)
EVAL_SCENARIO=percentile_scaled_horizons_5min  bash scripts/pucpr_daycount/eval_perpetua_scenarios.sh
EVAL_SCENARIO=percentile_scaled_horizons_15min bash scripts/pucpr_daycount/eval_perpetua_scenarios.sh
EVAL_SCENARIO=percentile_scaled_horizons_5min  bash scripts/pucpr_daycount/eval_vision_llm_scenarios.sh
EVAL_SCENARIO=percentile_scaled_horizons_15min bash scripts/pucpr_daycount/eval_vision_llm_scenarios.sh

# 4. plots + LaTeX tables — 6 scenarios × 3 groupings × 4 metrics
bash scripts/pucpr_daycount/run_render_all.sh
```

---

## 1. Data prep (one-time per split)

The 50/20/30 temporal split lives at
`/workspace/data/input/new_pklot/postprocessed/annotations/PUCPR_repaired_split_flanks_fixed/splits/50_20_30/`:

| File | Purpose |
|---|---|
| `train_through_2012-10-15.json` | 50% train (~35 days from 2012-09-11) |
| `val_from_2012-10-16_through_2012-10-29.json` | 20% val (~14 days) |
| `test_from_2012-10-30.json` | 30% test (~22 days, ≥ 1000 tracks) |
| `test_from_2012-10-30.track_id_to_location_cell128.json` | Per-track location sidecar (cell_size_px=128) for location-grouped inferencers |
| `split_metadata.json` | Provenance |

### 1a. (Re)generate the temporal split

Dry-run viability check first (aborts if test < 1000 tracks):
```bash
python generate_temporal_split.py \
  --manifest /workspace/data/input/new_pklot/postprocessed/annotations/PUCPR_repaired_split_flanks_fixed/export_augmented.json \
  --train-fraction 0.5 --val-fraction 0.2 --min-test-tracks 1000 --dry-run
```

Then materialize + build the test-side per-track location sidecar:
```bash
python generate_temporal_split.py \
  --manifest /workspace/data/input/new_pklot/postprocessed/annotations/PUCPR_repaired_split_flanks_fixed/export_augmented.json \
  --train-fraction 0.5 --val-fraction 0.2 --min-test-tracks 1000

python build_track_location_index.py \
  --manifest .../splits/50_20_30/test_from_2012-10-30.json \
  --cell-size-px 128
```

Track JSONs are referenced by the manifests in place — no per-track duplication.

### 1b. Materialize daycount training subsets

Each `N` gets a self-contained subdirectory under `.../50_20_30/daycount/` with
truncated per-track JSONs (observations clipped to `[train_start, train_start
+ N)`; tracks with no positive observation in window are dropped; boundaries
recomputed from kept observations).

```bash
python generate_daycount_train_manifests.py \
  --source-train-manifest .../50_20_30/train_through_2012-10-15.json \
  --days 0.5 1 2 3 4 5 7 10 14 21 28 35 --force
```

Output layout — `splits/50_20_30/daycount/days_<slug>/`:
- `train_first_<N>d_through_<cutoff_minus_one>.json` (integer N) or `train_first_<slug>d.json` (fractional N)
- `tracks/<track_id>.json` — truncated per-track JSONs
- `daycount_metadata.json` — provenance (source sha, train_start, kept/dropped counts, cutoff timestamp)

**Slug naming** — integer N uses zero-padded `days_0001/`, `days_0007/`, …;
fractional N uses dot-free slugs like `days_0p5/` so paths stay
filesystem-safe across all consumers.

**Fractional-day semantics** — for `--days 0.5` (more generally `K + f`,
`0 < f < 1`), the cutoff is anchored at the empirical `f`-quantile of image
timestamps on the partial day, deduped across tracks. So `--days 0.5` keeps
all observations whose timestamp is < the median image of day K. This
respects the fact that PKLot only collects during daylight — "half a day" is
"half the images of that day", not "midnight + 12h".

For reference, kept-track counts at the default grid: N=1→342 tracks,
N=2→993, N=3→1565, N=5→2494, N=7→3160, N=14→5576, N=21→6170, N=35→7333
(full).

### 1c. Per-track-balanced anchor sampling at eval time

Every eval cell uses
[configs/train_confs/benchmark_dataset/annotation_tool_pucpr_split_50_20_30_test_balanced.yaml](configs/train_confs/benchmark_dataset/annotation_tool_pucpr_split_50_20_30_test_balanced.yaml):

```yaml
max_anchor_observations: 1000
per_track_balanced: true
shuffle_seed: 42
```

The framework picks a per-track cap such that capping yields at least 1000
anchors total, then trims to exactly 1000 after a seeded shuffle. This
prevents long-lived tracks (e.g. parking spots that are empty all month) from
dominating the headline numbers. If you want an unbalanced control, point
the eval at a benchmark dataset YAML with `per_track_balanced: false` (the
archived `_ON_TRAIN` and 50/50 configs used the unbalanced form).

---

## 2. Method families

Three families produce a `metric_results.json` per (D, scenario, noise) cell;
the renderer reads aggregate sweep CSVs.

| Family | Backbone / source | Trained per D? | Where models / inference artifacts land |
|---|---|---|---|
| Perpetua | survival-fit (EM/AIC); `mixture_persistence_filter` for mixtures, `rosen` for singles | yes (8 variants × D) | `/workspace/data/output/perpetua/perpetua_*` |
| Vision-only | DINOv2 ViT-L/14 (frozen) + SelaVPR adapter + scalar head | yes (one per D) | `/workspace/data/output/perpetua/vision_anchor_midpoint_*` |
| PreSIST-Lang (gemma) | gemma 4.31b with `cap_proc_long7__gemmaVLMDesc_blur_bg_context_crop_mask_t3_cmd1_no_holes_longPrompt__ph_v3IqrBehav4` | no (inference-only, no D axis) | `/workspace/data/output/persistence_eval_outputs/.../gemma_*` |

Perpetua expands to 8 variants: cartesian product of `{single_exp,
single_lognormal, mixture_exp, mixture_lognormal} × {class, location}`.
- Train cfg pattern: `pucpr_train_config_perpetua_<dist>_<grouping>_split_50_20_30_daycount[.yaml]`
- Eval cfg pattern: `pucpr_eval_config_perpetua_<dist>_<grouping>_split_50_20_30_daycount_balanced[.yaml]`
- `experiment.name`: `perpetua_<dist>_<grouping>_pucpr_split_50_20_30_daycount_{fullK_mins5_p0p01,emk1_noise0p01_mins5}`

Vision:
- Train cfg: `pucpr_train_config_vision_anchor_midpoint_split_50_20_30_daycount_balanced_train.yaml`
- Eval cfg: `pucpr_eval_config_vision_anchor_midpoint_split_50_20_30_daycount_balanced_train_balanced_eval.yaml`

LLM:
- Eval cfg: `pucpr_eval_config_llm_split_50_20_30_gemma_v3iqrBehav4_mO20_percentile_balanced.yaml`
- Inference artifacts already exist on disk (no training step here).

---

## 3. Training + default-scenario eval (`percentile_horizons`, matched noise = 0.01)

The default eval scenario is `percentile_horizons` (anchor-only — no
observations fed to the filter), at matched noise `p_m = p_f = 0.01`.

### 3a. Perpetua — no GPU
```bash
bash scripts/pucpr_daycount/train_eval_perpetua.sh
```
Runs train + eval for all 8 variants × all default days. Idempotent — each
cell skips re-train if `persistence_params.json` already exists and skips
re-eval if a fresh `metric_results.json` exists newer than the model.

### 3b. Vision — GPU
```bash
bash scripts/pucpr_daycount/train_vision.sh <GPU_IDS> --jobs 4
bash scripts/pucpr_daycount/eval_vision.sh <GPU_IDS>
```
`eval_vision.sh` produces the `inference_runs/<timestamp>/` artifact that
`eval_vision_llm_scenarios.sh` reuses for alternate scenarios.

### 3c. LLM
Inference-only and the artifacts are already on disk; the default-scenario
eval is run from §4 (it's just another `EVAL_SCENARIO=percentile_horizons`
invocation against existing inference_runs).

---

## 4. Alternate-scenario eval (5min / 15min)

Two flavours, both with `recency_cap` ∈ {300s, 900s}:

- **`percentile_scaled_horizons_{5,15}min`** — per-query causal mask. Each
  query at `dt_q` sees observations in `[anchor, anchor + recency_cap]` AND
  strictly earlier than `t_q − min(recency_cap, dt_q/2)`.
- **`percentile_anchor_window_horizons{,_5min,_15min}`** — symmetric anchor
  window. Each query at `dt_q` sees observations in
  `[anchor, anchor + min(recency_cap, dt_q/2)]`.

Eval-only — uses pre-existing trained perpetua models and pre-existing vision
+ LLM inference_runs:

```bash
EVAL_SCENARIO=percentile_scaled_horizons_5min  bash scripts/pucpr_daycount/eval_perpetua_scenarios.sh
EVAL_SCENARIO=percentile_scaled_horizons_15min bash scripts/pucpr_daycount/eval_perpetua_scenarios.sh
EVAL_SCENARIO=percentile_scaled_horizons_5min  bash scripts/pucpr_daycount/eval_vision_llm_scenarios.sh
EVAL_SCENARIO=percentile_scaled_horizons_15min bash scripts/pucpr_daycount/eval_vision_llm_scenarios.sh
# (similar invocations for the three percentile_anchor_window_horizons variants)
```

---

## 5. Aggregation & plotting

```bash
bash scripts/pucpr_daycount/run_render_all.sh
```

Produces — for each of 6 scenarios × 3 groupings × 4 metrics = 72 PDFs + 72
LaTeX tables — under
`/workspace/data/output/persistence_eval_plots/daycount_<scenario>_matched_noise_<grouping>/{bacc,accuracy,f1,mae}.{pdf,tex}`.

Grouping modes:
- `both` — Perpetua class (G) + location (L) overlaid; vision + PreSIST-Lang as separate lines
- `class` — Perpetua class-grouped only + vision + PreSIST-Lang
- `location` — Perpetua location-grouped only + vision + PreSIST-Lang

Plot styling: 10.5×5.5 figsize, linear x-axis with `xlim(0, 36)`, 45° tick
rotation, 3-column legend (singles | mixtures | Vision + PreSIST-Lang),
brown+tan / green / red / purple / orange / teal palette, best-in-column
bolded in the LaTeX tables.

Single-panel invocation:
```bash
python scripts/pucpr_daycount/render_sweep_plots.py \
  --perp-csv <perp_csv> --vl-csv <vl_csv> \
  --grouping {both,class,location} --out-dir <dir>
```

---

## 6. Knob reference

| Knob | Scripts | Default | When to change |
|---|---|---|---|
| `DAYS` | all sweep scripts | `0.5 1 2 3 4 5 7 10 14 21 28 35` | Restrict the D grid (e.g. `DAYS="0.5"` for a single new D). Re-materialize the daycount manifest first for fractional values not yet on disk. |
| `KINDS` | perpetua scripts | `mixture single` | Limit to mixture variants only or singles only. |
| `METHODS` | `eval_vision_llm_scenarios.sh` | `vision llm` | Skip LLM (or vision) re-eval. |
| `JOBS` | all sweep scripts | `1` | Parallelism; `JOBS=4` is the typical setting we used. |
| `EVAL_SCENARIO` | both eval-only scripts | `percentile_horizons` | Switch to any `eval_scenario/*.yaml` (e.g. `percentile_scaled_horizons_{5,15}min`, `percentile_anchor_window_horizons*`, or a new scenario you add). |
| `EVAL_NOISE_PM` / `EVAL_NOISE_PF` | both eval-only scripts | unset (uses YAML default of 0.01) | Override the persistence-filter noise model; outputs land in noise-tagged CSV / log paths. |
| `FORCE_TRAIN` | `train_eval_perpetua.sh`, `train_vision.sh` | unset | Retrain even if `persistence_params.json` / `best_model.pt` exists. |
| `FORCE_EVAL` / `FORCE_REEVAL` | eval scripts | unset | Re-run eval even if metrics already exist for the cell. |
| `EXP_TAG` | `train_eval_perpetua.sh` | unset | Version suffix on model_dir + CSV (e.g. `EXP_TAG=v2` for an A/B comparison). |
| `MAX_ANCHOR_OBS` | `eval_vision_llm_scenarios.sh` | `50` | Cap on anchor observations passed to the vision/LLM benchmark dataset. |
| `OUT_CSV` | all sweep scripts | derived from scenario + noise tag | Direct the aggregate CSV elsewhere. |
| `LOG_DIR` | all sweep scripts | sibling of `OUT_CSV` | Direct per-cell logs elsewhere. |
| `FILTER_OVERRIDE` | `eval_perpetua_scenarios.sh` | unset | Override which `persistence_filter/*.yaml` to use (e.g. ablations). |

---

## 7. Adding a new variant

- **New D value** — add to `DAYS` (re-materialize the split first for fractional values via `generate_daycount_train_manifests.py --days <value> --force`).
- **New eval scenario** — drop a YAML under `configs/train_confs/eval_scenario/`, optionally add a new provider class under `persistence_eval/sampling/`, then `EVAL_SCENARIO=<name> bash …_scenarios.sh`. Append the scenario name to `SCENARIOS=( … )` in `scripts/pucpr_daycount/run_render_all.sh` to include it in the plot output.
- **New noise level** — `EVAL_NOISE_PM=… EVAL_NOISE_PF=… bash …_scenarios.sh`. Outputs land in a noise-tagged CSV so they don't overwrite the matched-noise headline.
- **New survival family or K range** — edit the corresponding `pucpr_train_config_perpetua_*.yaml` (or the `trainer/persistence_fit_*.yaml` it inherits from) and re-run with `FORCE_TRAIN=1`. The renderer auto-includes rows in the CSV matching the known method tokens; for genuinely new methods (not in `single_exp / single_lognormal / mixture_exp / mixture_lognormal / vision_anchor / llm`), add an entry to the `METHODS` / colour / marker tables in `render_sweep_plots.py`.
- **New grouping** — would need plumbing in `inferencer/` and the eval configs; not a quick knob.

---

## 8. Output locations

| Artifact | Path (container) |
|---|---|
| Trained perpetua params | `/workspace/data/output/perpetua/<EXP_NAME>/days<N>_seed<S>/persistence_params.json` |
| Trained vision checkpoints | `/workspace/data/output/perpetua/vision_anchor_midpoint_*/days<N>_seed<S>/best_model.pt` |
| Inference runs (vision/LLM) | `/workspace/data/output/persistence_eval_outputs/<dataset_name>/<EXP_NAME>_days<N>_seed<S>/.../inference_runs/<timestamp>/` |
| Eval metrics | `inference_runs/<timestamp>/evaluation_runs/survival_based__<scenario>/<timestamp>/metric_results.json` |
| Aggregate sweep CSVs | `/workspace/data/output/persistence_eval_outputs/sweep_metrics_pucpr_split_50_20_30_daycount_p0p01_mins5_{all8,vision_llm}_<scenario>[_pm0p01_pf0p01].csv` |
| Plots + LaTeX tables | `/workspace/data/output/persistence_eval_plots/daycount_<scenario>_matched_noise_<grouping>/{bacc,accuracy,f1,mae}.{pdf,tex}` |

Inference and eval land under distinct `<dataset_name>` subdirs (driven by the benchmark dataset's `name:` field) so eval-on-train diagnostics never collide with held-out-test results.

---

## 9. Debug / inspection utilities

| Script | What it does |
|---|---|
| [scripts/inspect_trained_model.py](scripts/inspect_trained_model.py) `<model_dir>` | Health checks on a trained Perpetua model: degenerate fits, implausible lifetimes, severely skewed bucket sizes. Exit 0 if all sanity checks pass; `--verbose` for details. |
| [scripts/check_truth_seed_variance.py](scripts/check_truth_seed_variance.py) | Re-derives the percentile_horizons truth distribution offline and sweeps `shuffle_seed` values. Use when you suspect the test sampling is shifting your truth ECDF. |
| [scripts/inspect_anchor_class_images.py](scripts/inspect_anchor_class_images.py) | For closed-set CLIP runs: dumps RGB + masked crop for any anchor whose assigned class fails a filter. Diagnoses CLIP misclassifications. |
| [scripts/debug_clip_alignment.py](scripts/debug_clip_alignment.py) | For a given image + segmentation cache, prints top-5 CLIP-matching classes per mask. |

Typical workflow when a sweep produces surprising numbers:
1. Spot-check a single trained checkpoint with `inspect_trained_model.py`.
2. Open the offending eval run dir under `persistence_eval_outputs/` and inspect its `prediction_records.json`.
3. If the truth ECDF looks shifted, run `check_truth_seed_variance.py` to confirm it's not seed-driven.

For filter / inferencer / scenario internals see [persistence_eval/README.md](persistence_eval/README.md).

---

## 10. Archived configs and scripts

These directories hold material that's no longer part of the active workflow
but is kept for reference / re-running historical experiments:

- [configs/train_confs/_archive_pklot/](configs/train_confs/_archive_pklot/) — pre-flanks-fix PKLot configs (transient-track labelling defect; do not use for new work).
- [configs/train_confs/_archive_pucpr/](configs/train_confs/_archive_pucpr/) — legacy 50/50 fraction-sweep train + eval configs, the `_daycount_emk1` train + eval pair (em_k1 mode is now wired through the keep-list train configs), and the `_ON_TRAIN` eval-on-train diagnostics.
- [configs/train_confs/eval_scenario/_archive/](configs/train_confs/eval_scenario/_archive/) — `dynamic_anchor_only`, `percentile_windowed_horizons`, `scaled_horizon`, and the unsuffixed 30-min `percentile_scaled_horizons` (superseded by the 5min/15min variants + the anchor-window family).
- [configs/train_confs/persistence_filter/_archive/](configs/train_confs/persistence_filter/_archive/) — `dummy` (test only) and `mixture_prior_pick_filter` (ablation only).
- [configs/train_confs/survival_generator/_archive/](configs/train_confs/survival_generator/_archive/) — `dummy` and `general_purpose` (unused). `weibull` and `weibull_pucpr_llm` remain active (root eval template default + LLM eval respectively).
- [scripts/_archive/](scripts/_archive/) — legacy one-shot sweeps (`_oneshot_d1_*`, `_oneshot_d2_*`, `_oneshot_d35_*`, noise grids, mixture/track sweeps), legacy 50/50 sweep wrappers, the older daycount wrappers `sweep_train_eval_pucpr_split_50_20_30_daycount_balanced{,_em,_emk1_balanced}.sh`, the original `sweep_train_eval.sh`, and ad-hoc analysis probes `_tmp_*.py`.

Re-running anything in the archives requires either pointing it at the new
`scripts/pucpr_daycount/` entry points or accepting it as a historical
snapshot.

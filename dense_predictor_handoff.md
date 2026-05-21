# Dense persistence predictor — handoff

Continuing this work on another machine. Read in order; nothing below
assumes you've memorized prior context.

## 1. What this work is

A dense (per-pixel) variant of the existing
`PersistenceObjConditionedCrossAttn` model: image → `(3, H, W)` map
encoding `[log p25, log(p75/p25 - 1), conf]` per pixel. Supervised only
at pixels covered by labeled instance masks. Reuses the same frozen
DINOv2 + trainable SELA-adapter backbone; the new pieces are a DPT
decoder, a rasterization step that converts per-instance labels into a
dense supervision map, and a per-pixel loss.

Reference documentation: [dense_predictor.md](dense_predictor.md)
(architecture, knobs, troubleshooting). Read that doc after this one if
the high level here leaves anything unclear.

## 2. Current state

All five planned stages are implemented and verified by smoke tests.
See [the plan](/home/aaadkins/.claude/plans/i-want-to-create-glowing-sunset.md)
for the design decisions.

| Stage | What it built                                       | Verified by                                                       |
|-------|-----------------------------------------------------|-------------------------------------------------------------------|
| 1     | `TrainingTask` abstraction; trainer refactor        | Per-mask smoke run reproduces previous behavior end-to-end.       |
| 2     | Dense backbone copy + intermediate-layer exposure   | `scripts/verify_dense_encoder.py` (trainable-param parity + shapes). |
| 3     | DPT decoder + `DPTDensePersistencePredictor`        | `scripts/verify_dpt_predictor.py` (instantiation + forward shape).|
| 4     | `dense` dataset mode + rasterizer + cache + collate | `scripts/test_dense_rasterizer.py` + `scripts/smoke_dense_dataset.py`. |
| 5     | Adapter + dense loss + `DensePersistenceTask` + dispatch | 2-epoch end-to-end smoke training (24 train samples).             |

**Last smoke run output** (epoch 2, 24 train samples, 4 val batches):
`train_loss=32.09`, `val_agg=0.0276`, `val_on_train_agg=0.0484`. The
pipeline runs end-to-end; *whether the dense model actually trains well
on full data is the open empirical question*.

## 3. Setting up on the new machine

### 3.1 Code

```bash
git clone <this repo>
cd semantic-persistence-filter
# Check that the dense-predictor work is on the current branch.
ls persistence_parameter_networks/models/dpt_dense_persistence_predictor.py
ls dense_predictor.md
```

If the dense files are missing, the work is on a different branch — check
`backupCode` (the branch this was developed on at time of writing).

### 3.2 Container

The whole pipeline runs inside the project's Docker container:

```bash
bash docker_start_datasets_plus_spf.sh
podman exec -it pf_plus_datasets_v2 bash
# Inside the container:
cd /workspace/code/semantic-persistence-filter
export PYTHONPATH=/workspace/code/semantic-persistence-filter
```

`PYTHONPATH` is required — none of the package directories ship
`__init__.py` (implicit namespace packages).

### 3.3 Required data

The dense training config points at the same dataset the per-mask
config uses (`seg_langSam3Q8T_behav4_mO20`). Confirm these exist:

| Path inside container                                                      | What                                          |
|----------------------------------------------------------------------------|-----------------------------------------------|
| `/workspace/data/input/`                                                   | Raw images.                                   |
| `/workspace/data/output/segmentation_results/langGround_…`                 | Instance segmentation outputs (mask npz).    |
| `/workspace/data/output/full_results/seg_langSam3Q8T_…_behav4_mO20`        | Per-instance persistence labels (JSON).      |
| `train_dataset_manifests/seg_langSam3Q8T_behav4_mO20/{train,val}.txt`      | Image splits (in-repo, should ship with code).|
| `pretrained_models/dinov2_vitl14_pretrain.pth`                             | DINOv2 ViT-L/14 weights.                     |

If the DINOv2 file is missing, download it from the upstream DINOv2
release into `pretrained_models/` (the path is referenced from
[configs/train_confs/model/dpt_dense_predictor.yaml](configs/train_confs/model/dpt_dense_predictor.yaml)).

### 3.4 Output / cache locations

Defaults in the configs:

| Purpose                            | Default path                                                                          |
|------------------------------------|---------------------------------------------------------------------------------------|
| Training output root               | `/workspace/data/output/training_outputs/`                                            |
| Dense supervision cache            | `/workspace/data/output/dense_supervision_cache/seg_langSam3Q8T_behav4_mO20/`         |
| Per-run subdir                     | `<output_root>/<dataset_name>/<model_name>/<trainer_name>/<timestamp>/`                |

If the new machine has different paths, override either by editing the
configs or via Hydra command-line overrides:

```bash
python3 train.py --config-name dpt_dense_seg_langSam3Q8T_behav4_mO20_imgbal \
  train_dataset.dense_cache_dir=/some/other/path \
  model_output_root_dir=/some/other/path
```

The dense-supervision cache from the old machine **will not transfer**
automatically — it's keyed by a fingerprint that includes the
rasterizer config, but the cache files themselves are on local disk.
Either rsync the cache dir to the new machine *or* re-warm it (§4.3).

## 4. First runs on the new machine, in order

### 4.1 Sanity-check the codebase compiles

```bash
podman exec pf_plus_datasets_v2 bash -lc '
  cd /workspace/code/semantic-persistence-filter &&
  PYTHONPATH=/workspace/code/semantic-persistence-filter python3 -c "
from training.task import DensePersistenceTask, PerMaskPersistenceTask
from persistence_parameter_networks.models.dpt_dense_persistence_predictor import DPTDensePersistencePredictor
from persistence_parameter_networks.losses.dense_losses import DenseWeightedMultiLoss
from persistence_parameter_networks.inference.persistence_predictor_adapter import make_adapter
print(\"imports OK\")
"'
```

Expected: `imports OK`. If anything errors, the most likely cause is a
missing `PYTHONPATH` or a path mismatch in the container.

### 4.2 Stage-by-stage verifications (≈3 min total)

Each prints `PASS` on success. Run them once on the new machine to
confirm nothing rotted:

```bash
# Rasterizer unit tests (CPU-only, instant)
python3 scripts/test_dense_rasterizer.py

# Dense encoder requires_grad parity + shape (CUDA, ~30s)
CUDA_VISIBLE_DEVICES=<free GPU> python3 scripts/verify_dense_encoder.py

# DPT predictor instantiation + forward shape (CUDA, ~30s)
CUDA_VISIBLE_DEVICES=<free GPU> python3 scripts/verify_dpt_predictor.py

# Dataset smoke (CPU + ~30s of dataset-build for the full split index)
python3 scripts/smoke_dense_dataset.py
```

### 4.3 Pre-warm the supervision cache (optional but recommended)

```bash
CUDA_VISIBLE_DEVICES= \
python3 scripts/precompute_dense_supervision.py \
  --config-name dpt_dense_seg_langSam3Q8T_behav4_mO20_imgbal \
  +precompute_splits='[train,val]' +precompute_num_workers=8
```

CPU-only (`CUDA_VISIBLE_DEVICES=` empty). Iterates the dataset and
writes rasterized `(target_map, loss_mask, pixel_weights)` to disk per
image. Re-runnable; existing entries are not recomputed. Disk cost:
budget ~50 GB for the train split at 518² (estimate; measure first
time).

### 4.4 Pipeline smoke

```bash
CUDA_VISIBLE_DEVICES=<free GPU> PYTORCH_ALLOC_CONF=expandable_segments:True \
python3 train.py \
  --config-name dpt_dense_seg_langSam3Q8T_behav4_mO20_imgbal_smoke
```

Should finish in 2–4 minutes and print 2 epoch summaries. If it
crashes, that's the new machine's specific issue — share the traceback
and check §6 below first.

### 4.5 Real training

```bash
CUDA_VISIBLE_DEVICES=<free GPU> PYTORCH_ALLOC_CONF=expandable_segments:True \
python3 train.py \
  --config-name dpt_dense_seg_langSam3Q8T_behav4_mO20_imgbal
```

W&B is on by default in the non-smoke config. If you don't want W&B,
add `trainer.loop.use_wandb=false` to the command. The trainer writes
JSONL metrics to `metrics.jsonl` either way (W&B is a parallel sink).

## 5. Files this work changed

New files (all under the repo root):

```
persistence_parameter_networks/
  data/
    dense_supervision_cache.py
    dense_target_rasterizer.py
  inference/
    mask_poolers.py
    persistence_predictor_adapter.py
  losses/
    dense_losses.py
  models/
    dense_persistence_predictor.py
    dpt_dense_persistence_predictor.py
    img_encoders/selavpr/dense/
      dense_rich_img_encoder.py
      dense_rich_output_network.py
training/
  dataset/dense_collate.py
  task.py
configs/train_confs/
  dpt_dense_seg_langSam3Q8T_behav4_mO20_imgbal.yaml
  dpt_dense_seg_langSam3Q8T_behav4_mO20_imgbal_smoke.yaml
  model/dpt_dense_predictor.yaml
  train_dataset/seg_langSam3Q8T_behav4_mO20_imgbal_dense.yaml
  trainer/loss_function/dense_weighted_multi_loss.yaml
scripts/
  inspect_instance_counts.py
  precompute_dense_supervision.py
  smoke_dense_dataset.py
  test_dense_rasterizer.py
  verify_dense_encoder.py
  verify_dpt_predictor.py
dense_predictor.md
dense_predictor_handoff.md   ← this file
```

Modified files (existing code I touched — there are no untouched
deletions, only adds / refactors):

```
training/dataset/distillation_dataset.py   added `dense` mode + config fields
training/trainer.py                        accepts TrainingTask + val_on_train_loader
train.py                                   dispatches by model type; builds dense loaders + task
```

**Files explicitly NOT modified** (intentionally left alone):
`rich_img_encoder.py`, `rich_output_network.py`, `vision_transformer.py`,
`persistence_output_predictors.py`, `weighted_multi_loss.py`. The
existing per-mask model still uses these unchanged.

## 6. Known gotchas / open items

### Likely-to-bite issues on a new machine

1. **`PYTHONPATH` unset.** Every script needs
   `PYTHONPATH=/workspace/code/semantic-persistence-filter`. Symptom:
   `ModuleNotFoundError: No module named 'persistence_parameter_networks'`
   or `'dataset_generation'`.
2. **DINOv2 weights missing.** Symptom: backbone construction errors
   about `foundation_model_path`. Fix: see §3.3.
3. **GPU OOM during model build.** Other processes may be using GPU 0;
   pick a freer GPU with `nvidia-smi` and set
   `CUDA_VISIBLE_DEVICES=<idx>`. The model needs ~5 GB just to load and
   forward; training adds activations.
4. **Hydra defaults-list errors.** If you edit
   [dpt_dense_seg_langSam3Q8T_behav4_mO20_imgbal.yaml](configs/train_confs/dpt_dense_seg_langSam3Q8T_behav4_mO20_imgbal.yaml),
   note that `override /...` entries must come last in the defaults list.
5. **Cache directory permission issues.** First run creates
   `<cache_dir>/manifest.json` and per-fingerprint subdirs; make sure
   the user inside the container can write there.

### Open follow-ups (none block training)

- Val-on-train iterates the *full* train split sequentially (capped by
  `train_eval_max_batches`) instead of a fixed record-ID subset.
  Comparable epoch-to-epoch within a run; not strictly comparable
  across model variants. Custom sampler needed to fix.
- Dense per-pixel loss is reported from the training step but not as a
  separate validation pass. Validation only reports per-instance
  metrics (via the pooling facade). Adding a dense val pass is a small
  change.
- `DPTDensePersistencePredictor.__init__` accepts `**_unused_kwargs` to
  absorb the unused `pooler_cfg` that the base `config.yaml` injects
  via `model/cross_attn_pooler`. Cleaner fix is removing line 72 of
  `configs/train_confs/config.yaml`; deferred to avoid touching
  per-mask training paths.
- The "Loaded N images and indexed 0 instances" print in `dense` mode
  is cosmetic (counts the per_instance index; not what dense uses).
- Multi-GPU training is *not* wired through `TrainingTask`. Single-GPU
  only today. Adding DDP would mean wrapping `self.task.model` in
  `DistributedDataParallel` inside `Trainer.train`.

## 7. Empirical questions still open

The smoke confirms plumbing. None of these are answered yet:

- Does the dense model match or exceed the per-mask cross-attn val
  numbers on the full train split?
- Do `Lr` and `Lc` train meaningfully, or do they stay stuck while L50
  drops? (Smoke showed L50 dropping but Lr/Lc moving in the wrong
  direction over 2 epochs; not enough data to say if real.)
- Does the per-instance auxiliary loss help (set
  `DenseWeightedMultiLossWithInstanceAux` in the loss config with
  `aux_weight: 0.1`)? Try only if vanilla per-pixel doesn't train well.
- How much memory does aux-loss training cost on the long-tail
  high-variance instance counts (max 299 per image)? Profile before
  long runs.

## 8. Where to read what

| Doc                                                             | For                                                                                          |
|-----------------------------------------------------------------|----------------------------------------------------------------------------------------------|
| [dense_predictor.md](dense_predictor.md)                        | Architecture, all knobs, troubleshooting cheatsheet. Reference doc.                          |
| `/home/aaadkins/.claude/plans/i-want-to-create-glowing-sunset.md` | The plan / design decisions and rationale.                                                  |
| This file (`dense_predictor_handoff.md`)                        | Status + setup + first steps on a new machine.                                              |
| [scripts/inspect_instance_counts.py](scripts/inspect_instance_counts.py) output | Distribution of instances per image (justifies the flat/concat representation). Run once. |

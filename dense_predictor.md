# Dense persistence predictor

Per-pixel persistence-parameter prediction over an image, supervised only at
pixels covered by labeled instance masks. The dense model is a sibling to the
existing per-mask `PersistenceObjConditionedCrossAttn`: same DINOv2 + SELA
backbone, but it ingests only the image and produces a `(3, H, W)` map (the
three channels are `[log p25, log(p75/p25 - 1), conf]`, matching the per-mask
output convention).

All commands run inside the project's docker container. Start it with
`bash docker_start_datasets_plus_spf.sh`; the workspace is mounted at
`/workspace/code/semantic-persistence-filter/`.

---

## 1. Quickstart

```bash
# Pipeline smoke (≈2 epochs on 0.1% subset, no W&B, no cache writes).
PYTHONPATH=/workspace/code/semantic-persistence-filter \
CUDA_VISIBLE_DEVICES=4 \
python3 train.py \
  --config-name dpt_dense_seg_langSam3Q8T_behav4_mO20_imgbal_smoke

# Full training run.
PYTHONPATH=/workspace/code/semantic-persistence-filter \
CUDA_VISIBLE_DEVICES=0 \
python3 train.py \
  --config-name dpt_dense_seg_langSam3Q8T_behav4_mO20_imgbal

# Optional: pre-warm the dense-supervision cache (rasterized per-image
# targets) before training so DataLoader workers don't pay for it at
# epoch start.
PYTHONPATH=/workspace/code/semantic-persistence-filter \
CUDA_VISIBLE_DEVICES= \
python3 scripts/precompute_dense_supervision.py \
  --config-name dpt_dense_seg_langSam3Q8T_behav4_mO20_imgbal \
  +precompute_splits='[train,val]' +precompute_num_workers=8
```

The trainer writes logs and checkpoints to
`/workspace/data/output/training_outputs/<dataset>/<model>/<trainer>/<timestamp>/`.

---

## 2. Architecture

```
   image (3, 518, 518)
          │
   ┌──────▼───────────────────────┐
   │ DINOv2 ViT-L/14 backbone     │   24 transformer blocks
   │  + SELA adapters in-block    │   DINOv2 frozen; adapters trainable
   │  (frozen body)               │
   └──────┬───────────────────────┘
          │  taps at blocks 6, 12, 18, 24
          ▼
   4 × (B, 1024, 37, 37) intermediate token maps
          │
   ┌──────▼───────────────────────┐
   │ DPT decoder                  │   Reassemble (1×1 + transposed/strided
   │  - Reassemble per stage      │   conv) puts the 4 stages at scales
   │    [2×, 1×, 0.5×, 0.25×]     │   relative to the patch grid: roughly
   │  - 4 FusionBlocks            │   (256, 74, 74), (256, 37, 37),
   │    (Residual Conv Units +    │   (256, 19, 19), (256, 10, 10)
   │     ×2 bilinear upsample)    │
   └──────┬───────────────────────┘
          │
   ┌──────▼───────────────────────┐
   │ Head:                        │
   │  Conv(256→128, 3×3) → ReLU   │
   │  Conv(128→3, 1×1)            │
   │  Bilinear upsample to 518²   │
   │  Sigmoid on conf channel†    │
   └──────┬───────────────────────┘
          │
   output (3, 518, 518)
```

† Only when `confidence_from_logits=false` (the default), matching the
per-mask `PersistenceOutputPredictor`.

**Backbone parity.** The dense path lives in *copies* of the SELA wrapper
files under
[persistence_parameter_networks/models/img_encoders/selavpr/dense/](persistence_parameter_networks/models/img_encoders/selavpr/dense/),
so the originals (and the per-mask model) are untouched. SELA adapters
live inside each DINOv2 block, so every tapped intermediate layer's
output already includes their contribution. The freeze rule
(`requires_grad=False` for params not containing `"adapter"`) is
preserved verbatim; under default config there are **192 trainable
adapter parameters** (≈ 50 M elements), exactly matching the per-mask
encoder.

**Trainable-parameter budget** (default config):

| Component        | Parameters |
|------------------|-----------:|
| SELA adapters    | 50.4 M     |
| DPT decoder      | 12.2 M     |
| Output head      | 0.3 M      |
| **Total**        | **62.9 M** |

DINOv2 body params (~300 M) stay frozen.

---

## 3. Supervision and loss

### 3.1 Rasterizing per-instance labels into a dense map

The dataset's `dense` mode loads one image plus all its valid instances
and asks a `TargetRasterizer` to produce three per-pixel tensors:

| Tensor          | Shape         | Meaning                                                |
|-----------------|---------------|--------------------------------------------------------|
| `target_map`    | `(3, H, W)`   | Per-pixel `[log p25, log IQR-ratio, conf]` target.    |
| `loss_mask`     | `(1, H, W)` bool | Which pixels participate in the loss.               |
| `pixel_weights` | `(1, H, W)`   | Per-pixel weight applied before reduction.            |

These tensors are deterministic functions of the image's instance set +
the rasterizer/normalizer config + the label transform, so the
`DenseSupervisionCache` writes them to disk keyed by a sha256
fingerprint of those inputs (plus per-class `__version__` attributes
bumped only on behavior-incompatible changes).

### 3.2 Overlap policy and normalization

Two orthogonal knobs control rasterization. The training-config default
is **SmallerWinsRasterizer + PerInstanceAreaNormalizer** (selected after
the brainstorm; see comments in
[train_dataset/…_imgbal_dense.yaml](configs/train_confs/train_dataset/seg_langSam3Q8T_behav4_mO20_imgbal_dense.yaml)).

**Overlap rasterizers**
([persistence_parameter_networks/data/dense_target_rasterizer.py](persistence_parameter_networks/data/dense_target_rasterizer.py)):

| Class                          | Behavior at overlap pixels                                                                         |
|--------------------------------|----------------------------------------------------------------------------------------------------|
| `SmallerWinsRasterizer`        | Pixel gets the smaller-area instance's label. *Default — small masks are usually more specific.*   |
| `LargerWinsRasterizer`         | Pixel gets the larger-area instance's label.                                                       |
| `ExcludeOverlapRasterizer`     | Multiply-covered pixels are marked invalid in `loss_mask`.                                         |
| `AverageOverlapRasterizer`     | Per-pixel target = unweighted mean of covering instances' labels.                                  |
| `AreaWeightedAverageRasterizer`| Weighted mean of covering instances' labels (smaller instances dominate via `1/area`).             |

**Pixel-weight normalizers**:

| Class                       | Per-pixel weight inside instance i                                              |
|-----------------------------|---------------------------------------------------------------------------------|
| `NoNormalization`           | `1.0` (large instances dominate by area).                                       |
| `PerInstanceAreaNormalizer` | `1 / area(i)` so each instance contributes equal total weight. *Default.*       |
| `SqrtAreaNormalizer`        | `1 / sqrt(area(i))` — softer.                                                   |

Both pieces are hydra-instantiated via `_target_` so you can drop in
your own subclass without changing dataset code.

### 3.3 Dense loss

[persistence_parameter_networks/losses/dense_losses.py](persistence_parameter_networks/losses/dense_losses.py).
All losses share the signature

```python
loss(pred_map, target_map, loss_mask, pixel_weights,
     instance_masks=None, batch_index=None, instance_labels=None)
    -> (scalar, metrics_dict)
```

Per-instance tensors only travel in the training batch when the
configured loss declares `requires_per_instance_tensors() == True`, so
the default training path is decoupled from per-image instance-count
variance.

| Class                                    | What it adds                                                                                                   |
|------------------------------------------|----------------------------------------------------------------------------------------------------------------|
| `DenseWeightedMultiLoss` *(default)*     | Per-pixel analogue of `WeightedMultiLoss`: MSE / Huber on y50, yr; MSE / BCE / Huber on conf. Weighted reduce. |
| `DenseWeightedMultiLossWithInstanceAux`  | Adds a per-instance pooled term (masked-mean → `WeightedMultiLoss`) as a calibration anchor. Off by default.   |

Weighted reduction inside the loss:

```text
loss = sum(loss_mask · pixel_weights · per_pixel_loss) /
       max(sum(loss_mask · pixel_weights), eps)
```

---

## 4. Configuration knobs

### 4.1 Top-level config

[`configs/train_confs/dpt_dense_seg_langSam3Q8T_behav4_mO20_imgbal.yaml`](configs/train_confs/dpt_dense_seg_langSam3Q8T_behav4_mO20_imgbal.yaml)
composes the dense pieces:

```yaml
defaults:
  - train_dataset: seg_langSam3Q8T_behav4_mO20_imgbal_dense
  - eval_dataset: mirror_train
  - model: dpt_dense_predictor
  - registry_info: registry_default
  - trainer: default
  - _self_
  - override /trainer/loss_function: dense_weighted_multi_loss
```

The smoke variant
[`…_smoke.yaml`](configs/train_confs/dpt_dense_seg_langSam3Q8T_behav4_mO20_imgbal_smoke.yaml)
overrides subsample fractions, batch sizes, and val caps for a fast
pipeline check.

### 4.2 Trainer knobs (selected)

Defined in
[`configs/train_confs/trainer/default.yaml`](configs/train_confs/trainer/default.yaml)
and overridable in the top-level config under `trainer.loop`:

| Knob                          | Notes                                                                                       |
|-------------------------------|---------------------------------------------------------------------------------------------|
| `batch_size` / `val_batch_size` | Dense ViT-L on a 80 GB H100 fits roughly `batch_size: 8` at 518² with mixed precision off.  |
| `image_balanced_sampling`     | Leave `false` for dense (dense draws one image per sample by construction).                 |
| `train_samples_per_epoch`     | Leave `null` for dense (irrelevant without `WeightedRandomSampler`).                        |
| `val_max_batches`             | Cap val passes for wall-time.                                                               |
| `train_eval_max_batches`      | Cap val-on-train pass.                                                                      |
| `use_amp`                     | Same as per-mask. Currently off by default.                                                 |

### 4.3 Model knobs

[`configs/train_confs/model/dpt_dense_predictor.yaml`](configs/train_confs/model/dpt_dense_predictor.yaml):

| Knob                       | Default                          | Notes                                                                                             |
|----------------------------|----------------------------------|---------------------------------------------------------------------------------------------------|
| `backbone_cfg.freeze_dino` | `true`                           | Keep `true` to match the per-mask config; only SELA adapters train.                              |
| `backbone_cfg.use_adapter` | `true`                           | Required — without adapters the backbone has no trainable params.                                |
| `backbone_cfg.image_size`  | `518`                            | Yields a 37×37 patch grid for ViT-L/14.                                                          |
| `decoder_dim`              | `256`                            | Per-stage channel width in the DPT decoder.                                                      |
| `scales`                   | `[2.0, 1.0, 0.5, 0.25]`          | Spatial scales for the 4 reassembled stages (shallow → deep).                                    |
| `confidence_from_logits`   | `false`                          | When `false`, sigmoid is applied to the conf channel of the output (the rasterized target is in `[0,1]`). |

Intermediate-layer indices `[5, 11, 17, 23]` are hard-coded in
[DenseGeoLocalizationNet](persistence_parameter_networks/models/img_encoders/selavpr/dense/dense_rich_output_network.py)
for now; expose them in the config if you want to experiment.

### 4.4 Dense-supervision knobs (dataset config)

[`configs/train_confs/train_dataset/seg_langSam3Q8T_behav4_mO20_imgbal_dense.yaml`](configs/train_confs/train_dataset/seg_langSam3Q8T_behav4_mO20_imgbal_dense.yaml):

```yaml
dense_rasterizer:
  _target_: persistence_parameter_networks.data.dense_target_rasterizer.SmallerWinsRasterizer
  normalizer:
    _target_: persistence_parameter_networks.data.dense_target_rasterizer.PerInstanceAreaNormalizer
dense_cache_dir: /workspace/data/output/dense_supervision_cache/<dataset>
dense_cache_enabled: true
```

To swap policies: change `dense_rasterizer._target_` (and/or `normalizer._target_`)
to any subclass listed in §3.2. Switching policies produces a new cache
fingerprint automatically (sha256 over class name + version + config),
so cached tensors from a previous policy are not silently reused.

`dense_cache_enabled: false` recomputes rasterization on every
`__getitem__` (useful for smoke runs where the cache dir would be
otherwise polluted).

### 4.5 Loss knobs

[`configs/train_confs/trainer/loss_function/dense_weighted_multi_loss.yaml`](configs/train_confs/trainer/loss_function/dense_weighted_multi_loss.yaml):

```yaml
_target_: persistence_parameter_networks.losses.dense_losses.DenseWeightedMultiLoss
confidence_loss: mse           # bce | mse | huber
confidence_from_logits: false  # keep in sync with model.confidence_from_logits
# w50, wr, wc, useHuberForLogVals, delta_50, delta_r, delta_c, eps  — same as WeightedMultiLoss
```

To enable the per-instance auxiliary term, point the loss config at
`DenseWeightedMultiLossWithInstanceAux` and set `aux_weight`. The
collate will automatically start emitting `instance_masks`,
`batch_index`, and `instance_labels` in the training batch (gated by
`loss_fn.requires_per_instance_tensors()`).

---

## 5. Data layer

| Mode            | One sample is               | Used for                                                                            |
|-----------------|-----------------------------|-------------------------------------------------------------------------------------|
| `per_instance`  | one `(image, mask, label)`  | Per-mask training, dense val-on-val, dense val-on-train (all keep same population). |
| `dense`         | one image + all instances   | Dense training only.                                                                |
| `per_image`     | (stub)                      | Unused.                                                                             |

The dense training loop builds **three loaders** (see
[`createDenseDataloaders`](train.py)):

- **train**: `dense` mode + flat/concat collate
  ([training/dataset/dense_collate.py](training/dataset/dense_collate.py)).
  Per-instance tensors travel only when the loss requires them.
- **val**: `per_instance` mode on the val split — identical to the
  per-mask model's val loader, so dense-vs-cross-attn val numbers stay
  apples-to-apples.
- **val-on-train**: `per_instance` mode on the train split, sequential
  pass (no `WeightedRandomSampler`) so the diagnostic is reproducible
  across epochs.

The flat/concat representation packs all instances in a batch into
`instance_masks: (Σ N_i, H, W)` + `batch_index: (Σ N_i,)`, motivated by
the measured train-set instance-count distribution (mean 45 / median 30
/ max 299 / p99-to-median 7.3×, see
[scripts/inspect_instance_counts.py](scripts/inspect_instance_counts.py)).

---

## 6. Validation

The existing per-instance validator
([training/validator.py](training/validator.py)) consumes `(image,
mask) → (y50_hat, yr_hat, yc_hat)` per sample. The dense task adapts to
that signature via a `_DensePerMaskFacade` (in
[training/task.py](training/task.py)) that:

1. Runs `predict_dense(image)` once for the batch.
2. Pools the resulting `(B, 3, H, W)` map inside each per-instance mask
   using a `MaskPooler` (`MaskedMeanPooler` by default).

So validation evaluates the **same** population of `(image, instance,
label)` triples for both per-mask and dense models — the only
difference is that the dense model arrives at each per-instance vector
by pooling rather than by per-mask forward.

The validator's loss is a paired `WeightedMultiLoss` constructed from
the dense loss's config (see `validator_loss_fn` in [train.py](train.py)),
so the val aggregate score is comparable across model variants.

---

## 7. File map

| Concern                              | Path                                                                                                                         |
|--------------------------------------|------------------------------------------------------------------------------------------------------------------------------|
| Dense model ABC                      | [persistence_parameter_networks/models/dense_persistence_predictor.py](persistence_parameter_networks/models/dense_persistence_predictor.py) |
| DPT implementation                   | [persistence_parameter_networks/models/dpt_dense_persistence_predictor.py](persistence_parameter_networks/models/dpt_dense_persistence_predictor.py) |
| Backbone copy (intermediate layers)  | [persistence_parameter_networks/models/img_encoders/selavpr/dense/](persistence_parameter_networks/models/img_encoders/selavpr/dense/)         |
| Rasterizer + normalizer family       | [persistence_parameter_networks/data/dense_target_rasterizer.py](persistence_parameter_networks/data/dense_target_rasterizer.py)               |
| Disk supervision cache               | [persistence_parameter_networks/data/dense_supervision_cache.py](persistence_parameter_networks/data/dense_supervision_cache.py)               |
| Dense loss family                    | [persistence_parameter_networks/losses/dense_losses.py](persistence_parameter_networks/losses/dense_losses.py)                                |
| Mask poolers                         | [persistence_parameter_networks/inference/mask_poolers.py](persistence_parameter_networks/inference/mask_poolers.py)                          |
| Inference adapter (`predict_persistence`) | [persistence_parameter_networks/inference/persistence_predictor_adapter.py](persistence_parameter_networks/inference/persistence_predictor_adapter.py) |
| Dataset `dense` mode + collate       | [training/dataset/distillation_dataset.py](training/dataset/distillation_dataset.py), [training/dataset/dense_collate.py](training/dataset/dense_collate.py) |
| Task wrapper                         | [training/task.py](training/task.py) (`DensePersistenceTask`, `_DensePerMaskFacade`)                                          |
| Trainer hooks                        | [training/trainer.py](training/trainer.py) (uses `task.make_validation_model()`, takes `val_on_train_loader`)                 |
| Cache-warmer script                  | [scripts/precompute_dense_supervision.py](scripts/precompute_dense_supervision.py)                                            |
| Instance-count inspector             | [scripts/inspect_instance_counts.py](scripts/inspect_instance_counts.py)                                                      |
| Rasterizer unit tests                | [scripts/test_dense_rasterizer.py](scripts/test_dense_rasterizer.py)                                                          |
| End-to-end dataset smoke             | [scripts/smoke_dense_dataset.py](scripts/smoke_dense_dataset.py)                                                              |

---

## 8. Troubleshooting

Symptom-to-knob cheatsheet. Each row points at the config field most
likely to help; the file paths show where the field lives.

### 8.1 Training dynamics

| Symptom                                                            | Try                                                                                                                                                                                                                  | Why                                                                                                                                          |
|--------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------|
| Total loss not decreasing at all                                   | Lower `lr` in `trainer.optimizer`; check the rasterized `target_map` looks sensible (run `scripts/smoke_dense_dataset.py` and inspect `target_map.min/max`).                                                          | Catastrophic LR or broken supervision usually shows up immediately.                                                                          |
| L50 drops but `Lr` / `Lc` don't                                    | Bump `wr` / `wc` in the loss config; or switch log losses to Huber by setting `useHuberForLogVals: true`.                                                                                                            | The default `(w50, wr, wc) = (1, 1, 0.5)` lets the dominant-magnitude term (usually L50) absorb most of the gradient.                        |
| Conf channel stuck near 0.5                                        | Set both `model.confidence_from_logits: true` and `trainer.loss_function.confidence_loss: bce` (with `confidence_from_logits: true`).                                                                                | BCE on logits gives sharper gradients on confidence than sigmoid-then-MSE.                                                                  |
| Gradient norms explode (e.g. `GN > 100` repeatedly)                | Lower `trainer.loop.grad_clip_norm` (e.g. `0.5`); reduce `lr`; turn `use_amp` off if loss is overflowing under fp16.                                                                                                  | Dense decoders can swing hard early in training; tight clipping stabilizes the first few epochs.                                             |
| NaN loss                                                           | Check label values for `log(0)`-style degeneracies (`p25 = 0` rows); set `clamp_confidence_probs: true` in the loss; lower `lr`.                                                                                     | Most NaNs trace back to a single bad target value or fp16 overflow.                                                                          |

### 8.2 Prediction quality

| Symptom                                                                                      | Try                                                                                                                                                                                                                                                  | Why                                                                                                                                                                                                |
|----------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Per-instance val numbers look much worse than the per-mask cross-attn baseline              | Enable the per-instance auxiliary loss: switch loss to `DenseWeightedMultiLossWithInstanceAux` with `aux_weight: 0.1–1.0`.                                                                                                                            | The aux term pools the dense output inside each mask and applies the per-mask loss, anchoring the dense model's per-instance mean to the same target the per-mask model trains against.            |
| Predictions blurry; small structures lost                                                    | Tap shallower ViT blocks in `DenseGeoLocalizationNet.__init__` (e.g. `[3, 8, 16, 23]` instead of `[5, 11, 17, 23]`); also try `scales: [4.0, 2.0, 1.0, 0.5]` to push pyramid resolution up.                                                            | Shallow ViT layers retain finer spatial detail; the DPT pyramid resolution is bounded by the shallowest tap.                                                                                       |
| Large instances dominate; small instances under-predicted                                    | Confirm `dense_rasterizer.normalizer` is `PerInstanceAreaNormalizer` (default); switch overlap policy to `SmallerWinsRasterizer` if not already.                                                                                                       | `PerInstanceAreaNormalizer` weights each instance equally regardless of area; `SmallerWinsRasterizer` ensures small instances get to assert their label at overlap pixels.                          |
| Overlapping mask labels appear to fight each other                                           | Switch `dense_rasterizer` to `ExcludeOverlapRasterizer` (drop ambiguous pixels) or `AreaWeightedAverageRasterizer` (soft average).                                                                                                                    | The default `SmallerWins` commits to one label at overlaps; if labels genuinely contradict, excluding or averaging may be better.                                                                  |
| Conf predictions saturate at 0 or 1                                                          | Use `confidence_loss: huber` instead of MSE, or BCE if `confidence_from_logits: true`.                                                                                                                                                                | Saturation usually means MSE-on-sigmoid has gone flat — BCE / Huber give better gradients near 0/1.                                                                                                |
| Predictions look right when pooled per-instance but ugly per-pixel                           | Lower `pixel_weight_normalization` magnitude (e.g. `SqrtAreaNormalizer`) and reduce the per-instance aux weight if enabled; consider tapping a shallower block for finer detail.                                                                       | Strong per-instance normalization can over-smooth dense outputs because every instance contributes equal *total* weight regardless of area.                                                        |

### 8.3 Memory and speed

| Symptom                                                       | Try                                                                                                                                                                                                                | Why                                                                                                                                                                                                 |
|---------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| CUDA OOM                                                      | Lower `batch_size`; lower `val_batch_size`; ensure your loss has `requires_per_instance_tensors() == False` (the default); reduce `decoder_dim`.                                                                    | The per-instance tensor stream (`instance_masks: (Σ N_i, H, W)`) is the biggest non-image memory cost when the aux loss is on.                                                                       |
| GPU idle between batches                                      | Pre-warm the cache (`scripts/precompute_dense_supervision.py`); bump `num_workers` and `val_num_workers`; turn `use_amp: true` if numerics allow.                                                                  | Rasterization dominates CPU cost on first epoch; once cached the bottleneck moves to GPU.                                                                                                            |
| Training slow even with cache hot                             | Enable AMP; increase `batch_size`; consider `decoder_dim: 128` for cheaper decoder.                                                                                                                                | DPT decoder + head is ~12 M params at `decoder_dim=256`; halving it ≈ 4× decoder cost.                                                                                                              |
| Validation pass dominates wall time                           | Lower `val_max_batches` and `train_eval_max_batches`; for val-on-train, raise the cap only if the diagnostic actually matters that epoch.                                                                          | Val/val-on-train run with the per-instance facade — each pass forwards `B` images then pools `B` masks.                                                                                             |
| Cache directory ballooning                                    | `dense_cache_enabled: false` during exploratory runs; otherwise inspect `manifest.json` and delete orphaned fingerprint subdirs (the manifest shows what configs each fingerprint came from).                       | Every rasterizer / normalizer / label-transform combo gets its own fingerprint subdir; experimenting with policies can quickly accumulate stale dirs.                                               |

### 8.4 Backbone / setup

| Symptom                                                                  | Try                                                                                                                                                                  | Why                                                                                                                                                              |
|--------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Backbone outputs look frozen-untrained (val score not moving)            | Confirm `backbone_cfg.use_adapter: true`; print trainable params and check the set contains `model.backbone.model.backbone.blocks.*.adapter*` keys.                  | The only trainable backbone params are the SELA adapters. Without `use_adapter=true` there's no path for the backbone to fine-tune anything.                     |
| `pretrained_models/dinov2_vitl14_pretrain.pth` missing                   | Either download it from the upstream DINOv2 repo into `pretrained_models/`, or override `backbone_cfg.foundation_model_path`.                                        | The encoder fails-fast if the foundation-model path is unset and there's no resume checkpoint.                                                                  |
| Unexpected `pooler_cfg` kwarg error from a non-DPT model in dense config | Make sure the model class accepts `**kwargs` (the DPT class does) or remove `model/cross_attn_pooler` from the base `config.yaml` defaults list.                     | The base config unconditionally adds the per-mask pooler default; any non-per-mask model needs to absorb or strip it.                                            |
| Multi-GPU "loss not connected to graph"                                  | Confirm `task.compute_loss(batch)` returns the loss directly from the dense forward (no `detach` calls). The trainer doesn't currently shard; single-GPU only today. | Multi-GPU isn't wired through `TrainingTask` yet.                                                                                                                |

### 8.5 Smoke / debugging

| Symptom                                                  | Try                                                                                                                                                                                              |
|----------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Want a one-minute iteration loop                         | Use `dpt_dense_seg_langSam3Q8T_behav4_mO20_imgbal_smoke` with `dense_cache_enabled: false` and `train_subsample_fraction: 0.0005`.                                                                |
| Want to verify the data layer only                        | `python3 scripts/smoke_dense_dataset.py` (no model forward).                                                                                                                                     |
| Want to verify rasterizer correctness                    | `python3 scripts/test_dense_rasterizer.py` (hand-crafted 2-instance overlap unit tests for every policy).                                                                                        |
| Want to confirm the dense backbone still respects freeze | `python3 scripts/verify_dense_encoder.py` (asserts trainable-param set matches the per-mask encoder).                                                                                            |
| Want to dump the resolved config                          | Add `--cfg job` to the `train.py` command; Hydra prints the fully resolved config and exits before training starts.                                                                              |

---

## 9. Caveats and known follow-ups

- The "Loaded N images and indexed 0 instances" print in `dense` mode
  counts the `per_instance` index (empty for dense). Cosmetic only.
- `DPTDensePersistencePredictor.__init__` swallows the unused
  `pooler_cfg` kwarg that the base `config.yaml` injects via
  `model/cross_attn_pooler`. A cleaner fix would be removing that
  redundant entry from the base config, but it's defaulted in both
  the base and the per-mask model config, so dropping it from the base
  is low-risk but more invasive than necessary for now.
- Val-on-train uses a sequential per-instance pass over the *full* train
  split (capped by `train_eval_max_batches`). The plan called out a
  record-ID-defined fixed subset for tighter cross-mode comparability;
  not yet implemented.
- The dense per-pixel loss is logged from the training step but not as
  a separate val pass. Adding one would mean a dense val loader and a
  small extension to `_validate`.
- Switching to the per-instance auxiliary loss enables per-image
  variance exposure (`instance_masks` flow into the batch). With the
  measured high train-set variance (p99/median 7.3×), test memory
  before launching long aux-loss runs.

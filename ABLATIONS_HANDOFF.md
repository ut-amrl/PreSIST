# Learned-model ablations — handoff

Set of ablation training runs defending the design of the current best
learned model: [configs/train_confs/cross_attn_seg_langSam3Q8T_behav4_mO20_imgbal_lowlr_huber_norm_loose50.yaml](configs/train_confs/cross_attn_seg_langSam3Q8T_behav4_mO20_imgbal_lowlr_huber_norm_loose50.yaml)
(referred to as "Ours" below).

Each ablation isolates one design choice with **minimum delta** vs
Ours, inheriting via Hydra `defaults:` so re-pointing "Ours" later is
a one-line change per ablation file.

---

## Context — what "Ours" is

"Ours" stacks:
- **Backbone:** DINOv2-L (pretrained, **frozen**) with learned
  bottleneck **adapters** inserted into each transformer block. The
  only trainable backbone params are the adapters.
- **Pooler:** mask-downsample-soft (bilinear) — produces a per-mask
  region descriptor `q` of shape `(B, K, C)` from the patch token
  grid.
- **Head:** cross-attention with `q` as query attending to all image
  patch tokens (KV), followed by FF + LayerNorm + `Linear(attn_dim, 3)`.
- **Output:** 3 channels — `(y50, yr, conf)`.
- **Target parameterization (per the `label_transform_log_ratio_p25_p75`
  transform):** `(log p25, log(p75/p25 − 1), conf)`. The second
  channel is a scale-invariant ratio of the inter-quartile gap to
  p25.
- **Normalization:** z-score per channel using train-set
  `(y50_mean, y50_std, yr_mean, yr_std)`. Un-applied at validator
  decode (`DenormalizingValidationModel`) and inferencer wrappers.
- **Loss:** Huber on channels 0 and 1 (`delta_50=3.0` for the
  loose50 variant, `delta_r=1.0`), MSE on the (sigmoid-activated)
  confidence channel. Weights `(w50, wr, wc) = (1, 1, 0.5)`.

The ablations below each isolate one of those design choices.

---

## Naming convention

All configs under [configs/train_confs/](configs/train_confs/) named
`<ours>__<tag>.yaml`, inheriting Ours via:
```yaml
defaults:
  - cross_attn_seg_langSam3Q8T_behav4_mO20_imgbal_lowlr_huber_norm_loose50
  - _self_
```

To re-base on a different "Ours" later: change that one line per
file.

---

## Phase A — ready to train (config-only, no code changes)

Six configs sized for ~96 GB GPUs (H100 96 GB-class). The five
frozen-DINO ablations match Ours' inherited `batch_size=32` → fully
apples-to-apples on optimizer dynamics. `__no_pretrain` halves to 16
because full ViT-L gradients cost ~2-3× more activation memory.

| Tag | Variable isolated | batch_size |
|---|---|---|
| `__no_conf` | `wc=0` (functionally equivalent to dropping the conf channel) | 32 |
| `__no_adapter` | adapters off, DINO frozen, head-only training | 32 |
| `__pooler_mask_downsample_nearest` | nearest vs Ours' bilinear mask downsample | 32 |
| `__pooler_dilated_token_soft` | dilated-token pooler | 32 |
| `__pooler_local_map` | local-map pooler (also flips `backbone_cfg.use_local: True`) | 32 |
| `__no_pretrain` | random-init DINO, unfrozen end-to-end (adapters kept) | 16 |

**If running on smaller GPUs** (48 GB): edit each
`trainer.loop.batch_size` in the per-ablation configs, or pass
`-- trainer.loop.batch_size=16 trainer.loop.val_batch_size=16` to the
launcher. Sensible per-tier:
- 96 GB GPUs: current settings (32 / 16 for no_pretrain).
- 48 GB GPUs: 16 for frozen-DINO ablations, 4 for no_pretrain.
- 24 GB GPUs: 8 for frozen-DINO, 2 for no_pretrain.

### Per-ablation rationale

**`__no_conf`** (`wc: 0.0`). Functionally equivalent to literally
dropping the conf channel: with `wc=0` the gradient on the conf row
of the final `Linear(attn_dim, 3)` is zero everywhere → those weights
stay at init → channel 2 becomes meaningless. Only differences vs
`outputLength=2`: one wasted row of `Linear` weights (negligible)
and the validator still logs diagnostic conf metrics. Treat conf
metrics as diagnostic-only in the writeup, not headline.

**`__no_adapter`** (`backbone_cfg.use_adapter: false`). DINO stays
frozen (inherited `freeze_dino: true`), so only the cross-attn head
+ pooler train. Quantifies what adapters add on top of frozen DINO
+ head.

**`__pooler_*` (3 variants)**. Cheapest possible ablation — pooler
swap via `defaults: override /model/cross_attn_pooler: <variant>`.
Tests how much the per-mask-query construction matters. `local_map`
also needs `backbone_cfg.use_local: True` (already set in its
config; the model config comment in
[configs/train_confs/model/obj_conditioned_cross_attn.yaml](configs/train_confs/model/obj_conditioned_cross_attn.yaml)
documents this requirement).

**`__no_pretrain`** (`foundation_model_path: null`, `freeze_dino: false`,
adapters kept). Adapters kept to minimize delta vs Ours — isolates
the "pretrained weights" variable. The assert in
[persistence_parameter_networks/models/img_encoders/selavpr/rich_output_network.py:245-252](persistence_parameter_networks/models/img_encoders/selavpr/rich_output_network.py#L245-L252)
short-circuits cleanly (`resume` defaults to `False` not `None`, so
the `null and None` conjunction is False); the subsequent
`if args.foundation_model_path:` skips weight loading and the
backbone keeps its random init. **Caveat:** much larger trainable
surface; lr may need re-tuning. Treat as upper-cost baseline, not a
like-for-like swap.

### Excluded from the wrapper

`__pooler_feature_upsample.yaml` exists in the configs dir but is
commented out of [run_all_ablations.sh](run_all_ablations.sh). At
518² image size, this pooler tries to upsample
`[B*K, 1024, 518, 518]` ≈ 8.8 B elements, exceeding PyTorch's
`upsample_bilinear2d_nhwc` INT_MAX (~2.15 B) limit. Would need
`batch_size <= 1` to fit, which isn't a fair comparison. Note in the
writeup as "infeasible at our 518² image size due to a PyTorch
upsample kernel limit; excluded from comparison."

---

## Phase B — GPU job launcher (done)

[run_ablations.py](run_ablations.py) — single-host job pool. One
python child per GPU via `CUDA_VISIBLE_DEVICES`, queues remaining
configs, dispatches next when a GPU frees up. Features:

- **Pre-flight nvidia-smi check**: refuses to use a GPU with
  `memory.used >= --busy-gb` (default 10 GB). `--on-busy
  {skip,abort,poll}`; `poll` waits `--busy-poll-minutes` and rechecks.
- **`.done` markers**: writes `ablation_logs/<cfg>.done` on rc=0; future
  runs skip configs whose marker exists. `--force` re-runs all.
- **Failure handling**: rc != 0 → no `.done` marker, launcher
  dispatches next queued config onto the freed GPU. Re-running picks
  up failed configs automatically.
- **Ctrl-C**: SIGTERM → 30 s → SIGKILL on children.
- **Extra-args forwarding**: anything after a bare `--` is appended
  to every `train.py --config-name=<cfg>` invocation.

[run_all_ablations.sh](run_all_ablations.sh) — thin bash wrapper
holding the canonical config list in priority order. Exports
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` to reduce
fragmentation OOMs on long runs.

### Launching

Run from a container shell (must have `nvidia-smi`, the project
Python env, dataset and checkpoint paths from
[docker_start_datasets_plus_spf_handannotated_scratch.sh](docker_start_datasets_plus_spf_handannotated_scratch.sh)
or equivalent):

```bash
# Full set:
./run_all_ablations.sh 0,1,2,3

# Wait for busy GPUs to free up instead of skipping them:
./run_all_ablations.sh 0,1 --on-busy poll --busy-poll-minutes 10

# Force re-run completed configs (e.g. after editing the config):
./run_all_ablations.sh 0,1,2,3 --force

# Smoke run (one epoch, tiny epoch, no val) — separate log dir so
# smoke .done markers don't block real runs:
python run_ablations.py --gpus 0 --log-dir ablation_logs_smoke \
  --configs cross_attn_seg_langSam3Q8T_behav4_mO20_imgbal_lowlr_huber_norm_loose50__no_conf \
  -- trainer.loop.train_samples_per_epoch=200 trainer.loop.max_epochs=1 \
     trainer.loop.val_max_batches=0 trainer.loop.train_eval_max_batches=0
```

**Important about smoke runs:** `train_subsample_fraction` does NOT
shorten the epoch in this config family — the base config sets
`image_balanced_sampling: true` with `train_samples_per_epoch: 64000`,
so `WeightedRandomSampler` draws 64 k samples per epoch from
whatever-size pool (with replacement). For a real smoke, override
`trainer.loop.train_samples_per_epoch` directly (as in the example
above) plus `val_max_batches` / `train_eval_max_batches` to skip
validation.

### Watching progress

```bash
# Launcher stdout shows dispatch + completion events.
# Per-run train.py logs:
ls -lt ablation_logs/*.log | head
tail -F ablation_logs/<cfg>.gpu<N>.<ts>.log

# Per-GPU live state:
watch -n 5 nvidia-smi
```

### Verifying configs without launching

In-container Hydra compose probe (catches override typos cheaply):

```bash
python -c "
from hydra import initialize, compose
from omegaconf import OmegaConf
NAMES = [
    'cross_attn_seg_langSam3Q8T_behav4_mO20_imgbal_lowlr_huber_norm_loose50__no_conf',
    'cross_attn_seg_langSam3Q8T_behav4_mO20_imgbal_lowlr_huber_norm_loose50__no_adapter',
    'cross_attn_seg_langSam3Q8T_behav4_mO20_imgbal_lowlr_huber_norm_loose50__no_pretrain',
    'cross_attn_seg_langSam3Q8T_behav4_mO20_imgbal_lowlr_huber_norm_loose50__pooler_mask_downsample_nearest',
    'cross_attn_seg_langSam3Q8T_behav4_mO20_imgbal_lowlr_huber_norm_loose50__pooler_dilated_token_soft',
    'cross_attn_seg_langSam3Q8T_behav4_mO20_imgbal_lowlr_huber_norm_loose50__pooler_local_map',
]
sel = OmegaConf.select
with initialize(version_base=None, config_path='configs/train_confs'):
    for n in NAMES:
        cfg = compose(config_name=n)
        print(f'OK: {n}')
        print(f'    pooler={sel(cfg, \"model.pooler_cfg.kind\")}'
              f'  bs={sel(cfg, \"trainer.loop.batch_size\")}'
              f'  wc={sel(cfg, \"trainer.loss_function.wc\", default=\"<default>\")}'
              f'  use_adapter={sel(cfg, \"model.backbone_cfg.use_adapter\")}'
              f'  freeze_dino={sel(cfg, \"model.backbone_cfg.freeze_dino\")}'
              f'  fmp={sel(cfg, \"model.backbone_cfg.foundation_model_path\")}')
"
```

---

## Phase C — code-touching ablations (NOT yet implemented)

Three additional ablations need source-file changes before they can
train. Full design + checklists below.

### `__masked_only` — skip cross-attention to full image

Current predictor pools per-mask query `q`, then cross-attends `q` to
**all** image patch tokens
([attention_based_predictor.py:335-360](persistence_parameter_networks/models/attention_based_predictor.py#L335-L360)).
The masked-only ablation drops that attention pathway and feeds `q`
straight into FF + head — tests whether the model needs image context
outside the mask region.

**Code change** in
[persistence_parameter_networks/models/attention_based_predictor.py](persistence_parameter_networks/models/attention_based_predictor.py)
`MultiMaskObjectConditionedCrossAttnPredictor`:
- Add `use_cross_attn: bool = True` to `__init__`. When False:
  - Don't construct `self.kv_proj` or `self.attn`.
- In `forward`, when False: replace lines 346-355 with
  `x = self.q_proj(q).reshape(B*K, self.attn_dim)` and continue to
  `self.ff` + `self.head`.

Thread the flag through
[persistence_output_predictors.py:45-78](persistence_parameter_networks/models/persistence_output_predictors.py#L45-L78)
(`PersistenceObjConditionedCrossAttn.__init__` accepts
`use_cross_attn`, forwards to the base predictor).

**Config:** `<ours>__masked_only.yaml`
```yaml
defaults:
  - cross_attn_seg_langSam3Q8T_behav4_mO20_imgbal_lowlr_huber_norm_loose50
  - _self_
trainer:
  name: lowlr_huber_norm_loose50_masked_only
model:
  use_cross_attn: false
```

(Considered "keep cross-attn but mask KV to in-mask patches only" as
an alternative — dropped. The cleaner "no image context at all"
variant isolates the cross-attn pathway's contribution end-to-end.)

### `__p25p75_linear` and `__p25p75_loggap` — direct percentile prediction

**Reviewer-facing framing:** Ours uses `(log p25, log(p75/p25 − 1))`,
which puts the gap in log space AND normalizes it by p25 (multiplicative
"ratio" prior). We strip one of those two choices each:

| Variant | Scale | Gap coupling | Ordering / positivity enforced by |
|---|---|---|---|
| Ours | log | **multiplicative** ratio (scale-invariant) | `p25=exp(c0)>0`; `p75 = p25·(1+exp(c1)) > p25` ✓ |
| 5a `p25p75_linear` | **linear** | additive gap | `softplus(c0)≥0` ⇒ `p25≥0`; `softplus(c1)≥0` ⇒ `gap≥0` ⇒ `p75≥p25≥0` ✓ |
| 5b `p25p75_loggap` | log | **additive** gap (in log) | `p25=exp(c0)>0`; `p75 = p25 + exp(c1) > p25` ✓ |

- **5a vs Ours** tests "log vs linear scale" (plus a free
  additive-vs-multiplicative switch; linear has no natural
  multiplicative form).
- **5b vs Ours** tests "is the scale-invariant ratio prior doing
  useful work, or would a plain log-space gap have sufficed?" In
  this domain p25 likely varies over many orders of magnitude
  (minutes for ripe food, years for furniture) and the gap tends to
  scale with p25; if that prior is real, Ours should outperform 5b,
  and the gap should be larger on objects whose lifetime varies most
  across instances.

Considered a third "independent `(log p25, log p75)` with post-hoc
sort" variant — dropped because the post-hoc sort makes the loss
objective and the eval-time decode inconsistent (model is optimized
without ordering, then forced into it at eval), which muddles the
message. 5b achieves "independent direct prediction" cleanly while
keeping the loss objective and the decode aligned.

#### Shared scaffolding (both 5a and 5b)

**Validator decode**: add `target_parameterization` field on
`ValConfig` and branch the `compute_original_time_metrics` block in
[training/validator.py:412-430](training/validator.py#L412-L430) on
a small 3-way dispatch:
- `"log_p25_logratio"` (default — current Ours, unchanged):
  `p_lo = exp(c0); p_hi = p_lo * (1 + exp(c1))`
- `"log_p25_loggap"` (5b):
  `p_lo = exp(c0); p_hi = p_lo + exp(c1)`
- `"linear_p25_gap"` (5a; head outputs are already post-softplus):
  `p_lo = c0; p_hi = c0 + c1`

**One-off stats script** (not committed): `compute_target_stats.py`
that iterates the train dataset once with no normalization,
printing `(mean, std)` for channels 0 and 1. Used to fill in the
`<fill_in>` placeholders in 5a/5b's configs.

#### 5a — `p25p75_linear` (linear, positivity + ordering via softplus)

Target tensor: `(p25_hours, p75_hours − p25_hours, conf)`. Model
outputs raw `(o0, o1, o2)`. Apply `softplus` to channels 0 and 1 at
the predictor wrapper, sigmoid still on channel 2. Recall
`softplus(x) = log(1 + exp(x))` — smooth always-positive (→0 as
x→−∞, →x as x→+∞). After activation:
- `p_lo = softplus(o0) ≥ 0` ⇒ p25 non-negative
- `gap = softplus(o1) ≥ 0` ⇒ p75 ≥ p25

Both constraints (positivity AND ordering) fall out of one choice.

**Code changes:**

1. In
   [training/dataset/distillation_dataset.py](training/dataset/distillation_dataset.py):
   ```python
   def label_transform_linear_p25_gap(raw: dict) -> torch.Tensor:
       p25 = float(raw["p25_persistence_hours"])
       p75 = float(raw["p75_persistence_hours"])
       gap = max(p75 - p25, 0.0)
       return torch.tensor([p25, gap, float(raw["persistence_conf"])],
                           dtype=torch.float32)
   ```
   Register in `LABEL_TRANSFORM_REQUIRED_FIELDS` (lines 199-202)
   with `("p25_persistence_hours", "p75_persistence_hours", "persistence_conf")`.

2. In
   [persistence_parameter_networks/models/persistence_output_predictors.py](persistence_parameter_networks/models/persistence_output_predictors.py):
   add an `output_activations` config field on
   `PersistenceOutputPredictor` (or a sibling subclass
   `PersistenceLinearPercentilePredictor`) that applies softplus to
   channels 0 and 1 before returning, instead of leaving them raw.
   Default value matches today's behavior (raw on 0/1, sigmoid on 2).
   Activated by config for this ablation only.

3. Loss config: **keep Huber on** (the flag name
   `useHuberForLogVals` is misleading — internally it just routes
   between MSE and Huber on channels 0/1 regardless of whether those
   channels are in log or linear space). Linear-hour residuals have
   a much heavier tail than log-hour residuals (chairs hit ~10⁵
   hours), so Huber is *more* needed here. Likely want a larger
   `delta` than Ours' `delta_50=3.0` — start with `delta_50=5.0`,
   `delta_r=5.0` and revisit after smoke if gradients explode.

4. Normalization: linear hours span minutes to years. Z-score
   normalization (see explanation below) is essential. Run the stats
   script and fill in.

**Config:** `<ours>__p25p75_linear.yaml`
```yaml
defaults:
  - cross_attn_seg_langSam3Q8T_behav4_mO20_imgbal_lowlr_huber_norm_loose50
  - _self_
trainer:
  name: lowlr_huber_norm_loose50_p25p75_linear
  loss_function:
    delta_50: 5.0
    delta_r: 5.0
  validationConfig:
    target_parameterization: linear_p25_gap
model:
  output_activations: linear_softplus   # softplus on c0,c1; sigmoid on c2
train_dataset:
  label_transform:
    _target_: training.dataset.distillation_dataset.label_transform_linear_p25_gap
  y50_mean: <fill_in>
  y50_std:  <fill_in>
  yr_mean:  <fill_in>
  yr_std:   <fill_in>
```

(The field names `y50_*` / `yr_*` are inherited generic z-score
buffers — they apply to channels 0/1 regardless of semantics; see
[train.py:324-329](train.py#L324-L329).)

#### 5b — `p25p75_loggap` (log-space additive gap)

Target tensor: `(log p25, log(p75 − p25), conf)`. Model outputs raw.
Decode: `p25 = exp(c0)`, `p75 = p25 + exp(c1)` — both `exp(·) > 0`
gives positivity and ordering for free. Differs from Ours only in
that the gap is in absolute time units (additive log) rather than
normalized by p25 (multiplicative ratio).

**Code changes:**

1. In
   [training/dataset/distillation_dataset.py](training/dataset/distillation_dataset.py):
   ```python
   def label_transform_log_p25_log_gap(raw: dict) -> torch.Tensor:
       eps = 1e-8
       tlo = max(float(raw["p25_persistence_hours"]), eps)
       thi = max(float(raw["p75_persistence_hours"]), eps)
       y_p25 = math.log(tlo)
       y_gap = math.log(max(thi - tlo, eps))
       return torch.tensor([y_p25, y_gap, float(raw["persistence_conf"])],
                           dtype=torch.float32)
   ```
   Register in `LABEL_TRANSFORM_REQUIRED_FIELDS`.

2. Recompute z-score stats for `(log p25, log(p75 − p25))` over train.
   Channel-0 stats should match Ours' y50 stats closely (same
   target). Channel-1 stats will differ from Ours' yr stats
   (different gap parameterization).

3. Validator decode branch (`log_p25_loggap`) as above.

4. No predictor changes (raw outputs). No loss changes (Huber on
   log-values applies).

5. **Suggested diagnostic in writeup:** bucket val MAE on `p75` by
   how much `p75/p25` varies across the bucket. If Ours'
   multiplicative prior is doing real work, expect Ours to beat 5b
   on high-spread buckets and tie elsewhere.

**Config:** `<ours>__p25p75_loggap.yaml`
```yaml
defaults:
  - cross_attn_seg_langSam3Q8T_behav4_mO20_imgbal_lowlr_huber_norm_loose50
  - _self_
trainer:
  name: lowlr_huber_norm_loose50_p25p75_loggap
  validationConfig:
    target_parameterization: log_p25_loggap
train_dataset:
  label_transform:
    _target_: training.dataset.distillation_dataset.label_transform_log_p25_log_gap
  y50_mean: <fill_in>
  y50_std:  <fill_in>
  yr_mean:  <fill_in>
  yr_std:   <fill_in>
```

**Loss-class note:** no new loss class is needed for any ablation.
5a and 5b both use the existing parameter-agnostic
[persistence_parameter_networks/losses/weighted_multi_loss.py](persistence_parameter_networks/losses/weighted_multi_loss.py)
— it just does Huber/MSE on the three output channels regardless of
their semantic meaning.

### Phase C — files to add / modify summary

**New configs:**
- `<ours>__masked_only.yaml`
- `<ours>__p25p75_linear.yaml`
- `<ours>__p25p75_loggap.yaml`

**Code changes:**
- [persistence_parameter_networks/models/attention_based_predictor.py](persistence_parameter_networks/models/attention_based_predictor.py):
  add `use_cross_attn` to `MultiMaskObjectConditionedCrossAttnPredictor`.
- [persistence_parameter_networks/models/persistence_output_predictors.py](persistence_parameter_networks/models/persistence_output_predictors.py):
  thread `use_cross_attn`; add `output_activations` (or sibling
  subclass) to apply softplus on channels 0/1 for 5a.
- [training/dataset/distillation_dataset.py](training/dataset/distillation_dataset.py):
  add `label_transform_linear_p25_gap` and
  `label_transform_log_p25_log_gap`; register both in
  `LABEL_TRANSFORM_REQUIRED_FIELDS`.
- [training/validator.py](training/validator.py): add
  `target_parameterization` field on `ValConfig`; branch the
  `compute_original_time_metrics` decode (3-way dict of callables).

**One-off helper (not committed):** `compute_target_stats.py` for
filling in 5a/5b normalization placeholders.

**Add to** [run_all_ablations.sh](run_all_ablations.sh)'s `CONFIGS`
array once the above lands.

---

## Background: what "z-score normalization" is

Transform each target channel by `(x − μ_channel) / σ_channel`, with
`μ` and `σ` computed once over the training set. Targets become
mean-0, std-1 per channel. Two reasons we use it:

1. **Channel scale balance**: `log p25` ranges over ~18 in this
   dataset; `log(p75/p25 − 1)` ranges over ~9. Without z-scoring,
   one channel would dominate the loss; weights would need
   handcrafted balancing. Z-scoring makes a single set of weights
   apply across channels.
2. **Gradient stability**: comparable gradient magnitudes per
   channel.

Applied via the `train_dataset.{y50,yr}_{mean,std}` fields on each
config; un-applied at validator decode
(`DenormalizingValidationModel`) and at inferencer wrappers, so
val/inference outputs are reported in target units.

The existing `__no_target_norm` ablation is effectively **already
done** by the parent config
[configs/train_confs/cross_attn_seg_langSam3Q8T_behav4_mO20_imgbal_lowlr_huber.yaml](configs/train_confs/cross_attn_seg_langSam3Q8T_behav4_mO20_imgbal_lowlr_huber.yaml)
— that's Ours **without** `_norm` and `_loose50`, using
`delta_50=1.0` (the natural choice for unnormalized log-hour space)
and no target normalization. If you have a trained run of that
config, label it as the no-target-normalization ablation in the
table. No new file needed.

---

## Verification (per-config, before full-length training)

1. **Compose probe** — see the in-container Hydra snippet in the
   launcher section. Catches override typos.

2. **Smoke run** — separate `--log-dir` so smoke `.done` markers
   don't block real runs:
   ```bash
   python run_ablations.py --gpus 0 --log-dir ablation_logs_smoke \
     --configs <cfg> \
     -- trainer.loop.train_samples_per_epoch=200 trainer.loop.max_epochs=1 \
        trainer.loop.val_max_batches=0 trainer.loop.train_eval_max_batches=0
   ```
   Confirms run starts, loss decreases, wandb name appears.

3. **Trainable param sanity-check** (print once at trainer init):
   - `no_pretrain`: full ViT-L + adapters + head/pooler trainable.
   - `no_adapter`: head/pooler only.
   - `masked_only`: head/pooler only, `attn` / `kv_proj` absent.
   - `no_conf`: same param count as Ours; confirm `wc=0` shows in
     logged config.
   - p25p75 variants: same param count as Ours; for 5a confirm the
     output activations include softplus on channels 0/1.

4. **First-batch target inspection** (5a/5b): print one batch's
   targets next to the raw JSON percentiles to confirm the transform
   does what it claims.

5. **First-batch decode inspection** (5a/5b): print one batch's
   predicted `(p_lo, p_hi)` after the validator decode and confirm
   `p_hi ≥ p_lo` and `p_lo ≥ 0` always — both fall out by
   construction in 5a (softplus on both channels) and 5b (`exp(·)`
   on both channels).

6. Full-length training for each ablation and Ours on the same
   dataset/seed for the headline table.

---

## Evaluation on output datasets (TODO)

After training, each ablation's checkpoint needs to be evaluated on
the same set of held-out / benchmark datasets used for Ours in the
paper. **Fill in before kicking off evals:**

- The list of eval datasets. From the working-tree status,
  candidates include `ahg_amanda_kinect_test`, `hd_epic_P01_test`,
  `robotcar_loc01_test` (see
  [configs/train_confs/benchmark_dataset/](configs/train_confs/benchmark_dataset/)
  and the `eval_config_closed_set_clip_p25_p75_*` files in
  `configs/train_confs/`). Confirm the canonical paper-table list.
- The eval entrypoint script(s) — likely
  [eval_persistence.py](eval_persistence.py) and/or
  [evaluate.py](evaluate.py); confirm which.
- Either a second launcher script or a `--mode eval` extension of
  `run_ablations.py` that takes each ablation's final checkpoint
  and runs eval across all benchmark datasets, again with a GPU
  pool. The `.done` marker convention should include the dataset
  so train and eval completion don't collide:
  `<cfg>.train.done` and `<cfg>.eval.<dataset>.done`.
- The aggregation script that ingests all per-(cfg, dataset) eval
  outputs and produces the final ablation table for the paper.
- For the p25/p75 variants, the eval path needs to apply the same
  decode (`exp` / `softplus`) that the validator uses — otherwise
  the eval metrics will be in target-space, not time-space, and
  won't be comparable to Ours' time-space numbers. Confirm the
  eval pipeline either reuses `target_parameterization` from the
  training config or has an equivalent decode hook.

Treat as a separate planning round once Phase A training is at least
partially in flight — the eval pipeline depends on which checkpoints
land where and on the validator-decode work being done as part of
ablations 5a/5b.

---

## Recovery / restart

- A failed config (rc != 0) leaves no `.done` marker. Re-running the
  wrapper picks it up automatically.
- A completed config writes `ablation_logs/<cfg>.done` and is skipped
  on re-runs. `--force` re-runs everything regardless.
- Ctrl-C: SIGTERM → 30 s → SIGKILL on children.
- Interrupted runs don't get `.done` markers.

## State at handoff

- No `.done` markers from prior launches (all earlier attempts hit
  OOM on a 48 GB-class machine before completing).
- One `__no_adapter` run may have been left running on the old
  machine at epoch ~3/100 — was on a different filesystem, will not
  transfer.
- No checkpoints from these configs need to move with the code.
  Start fresh on the new machine.

## Caveats for the writeup

- **Batch-size delta vs Ours:** the five frozen-DINO ablations now
  match Ours at `batch_size=32` → fully apples-to-apples.
  `__no_pretrain` at `batch_size=16` is the one nuisance — flag
  explicitly. Optimizer lr was tuned at bs=32; bs=16 changes
  gradient noise and per-epoch step count.
- **`__no_pretrain`:** full-FT ViT-L is a much larger trainable
  surface than Ours (frozen-DINO + adapters). Treat as an
  upper-cost baseline, not a like-for-like swap. lr may need
  re-tuning if training is unstable.
- **`__pooler_feature_upsample`** infeasible at 518² (kernel limit).
- **`__no_conf` ≡ dropping the conf channel:** validator still logs
  diagnostic conf metrics — diagnostic-only, not headline.
- **p25/p75 ablations**: 5a's `delta_50=5.0`, `delta_r=5.0` is a
  guess for the heavier linear-space tail; revisit after smoke. 5b
  vs Ours diagnostic (val MAE bucketed by p75/p25 spread) directly
  tests whether the multiplicative ratio prior matters.

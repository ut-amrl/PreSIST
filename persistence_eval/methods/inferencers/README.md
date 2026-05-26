# Persistence Parameter Inferencers

Implementations of [`AbsPersistenceParameterInferencer`](base.py) used by
`evaluate.py` to predict per-object persistence parameters at evaluation time.
Each inferencer consumes an `ImageMaskGroup` (image + K masks + track ids) and
returns one `PersistenceParameterEstimate` per mask, in the same order as
`group.track_ids`.

Configs that select an inferencer for a run live under
[`configs/train_confs/`](../../../configs/train_confs/) as `eval_config_*.yaml`.

## Base interface

[`base.py`](base.py)

- `infer(group: ImageMaskGroup) -> list[PersistenceParameterEstimate]` — required.
- `set_run_dir(run_dir: Path) -> None` — optional. Called once by
  `InferenceRunner` before iteration begins. Override to opt in to per-run
  debug output; supported inferencers write artifacts under
  `run_dir/debug_vis/`.

## Catalogue

| Inferencer | What it does | Key ctor args |
|---|---|---|
| [`ConstantParamInferencer`](constant_param.py) | Emits identical fixed params for every track. Baseline. | `**params` |
| [`ClassKeyedLookupInferencer`](class_keyed_lookup_inferencer.py) | Cached params for a single hard-coded semantic class; no CLIP step. | `class_lookup_file`, `class_id`, `class_name?` |
| [`ClosedSetCLIPInferencer`](closed_set_clip_inferencer.py) | CLIP-aligns each mask to a closed-set class, then looks up precomputed persistence params. | `clipAligner`, `semanticClassPersistenceParamLookup` |
| [`LocationKeyedLookupInferencer`](location_keyed_lookup_inferencer.py) | Looks up params by track location via sidecar files. Falls back to global defaults. | `location_lookup_file`, `track_id_to_location_file` |
| [`LearnedModelInferencer`](learned_model.py) | Runs a trained per-mask or dense checkpoint; handles normalization and output-key resolution. | `model_cfg`, `checkpoint_path`, `device`, `output_param_keys?`, `pooler?` |
| [`VisionAnchorMidpointInferencer`](vision_anchor_midpoint_inferencer.py) | Runs a checkpoint that predicts time-to-midpoint persistence (t50) from `(image, mask)`. One forward per mask. | `model_cfg`, `checkpoint_path`, `device`, `image_size` |
| [`DirectVLMQueryInferencer`](direct_vlm_query.py) | Records image and mask paths for later offline VLM querying; caches masks if not on disk. Not a live VLM call. | (none required) |
| [`LLMPipelineInferencer`](llm_pipeline.py) | Three stages per anchor: scene caption (VLM) → object descriptions (VLM, per object) → persistence params (LLM or VLM). Mirrors the dataset-generation pipeline. | `sceneCaptioner`, `objectDescriber`, `llmPersistenceQuerier`, `param_keys?`, `debug_log`, `debug_visualizations`, `strict_object_coverage` |
| [`DirectVLMPipelineInferencer`](direct_vlm_pipeline.py) | One VLM call per object that directly returns strict-JSON persistence parameters, skipping the caption and description stages. Lets the VLM see the image when reasoning about persistence. | `llmPersistenceQuerier`, `param_keys?`, `debug_log`, `debug_visualizations`, `strict_object_coverage` |

## When to use which

- **Baseline / sanity check**: `ConstantParamInferencer`, then
  `ClassKeyedLookupInferencer`.
- **Closed-set semantic priors + CLIP**: `ClosedSetCLIPInferencer`.
- **Location-conditioned priors**: `LocationKeyedLookupInferencer`.
- **Trained model**: `LearnedModelInferencer` (full param dict) or
  `VisionAnchorMidpointInferencer` (t50 only).
- **VLM zero-shot, single fast pass per object**: `DirectVLMPipelineInferencer`.
- **VLM/LLM zero-shot, willing to pay 3 model passes for the most control
  over scene + object reasoning**: `LLMPipelineInferencer`.
- **Deferred / offline VLM querying** (just dump masks now, query later):
  `DirectVLMQueryInferencer`.

## Supporting subdirectories

These are not inferencers themselves — they hold lookups and infrastructure
imported by the inferencers above.

- [`closed_set_clip_align/`](closed_set_clip_align/) — CLIP alignment for
  closed-set semantic class matching.
- [`closed_set_semantics/`](closed_set_semantics/) — class-keyed persistence
  parameter lookup tables and precomputation.
- [`location_keyed/`](location_keyed/) — location-indexed persistence param
  lookup.
- [`permanence_buckets/`](permanence_buckets/) — permanence-bucket utilities
  used by the semantic-class path.

## Debug output convention

Inferencers that override `set_run_dir(run_dir)` write per-anchor artifacts
under `run_dir/debug_vis/<inferencer_subdir>/`. Currently:

- `LLMPipelineInferencer` → `debug_vis/` (per-anchor JSON of caption +
  descriptions + raw estimates, plus mask-overlay PNGs).
- `DirectVLMPipelineInferencer` → `debug_vis/direct_vlm/` (per-object prompt
  images, raw VLM responses if `debug_log=True`, parsed JSON).

Both also surface the querier's `DebugPayload` in-memory (no files written
unless explicitly flushed) when `debug_log=True`.

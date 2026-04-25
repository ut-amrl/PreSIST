# persistence_eval

Evaluation framework for semantic persistence filters. Given an observation of an object
(image + segmentation mask), the system estimates how long that object is likely to remain
present, then tests those predictions against ground-truth track data.

---

## Concepts

**Persistence parameter inference** — An inferencer looks at an image and mask and outputs
a small set of named parameters (e.g. `{"t50": 120.0, "t90": 400.0}`) that describe the
predicted lifetime distribution of the object.

**Persistence filter** — A Bayesian filter that takes those parameters, a survival function
built from them, and a history of prior observations, and outputs P(object exists) at one or
more future query times.

**Episode** — One evaluation unit: a single (artifact, context) pair consisting of a set of
inferred persistence parameters plus the prior observations, prediction horizons, and ground
truth needed to evaluate them.

---

## Pipeline Overview

```
┌─────────────────────── INFERENCE ───────────────────────┐
│                                                          │
│  ImageMaskView                                           │
│      │ iter_image_groups()  ← one item per frame         │
│      ▼                                                   │
│  InferenceDataset (IterableDataset)                      │
│      │ yields ImageMaskGroup (K×H×W masks + image)       │
│      ▼                                                   │
│  DataLoader ──► InferenceRunner                          │
│                     │ inferencer.infer(group)            │
│                     │   → list[PersistenceParameterEstimate]
│                     ▼                                    │
│               InferenceArtifact (track_id, anchor_time,  │
│                   estimate, latency) × N                 │
│                     │ saved every N via ArtifactStore    │
└─────────────────────────────────────────────────────────┘

┌─────────────────────── EVALUATION ──────────────────────┐
│                                                          │
│  ArtifactStore.load()  +  TrackBasedEpisodeProvider      │
│      │                        │ get_contexts(track_id,   │
│      │                        │   anchor_time)           │
│      │                        │   → EvaluationContext    │
│      │                        │   (prior_obs, query_times│
│      │                        │    bounds, metadata)     │
│      ▼                        ▼                          │
│  EvaluationDataset  yields (artifact, context) pairs     │
│      ▼                                                   │
│  DataLoader ──► EvaluationRunner                         │
│                     │ survival_fn = gen.build(estimate)  │
│                     │ probs = filter.predict(survival_fn,│
│                     │   prior_obs, query_times)          │
│                     │ accumulator.update(...)            │
│                     ▼                                    │
│               MetricAccumulator.compute()                │
│                     │ → list[MetricResult]               │
│                     ▼                                    │
│               MetricsAnalyzer.analyze(results, batch,    │
│                   output_dir)                            │
└─────────────────────────────────────────────────────────┘
```

---

## Directory Structure

```
persistence_eval/
├── core/
│   ├── domain.py          # All dataclasses (ImageMaskGroup, InferenceArtifact,
│   │                      #   EvaluationContext, EvaluationBatch, ...)
│   └── protocols.py       # All swappable interfaces (PersistenceParameterInferencer,
│                          #   EpisodeProvider, MetricAccumulator, ...)
│
├── datasets/
│   ├── backing.py              # AbsBackingDataset — raw data access
│   ├── image_mask_view.py      # AbsImageMaskDataset / ImageMaskView — yields
│   │                           #   AnchorObservation and ImageMaskGroup
│   ├── track_view.py           # AbsTrackDataset / TrackView — yields track sequences
│   ├── inference_dataset.py    # InferenceDataset(IterableDataset) wrapping ImageMaskView
│   ├── evaluation_dataset.py   # EvaluationDataset(IterableDataset) joining artifacts
│   │                           #   with EvaluationContexts
│   ├── annotation_tool.py      # Concrete dataset backed by annotation-tool JSON exports
│   └── dummy.py                # In-memory dummy dataset + DummyInferencer for tests
│
├── sampling/
│   └── episode_provider.py     # TrackBasedEpisodeProvider — maps (track_id, anchor_time)
│                               #   to EvaluationContext(s) with prior obs + query horizons
│
├── methods/
│   ├── inferencers/
│   │   ├── base.py             # AbsPersistenceParameterInferencer
│   │   ├── learned_model.py    # Neural network checkpoint inferencer
│   │   └── llm_pipeline.py     # Scene caption → object description → LLM querier
│   ├── survival/
│   │   ├── exponential.py      # Exp(lam); fits from 'mean' or 't50'
│   │   ├── weibull.py          # Weibull(k, lam); fits from two percentiles
│   │   ├── log_normal.py       # LogNormal(mu, sigma); fits from two percentiles
│   │   └── general_purpose.py  # Log-uniform prior over exponential rates
│   ├── filters/
│   │   └── rosen_filter.py     # Bayesian persistence filter (Rosen et al., ICRA 2016)
│   ├── evaluation_method.py    # EvaluationMethod dataclass (survival_gen + filter)
│   └── benchmark_method.py     # BenchmarkMethod dataclass (EvaluationMethod + inferencer)
│
├── metrics/
│   ├── base.py            # Metric (legacy batch), AbsMetricAccumulator,
│   │                      #   BatchingMetricAccumulator
│   ├── classification.py  # ThresholdAccuracy
│   ├── existence.py       # ExistenceMAE
│   └── latency.py         # MeanQueryLatencyMs, MeanImageTotalLatencyMs
│
├── runners/
│   ├── inference.py       # InferenceRunner — DataLoader loop + periodic checkpointing
│   ├── evaluation.py      # EvaluationRunner — DataLoader loop + accumulator streaming
│   └── orchestrator.py    # BenchmarkOrchestrator — top-level coordinator (Hydra entry)
│
├── storage/
│   └── artifact_store.py  # JsonArtifactStore — save/load/append InferenceArtifacts
│
└── analysis/
    └── reporting.py       # BasicMetricsAnalyzer — writes JSON + CSV output files
```

---

## Running the Pipeline

### Inference only

Runs the inferencer over all image+mask observations and writes artifacts to disk.
Artifacts are checkpointed every `save_every_n` image groups so progress is not lost
on long runs.

```python
from pathlib import Path
from persistence_eval.datasets.annotation_tool import (
    AnnotationToolBackingDataset, AnnotationToolImageMaskView
)
from persistence_eval.datasets.inference_dataset import InferenceDataset
from persistence_eval.methods.inferencers.llm_pipeline import LLMPipelineInferencer
from persistence_eval.runners.inference import InferenceRunner
from persistence_eval.storage.artifact_store import JsonArtifactStore

backing = AnnotationToolBackingDataset(Path("data/manifest.json"))
view = AnnotationToolImageMaskView(backing)

# mode="grouped"    → one item per image frame, all masks batched together (default)
# mode="individual" → one item per (image, mask) pair
dataset = InferenceDataset(view, mode="grouped")

inferencer = LLMPipelineInferencer(
    sceneCaptioner=my_captioner,
    objectDescriber=my_describer,
    llmPersistenceQuerier=my_querier,
)

runner = InferenceRunner(inferencer, JsonArtifactStore(), save_every_n=50)
artifacts = runner.run(dataset, run_dir=Path("outputs/run_001"))
# → outputs/run_001/inference_artifacts.json written incrementally
```

### Evaluation only (from saved artifacts)

Loads artifacts from a previous inference run and evaluates them against track data.

```python
from persistence_eval.datasets.annotation_tool import (
    AnnotationToolBackingDataset, AnnotationToolTrackView
)
from persistence_eval.datasets.evaluation_dataset import EvaluationDataset
from persistence_eval.sampling.episode_provider import TrackBasedEpisodeProvider
from persistence_eval.methods.evaluation_method import EvaluationMethod
from persistence_eval.methods.survival.weibull import WeibullSurvivalFunctionGenerator
from persistence_eval.methods.filters.rosen_filter import RosenPersistenceFilter
from persistence_eval.metrics.base import BatchingMetricAccumulator
from persistence_eval.metrics.existence import ExistenceMAE
from persistence_eval.metrics.classification import ThresholdAccuracy
from persistence_eval.runners.evaluation import EvaluationRunner
from persistence_eval.storage.artifact_store import JsonArtifactStore
from persistence_eval.analysis.reporting import BasicMetricsAnalyzer

backing = AnnotationToolBackingDataset(Path("data/manifest.json"))
track_view = AnnotationToolTrackView(backing)

artifacts = JsonArtifactStore().load(Path("outputs/run_001"))

provider = TrackBasedEpisodeProvider(
    track_dataset=track_view,
    query_horizons=[1.0, 5.0, 10.0, 30.0, 60.0],  # seconds after anchor
)

method = EvaluationMethod(
    name="weibull_rosen",
    survival_generator=WeibullSurvivalFunctionGenerator(),
    persistence_filter=RosenPersistenceFilter(
        false_detection_prob=0.05,
        missed_detection_prob=0.05,
    ),
)

accumulators = [
    BatchingMetricAccumulator(ExistenceMAE()),
    BatchingMetricAccumulator(ThresholdAccuracy(threshold=0.5)),
]

eval_dataset = EvaluationDataset(artifacts, provider)
runner = EvaluationRunner(method, accumulators)
batch, results = runner.run(eval_dataset)

BasicMetricsAnalyzer().analyze(results, batch, Path("outputs/run_001/eval"))
# → outputs/run_001/eval/metric_results.json
# → outputs/run_001/eval/prediction_records.json
# → outputs/run_001/eval/group_summary.csv
```

### Inference + evaluation in one pass

```python
from persistence_eval.methods.benchmark_method import BenchmarkMethod
from persistence_eval.runners.orchestrator import BenchmarkOrchestrator

# BenchmarkOrchestrator.from_cfg(cfg) is the Hydra entry point.
# For programmatic use, construct directly:
benchmark_method = BenchmarkMethod(evaluation_method=method, inferencer=inferencer)

orchestrator = BenchmarkOrchestrator(
    inference_dataset=InferenceDataset(view),
    episode_provider=provider,
    benchmark_method=benchmark_method,
    accumulators=accumulators,
    analysis=BasicMetricsAnalyzer(),
    output_layout=layout,
    explicit_inference_run_dir=None,
    eval_type_name="weibull",
    timestamp="20240101_120000",
)

orchestrator.run_all()
# or:
artifacts, run_dir = orchestrator.run_inference()
batch, results   = orchestrator.run_evaluation()
```

---

## Key Interfaces

Each interface can be swapped independently. You only need to implement one method.

### `PersistenceParameterInferencer`
**Where:** `core/protocols.py` | **Used by:** `InferenceRunner`

```python
def infer(self, group: ImageMaskGroup) -> list[PersistenceParameterEstimate]: ...
```

`group.masks` is `(K, H, W)` bool — all masks from a single image frame.
Returns one estimate per mask (same order as `group.track_ids`).

| Implementation | When to use |
|---|---|
| `LLMPipelineInferencer` | LLM/VLM-based estimation via scene captioning + object description |
| `LearnedModelInferencer` | Trained neural network checkpoint |
| `DummyInferencer` | Smoke tests / baseline comparisons |
| Custom static heuristic | Fixed distribution regardless of image content (e.g. "always t50=60s") |

**To add a new inferencer:** Subclass `AbsPersistenceParameterInferencer` and implement
`infer()`. The output params dict keys (e.g. `"t50"`, `"t90"`, `"mean"`) must match what
your chosen `SurvivalFunctionGenerator` expects.

---

### `SurvivalFunctionGenerator`
**Where:** `core/protocols.py` | **Used by:** `EvaluationRunner`

```python
def build(self, estimate: PersistenceParameterEstimate) -> SurvivalFunction: ...
```

Converts the named parameters from an artifact into a callable survival function S(dt).

| Implementation | Params required | Distribution |
|---|---|---|
| `ExponentialSurvivalFunctionGenerator` | `"mean"` or `"t50"` | Exponential |
| `WeibullSurvivalFunctionGenerator` | `"t50"`, `"t90"` (configurable) | Weibull |
| `LogNormalSurvivalFunctionGenerator` | `"t50"`, `"t90"` (configurable) | Log-normal |
| `GeneralPurposeSurvivalFunctionGenerator` | `"lambda_l"`, `"lambda_u"` | Log-uniform prior over Exp rates |

**When to swap:** Choose the distribution that best matches your inferencer's output format.
Weibull and log-normal both accept two percentile estimates and differ in tail behavior.
The general-purpose generator is useful when the inferencer returns rate bounds rather
than percentiles.

---

### `PersistenceFilter`
**Where:** `core/protocols.py` | **Used by:** `EvaluationRunner`

```python
def predict(self, survival_fn, prior_observations, query_times) -> torch.Tensor: ...
```

| Implementation | When to use |
|---|---|
| `RosenPersistenceFilter` | Standard Bayesian filter with false/missed detection probabilities |
| Custom | Different observation noise model, non-Bayesian fusion, etc. |

---

### `EpisodeProvider`
**Where:** `core/protocols.py` | **Used by:** `EvaluationDataset`

```python
def get_contexts(self, track_id: str, anchor_time: float) -> Iterable[EvaluationContext]: ...
```

Given an artifact's `(track_id, anchor_time)`, returns the prior observations, query
horizons, ground-truth bounds, and metadata needed to evaluate it.

| Implementation | When to use |
|---|---|
| `TrackBasedEpisodeProvider` | Standard: looks up track data by ID, applies fixed query horizons |
| Custom | Multiple horizon sets per artifact; filtering by track properties; synthetic evaluation contexts |

**Key constructor options for `TrackBasedEpisodeProvider`:**
- `query_horizons` — list of dt offsets (seconds) to predict at
- `require_disappearance_observed` — skip tracks where the object never disappears
- `include_prior_history` — whether to pass prior observations to the filter
- `max_prior_observations` — cap the number of prior observations (recency bias)

---

### `MetricAccumulator`
**Where:** `core/protocols.py` | **Used by:** `EvaluationRunner`

```python
def update(self, predicted_probs, ground_truth, query_times,
           num_prior_observations, prior_observation_timespan_s,
           inference_latency_ms, filter_latency_ms, metadata) -> None: ...
def compute(self) -> list[MetricResult]: ...
```

Called once per (artifact, context) pair with tensors of shape `(Q,)` where Q is the
number of query horizons. Returns `MetricResult` objects with an optional `group` dict
for stratification.

**Stratification:** An accumulator can bucket results by any input field and return
multiple `MetricResult`s — e.g. one per query horizon bucket, one per prior-history
length range. Set `MetricResult.group = {"query_time_bucket": "0-10s"}` to enable
downstream grouping in the analyzer.

| Implementation | When to use |
|---|---|
| `BatchingMetricAccumulator(metric)` | Wraps any legacy `Metric` (e.g. `ExistenceMAE`, `ThresholdAccuracy`) for streaming use |
| Custom `AbsMetricAccumulator` | Streaming computation, calibration curves, per-horizon breakdown, confidence-weighted metrics |

---

### `ArtifactStore`
**Where:** `core/protocols.py` | **Used by:** `InferenceRunner`, `BenchmarkOrchestrator`

```python
def save(self, artifacts, run_dir) -> None: ...
def append(self, artifacts, run_dir) -> None: ...   # called every N groups during inference
def load(self, run_dir) -> list[InferenceArtifact]: ...
```

| Implementation | When to use |
|---|---|
| `JsonArtifactStore` | Default — human-readable, easy to inspect |
| Custom | Binary format for large runs, database-backed storage, cloud blob storage |

---

### `MetricsAnalyzer`
**Where:** `core/protocols.py` | **Used by:** `BenchmarkOrchestrator`

```python
def analyze(self, metrics, batch, output_dir) -> None: ...
```

Receives the final `list[MetricResult]` and the full `EvaluationBatch` (flat tensors of
all predictions). Runs completely after evaluation, so it can be re-run offline on saved
outputs without re-running inference or evaluation.

| Implementation | When to use |
|---|---|
| `BasicMetricsAnalyzer` | Writes `metric_results.json`, `prediction_records.json`, `group_summary.csv` |
| Custom | Calibration plots, per-class breakdowns, debug overlay images, Weights & Biases logging |

---

## Metrics

### Default metrics (always reported by `BenchmarkOrchestrator.from_cfg`)

Every evaluation run automatically reports:

| Metric name | Class | Description |
|---|---|---|
| `existence_mae` | `ExistenceMAE` | MAE between predicted probability and 0/1 ground truth |
| `accuracy@0.5` | `ThresholdAccuracy` | Fraction of correct predictions at threshold 0.5 |
| `balanced_accuracy@0.5` | `ThresholdAccuracy` | Mean of TPR and TNR — accounts for class imbalance |
| `f1@0.5` | `ThresholdAccuracy` | Harmonic mean of precision and recall |
| `precision@0.5` | `ThresholdAccuracy` | TP / (TP + FP) |
| `recall@0.5` | `ThresholdAccuracy` | TP / (TP + FN) |
| `mean_image_total_latency_ms` | `MeanImageTotalLatencyMs` | Per-image mean of inference + filter latency |

All classification metrics skip records where ground truth is unknown (NaN).
Degenerate cases (e.g. no positive predictions) return 0.0 rather than raising.

### Adding metrics via config

Pass additional `Metric` instances in `cfg.metrics` — they are appended after the defaults:

```yaml
# configs/train_confs/eval_config.yaml
metrics:
  - _target_: persistence_eval.metrics.classification.ThresholdAccuracy
    threshold: 0.3
  - _target_: persistence_eval.metrics.latency.MeanQueryLatencyMs
```

This produces the seven default metrics **plus** `accuracy@0.3`, `balanced_accuracy@0.3`, etc. and `mean_query_latency_ms`.

### Opting out of defaults

To run with a fully custom metric set, construct `BenchmarkOrchestrator` directly (bypassing `from_cfg`) and supply your own `accumulators` list:

```python
from persistence_eval.metrics.base import BatchingMetricAccumulator, CompositeMetricSuite
from persistence_eval.metrics.existence import ExistenceMAE

accumulators = [BatchingMetricAccumulator(CompositeMetricSuite([ExistenceMAE()]))]
orchestrator = BenchmarkOrchestrator(..., accumulators=accumulators)
```

---

## Dataset: Annotation Tool Format

The on-disk format is a manifest JSON pointing to per-track JSON files:

```
manifest.json          {"run_id": "...", "location_root": "...", "track_files": [...]}
tracks/track_0001.json {"track_id": "...", "observations": [...], "boundaries": {...}}
```

**Splitting by time** (e.g. train/test split at a timestamp boundary):

```python
from persistence_eval.datasets.annotation_tool import (
    AnnotationToolBackingDataset, split_at_boundary,
    AnnotationToolImageMaskView, AnnotationToolTrackView,
)

backing = AnnotationToolBackingDataset(Path("data/manifest.json"))
before, after = split_at_boundary(backing, boundary_timestamp=3600.0)  # seconds

# Use 'before' for training / inference, 'after' for evaluation
infer_view = AnnotationToolImageMaskView(before)
eval_track_view = AnnotationToolTrackView(after)
```

Tracks with positive observations on both sides of the boundary are excluded from both
halves to avoid label leakage.

---

## Closed-Set CLIP Inferencer: Cache Preparation

`ClosedSetCLIPInferencer` assigns each object to a semantic class via CLIP cosine similarity, then
looks up pre-computed persistence parameters for that class. Two offline caches must be built before
this inferencer can be used:

1. **Persistence parameter lookup** — ask a VLM (Qwen3-VL) once per class what the typical
   persistence is, save the answers as a JSON lookup table.
2. **CLIP alignment cache** — embed every class name with CLIP's text encoder once, save the
   normalized vectors so the inferencer can find the nearest class at runtime without reloading CLIP.

Both caches depend on a **class list file** (one class name per line). Build the persistence cache
first because it is model-dependent and may require GPU time; the CLIP cache can be built any time.

---

### Step 1 — Write a class list file

```
chair
desk
laptop
car
person
wall
```

Save it somewhere stable, e.g. `configs/class_lists/my_classes.txt`.

---

### Step 2 — Build the persistence parameter lookup (Qwen3-VL)

```bash
python -m persistence_eval.methods.inferencers.closed_set_semantics.semantic_class_persistence_precomputer \
    --class_list_file configs/class_lists/my_classes.txt \
    --output_file outputs/caches/persistence_params.json \
    --prompt_template_file prompt_files/persistence_precomputer/class_persistence_query.txt \
    --param_keys p50_persistence_hours p90_persistence_hours \
    --output_param_keys t50 t90 \
    --model_name Qwen/Qwen3-VL-8B-Thinking \
    --use_vllm \
    --vllm_port 9001
```

**Key flags:**

| Flag | Purpose |
|---|---|
| `--prompt_template_file` | Text file with a `{class_name}` placeholder. The default template at `prompt_files/persistence_precomputer/class_persistence_query.txt` asks for `p50_persistence_hours` and `p90_persistence_hours`. |
| `--param_keys` | Keys to read from the VLM's JSON response — must match the keys your prompt asks for. |
| `--output_param_keys` | Keys to store in the lookup JSON — must match what your `SurvivalFunctionGenerator` expects (e.g. `t50 t90` for `WeibullSurvivalFunctionGenerator`). |
| `--time_unit` | `hours` (default) multiplies VLM values × 3600 before storing. Use `seconds` if your prompt already returns seconds. |
| `--use_vllm` | Connect to a running vLLM server instead of loading the model in-process. |

If any class fails to parse (malformed JSON in the response), the script prints a warning and exits
with code 1. Fix failures manually by adding the class to the JSON file or re-running on the subset:

```bash
# Inspect the output, then patch manually or re-run on failing classes only
python -m persistence_eval.methods.inferencers.closed_set_semantics.semantic_class_persistence_precomputer \
    --class_list_file configs/class_lists/failing_classes.txt \
    --output_file outputs/caches/persistence_params_patch.json \
    ...
```

Merge patch files manually into the main JSON before building the CLIP cache.

---

### Step 3 — Build the CLIP alignment cache

```bash
python -m persistence_eval.methods.inferencers.closed_set_clip_align.class_clipalignment_cacher \
    --class_list_file configs/class_lists/my_classes.txt \
    --output_file outputs/caches/clip_cache.pt \
    --model_name openai/clip-vit-large-patch14 \
    --device cuda
```

This embeds every class name with CLIP's text encoder and saves a `dict[class_name → tensor]` to
the `.pt` file. Re-run whenever the class list changes. The CLIP model is ~1.7 GB; the cache itself
is small (D-dimensional float32 vector per class, typically a few KB total).

---

### Step 4 — Wire up the inferencer

```python
from persistence_eval.methods.inferencers.closed_set_semantics.closed_set_semantics import ClosedSetSemanticClassList
from persistence_eval.methods.inferencers.closed_set_clip_align.closed_set_clip_embedder import (
    CLIPEmbedder, CachedCLIPAlignment, CLIPAligner,
)
from persistence_eval.methods.inferencers.closed_set_semantics.semantic_class_persistence_param_lookup import (
    SemanticClassPersistenceParamLookup,
)
from persistence_eval.methods.inferencers.closed_set_clip_inferencer import ClosedSetCLIPInferencer

# Load class list (provides id ↔ name mapping used by both caches)
class_list = ClosedSetSemanticClassList(class_text_file="configs/class_lists/my_classes.txt")

# Load caches
cache = CachedCLIPAlignment()
cache.load_cache("outputs/caches/clip_cache.pt")

lookup = SemanticClassPersistenceParamLookup()
lookup.load_params_from_file("outputs/caches/persistence_params.json")

# Build inferencer
embedder = CLIPEmbedder(model_name="openai/clip-vit-large-patch14", device="cuda")
aligner = CLIPAligner(clip_embedder=embedder, cache=cache, class_list=class_list)
inferencer = ClosedSetCLIPInferencer(clipAligner=aligner, semanticClassPersistenceParamLookup=lookup)
```

The CLIP model is lazy-loaded on first call to `find_closest_class`, so construction is fast.
The `class_list` is the source of truth for the integer ID ↔ name mapping: the CLIP cache uses
names (for text embedding), and the persistence lookup and aligner output use integer IDs.

---

### Cache summary

| Cache file | Built by | Rebuild when |
|---|---|---|
| `persistence_params.json` | `semantic_class_persistence_precomputer.py` | Class list changes, or prompt / VLM updated |
| `clip_cache.pt` | `class_clipalignment_cacher.py` | Class list or CLIP model changes |

---

## Adding a New Inferencer End-to-End

1. **Implement** `AbsPersistenceParameterInferencer.infer(group: ImageMaskGroup)`.
   Output `params` keys must match your chosen `SurvivalFunctionGenerator`.

2. **Choose** a `SurvivalFunctionGenerator` that accepts those keys, or implement one.

3. **Run inference:**
   ```python
   runner = InferenceRunner(my_inferencer, JsonArtifactStore(), save_every_n=100)
   artifacts = runner.run(InferenceDataset(view, mode="grouped"), run_dir)
   ```

4. **Evaluate:**
   ```python
   provider = TrackBasedEpisodeProvider(track_view, query_horizons=[5.0, 30.0, 120.0])
   runner = EvaluationRunner(EvaluationMethod("my_method", my_gen, my_filter), accumulators)
   batch, results = runner.run(EvaluationDataset(artifacts, provider))
   ```

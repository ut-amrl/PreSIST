"""End-to-end smoke test for the refactored inference + evaluation pipeline."""
import pathlib
import tempfile

from persistence_eval.datasets.dummy import DummyBackingDataset, DummyImageMaskView, DummyTrackView, DummyInferencer
from persistence_eval.datasets.inference_dataset import InferenceDataset
from persistence_eval.datasets.evaluation_dataset import EvaluationDataset
from persistence_eval.sampling.episode_provider import TrackBasedEpisodeProvider
from persistence_eval.storage.artifact_store import JsonArtifactStore
from persistence_eval.runners.inference import InferenceRunner
from persistence_eval.runners.evaluation import EvaluationRunner
from persistence_eval.methods.survival.weibull import WeibullSurvivalFunctionGenerator
from persistence_eval.methods.filters.rosen_filter import RosenPersistenceFilter
from persistence_eval.methods.predictors.survival_based import SurvivalBasedPredictor
from persistence_eval.methods.evaluation_method import EvaluationMethod
from persistence_eval.metrics.base import BatchingMetricAccumulator
from persistence_eval.metrics.existence import ExistenceMAE

backing = DummyBackingDataset()
view = DummyImageMaskView(backing)
track_view = DummyTrackView(backing)
inferencer = DummyInferencer()
store = JsonArtifactStore()
provider = TrackBasedEpisodeProvider(track_view, query_horizons=[1.0, 3.0, 5.0])

inf_dataset = InferenceDataset(view, mode="grouped")
with tempfile.TemporaryDirectory() as tmpdir:
    run_dir = pathlib.Path(tmpdir)
    runner = InferenceRunner(inferencer, store, save_every_n=10)
    artifacts = runner.run(inf_dataset, run_dir)
    print(f"Inference artifacts: {len(artifacts)}")
    print(f"First artifact: track_id={artifacts[0].track_id}, anchor_time={artifacts[0].anchor_time}")
    loaded = store.load(run_dir)
    assert len(loaded) == len(artifacts), f"Expected {len(artifacts)} loaded, got {len(loaded)}"
    print(f"Loaded from disk: {len(loaded)} OK")

method = EvaluationMethod(
    name="test",
    predictor=SurvivalBasedPredictor(
        survival_generator=WeibullSurvivalFunctionGenerator(),
        persistence_filter=RosenPersistenceFilter(false_detection_prob=0.05, missed_detection_prob=0.05),
    ),
)
acc = BatchingMetricAccumulator(ExistenceMAE())
eval_dataset = EvaluationDataset(artifacts, provider)
eval_runner = EvaluationRunner(method, [acc])
batch, results = eval_runner.run(eval_dataset)
print(f"Batch records: {len(batch)}")
print(f"Metric results: {results}")
assert len(batch) > 0, "Expected non-empty batch"
assert len(results) > 0, "Expected non-empty metric results"
print("SMOKE TEST PASSED")

from __future__ import annotations

from pathlib import Path

from omegaconf import DictConfig

from persistence_eval.core.domain import EvaluationBatch, InferenceArtifact, MetricResult
from persistence_eval.datasets.evaluation_dataset import EvaluationDataset
from persistence_eval.datasets.inference_dataset import InferenceDataset
from persistence_eval.methods.evaluation_method import EvaluationMethod
from persistence_eval.methods.benchmark_method import BenchmarkMethod
from persistence_eval.metrics.base import BatchingMetricAccumulator
from persistence_eval.runners.evaluation import EvaluationRunner
from persistence_eval.runners.inference import InferenceRunner
from persistence_eval.sampling.episode_provider import TrackBasedEpisodeProvider
from persistence_eval.storage.artifact_store import JsonArtifactStore
from persistence_eval.utils.paths import make_timestamp, resolve_output_layout


class BenchmarkOrchestrator:
    """
    Top-level coordinator for inference and evaluation runs.
    Constructed from a Hydra DictConfig via from_cfg(); supports three modes:
      infer      — run inferencer, write artifacts to disk
      evaluate   — load artifacts from disk, run eval pipeline
      run_all    — run both in sequence without a disk round-trip between them
    """

    def __init__(
        self,
        inference_dataset: InferenceDataset,
        episode_provider: TrackBasedEpisodeProvider,
        benchmark_method: BenchmarkMethod,
        accumulators,
        analysis,
        output_layout,
        explicit_inference_run_dir: Path | None,
        eval_type_name: str,
        timestamp: str,
        save_every_n: int = 100,
        num_workers: int = 0,
        halt_on_incompatible_estimate: bool = True,
    ) -> None:
        self._inference_dataset = inference_dataset
        self._episode_provider = episode_provider
        self._benchmark_method = benchmark_method
        self._accumulators = accumulators
        self._analysis = analysis
        self._layout = output_layout
        self._explicit_inference_run_dir = explicit_inference_run_dir
        self._eval_type_name = eval_type_name
        self._timestamp = timestamp
        self._num_workers = num_workers
        self._store = JsonArtifactStore()
        self._inference_runner = InferenceRunner(
            inferencer=benchmark_method.inferencer,
            artifact_store=self._store,
            save_every_n=save_every_n,
        )
        self._evaluation_runner = EvaluationRunner(
            method=benchmark_method.evaluation_method,
            accumulators=accumulators,
            halt_on_incompatible_estimate=halt_on_incompatible_estimate,
        )

    @staticmethod
    def from_cfg(cfg: DictConfig) -> "BenchmarkOrchestrator":
        import hydra.utils

        from persistence_eval.datasets.dummy import DummyBackingDataset, DummyImageMaskView, DummyTrackView
        from persistence_eval.metrics.base import CompositeMetricSuite

        # Dataset views — _recursive_=False so Hydra doesn't try to instantiate
        # image_mask_view/track_view sub-configs (they need `backing`, passed below).
        print("Creating dataset views")
        backing = hydra.utils.instantiate(cfg.benchmark_dataset, _recursive_=False)
        image_mask_view = (
            hydra.utils.instantiate(cfg.benchmark_dataset.image_mask_view, backing=backing)
            if hasattr(cfg.benchmark_dataset, "image_mask_view")
            else DummyImageMaskView(backing)
        )
        track_view = (
            hydra.utils.instantiate(cfg.benchmark_dataset.track_view, backing=backing)
            if hasattr(cfg.benchmark_dataset, "track_view")
            else DummyTrackView(backing)
        )

        # Dataset mode (grouped or individual)
        dataset_mode = cfg.benchmark.get("dataset_mode", "grouped")
        inference_dataset = InferenceDataset(image_mask_view, mode=dataset_mode)

        print("Creating episode provider")
        # Episode provider: prefer top-level eval_scenario, fall back to benchmark.sampler
        if hasattr(cfg, "eval_scenario"):
            episode_provider = hydra.utils.instantiate(cfg.eval_scenario, track_dataset=track_view)
        else:
            sampler_cfg = cfg.benchmark.get("sampler", {})
            episode_provider = TrackBasedEpisodeProvider(
                track_dataset=track_view,
                query_horizons=list(sampler_cfg.get("query_horizons", [1.0, 3.0, 5.0])),
                require_disappearance_observed=sampler_cfg.get("require_disappearance_observed", False),
            )

        print("Creating inferencer")
        # Inferencer
        inferencer_cfg = cfg.inferencer
        model_cfg = cfg.get("model", None)
        inferencer = hydra.utils.instantiate(
            inferencer_cfg,
            **({"model_cfg": model_cfg} if model_cfg is not None else {}),
        )

        # Compose method name from LLM pipeline sub-component names, or fall back to _target_
        sc_name = cfg.inferencer.get("sceneCaptioner", {}).get("name", "")
        od_name = cfg.inferencer.get("objectDescriber", {}).get("name", "")
        ph_name = cfg.inferencer.get("llmPersistenceQuerier", {}).get("promptHandler", {}).get("name", "")
        lq_name = cfg.inferencer.get("llmPersistenceQuerier", {}).get("llmQuerier", {}).get("name", "")
        pq_name = f"ph_{ph_name}__{lq_name}" if (ph_name and lq_name) else ""
        if sc_name and od_name and pq_name:
            method_name = f"{sc_name}__{od_name}__{pq_name}"
        else:
            method_name = cfg.inferencer.get("name") or cfg.inferencer.get("_target_", "unknown").split(".")[-1]

        print("Creating predictor")

        # EvaluationMethod
        predictor = hydra.utils.instantiate(cfg.predictor)
        eval_method = EvaluationMethod(
            name=method_name,
            predictor=predictor,
        )
        benchmark_method = BenchmarkMethod(evaluation_method=eval_method, inferencer=inferencer)

        # Metrics — always include the default suite, then any extras from config
        from persistence_eval.metrics.existence import ExistenceMAE
        from persistence_eval.metrics.classification import ThresholdAccuracy
        from persistence_eval.metrics.latency import MeanImageTotalLatencyMs

        metric_cfgs = cfg.get("metrics", [])
        extra_metrics = [hydra.utils.instantiate(m) for m in metric_cfgs] if metric_cfgs else []
        raw_metrics = [
            ExistenceMAE(),
            ThresholdAccuracy(threshold=0.5),
            MeanImageTotalLatencyMs(),
            *extra_metrics,
        ]
        print("Creating analysis objects")
        metric_suite = CompositeMetricSuite(raw_metrics)
        accumulators = [BatchingMetricAccumulator(metric_suite)]

        # Analysis
        analysis = hydra.utils.instantiate(cfg.analysis)

        # Output paths
        ts = cfg.runtime.get("timestamp") or make_timestamp()
        dataset_name = cfg.benchmark_dataset.get("name") or track_view.name
        pred_name = cfg.predictor.get("name") or cfg.predictor.get("_target_", "pred").split(".")[-1]
        ep_name = cfg.eval_scenario.get("name") if hasattr(cfg, "eval_scenario") else ""
        eval_type = f"{pred_name}__{ep_name}" if ep_name else pred_name
        layout = resolve_output_layout(
            output_root=Path(cfg.paths.output_root),
            dataset_name=dataset_name,
            method_name=eval_method.name,
            benchmark_name=cfg.benchmark.get("name", "default"),
        )
        explicit_dir = (
            Path(cfg.paths.explicit_inference_run_dir)
            if cfg.paths.get("explicit_inference_run_dir")
            else None
        )

        print("Creating orchestrator")
        return BenchmarkOrchestrator(
            inference_dataset=inference_dataset,
            episode_provider=episode_provider,
            benchmark_method=benchmark_method,
            accumulators=accumulators,
            analysis=analysis,
            output_layout=layout,
            explicit_inference_run_dir=explicit_dir,
            eval_type_name=eval_type,
            timestamp=ts,
            save_every_n=cfg.benchmark.get("save_every_n", 100),
            num_workers=cfg.benchmark.get("num_workers", 0),
            halt_on_incompatible_estimate=cfg.runtime.get("halt_on_incompatible_estimate", True),
        )

    def run_inference(self) -> tuple[list[InferenceArtifact], Path]:
        run_dir = self._layout.inference_run_dir(self._timestamp)
        artifacts = self._inference_runner.run(
            self._inference_dataset,
            run_dir,
            num_workers=self._num_workers,
        )
        return artifacts, run_dir

    def run_evaluation(self) -> tuple[EvaluationBatch, list[MetricResult]]:
        inference_dir = self._explicit_inference_run_dir or self._layout.latest_inference_run()
        artifacts = self._store.load(inference_dir)

        eval_dataset = EvaluationDataset(artifacts, self._episode_provider)
        batch, metric_results = self._evaluation_runner.run(
            eval_dataset,
            num_workers=self._num_workers,
        )

        eval_dir = inference_dir / "evaluation_runs" / self._eval_type_name / self._timestamp
        eval_dir.mkdir(parents=True, exist_ok=True)
        print("Starting analysis ")
        self._analysis.analyze(metric_results, batch, eval_dir)
        return batch, metric_results

    def run_all(self) -> None:
        artifacts, run_dir = self.run_inference()

        eval_dataset = EvaluationDataset(artifacts, self._episode_provider)
        batch, metric_results = self._evaluation_runner.run(
            eval_dataset,
            num_workers=self._num_workers,
        )

        eval_dir = run_dir / "evaluation_runs" / self._eval_type_name / self._timestamp
        eval_dir.mkdir(parents=True, exist_ok=True)
        self._analysis.analyze(metric_results, batch, eval_dir)

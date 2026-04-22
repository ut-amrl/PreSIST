from __future__ import annotations

from pathlib import Path

from omegaconf import DictConfig

from persistence_eval.core.domain import (
    InferenceArtifact,
    MetricResult,
    PredictionRecord,
)
from persistence_eval.methods.evaluation_method import EvaluationMethod
from persistence_eval.methods.benchmark_method import BenchmarkMethod
from persistence_eval.runners.evaluation import EvaluationRunner
from persistence_eval.runners.inference import InferenceRunner
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
        episode_pairs,
        benchmark_method: BenchmarkMethod,
        metrics,
        analysis,
        output_layout,
        explicit_inference_run_dir: Path | None,
        eval_type_name: str,
        timestamp: str,
    ) -> None:
        self._episode_pairs = episode_pairs
        self._benchmark_method = benchmark_method
        self._metrics = metrics
        self._analysis = analysis
        self._layout = output_layout
        self._explicit_inference_run_dir = explicit_inference_run_dir
        self._eval_type_name = eval_type_name
        self._timestamp = timestamp
        self._store = JsonArtifactStore()
        self._inference_runner = InferenceRunner()
        self._evaluation_runner = EvaluationRunner()

    @staticmethod
    def from_cfg(cfg: DictConfig) -> "BenchmarkOrchestrator":
        import hydra.utils

        from persistence_eval.datasets.dummy import DummyBackingDataset, DummyTrackView
        from persistence_eval.sampling.specs import EpisodeSamplerSpec
        from persistence_eval.sampling.track_episode_sampler import SimpleTrackEpisodeSampler
        from persistence_eval.metrics.base import CompositeMetricSuite

        # Dataset + episode sampling
        backing = hydra.utils.instantiate(cfg.benchmark_dataset)
        track_view = hydra.utils.instantiate(cfg.benchmark_dataset.track_view, backing=backing) \
            if hasattr(cfg.benchmark_dataset, "track_view") else DummyTrackView(backing)

        sampler_spec = EpisodeSamplerSpec(**cfg.benchmark.get("sampler", {}))
        sampler = SimpleTrackEpisodeSampler(sampler_spec)
        episode_pairs = sampler.sample(track_view)

        # Inferencer (model_cfg passed explicitly when available)
        inferencer_cfg = cfg.inferencer
        model_cfg = cfg.get("model", None)
        inferencer = hydra.utils.instantiate(
            inferencer_cfg,
            **({"model_cfg": model_cfg} if model_cfg is not None else {}),
        )

        # EvaluationMethod
        survival_gen = hydra.utils.instantiate(cfg.survival_generator)
        pfilter = hydra.utils.instantiate(cfg.persistence_filter)
        eval_method = EvaluationMethod(
            name=cfg.inferencer.get("_target_", "unknown").split(".")[-1],
            survival_generator=survival_gen,
            persistence_filter=pfilter,
        )
        benchmark_method = BenchmarkMethod(evaluation_method=eval_method, inferencer=inferencer)

        # Metrics
        metric_cfgs = cfg.get("metrics", [])
        metrics = [hydra.utils.instantiate(m) for m in metric_cfgs] if metric_cfgs else []
        metric_suite = CompositeMetricSuite(metrics)

        # Analysis
        analysis = hydra.utils.instantiate(cfg.analysis)

        # Output paths
        ts = cfg.runtime.get("timestamp") or make_timestamp()
        layout = resolve_output_layout(
            output_root=Path(cfg.paths.output_root),
            dataset_name=track_view.name,
            method_name=eval_method.name,
            benchmark_name=cfg.benchmark.get("name", "default"),
        )
        explicit_dir = (
            Path(cfg.paths.explicit_inference_run_dir)
            if cfg.paths.get("explicit_inference_run_dir")
            else None
        )
        eval_type = f"{cfg.survival_generator.get('_target_', 'surv').split('.')[-1]}"

        return BenchmarkOrchestrator(
            episode_pairs=episode_pairs,
            benchmark_method=benchmark_method,
            metrics=[metric_suite],
            analysis=analysis,
            output_layout=layout,
            explicit_inference_run_dir=explicit_dir,
            eval_type_name=eval_type,
            timestamp=ts,
        )

    def run_inference(self) -> tuple[list[InferenceArtifact], Path]:
        episodes = [ep for ep, _ in self._episode_pairs]
        artifacts = self._inference_runner.run(episodes, self._benchmark_method.inferencer)
        run_dir = self._layout.inference_run_dir(self._timestamp)
        run_dir.mkdir(parents=True, exist_ok=True)
        self._store.save(artifacts, run_dir)
        return artifacts, run_dir

    def run_evaluation(self) -> tuple[list[PredictionRecord], list[MetricResult]]:
        inference_dir = self._explicit_inference_run_dir or self._layout.latest_inference_run()
        artifacts_list = self._store.load(inference_dir)
        artifacts = {a.episode_id: a for a in artifacts_list}

        records, metric_results = self._evaluation_runner.run(
            episode_pairs=self._episode_pairs,
            artifacts=artifacts,
            method=self._benchmark_method.evaluation_method,
            metrics=self._metrics,
        )

        eval_dir = inference_dir / "evaluation_runs" / self._eval_type_name / self._timestamp
        eval_dir.mkdir(parents=True, exist_ok=True)
        self._analysis.analyze(metric_results, records, eval_dir)
        return records, metric_results

    def run_all(self) -> None:
        episodes = [ep for ep, _ in self._episode_pairs]
        artifacts_list = self._inference_runner.run(episodes, self._benchmark_method.inferencer)

        run_dir = self._layout.inference_run_dir(self._timestamp)
        run_dir.mkdir(parents=True, exist_ok=True)
        self._store.save(artifacts_list, run_dir)

        artifacts = {a.episode_id: a for a in artifacts_list}
        records, metric_results = self._evaluation_runner.run(
            episode_pairs=self._episode_pairs,
            artifacts=artifacts,
            method=self._benchmark_method.evaluation_method,
            metrics=self._metrics,
        )

        eval_dir = run_dir / "evaluation_runs" / self._eval_type_name / self._timestamp
        eval_dir.mkdir(parents=True, exist_ok=True)
        self._analysis.analyze(metric_results, records, eval_dir)

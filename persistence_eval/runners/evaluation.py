from __future__ import annotations

import time
import warnings

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from persistence_eval.core.domain import EvaluationBatch, MetricResult
from persistence_eval.core.exceptions import IncompatibleEstimateError
from persistence_eval.datasets.evaluation_dataset import EvaluationDataset
from persistence_eval.methods.evaluation_method import EvaluationMethod
from persistence_eval.metrics.base import AbsMetricAccumulator


class EvaluationRunner:
    """
    Iterates an EvaluationDataset via a DataLoader, runs the persistence filter for each
    (artifact, context) pair, streams results to metric accumulators, and buffers records
    for EvaluationBatch construction (used by the analyzer).
    """

    def __init__(
        self,
        method: EvaluationMethod,
        accumulators: list[AbsMetricAccumulator],
        *,
        halt_on_incompatible_estimate: bool = True,
    ) -> None:
        self._method = method
        self._accumulators = accumulators
        self._halt_on_incompatible_estimate = halt_on_incompatible_estimate

    @staticmethod
    def _describe_incompatible_estimate(artifact) -> str:
        """
        Produces a human-readable diagnosis for an IncompatibleEstimateError.

        Three distinct situations produce this error:
          - Empty params ({}): the inferencer stored a sentinel artifact when all LLM
            retries failed — no persistence estimate was ever produced for this episode.
          - Misformatted params: the inferencer ran and returned params, but under key
            names the survival generator does not recognise (e.g. a different prompt
            template or querier produced differently-named keys).
          - Fully-absent artifact: the artifact was never written to the store at all
            (e.g. inference was interrupted).  These episodes are silently skipped by
            EvaluationDataset and never reach this method, so they cannot be diagnosed
            here.  If you suspect missing coverage, compare the artifact store against
            the inference dataset.
        """
        episode = f"{artifact.track_id}@{artifact.anchor_time:.3f}s"
        if not artifact.estimate.params:
            return (
                f"Episode {episode} has empty params {{}}.  "
                "This is the sentinel written by LLMPipelineInferencer when all "
                "querier retries are exhausted — the LLM produced no usable estimate "
                "for this episode during inference."
            )
        return (
            f"Episode {episode} has params with unrecognised keys: "
            f"{sorted(artifact.estimate.params)}.  "
            "The survival generator requires different keys — check that the "
            "prompt handler and survival generator config are matched (e.g. "
            "the querier returned 'p50_persistence_hours' but the generator "
            "expects 'p25_persistence_hours'/'p75_persistence_hours')."
        )

    def run(
        self,
        dataset: EvaluationDataset,
        *,
        num_workers: int = 0,
    ) -> tuple[EvaluationBatch, list[MetricResult]]:
        loader = DataLoader(
            dataset,
            batch_size=None,
            collate_fn=lambda x: x,
            num_workers=num_workers,
        )

        episode_ids: list[str] = []
        query_times_list: list[float] = []
        preds_list: list[float] = []
        gt_list: list[float] = []
        infer_lat_list: list[float] = []
        filter_lat_list: list[float] = []
        meta_list: list[dict] = []

        skipped_empty = 0
        skipped_misformatted = 0

        for artifact, ctx in tqdm(loader, desc="Evaluation", unit="episode"):
            try:
                survival_fn = self._method.survival_generator.build(artifact.estimate)
            except IncompatibleEstimateError as e:
                diagnosis = self._describe_incompatible_estimate(artifact)
                if self._halt_on_incompatible_estimate:
                    raise IncompatibleEstimateError(diagnosis) from e
                if not artifact.estimate.params:
                    skipped_empty += 1
                else:
                    skipped_misformatted += 1
                warnings.warn(f"Skipping episode: {diagnosis}", stacklevel=2)
                continue
            query_times_t = torch.tensor(ctx.query_times, dtype=torch.float64)

            t0 = time.perf_counter()
            probs_t = self._method.persistence_filter.predict(
                survival_fn, ctx.prior_observations, query_times_t
            )
            filter_lat_ms = (time.perf_counter() - t0) * 1000.0

            gt_t = torch.tensor(
                [
                    float(label) if (label := ctx.bounds.exists_at(artifact.anchor_time + qt)) is not None
                    else float("nan")
                    for qt in ctx.query_times
                ],
                dtype=torch.float64,
            )

            full_meta = {
                "track_id": artifact.track_id,
                "anchor_time": artifact.anchor_time,
                **ctx.metadata,
            }
            prior_span = (
                artifact.anchor_time - ctx.prior_observations[0].timestamp
                if ctx.prior_observations else 0.0
            )

            for acc in self._accumulators:
                acc.update(
                    probs_t,
                    gt_t,
                    query_times_t,
                    len(ctx.prior_observations),
                    prior_span,
                    artifact.inference_latency_ms,
                    filter_lat_ms,
                    full_meta,
                )

            episode_id = f"{artifact.track_id}:{artifact.anchor_time}"
            per_q_lat = filter_lat_ms / max(len(ctx.query_times), 1)
            for i, qt in enumerate(ctx.query_times):
                episode_ids.append(episode_id)
                query_times_list.append(qt)
                preds_list.append(probs_t[i].item())
                gt_list.append(gt_t[i].item())
                infer_lat_list.append(artifact.inference_latency_ms)
                filter_lat_list.append(per_q_lat)
                meta_list.append(full_meta)

        total_skipped = skipped_empty + skipped_misformatted
        if total_skipped:
            parts = []
            if skipped_empty:
                parts.append(f"{skipped_empty} with empty params (inference failure)")
            if skipped_misformatted:
                parts.append(f"{skipped_misformatted} with misformatted params")
            warnings.warn(
                f"Evaluation complete: skipped {total_skipped} episode(s) due to "
                f"IncompatibleEstimateError ({', '.join(parts)}).",
                stacklevel=2,
            )

        batch = EvaluationBatch(
            episode_ids=episode_ids,
            query_times=torch.tensor(query_times_list, dtype=torch.float64),
            predicted_probs=torch.tensor(preds_list, dtype=torch.float64),
            ground_truth=torch.tensor(gt_list, dtype=torch.float64),
            inference_latency_ms=torch.tensor(infer_lat_list, dtype=torch.float64),
            filter_latency_ms=torch.tensor(filter_lat_list, dtype=torch.float64),
            metadata=meta_list,
        )

        results: list[MetricResult] = [r for acc in self._accumulators for r in acc.compute()]
        return batch, results

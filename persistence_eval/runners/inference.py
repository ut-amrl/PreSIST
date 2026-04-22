from __future__ import annotations

import time
from collections import defaultdict
from pathlib import Path

import numpy as np

from persistence_eval.core.domain import (
    AnchorObservation,
    InferenceArtifact,
    PersistenceFilterInput,
)
from persistence_eval.core.protocols import PersistenceParameterInferencer


class InferenceRunner:
    """
    Runs the persistence parameter inferencer over a list of episodes.
    Groups episodes sharing the same anchor image to exploit infer_batch() when possible,
    reducing per-image preprocessing overhead for multi-object frames.
    """

    def run(
        self,
        episodes: list[PersistenceFilterInput],
        inferencer: PersistenceParameterInferencer,
    ) -> list[InferenceArtifact]:
        # Group by (anchor_time, track_id) — same image implies same track+time here.
        # For real datasets that provide pre-loaded images, group by image identity.
        # Fallback: call infer() per episode when image data is unavailable for batching.
        artifacts: list[InferenceArtifact] = []

        # Group episodes whose anchor has an actual image array (not None) by array id,
        # so multi-mask batching can be applied when the same image appears more than once.
        by_image: dict[int, list[PersistenceFilterInput]] = defaultdict(list)
        no_image: list[PersistenceFilterInput] = []

        for ep in episodes:
            if ep.anchor.image is not None:
                by_image[id(ep.anchor.image)].append(ep)
            else:
                no_image.append(ep)

        for group in by_image.values():
            if len(group) == 1:
                artifacts.append(self._infer_single(group[0], inferencer))
            else:
                artifacts.extend(self._infer_batch(group, inferencer))

        for ep in no_image:
            artifacts.append(self._infer_single(ep, inferencer))

        return artifacts

    @staticmethod
    def _infer_single(
        episode: PersistenceFilterInput,
        inferencer: PersistenceParameterInferencer,
    ) -> InferenceArtifact:
        t0 = time.perf_counter()
        estimate = inferencer.infer(episode.anchor)
        latency_ms = (time.perf_counter() - t0) * 1000.0
        return InferenceArtifact(
            episode_id=episode.episode_id,
            estimate=estimate,
            inference_latency_ms=latency_ms,
        )

    @staticmethod
    def _infer_batch(
        group: list[PersistenceFilterInput],
        inferencer: PersistenceParameterInferencer,
    ) -> list[InferenceArtifact]:
        anchor0 = group[0].anchor
        masks = [ep.anchor.mask for ep in group]
        track_ids = [ep.anchor.track_id for ep in group]

        t0 = time.perf_counter()
        estimates = inferencer.infer_batch(
            image=anchor0.image,
            masks=masks,
            anchor_time=anchor0.anchor_time,
            track_ids=track_ids,
        )
        total_ms = (time.perf_counter() - t0) * 1000.0
        per_ms = total_ms / len(group)

        return [
            InferenceArtifact(
                episode_id=ep.episode_id,
                estimate=est,
                inference_latency_ms=per_ms,
            )
            for ep, est in zip(group, estimates)
        ]

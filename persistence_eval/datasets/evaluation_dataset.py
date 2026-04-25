from __future__ import annotations

from typing import Iterable, Iterator

from torch.utils.data import IterableDataset

from persistence_eval.core.domain import EvaluationContext, InferenceArtifact
from persistence_eval.core.protocols import EpisodeProvider


class EvaluationDataset(IterableDataset):
    """
    Joins InferenceArtifacts with EvaluationContexts via an EpisodeProvider.
    Each yielded item is (artifact, context). One artifact may yield multiple contexts
    if the provider returns multiple prediction horizon sets per observation.
    Artifacts with no matching context (unknown track_id or anchor_time) are skipped.
    """

    def __init__(
        self,
        artifacts: Iterable[InferenceArtifact],
        provider: EpisodeProvider,
    ) -> None:
        self._artifacts = artifacts
        self._provider = provider

    def __iter__(self) -> Iterator[tuple[InferenceArtifact, EvaluationContext]]:
        for artifact in self._artifacts:
            for ctx in self._provider.get_contexts(artifact.track_id, artifact.anchor_time):
                yield artifact, ctx

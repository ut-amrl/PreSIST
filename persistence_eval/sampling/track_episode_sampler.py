from __future__ import annotations

import random
from abc import ABC, abstractmethod

from persistence_eval.core.domain import (
    AnchorObservation,
    BinaryObservationEvent,
    PersistenceFilterInput,
    TrackTemporalBounds,
)
from persistence_eval.core.protocols import TrackDataset
from persistence_eval.sampling.specs import EpisodeSamplerSpec


class AbsEpisodeSampler(ABC):
    @abstractmethod
    def sample(
        self, dataset: TrackDataset
    ) -> list[tuple[PersistenceFilterInput, TrackTemporalBounds]]:
        """Return (filter_input, ground_truth) pairs sampled from the dataset."""
        ...


class SimpleTrackEpisodeSampler(AbsEpisodeSampler):
    def __init__(self, spec: EpisodeSamplerSpec) -> None:
        self._spec = spec

    def sample(
        self, dataset: TrackDataset
    ) -> list[tuple[PersistenceFilterInput, TrackTemporalBounds]]:
        results: list[tuple[PersistenceFilterInput, TrackTemporalBounds]] = []
        for track_id, observations, bounds in dataset.iter_tracks():
            if self._spec.require_disappearance_observed and bounds.first_missing_time is None:
                continue

            positive_obs = [o for o in observations if o.is_positive]
            anchors = self._select_anchors(positive_obs, track_id, dataset.name)
            if not anchors:
                continue

            for anchor_idx, anchor_obs in enumerate(anchors):
                episode_id = f"{dataset.name}:{track_id}:{anchor_idx}"
                prior = self._build_prior(observations, anchor_obs.anchor_time)
                results.append((
                    PersistenceFilterInput(
                        episode_id=episode_id,
                        anchor=anchor_obs,
                        prior_observations=prior,
                        query_times=list(self._spec.query_horizons),
                        metadata={"dataset_name": dataset.name, "track_id": track_id},
                    ),
                    bounds,
                ))
        return results

    def _select_anchors(
        self, positive_obs: list[BinaryObservationEvent], track_id: str, dataset_name: str
    ) -> list[AnchorObservation]:
        if not positive_obs:
            return []
        strategy = self._spec.anchor_strategy
        if strategy == "first_positive":
            selected = [positive_obs[0]]
        elif strategy == "random_positive":
            selected = [random.choice(positive_obs)]
        else:  # all_positive
            selected = positive_obs

        return [
            AnchorObservation(
                image=None,  # type: ignore[arg-type]  # filled by dataset loader
                mask=None,   # type: ignore[arg-type]
                anchor_time=obs.timestamp,
                track_id=track_id,
            )
            for obs in selected
        ]

    def _build_prior(
        self,
        observations: list[BinaryObservationEvent],
        anchor_time: float,
    ) -> list[BinaryObservationEvent]:
        if not self._spec.include_prior_history:
            return []
        prior = [o for o in observations if o.timestamp < anchor_time]
        if self._spec.max_prior_observations is not None:
            prior = prior[-self._spec.max_prior_observations:]
        return prior

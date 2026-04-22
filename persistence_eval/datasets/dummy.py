from __future__ import annotations

from typing import Iterable

import numpy as np

from persistence_eval.core.domain import (
    AnchorObservation,
    BinaryObservationEvent,
    TrackTemporalBounds,
)
from persistence_eval.datasets.backing import AbsBackingDataset
from persistence_eval.datasets.image_mask_view import ImageMaskView
from persistence_eval.datasets.track_view import TrackView

_TRACK_ID = "dummy_track_0"
_ANCHOR_TIME = 0.0
_IMAGE_H, _IMAGE_W = 64, 64


class DummyBackingDataset(AbsBackingDataset):
    """In-memory dataset with one synthetic track. Requires no I/O; used for smoke tests."""

    @property
    def name(self) -> str:
        return "dummy"


class DummyImageMaskView(ImageMaskView):
    def iter_anchor_observations(self) -> Iterable[AnchorObservation]:
        image = np.zeros((_IMAGE_H, _IMAGE_W, 3), dtype=np.uint8)
        mask = np.ones((_IMAGE_H, _IMAGE_W), dtype=bool)
        yield AnchorObservation(image=image, mask=mask, anchor_time=_ANCHOR_TIME, track_id=_TRACK_ID)


class DummyTrackView(TrackView):
    def iter_tracks(
        self,
    ) -> Iterable[tuple[str, list[BinaryObservationEvent], TrackTemporalBounds]]:
        observations = [
            BinaryObservationEvent(timestamp=0.0, is_positive=True),
            BinaryObservationEvent(timestamp=1.0, is_positive=True),
            BinaryObservationEvent(timestamp=2.0, is_positive=False),
        ]
        bounds = TrackTemporalBounds(
            first_observed_time=0.0,
            last_observed_time=1.0,
            first_absent_before_observation=None,
            first_missing_time=2.0,
        )
        yield _TRACK_ID, observations, bounds

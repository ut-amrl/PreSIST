"""Inferencer that looks up per-location persistence params by track_id.

Resolves each track in the input group through a `track_id -> location_key`
sidecar (built once by build_track_location_index.py from the test manifest)
into a `LocationPersistenceParamLookup`. Falls back to the lookup's
`__global__` entry when a track's location key is unknown.
"""

from __future__ import annotations

import json
from typing import Optional

from persistence_eval.core.domain import ImageMaskGroup, PersistenceParameterEstimate
from persistence_eval.methods.inferencers.base import AbsPersistenceParameterInferencer
from persistence_eval.methods.inferencers.location_keyed.location_persistence_param_lookup import (
    LocationPersistenceParamLookup,
)


class LocationKeyedLookupInferencer(AbsPersistenceParameterInferencer):
    def __init__(
        self,
        location_lookup_file: str,
        track_id_to_location_file: str,
        name: str | None = None,
    ) -> None:
        self._lookup = LocationPersistenceParamLookup(params_file=location_lookup_file)
        with open(track_id_to_location_file, "r") as f:
            self._track_id_to_location: dict[str, str] = json.load(f)

    def infer(
        self, group: ImageMaskGroup, *, one_off: bool = False,
    ) -> tuple[list[PersistenceParameterEstimate], Optional[float]]:
        estimates: list[PersistenceParameterEstimate] = []
        for track_id in group.track_ids:
            location_key = self._track_id_to_location.get(track_id)
            if location_key is None:
                # No mapping for this track: fall back via a synthetic key.
                shared, used_fallback = self._lookup.get_params_with_fallback(
                    f"__missing_track__:{track_id}"
                )
                location_key = "<unknown>"
            else:
                shared, used_fallback = self._lookup.get_params_with_fallback(location_key)
            estimates.append(
                PersistenceParameterEstimate(
                    params=shared.params,
                    confidence=shared.confidence,
                    metadata={
                        **shared.metadata,
                        "location_key": location_key,
                        "fallback": "global" if used_fallback else "hit",
                    },
                )
            )
        return estimates, None

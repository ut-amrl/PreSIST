"""String-keyed analog of SemanticClassPersistenceParamLookup.

Persistence parameters keyed by a free-form string (typically a `space_id` or a
discretized image-cell key produced by location_keying.location_key_for_track).
The reserved key "__global__" holds the pooled fallback used when a queried
key is missing from the lookup.
"""

import json

from persistence_eval.core.domain import PersistenceParameterEstimate


GLOBAL_FALLBACK_KEY = "__global__"


class LocationPersistenceParamLookup:
    """Maps location keys (strings) to PersistenceParameterEstimate objects."""

    def __init__(self, params_file: str | None = None):
        self.key_to_params: dict[str, PersistenceParameterEstimate] = {}
        if params_file is not None:
            self.load_params_from_file(params_file)

    def get_params_for_key(self, location_key: str) -> PersistenceParameterEstimate:
        if location_key in self.key_to_params:
            return self.key_to_params[location_key]
        raise KeyError(f"Location key {location_key!r} not found in parameter lookup.")

    def get_params_with_fallback(
        self, location_key: str
    ) -> tuple[PersistenceParameterEstimate, bool]:
        """Return (estimate, used_fallback). Falls back to GLOBAL_FALLBACK_KEY."""
        if location_key in self.key_to_params:
            return self.key_to_params[location_key], False
        if GLOBAL_FALLBACK_KEY in self.key_to_params:
            return self.key_to_params[GLOBAL_FALLBACK_KEY], True
        raise KeyError(
            f"Location key {location_key!r} missing and no {GLOBAL_FALLBACK_KEY!r} entry."
        )

    def set_params_for_key(
        self, location_key: str, params: PersistenceParameterEstimate
    ) -> None:
        self.key_to_params[location_key] = params

    def load_params_from_file(self, file_path: str) -> None:
        with open(file_path, "r") as f:
            data = json.load(f)
        self.key_to_params = {
            location_key: PersistenceParameterEstimate(
                params=entry["params"],
                confidence=entry.get("confidence"),
                metadata=entry.get("metadata", {}),
            )
            for location_key, entry in data.items()
        }

    def save_params_to_file(self, file_path: str) -> None:
        data = {
            location_key: {
                "params": estimate.params,
                "confidence": estimate.confidence,
                "metadata": estimate.metadata,
            }
            for location_key, estimate in self.key_to_params.items()
        }
        with open(file_path, "w") as f:
            json.dump(data, f, indent=2, sort_keys=True)

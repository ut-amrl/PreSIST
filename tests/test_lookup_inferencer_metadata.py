"""Tests that the location-keyed inferencer surfaces hit/fallback metadata."""

import json
import tempfile
from pathlib import Path

import torch

from persistence_eval.core.domain import ImageMaskGroup, PersistenceParameterEstimate
from persistence_eval.methods.inferencers.location_keyed.location_persistence_param_lookup import (
    GLOBAL_FALLBACK_KEY,
    LocationPersistenceParamLookup,
)
from persistence_eval.methods.inferencers.location_keyed_lookup_inferencer import (
    LocationKeyedLookupInferencer,
)


def _write_lookup_files(tmp: Path) -> tuple[str, str]:
    lookup = LocationPersistenceParamLookup()
    lookup.set_params_for_key(
        "spot_A", PersistenceParameterEstimate(params={"lambda_": 0.5, "mean": 2.0})
    )
    lookup.set_params_for_key(
        GLOBAL_FALLBACK_KEY,
        PersistenceParameterEstimate(params={"lambda_": 0.1, "mean": 10.0}),
    )
    params_path = tmp / "params.json"
    lookup.save_params_to_file(str(params_path))

    sidecar_path = tmp / "track_id_to_location.json"
    sidecar_path.write_text(
        json.dumps({"track_alpha": "spot_A", "track_beta": "spot_unknown"})
    )
    return str(params_path), str(sidecar_path)


def test_hit_and_fallback_metadata():
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        params_path, sidecar_path = _write_lookup_files(tmp)
        inferencer = LocationKeyedLookupInferencer(
            location_lookup_file=params_path,
            track_id_to_location_file=sidecar_path,
        )
        group = ImageMaskGroup(
            image=torch.zeros(3, 4, 4, dtype=torch.uint8),
            masks=torch.zeros(2, 4, 4, dtype=torch.bool),
            track_ids=["track_alpha", "track_beta"],
            anchor_time=0.0,
        )
        estimates = inferencer.infer(group)
        assert len(estimates) == 2
        assert estimates[0].metadata["fallback"] == "hit"
        assert estimates[0].metadata["location_key"] == "spot_A"
        assert estimates[0].params["lambda_"] == 0.5
        assert estimates[1].metadata["fallback"] == "global"
        assert estimates[1].params["lambda_"] == 0.1

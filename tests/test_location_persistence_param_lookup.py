"""Tests for LocationPersistenceParamLookup save/load roundtrip and fallback."""

import json
import tempfile

import pytest

from persistence_eval.core.domain import PersistenceParameterEstimate
from persistence_eval.methods.inferencers.location_keyed.location_persistence_param_lookup import (
    GLOBAL_FALLBACK_KEY,
    LocationPersistenceParamLookup,
)


def test_roundtrip_scalar_and_list_params():
    lookup = LocationPersistenceParamLookup()
    lookup.set_params_for_key(
        "persistent_-89",
        PersistenceParameterEstimate(params={"lambda_": 0.5}, confidence=0.9),
    )
    lookup.set_params_for_key(
        "persistent_-90",
        PersistenceParameterEstimate(
            params={"weights": [0.3, 0.7], "lambdas": [0.1, 0.9]},
            metadata={"model_type": "mixture_exponential"},
        ),
    )

    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        path = f.name
    lookup.save_params_to_file(path)

    reloaded = LocationPersistenceParamLookup(params_file=path)
    a = reloaded.get_params_for_key("persistent_-89")
    assert a.params == {"lambda_": 0.5}
    assert a.confidence == 0.9
    b = reloaded.get_params_for_key("persistent_-90")
    assert b.params == {"weights": [0.3, 0.7], "lambdas": [0.1, 0.9]}
    assert b.metadata["model_type"] == "mixture_exponential"


def test_global_fallback():
    lookup = LocationPersistenceParamLookup()
    lookup.set_params_for_key(
        GLOBAL_FALLBACK_KEY,
        PersistenceParameterEstimate(params={"lambda_": 0.1}),
    )
    estimate, used_fallback = lookup.get_params_with_fallback("missing_key")
    assert used_fallback is True
    assert estimate.params == {"lambda_": 0.1}


def test_hit_does_not_use_fallback():
    lookup = LocationPersistenceParamLookup()
    lookup.set_params_for_key(
        "k",
        PersistenceParameterEstimate(params={"lambda_": 0.5}),
    )
    lookup.set_params_for_key(
        GLOBAL_FALLBACK_KEY,
        PersistenceParameterEstimate(params={"lambda_": 0.1}),
    )
    estimate, used_fallback = lookup.get_params_with_fallback("k")
    assert used_fallback is False
    assert estimate.params == {"lambda_": 0.5}


def test_fallback_raises_when_no_global():
    lookup = LocationPersistenceParamLookup()
    lookup.set_params_for_key("k", PersistenceParameterEstimate(params={"lambda_": 0.5}))
    with pytest.raises(KeyError):
        lookup.get_params_with_fallback("missing")


def test_validation_rejects_non_numeric():
    with pytest.raises(TypeError):
        PersistenceParameterEstimate(params={"bad": "string"})  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        PersistenceParameterEstimate(params={"bad": [1, "two"]})  # type: ignore[list-item]

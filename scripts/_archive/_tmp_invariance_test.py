"""Anchor-time invariance check at the filter level for both Rosen and Mixture.

If our filters are correct, then for the same survival function + observations +
query horizons, shifting everything in time by the same offset must give
identical predictions. This rules out timestamp/unit bugs that would otherwise
silently corrupt eval numbers.
"""
import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")
import torch
import json
import math

from persistence_eval.core.domain import BinaryObservationEvent, PersistenceParameterEstimate
from persistence_eval.methods.filters.rosen_filter import RosenPersistenceFilter
from persistence_eval.methods.filters.mixture_persistence_filter import MixturePersistenceFilter
from persistence_eval.methods.survival.mixture import MixtureExponentialGenerator
from persistence_eval.methods.survival.exponential import ExponentialSurvivalFunction

# Helper: build a synthetic observation stream + query set, then predict at
# anchor=0 and anchor=1.35e9. Must give the same result.
def _obs(times, alives):
    return [BinaryObservationEvent(timestamp=t, is_positive=a) for t, a in zip(times, alives)]

def check(filter_name, filter_inst, survival_fn, queries, base_obs, label):
    """Compare predictions with anchor_time at 0.0 vs 1.35e9."""
    pred_at_zero = filter_inst.predict(
        survival_fn, anchor_time=0.0,
        query_times=queries, observations=_obs(*base_obs)
    ).numpy()
    OFFSET = 1.35e9
    shifted_obs = ([t + OFFSET for t in base_obs[0]], base_obs[1])
    pred_at_offset = filter_inst.predict(
        survival_fn, anchor_time=OFFSET,
        query_times=queries, observations=_obs(*shifted_obs)
    ).numpy()
    diff = abs(pred_at_zero - pred_at_offset).max()
    status = "OK" if diff < 5e-4 else "FAIL"
    print(f"  {filter_name:30s} {label:30s}  max|Δ| = {diff:.6e}  [{status}]")
    return diff

# ----- Test on a single-exponential survival (uses Rosen) -----
rate = torch.tensor(2e-5, dtype=torch.float64)  # mean ~14 hours
single_exp = ExponentialSurvivalFunction(rate=rate)
queries = torch.tensor([60.0, 600.0, 3600.0, 86400.0, 604800.0], dtype=torch.float64)
rosen = RosenPersistenceFilter(false_detection_prob=0.05, missed_detection_prob=0.1)

print("=== Rosen filter, single-exp survival ===")
check("Rosen", rosen, single_exp, queries, ([], []),            "no obs")
check("Rosen", rosen, single_exp, queries, ([300., 600., 1500.], [True, True, False]), "3 obs (2+ 1-)")
check("Rosen", rosen, single_exp, queries, ([60., 120., 180., 240., 300.], [True]*5), "5 positives")

# ----- Test on a K=2 mixture-exponential survival (uses MixturePersistenceFilter) -----
mix_filter = MixturePersistenceFilter(false_detection_prob=0.05, missed_detection_prob=0.1)
mix_surv = MixtureExponentialGenerator().build(
    PersistenceParameterEstimate(params={"weights": [0.4, 0.6], "lambdas": [2e-5, 5e-3]})
)

print()
print("=== MixturePersistenceFilter, K=2 mixture-exp survival ===")
check("MixturePersistenceFilter", mix_filter, mix_surv, queries, ([], []),            "no obs")
check("MixturePersistenceFilter", mix_filter, mix_surv, queries, ([300., 600., 1500.], [True, True, False]), "3 obs (2+ 1-)")
check("MixturePersistenceFilter", mix_filter, mix_surv, queries, ([60., 120., 180., 240., 300.], [True]*5), "5 positives")

# ----- Also: check Rosen with the MIXTURE survival (used by perpetua single-* configs) -----
print()
print("=== Rosen filter, mixture-exp survival (the single-bucket-perpetua pathway) ===")
check("Rosen", rosen, mix_surv, queries, ([], []),            "no obs")
check("Rosen", rosen, mix_surv, queries, ([300., 600., 1500.], [True, True, False]), "3 obs (2+ 1-)")
check("Rosen", rosen, mix_surv, queries, ([60., 120., 180., 240., 300.], [True]*5), "5 positives")

# ----- And: do the inputs reach the filter as expected? -----
print()
print("=== Edge: PKLot-scale timestamps (anchor = 1.35e9, dt up to 1 week) ===")
check("Rosen",                    rosen,      mix_surv, queries, ([0., 300., 600., 900., 1200., 1500.], [True, True, True, True, False, False]), "PKLot 5-min cadence, mid-window disappearance")
check("MixturePersistenceFilter", mix_filter, mix_surv, queries, ([0., 300., 600., 900., 1200., 1500.], [True, True, True, True, False, False]), "PKLot 5-min cadence, mid-window disappearance")

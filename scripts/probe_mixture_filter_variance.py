"""One-off probe: feed different fabricated observation histories into the
mixture filter and confirm τ_k and predicted survival diverge.

Uses the trained days=35 mixture_exp_class params (3 components).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from persistence_eval.core.domain import BinaryObservationEvent, PersistenceParameterEstimate
from persistence_eval.methods.filters.mixture_persistence_filter import MixturePersistenceFilter
from persistence_eval.methods.survival.mixture import MixtureExponentialGenerator

_CANDIDATES = [
    Path("/scratch/aaadkins/data/semantic-persistence-filter/output/perpetua/"
         "perpetua_mixture_exp_class_pucpr_split_50_20_30_daycount_fullK_mins5_p0p01/"
         "days35_seed0/persistence_params.json"),
    Path("/workspace/data/output/perpetua/"
         "perpetua_mixture_exp_class_pucpr_split_50_20_30_daycount_fullK_mins5_p0p01/"
         "days35_seed0/persistence_params.json"),
]
PARAMS_JSON = next(p for p in _CANDIDATES if p.exists())


def main() -> None:
    raw = json.loads(PARAMS_JSON.read_text())["0"]
    estimate = PersistenceParameterEstimate(
        confidence=float(raw["confidence"]),
        params=raw["params"],
        metadata=raw["metadata"],
    )
    surv = MixtureExponentialGenerator().build(estimate)
    print(f"lambdas = {surv.component_param('lambda_').tolist()}")
    print(f"pi (prior) = {surv.pi.tolist()}\n")

    filt = MixturePersistenceFilter(false_detection_prob=0.01, missed_detection_prob=0.01)
    anchor = 0.0
    # Query out to ~10h in coarse steps.
    query_times = torch.tensor(
        [0.0, 60.0, 600.0, 3600.0, 7200.0, 14400.0, 21600.0, 36000.0],
        dtype=torch.float64,
    )

    # Three fabricated observation histories representing different "true" survival.
    one_hour = 3600.0
    histories = {
        "no_obs": [],
        "anchor_only_positive": [
            BinaryObservationEvent(timestamp=0.0, is_positive=True),
        ],
        "short_lived (positive then negative quickly)": [
            BinaryObservationEvent(timestamp=0.0, is_positive=True),
            BinaryObservationEvent(timestamp=60.0, is_positive=True),
            BinaryObservationEvent(timestamp=600.0, is_positive=False),
            BinaryObservationEvent(timestamp=1200.0, is_positive=False),
        ],
        "long_lived (many positives over 6h)": [
            BinaryObservationEvent(timestamp=t, is_positive=True)
            for t in np.linspace(0.0, 6 * one_hour, 12)
        ],
        "very_long_lived (positives over 24h)": [
            BinaryObservationEvent(timestamp=t, is_positive=True)
            for t in np.linspace(0.0, 24 * one_hour, 25)
        ],
    }

    # Use the underlying jax filter directly so we can also inspect τ_k.
    from persistence_eval.methods.filters.perpetua_filter_adapter import _jax_modules
    mods = _jax_modules()
    jnp = mods["jnp"]
    mix_filter = mods["mixture_filter"]

    log_s, params_j, pi_j = surv.build_jax_state()

    print(f"{'history':50s} | τ_k posterior                                 "
          f"| P(alive @ 1h, 4h, 10h)")
    print("-" * 130)
    for name, obs in histories.items():
        state = mix_filter.init(log_s, params_j, pi_j, initialization_time=anchor)
        for o in sorted(obs, key=lambda o: o.timestamp):
            state = mix_filter.update(
                state, bool(o.is_positive), float(o.timestamp), 0.01, 0.01
            )
        tau = np.asarray(jnp.exp(state.log_tau))
        tau = tau / tau.sum()
        preds = filt.predict(
            surv, anchor_time=anchor, query_times=query_times, observations=obs
        ).numpy()
        # Indices: 3=1h, 5=4h, 7=10h
        p1, p4, p10 = preds[3], preds[5], preds[7]
        tau_str = "[" + ", ".join(f"{v:.3f}" for v in tau) + "]"
        print(f"{name:50s} | {tau_str:46s} | {p1:.4f}, {p4:.4f}, {p10:.4f}")


if __name__ == "__main__":
    main()

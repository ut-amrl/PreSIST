"""Tests for ScaledHorizonEpisodeProvider.

Validates the cutoff rule h_eff = min(X, dt_q/2), the prefix-grouping algorithm,
and the X-bounded observation collection.
"""

from __future__ import annotations

from persistence_eval.core.domain import (
    BinaryObservationEvent,
    TrackTemporalBounds,
)
from persistence_eval.sampling.scaled_horizon_episode_provider import (
    ScaledHorizonEpisodeProvider,
)


class _StubTrackDataset:
    name = "stub"

    def __init__(self, tracks):
        # tracks: list[(track_id, observations, bounds)]
        self._tracks = tracks

    def iter_tracks(self):
        yield from self._tracks


def _bounds():
    return TrackTemporalBounds(
        first_observed_time=0.0,
        last_observed_time=1000.0,
        first_absent_before_observation=None,
        first_missing_time=None,
    )


def _obs(t, pos=True):
    return BinaryObservationEvent(timestamp=t, is_positive=pos)


def test_grouping_by_prefix_with_three_distinct_groups():
    """Obs at [5, 10, 15] with X=20 (keeps all obs); queries chosen so cutoffs
    straddle different obs ⇒ three groups."""
    obs = [_obs(5.0), _obs(10.0), _obs(15.0)]
    track = ("t0", obs, _bounds())
    ds = _StubTrackDataset([track])

    # X=20. For dt_q ≤ 40, gap_floor = dt_q/2. Cutoffs:
    #   dt_q=12 ⇒ cutoff=6  ⇒ visible {5}        ⇒ k=1
    #   dt_q=14 ⇒ cutoff=7  ⇒ visible {5}        ⇒ k=1
    #   dt_q=24 ⇒ cutoff=12 ⇒ visible {5,10}     ⇒ k=2
    #   dt_q=40 ⇒ cutoff=20 ⇒ visible {5,10,15}  ⇒ k=3
    provider = ScaledHorizonEpisodeProvider(
        ds, query_horizons=[12.0, 14.0, 24.0, 40.0], recency_cap=20.0
    )
    contexts = list(provider.get_contexts("t0", anchor_time=0.0))

    assert len(contexts) == 3
    by_dt = {tuple(c.query_times): c for c in contexts}
    assert (12.0, 14.0) in by_dt
    assert (24.0,) in by_dt
    assert (40.0,) in by_dt

    g1 = by_dt[(12.0, 14.0)]
    assert [o.timestamp for o in g1.observations] == [5.0]
    g2 = by_dt[(24.0,)]
    assert [o.timestamp for o in g2.observations] == [5.0, 10.0]
    g3 = by_dt[(40.0,)]
    assert [o.timestamp for o in g3.observations] == [5.0, 10.0, 15.0]


def test_query_count_preserved_across_groups():
    obs = [_obs(t) for t in (1.0, 4.0, 7.0, 11.0, 19.0)]
    track = ("t0", obs, _bounds())
    ds = _StubTrackDataset([track])
    queries = [2.0, 4.0, 6.0, 8.0, 10.0, 14.0, 22.0, 30.0]
    provider = ScaledHorizonEpisodeProvider(
        ds, query_horizons=queries, recency_cap=10.0
    )
    contexts = list(provider.get_contexts("t0", anchor_time=0.0))
    total = sum(len(c.query_times) for c in contexts)
    assert total == len(queries)


def test_obs_bounded_to_recency_cap():
    """Observations strictly past anchor + recency_cap must be dropped from
    every emitted context."""
    obs = [_obs(t) for t in (5.0, 50.0, 200.0, 500.0)]
    track = ("t0", obs, _bounds())
    ds = _StubTrackDataset([track])
    provider = ScaledHorizonEpisodeProvider(
        ds, query_horizons=[10.0, 100.0, 1000.0], recency_cap=100.0
    )
    contexts = list(provider.get_contexts("t0", anchor_time=0.0))
    for c in contexts:
        for o in c.observations:
            assert 0.0 <= o.timestamp <= 100.0


def test_h_eff_for_short_query_is_dt_over_two():
    """For dt_q < 2X, gap_floor = dt_q/2. With an obs sitting just inside the
    cutoff, actual h_eff approximates dt_q/2."""
    # X=10, dt_q=10 ⇒ gap_floor=5, cutoff=5. Place obs at 4.999 (just visible).
    obs = [_obs(4.999)]
    track = ("t0", obs, _bounds())
    ds = _StubTrackDataset([track])
    provider = ScaledHorizonEpisodeProvider(
        ds, query_horizons=[10.0], recency_cap=10.0
    )
    contexts = list(provider.get_contexts("t0", anchor_time=0.0))
    assert len(contexts) == 1
    [h_eff] = contexts[0].effective_horizons
    # actual h_eff = 10 - 4.999 = 5.001
    assert abs(h_eff - 5.001) < 1e-9


def test_h_eff_for_long_query_floors_at_dt_minus_X_when_obs_is_at_cap():
    """For dt_q > 2X with an obs at anchor + X, actual h_eff = dt_q - X."""
    obs = [_obs(10.0)]   # at anchor + X
    track = ("t0", obs, _bounds())
    ds = _StubTrackDataset([track])
    provider = ScaledHorizonEpisodeProvider(
        ds, query_horizons=[30.0], recency_cap=10.0
    )
    contexts = list(provider.get_contexts("t0", anchor_time=0.0))
    assert len(contexts) == 1
    [h_eff] = contexts[0].effective_horizons
    # cutoff = t_q - X = 30 - 10 = 20; obs at 10 < 20 ⇒ visible.
    # actual h_eff = 30 - 10 = 20.
    assert abs(h_eff - 20.0) < 1e-9


def test_h_eff_with_no_visible_obs_equals_dt_q():
    obs = [_obs(50.0)]
    track = ("t0", obs, _bounds())
    ds = _StubTrackDataset([track])
    # X=10, dt_q=4 ⇒ gap_floor = 2, cutoff = 2. obs at 50 not visible.
    provider = ScaledHorizonEpisodeProvider(
        ds, query_horizons=[4.0], recency_cap=10.0
    )
    contexts = list(provider.get_contexts("t0", anchor_time=0.0))
    [h_eff] = contexts[0].effective_horizons
    assert h_eff == 4.0


def test_unknown_track_yields_nothing():
    provider = ScaledHorizonEpisodeProvider(
        _StubTrackDataset([]), query_horizons=[1.0], recency_cap=1.0
    )
    assert list(provider.get_contexts("missing", anchor_time=0.0)) == []

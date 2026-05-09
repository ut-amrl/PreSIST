from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class EpisodeSamplerSpec:
    anchor_strategy: str = "all_positive"     # "all_positive" | "first_positive" | "random_positive"
    query_horizons: list[float] = field(default_factory=lambda: [1.0, 3.0, 5.0])
    require_disappearance_observed: bool = False

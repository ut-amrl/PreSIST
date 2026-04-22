from __future__ import annotations

from dataclasses import dataclass

from persistence_eval.core.protocols import PersistenceFilter, SurvivalFunctionGenerator


@dataclass
class EvaluationMethod:
    """
    Sufficient to run evaluation from pre-cached inference artifacts.
    Does not require an inferencer — use when separating inference from eval.
    """
    name: str
    survival_generator: SurvivalFunctionGenerator
    persistence_filter: PersistenceFilter

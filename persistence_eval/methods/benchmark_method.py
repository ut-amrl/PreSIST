from __future__ import annotations

from dataclasses import dataclass

from persistence_eval.core.protocols import PersistenceParameterInferencer
from persistence_eval.methods.evaluation_method import EvaluationMethod


@dataclass
class BenchmarkMethod:
    """
    Full method for end-to-end benchmarking including inference.
    Composes EvaluationMethod with an inferencer for convenience.
    """
    evaluation_method: EvaluationMethod
    inferencer: PersistenceParameterInferencer

    @property
    def name(self) -> str:
        return self.evaluation_method.name

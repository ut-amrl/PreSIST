from __future__ import annotations

import json
from abc import ABC, abstractmethod
from pathlib import Path

from persistence_eval.core.domain import InferenceArtifact, PersistenceParameterEstimate

_ARTIFACTS_FILENAME = "inference_artifacts.json"


class ArtifactStore(ABC):
    @abstractmethod
    def save(self, artifacts: list[InferenceArtifact], run_dir: Path) -> None: ...

    @abstractmethod
    def load(self, run_dir: Path) -> list[InferenceArtifact]: ...


class JsonArtifactStore(ArtifactStore):
    """
    Saves/loads InferenceArtifacts as a JSON file.
    Mirrors the pattern of spf_io/persistence_io.py.
    """

    def save(self, artifacts: list[InferenceArtifact], run_dir: Path) -> None:
        path = run_dir / _ARTIFACTS_FILENAME
        data = [self._serialize(a) for a in artifacts]
        path.write_text(json.dumps(data, indent=2))

    def load(self, run_dir: Path) -> list[InferenceArtifact]:
        path = run_dir / _ARTIFACTS_FILENAME
        data = json.loads(path.read_text())
        return [self._deserialize(d) for d in data]

    @staticmethod
    def _serialize(a: InferenceArtifact) -> dict:
        return {
            "episode_id": a.episode_id,
            "inference_latency_ms": a.inference_latency_ms,
            "estimate": {
                "params": a.estimate.params,
                "confidence": a.estimate.confidence,
                "metadata": a.estimate.metadata,
            },
        }

    @staticmethod
    def _deserialize(d: dict) -> InferenceArtifact:
        est = d["estimate"]
        return InferenceArtifact(
            episode_id=d["episode_id"],
            inference_latency_ms=float(d["inference_latency_ms"]),
            estimate=PersistenceParameterEstimate(
                params={k: float(v) for k, v in est["params"].items()},
                confidence=est.get("confidence"),
                metadata=est.get("metadata", {}),
            ),
        )

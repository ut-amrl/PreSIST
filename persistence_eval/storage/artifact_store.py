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
    def append(self, artifacts: list[InferenceArtifact], run_dir: Path) -> None: ...

    @abstractmethod
    def load(self, run_dir: Path) -> list[InferenceArtifact]: ...


class JsonArtifactStore(ArtifactStore):
    """Saves/loads InferenceArtifacts as a JSON file keyed by (track_id, anchor_time)."""

    def save(self, artifacts: list[InferenceArtifact], run_dir: Path) -> None:
        path = run_dir / _ARTIFACTS_FILENAME
        data = [self._serialize(a) for a in artifacts]
        path.write_text(json.dumps(data, indent=2))

    def append(self, artifacts: list[InferenceArtifact], run_dir: Path) -> None:
        path = run_dir / _ARTIFACTS_FILENAME
        run_dir.mkdir(parents=True, exist_ok=True)
        existing: list[dict] = json.loads(path.read_text()) if path.exists() else []
        existing.extend(self._serialize(a) for a in artifacts)
        path.write_text(json.dumps(existing, indent=2))

    def load(self, run_dir: Path) -> list[InferenceArtifact]:
        path = run_dir / _ARTIFACTS_FILENAME
        data = json.loads(path.read_text())
        return [self._deserialize(d) for d in data]

    @staticmethod
    def _serialize(a: InferenceArtifact) -> dict:
        return {
            "track_id": a.track_id,
            "anchor_time": a.anchor_time,
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
            track_id=d["track_id"],
            anchor_time=float(d["anchor_time"]),
            inference_latency_ms=float(d["inference_latency_ms"]),
            estimate=PersistenceParameterEstimate(
                params={k: float(v) for k, v in est["params"].items()},
                confidence=est.get("confidence"),
                metadata=est.get("metadata", {}),
            ),
        )

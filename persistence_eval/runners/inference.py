from __future__ import annotations

import time
from pathlib import Path

from torch.utils.data import DataLoader
from tqdm import tqdm

from persistence_eval.core.domain import InferenceArtifact
from persistence_eval.core.protocols import ArtifactStore, PersistenceParameterInferencer
from persistence_eval.datasets.inference_dataset import InferenceDataset


class InferenceRunner:
    """
    Iterates an InferenceDataset via a DataLoader, calls the inferencer once per
    ImageMaskGroup, and checkpoints artifacts to disk every save_every_n groups.
    Returns the full list of artifacts after all groups are processed.
    """

    def __init__(
        self,
        inferencer: PersistenceParameterInferencer,
        artifact_store: ArtifactStore,
        save_every_n: int = 100,
    ) -> None:
        self._inferencer = inferencer
        self._store = artifact_store
        self._save_every_n = save_every_n

    def run(
        self,
        dataset: InferenceDataset,
        run_dir: Path,
        *,
        num_workers: int = 0,
    ) -> list[InferenceArtifact]:
        run_dir.mkdir(parents=True, exist_ok=True)
        self._inferencer.set_run_dir(run_dir)
        loader = DataLoader(
            dataset,
            batch_size=None,
            collate_fn=lambda x: x,
            num_workers=num_workers,
        )

        buffer: list[InferenceArtifact] = []
        all_artifacts: list[InferenceArtifact] = []

        for group in tqdm(loader, desc="Inference", unit="group"):
            t0 = time.perf_counter()
            estimates = self._inferencer.infer(group)
            latency_ms = (time.perf_counter() - t0) * 1000.0 / max(len(estimates), 1)

            for track_id, estimate in zip(group.track_ids, estimates):
                artifact = InferenceArtifact(
                    track_id=track_id,
                    anchor_time=group.anchor_time,
                    estimate=estimate,
                    inference_latency_ms=latency_ms,
                )
                buffer.append(artifact)
                all_artifacts.append(artifact)

            if len(buffer) >= self._save_every_n:
                self._store.append(buffer, run_dir)
                buffer.clear()

        if buffer:
            self._store.append(buffer, run_dir)

        return all_artifacts

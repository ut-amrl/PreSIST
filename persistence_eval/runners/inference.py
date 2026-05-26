from __future__ import annotations

import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from persistence_eval.core.domain import ImageMaskGroup, InferenceArtifact
from persistence_eval.core.protocols import ArtifactStore, PersistenceParameterInferencer
from persistence_eval.datasets.inference_dataset import InferenceDataset


def _build_warmup_group() -> ImageMaskGroup:
    H, W = 224, 224
    image = torch.zeros((3, H, W), dtype=torch.uint8)
    masks = torch.zeros((1, H, W), dtype=torch.bool)
    masks[0, H // 4 : 3 * H // 4, W // 4 : 3 * W // 4] = True
    return ImageMaskGroup(
        image=image,
        masks=masks,
        track_ids=["__warmup__"],
        anchor_time=0.0,
        image_path=None,
    )


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

        # Park the Gemini token-usage snapshot inside the per-run output dir.
        # No-op when Gemini isn't in use — the singleton just stores the path
        # and never writes anything until an engine records usage.
        from dataset_generation.llm_persistence_querying.gemini import _shared as _gemini_shared
        _gemini_shared.set_usage_log_file(run_dir / "gemini_token_usage.json")
        loader = DataLoader(
            dataset,
            batch_size=None,
            collate_fn=lambda x: x,
            num_workers=num_workers,
        )

        buffer: list[InferenceArtifact] = []
        all_artifacts: list[InferenceArtifact] = []

        # Exercise the inferencer on a synthetic group so first-call costs
        # (CUDA kernel JIT, lazy weight loads, TLS handshakes, etc.) land
        # outside the timed loop. one_off=True suppresses cache writes and
        # timer recording for this call, so it neither persists results nor
        # contaminates the timing aggregates.
        #
        # Inferencers opt out via ``wants_warmup() -> bool`` (default True on
        # ``AbsPersistenceParameterInferencer``). HTTP-only backends (e.g.
        # all-Gemini pipelines) override to False so we don't burn a real
        # billed API call to amortize a ~hundred-ms TCP handshake.
        if self._inferencer.wants_warmup():
            try:
                self._inferencer.infer(_build_warmup_group(), one_off=True)
            except Exception as e:
                print(f"[InferenceRunner] warmup infer failed (continuing): {e}", file=sys.stderr, flush=True)
        else:
            print("[InferenceRunner] skipping warmup (inferencer.wants_warmup() returned False)", file=sys.stderr, flush=True)

        for group in tqdm(loader, desc="Inference", unit="group", total=len(dataset)):
            t0 = time.perf_counter()
            estimates, reported_elapsed_ms = self._inferencer.infer(group)
            # Prefer the inferencer's reported elapsed (sum of original-
            # computation times across phases, including cached replays) so
            # cache hits don't collapse the latency. Fall back to wall-clock
            # for inferencers that don't track this.
            elapsed_ms = (
                reported_elapsed_ms
                if reported_elapsed_ms is not None
                else (time.perf_counter() - t0) * 1000.0
            )
            latency_ms = elapsed_ms / max(len(estimates), 1)

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

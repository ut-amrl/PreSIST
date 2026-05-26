from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

from persistence_eval.core.domain import ImageMaskGroup, PersistenceParameterEstimate


class AbsPersistenceParameterInferencer(ABC):
    """
    Estimates persistence parameters for all objects in an ImageMaskGroup.
    Returns (estimates, elapsed_ms): one PersistenceParameterEstimate per mask
    in the same order as group.track_ids, and the original-computation elapsed
    time in milliseconds (live perf_counter for cache misses, replayed cached
    elapsed for hits) so callers can report latency that reflects the underlying
    work rather than the cache lookup. Implementations without caches or
    internal timing can return None for the elapsed value — the runner falls
    back to wall-clock in that case. group.masks is (K, H, W) bool.
    """

    @abstractmethod
    def infer(
        self, group: ImageMaskGroup, *, one_off: bool = False,
    ) -> tuple[list[PersistenceParameterEstimate], Optional[float]]:
        """
        one_off: when True, callers are signalling this is a throwaway call
        (e.g. warmup). Implementations that have caches or timing instrumentation
        should skip cache reads/writes and skip recording into timing aggregates,
        so the call neither persists results nor contaminates measurements.
        Implementations without caches or timers can ignore the flag.
        """

    def set_run_dir(self, run_dir: Path) -> None:
        """Called by InferenceRunner before iteration begins. Override to opt in."""

    def wants_warmup(self) -> bool:
        """Whether the runner should call ``infer(..., one_off=True)`` on a
        synthetic group before the main loop, to absorb first-call costs
        (CUDA kernel JIT, lazy weight loads, TCP/TLS handshakes, etc.) outside
        the timed loop.

        Default: True — local model backends almost always benefit. HTTP-only
        backends (e.g. all-Gemini pipelines) should override to False so we
        don't burn a real billed API call to amortize a sub-second handshake.
        """
        return True

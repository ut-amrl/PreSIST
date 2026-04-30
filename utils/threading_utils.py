"""Threading utilities used across the dataset-generation pipeline.

The main export, ``ForwardSerializeLock``, is the lock + warn-once wrapper used
by local-PyTorch inference paths (Gemma / Qwen / InstructBLIP / SAM2 / SAM3).
Concurrent forward passes on a single PyTorch model from multiple host threads
are not safe; this lock makes the path correct-but-serialized and emits a
one-shot warning the first time it observes contention so the operator
notices the configuration is suboptimal rather than failing silently.

When the work is HTTP-bound (e.g., a vLLM-server client), no lock is needed —
httpx is documented thread-safe and a shared client is the recommended
pattern (https://github.com/encode/httpx/discussions/1633).
"""
from __future__ import annotations

import threading
import warnings
from contextlib import contextmanager
from typing import Iterator


class ForwardSerializeLock:
    """A lock that serializes a code region and warns once on first contention.

    Usage::

        # in __init__:
        self._forward_lock = ForwardSerializeLock(
            label=f"{type(self).__name__} forward",
            advice="use vLLM-server mode for real concurrency",
        )

        # wrapping the forward call:
        with self._forward_lock.acquire():
            out = self.model.generate(...)

    The warning is best-effort, not a correctness signal — ``Lock.locked()``
    is a non-atomic read-then-acquire, so the "did we observe contention?"
    decision can race. That's fine: the lock itself remains correct, and the
    warning fires the first time the race resolves in favor of "yes, contended."
    """

    def __init__(self, *, label: str, advice: str = ""):
        self._label = label
        self._advice = advice
        self._lock = threading.Lock()
        self._warned = False
        self._warned_lock = threading.Lock()

    @contextmanager
    def acquire(self) -> Iterator[None]:
        contended = self._lock.locked()
        with self._lock:
            if contended:
                self._maybe_warn()
            yield

    def _maybe_warn(self) -> None:
        # Two-phase check so we only emit the warning once per instance even if
        # several threads observe contention before any of them gets to set the
        # flag. The check-then-set is itself guarded by a tiny lock; this is
        # cheap because it only runs on contended entries.
        if self._warned:
            return
        with self._warned_lock:
            if self._warned:
                return
            self._warned = True
            msg = f"{self._label} called under thread contention; this path " \
                  f"is now serialized for correctness."
            if self._advice:
                msg += f" Advice: {self._advice}."
            warnings.warn(msg, stacklevel=4)

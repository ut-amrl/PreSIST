"""
Abstract interface for "direct future-existence" VLM queriers — components that,
given a single image, an object mask, and a list of future time horizons, ask a
VLM for the probability the object still exists at each horizon.

Concrete implementations decide:
- which VLM backend to call (Qwen vLLM, Gemma chat engine, etc.)
- what prompt template to use
- what visual overlays to render around the object (mask highlight, bounding box,
  zoomed crops, arrow, etc.)

Interface is intentionally minimal: `query(image, mask, query_times_seconds) ->
list[float]` — anything beyond image/mask/query_times is the implementation's
own responsibility.
"""
from __future__ import annotations

import json
import math
import re
from abc import ABC, abstractmethod
from typing import List, Sequence

import torch


def _format_horizon(dt_seconds: float) -> str:
    if dt_seconds < 60.0:
        return f"{dt_seconds:.1f}s"
    if dt_seconds < 3600.0:
        return f"{dt_seconds / 60.0:.1f}min"
    return f"{dt_seconds / 3600.0:.2f}h"


def format_horizons_block(query_times_seconds: Sequence[float]) -> str:
    """Format a list of horizons (seconds) for inclusion in a VLM prompt."""
    return "\n".join(
        f"- {dt:.3f} ({_format_horizon(float(dt))})" for dt in query_times_seconds
    )


def extract_prob_array(response_text: str) -> List[dict]:
    """
    Find the last JSON array of dicts in the response.

    Strips anything before the final ``</think>`` tag (handles thinking models),
    then collects every parseable ``[...]`` substring whose elements are dicts
    and returns the last such candidate. Raises ValueError if none found.
    """
    cleaned = response_text.strip()
    think_end = cleaned.rfind("</think>")
    if think_end != -1:
        cleaned = cleaned[think_end + len("</think>"):].strip()

    candidates: List[List[dict]] = []
    if cleaned.startswith("[") and cleaned.endswith("]"):
        try:
            obj = json.loads(cleaned)
            if isinstance(obj, list) and all(isinstance(x, dict) for x in obj):
                candidates.append(obj)
        except json.JSONDecodeError:
            pass

    for m in re.finditer(r"\[.*?\]", cleaned, flags=re.DOTALL):
        try:
            obj = json.loads(m.group(0))
        except json.JSONDecodeError:
            continue
        if isinstance(obj, list) and obj and all(isinstance(x, dict) for x in obj):
            candidates.append(obj)

    if not candidates:
        raise ValueError(
            f"Could not parse JSON probability array from response:\n{response_text}"
        )
    return candidates[-1]


def parse_single_prob(response_text: str) -> float:
    """
    Pull a single probability out of a VLM response.

    Accepts (in priority order, last match wins): a bare JSON number, a JSON
    object with a ``prob`` key, a singleton JSON array of one such object,
    or any of the above embedded inside a longer response (with optional
    ``</think>`` prefix). Returns the value clamped to [0, 1].
    """
    cleaned = response_text.strip()
    think_end = cleaned.rfind("</think>")
    if think_end != -1:
        cleaned = cleaned[think_end + len("</think>"):].strip()

    def _from_obj(obj) -> float | None:
        if isinstance(obj, dict):
            p = obj.get("prob")
            return float(p) if isinstance(p, (int, float)) else None
        if isinstance(obj, list) and obj and isinstance(obj[0], dict):
            p = obj[0].get("prob")
            return float(p) if isinstance(p, (int, float)) else None
        if isinstance(obj, (int, float)):
            return float(obj)
        return None

    candidates: List[float] = []
    try:
        v = _from_obj(json.loads(cleaned))
        if v is not None:
            candidates.append(v)
    except json.JSONDecodeError:
        pass
    for pattern in (r"\{[^{}]*\}", r"\[.*?\]"):
        for m in re.finditer(pattern, cleaned, flags=re.DOTALL):
            try:
                v = _from_obj(json.loads(m.group(0)))
            except json.JSONDecodeError:
                continue
            if v is not None:
                candidates.append(v)

    if not candidates:
        raise ValueError(
            f"Could not parse a single probability from response:\n{response_text}"
        )
    return max(0.0, min(1.0, candidates[-1]))


def parse_probs_aligned_to_query_times(
    response_text: str, query_times_seconds: Sequence[float]
) -> List[float]:
    """
    Parse a list of {"dt_seconds", "prob"} dicts from a VLM response and align
    them to the requested ``query_times_seconds`` order.

    Fast path: if lengths match, trust the order and read each ``prob``.
    Fallback: greedy nearest-``dt_seconds`` matching. Probabilities are clamped
    to [0, 1]; unmatched horizons come back as NaN.
    """
    arr = extract_prob_array(response_text)
    probs: List[float] = [float("nan")] * len(query_times_seconds)

    if len(arr) == len(query_times_seconds):
        for i, entry in enumerate(arr):
            p = entry.get("prob")
            if isinstance(p, (int, float)):
                probs[i] = max(0.0, min(1.0, float(p)))
        return probs

    seen = [False] * len(arr)
    for i, dt in enumerate(query_times_seconds):
        best_j = -1
        best_gap = math.inf
        for j, entry in enumerate(arr):
            if seen[j]:
                continue
            cand_dt = entry.get("dt_seconds")
            if not isinstance(cand_dt, (int, float)):
                continue
            gap = abs(float(cand_dt) - float(dt))
            if gap < best_gap:
                best_gap = gap
                best_j = j
        if best_j >= 0:
            seen[best_j] = True
            p = arr[best_j].get("prob")
            if isinstance(p, (int, float)):
                probs[i] = max(0.0, min(1.0, float(p)))
    return probs


class AbsDirectFutureExistenceVLMQuerier(ABC):
    """
    Abstract direct-future-existence VLM querier.

    Implementations receive only the single object's image, its mask, and the
    set of future horizons; they own all backend, prompt, and overlay choices.
    """

    def __init__(self, name: str) -> None:
        self.name = name

    @abstractmethod
    def query(
        self,
        image: torch.Tensor,
        mask: torch.Tensor,
        query_times_seconds: Sequence[float],
    ) -> List[float]:
        """
        Return a list of probabilities aligned (in order) to ``query_times_seconds``.
        Missing / unparseable entries should be NaN.
        """
        ...

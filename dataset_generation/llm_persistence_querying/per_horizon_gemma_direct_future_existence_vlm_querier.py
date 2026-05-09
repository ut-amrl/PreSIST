"""
Per-horizon variant of :class:`GemmaDirectFutureExistenceVLMQuerier`: rather
than asking the VLM for every horizon's probability in a single response,
issues one chat per horizon (fan-out via ``GemmaChatEngine.chat_many``) and
parses each response as a single probability.

Trade-off: more total VLM tokens (and time, modulo server-side concurrency),
but each horizon gets the model's full attention without competing for tokens
in a single output array. Tends to produce better-calibrated tails when there
are many horizons.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import torch
from PIL import Image

from dataset_generation.llm_persistence_querying.abs_direct_future_existence_vlm_querier import (
    parse_single_prob,
)
from dataset_generation.llm_persistence_querying.gemma.gemma_chat_engine import (
    GemmaChatRequest,
    pil_to_data_url_jpeg,
)
from dataset_generation.llm_persistence_querying.gemma_direct_future_existence_vlm_querier import (
    GemmaDirectFutureExistenceVLMQuerier,
)
from utils.image_utils import image_tensor_to_pil
from utils.timing import timer


_DEFAULT_PER_HORIZON_PROMPT = (
    "You are estimating whether one specific object will still exist at its current "
    "location at a single future time.\n\n"
    "Estimate the probability the highlighted object still exists at this location "
    "{horizon_seconds:.3f} seconds from now ({horizon_pretty}). "
    "Probability is a float in [0, 1].\n\n"
    "Respond with ONLY a JSON object: {{\"prob\": <float in [0,1]>}}.\n"
    "No markdown, no extra text."
)


def _format_horizon_pretty(dt_seconds: float) -> str:
    if dt_seconds < 60.0:
        return f"{dt_seconds:.1f}s"
    if dt_seconds < 3600.0:
        return f"{dt_seconds / 60.0:.1f}min"
    return f"{dt_seconds / 3600.0:.2f}h"


class PerHorizonGemmaDirectFutureExistenceVLMQuerier(GemmaDirectFutureExistenceVLMQuerier):
    """
    One Gemma VLM call per query horizon. Image rendering, overlay options,
    chat-engine setup, and prompt-prefix handling all match the parent class;
    only the prompt template, fan-out, and parser differ.

    The ``prompt`` / ``prompt_file`` constructor args are interpreted as the
    *per-horizon* prompt template — they must contain ``{horizon_seconds}``
    and ``{horizon_pretty}`` placeholders. ``prefix`` / ``prefix_file`` are
    prepended exactly as in the parent class.
    """

    def __init__(
        self,
        *,
        max_concurrency: int = 16,
        prompt: Optional[str] = None,
        prompt_file: Optional[str] = None,
        **parent_kwargs: Any,
    ) -> None:
        # Suppress the parent's all-horizons template handling — we maintain
        # our own per-horizon template below. Prefix handling is reused.
        super().__init__(prompt=None, prompt_file=None, **parent_kwargs)

        if prompt is not None and prompt_file is not None:
            raise ValueError("Only one of prompt or prompt_file should be provided.")
        if prompt_file is not None:
            with open(prompt_file, "r", encoding="utf-8") as f:
                prompt = f.read()

        self._per_horizon_prompt_template = (
            self._prefix + (prompt or _DEFAULT_PER_HORIZON_PROMPT)
        ).strip()
        self.max_concurrency = max(1, int(max_concurrency))

    # ------------------------------------------------------------------
    # AbsDirectFutureExistenceVLMQuerier interface
    # ------------------------------------------------------------------

    def query(
        self,
        image: torch.Tensor,
        mask: torch.Tensor,
        query_times_seconds: Sequence[float],
    ) -> List[float]:
        if not query_times_seconds:
            return []

        pil_image = image_tensor_to_pil(image)
        full_scene_for_engine: Union[Image.Image, str] = (
            pil_to_data_url_jpeg(pil_image) if self._engine.use_vllm_server else pil_image
        )
        # Render overlay images once and reuse across every per-horizon request;
        # only the text prompt differs.
        request_images = self._build_query_images(
            image=image,
            pil_image=pil_image,
            full_scene_for_engine=full_scene_for_engine,
            mask=mask,
        )

        requests = [
            GemmaChatRequest(
                prompt=self._per_horizon_prompt_template.format(
                    horizon_seconds=float(dt),
                    horizon_pretty=_format_horizon_pretty(float(dt)),
                ),
                images=request_images,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                top_p=self.top_p,
            )
            for dt in query_times_seconds
        ]

        with timer.time_block(self._per_query_timer_name, batch_count=len(requests)):
            results = self._engine.chat_many(
                requests, max_concurrency=self.max_concurrency
            )

        probs: List[float] = []
        for res in results:
            try:
                probs.append(parse_single_prob(res.text))
            except ValueError:
                probs.append(float("nan"))
        return probs

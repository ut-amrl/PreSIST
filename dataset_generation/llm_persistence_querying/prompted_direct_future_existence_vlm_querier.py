"""
PromptedDirectFutureExistenceVLMQuerier: direct-future-existence VLM querier
that wraps an :class:`AbsVLMQuerier` (Qwen-VL-style) plus a prompt template
file. Renders a fixed two-image input (original + mask-highlighted view) and
asks the VLM for per-horizon probabilities.

Implementation choices kept minimal so the class mirrors how
``ChainedVLMPersistenceQuerier`` plugs an ``AbsVLMQuerier`` into a prompt + image
pipeline. For backends that need their own concurrency / overlay options (e.g.
Gemma vLLM with bounding boxes and zoom crops) see
``GemmaDirectFutureExistenceVLMQuerier``.
"""
from __future__ import annotations

from typing import List, Optional, Sequence

import torch

from dataset_generation.llm_persistence_querying.abs_direct_future_existence_vlm_querier import (
    AbsDirectFutureExistenceVLMQuerier,
    format_horizons_block,
    parse_probs_aligned_to_query_times,
)
from dataset_generation.llm_persistence_querying.vlm_querying.abs_vlm_querier import AbsVLMQuerier
from utils.image_utils import image_tensor_to_pil, numpy_bgr_to_pil
from utils.visualization_utils import mask_highlight_image


class PromptedDirectFutureExistenceVLMQuerier(AbsDirectFutureExistenceVLMQuerier):
    """
    Loads a prompt template from disk, renders an original + mask-highlighted
    image pair, and calls the wrapped ``AbsVLMQuerier``.

    The prompt template must accept a ``{query_times_block}`` placeholder; no
    other context is provided to the VLM (per the abstract contract).
    """

    def __init__(
        self,
        vlmQuerier: AbsVLMQuerier,
        prompt_template_file: str,
        mask_style: str = "blur_bg_outline_fg",
        mask_highlight_color: Sequence[int] = (0, 255, 0),
        name: Optional[str] = None,
    ) -> None:
        super().__init__(name=name or f"prompted_direct_future_existence_{vlmQuerier.name}")
        with open(prompt_template_file, "r", encoding="utf-8") as f:
            self._template = f.read()
        self._vlm_querier = vlmQuerier
        self._mask_style = mask_style
        self._mask_highlight_color = tuple(int(x) for x in mask_highlight_color)

    def query(
        self,
        image: torch.Tensor,
        mask: torch.Tensor,
        query_times_seconds: Sequence[float],
    ) -> List[float]:
        prompt = self._template.format(
            query_times_block=format_horizons_block(query_times_seconds),
        )

        original_pil = image_tensor_to_pil(image)
        masked_np = mask_highlight_image(
            image, mask, style=self._mask_style, mask_highlight_color=self._mask_highlight_color
        )
        masked_pil = numpy_bgr_to_pil(masked_np)

        response, _debug = self._vlm_querier.queryVLM(prompt, [original_pil, masked_pil])
        return parse_probs_aligned_to_query_times(response, query_times_seconds)

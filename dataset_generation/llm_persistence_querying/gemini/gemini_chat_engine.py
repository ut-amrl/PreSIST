"""
Gemini chat engine (google-genai SDK). Sibling of GemmaChatEngine that fronts
the Google Gemini API and exposes the same ``chat(...)`` / ``chat_many(...)``
surface so other queriers can swap backends with minimal changes.

Returns (text, reasoning, finish_reason, usage) on every call. Cumulative
token usage and the snapshot file live on a process-wide shared singleton in
``gemini._shared`` — every engine instance feeds the same tracker, so a run
with multiple Gemini-backed components produces one unified usage report.
Configure the shared state once at process startup via
``configure_gemini_shared(...)``; if you skip that step, engine construction
lazily initializes it from env vars (``GEMINI_API_KEY`` / ``GOOGLE_API_KEY``).

Thinking mode is set primarily at construction (``thinking_config=...``); a
per-call override is allowed but optional. Thinking-capable models are
``gemini-2.5-*`` and ``gemini-3.1-pro-preview``. If a ``thinking_config`` is
supplied for a non-thinking model (e.g. ``gemini-3.5-flash``,
``gemini-3.1-flash-lite``) the engine warns once and drops it from the request.
NOTE: 2.5 models have thinking enabled by default — ``max_new_tokens`` must
cover both thoughts and the visible reply, so use a generous budget (e.g.
2048+) or disable thinking with ``ThinkingConfig(thinking_budget=0)``.

Default model is ``gemini-2.5-pro``. Allowed model IDs are enumerated in
``_SUPPORTED_MODELS`` below; the allow-list intentionally excludes audio/live,
TTS, image-generation (Nano Banana, Veo, Imagen), Lyria, embeddings, Computer
Use, Deep Research, and deprecated Gemini 2.0. ``gemini-robotics-er-*`` is
known-compatible but tuned for embodied spatial reasoning rather than free-text
captions; opt in with ``allow_unvetted_model=True`` and the explicit model ID.
"""
from __future__ import annotations

import io
import warnings
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

from PIL import Image

from dataset_generation.llm_persistence_querying.gemini import _shared
from dataset_generation.llm_persistence_querying.gemini._shared import TokenUsage


_SUPPORTED_MODELS = frozenset({
    "gemini-3.5-flash",
    "gemini-3.1-flash-lite",
    "gemini-3.1-pro-preview",
    "gemini-2.5-pro",
    "gemini-2.5-flash",
    # Robotics ER: text+image → text with embodied-spatial-reasoning bias.
    # Same pricing as 2.5-flash but with a thinking-tuned reasoning style.
    "gemini-robotics-er-1.6-preview",
})

# Models that honor a ThinkingConfig (either thinking_budget or
# thinking_level). Anything not in this set gets a one-shot warning and the
# config is dropped from the outgoing request.
_THINKING_CAPABLE_MODELS = frozenset({
    "gemini-2.5-pro",
    "gemini-2.5-flash",
    "gemini-3.1-pro-preview",
    "gemini-3.5-flash",
    "gemini-3.1-flash-lite",
    "gemini-robotics-er-1.6-preview",
})

# Per-model recipe for "disable thinking as much as the model allows".
# Picks the right parameter family (thinking_budget vs thinking_level) and
# the minimum permitted value, since the two model families and even
# individual models within a family differ in what they accept:
#   - Gemini 2.5 family + robotics ER use ``thinking_budget`` (int).
#   - Gemini 3 family uses ``thinking_level`` (str), and 3.1-pro-preview
#     specifically does NOT accept ``"minimal"`` — its minimum is ``"low"``.
#   - Gemini 2.5-pro cannot truly disable thinking; 128 is the documented
#     minimum budget and the closest we can get.
# https://ai.google.dev/gemini-api/docs/thinking
_THINKING_OFF_RECIPES: Dict[str, Tuple[str, Any]] = {
    "gemini-2.5-pro":                 ("budget", 128),       # minimum; cannot fully disable
    "gemini-2.5-flash":               ("budget", 0),
    "gemini-3.1-pro-preview":         ("level",  "low"),     # "minimal" not supported on pro
    "gemini-3.5-flash":               ("level",  "minimal"),
    "gemini-3.1-flash-lite":          ("level",  "minimal"),
    "gemini-robotics-er-1.6-preview": ("budget", 0),
}


def _build_disabled_thinking_config(model_name: str) -> Any:
    """Build the right ``ThinkingConfig`` for "disable thinking" given a
    model name. Falls back to ``thinking_budget=0`` with a one-shot warning
    for unknown models (typically harmless on budget-family models; may be
    silently ignored or error on level-family models).
    """
    from google.genai import types as genai_types

    recipe = _THINKING_OFF_RECIPES.get(model_name)
    if recipe is None:
        warnings.warn(
            f"no thinking-off recipe known for model {model_name!r}; "
            f"defaulting to ThinkingConfig(thinking_budget=0). Update "
            f"_THINKING_OFF_RECIPES if this model uses a different knob.",
            stacklevel=3,
        )
        return genai_types.ThinkingConfig(thinking_budget=0)
    kind, value = recipe
    if kind == "budget":
        return genai_types.ThinkingConfig(thinking_budget=value)
    if kind == "level":
        return genai_types.ThinkingConfig(thinking_level=value)
    raise RuntimeError(f"unknown thinking-off recipe kind {kind!r}")


def pil_to_jpeg_bytes(image: Image.Image) -> bytes:
    """JPEG-encode a PIL image. Shared by ``pil_to_part`` and the batch
    engine's content-hash so the same byte sequence drives the API call and
    the cache key.
    """
    buf = io.BytesIO()
    image.save(buf, format="JPEG", quality=95)
    return buf.getvalue()


def pil_to_part(image: Image.Image) -> Any:
    """Encode a PIL image as a Gemini ``Part`` (JPEG bytes).

    Public so callers can pre-encode an image once and pass the resulting Part
    into ``chat()`` / ``chat_many()`` repeatedly, avoiding redundant JPEG work
    on every request.
    """
    from google.genai import types as genai_types

    return genai_types.Part.from_bytes(data=pil_to_jpeg_bytes(image), mime_type="image/jpeg")


@dataclass
class GeminiChatResult:
    text: str
    reasoning: Optional[str] = None
    finish_reason: Optional[str] = None
    usage: TokenUsage = field(default_factory=TokenUsage)


@dataclass
class GeminiChatRequest:
    """One unit of work for chat_many: mirrors the kwargs of chat()."""
    prompt: str
    max_new_tokens: int
    temperature: float
    top_p: float
    images: Optional[List[Union[Image.Image, Any]]] = None
    stop: Optional[List[str]] = None
    seed: Optional[int] = None
    thinking_config: Optional[Any] = None  # genai.types.ThinkingConfig


class ChatManyError(RuntimeError):
    """Raised by ``chat_many`` when one or more requests failed.

    ``failures`` is a list of ``(input_index, exception)`` pairs in input
    order. The first failure is also chained as ``__cause__`` so a typical
    traceback shows it directly.
    """

    def __init__(self, failures: List[Tuple[int, BaseException]], total: int):
        self.failures = failures
        self.total = total
        lines = [f"chat_many: {len(failures)} of {total} requests failed:"]
        for idx, exc in failures:
            lines.append(f"  [request {idx}] {type(exc).__name__}: {exc}")
        super().__init__("\n".join(lines))


class GeminiChatEngine:
    def __init__(
        self,
        *,
        model_name: str = "gemini-2.5-pro",
        thinking_config: Optional[Any] = None,
        disable_thinking: bool = False,
        max_concurrent_requests: int = 16,
        extra_generation_config: Optional[Dict[str, Any]] = None,
        allow_unvetted_model: bool = False,
    ):
        if not allow_unvetted_model and model_name not in _SUPPORTED_MODELS:
            raise ValueError(
                f"model_name={model_name!r} is not in the default allow-list. "
                f"Allowed: {sorted(_SUPPORTED_MODELS)}. Pass "
                f"allow_unvetted_model=True to override (e.g. for "
                f"gemini-robotics-er-* or newer Google releases)."
            )

        self.model_name = model_name
        if disable_thinking and thinking_config is not None:
            raise ValueError(
                "GeminiChatEngine: pass either disable_thinking=True or an "
                "explicit thinking_config, not both."
            )
        if disable_thinking:
            thinking_config = _build_disabled_thinking_config(model_name)
        self.default_thinking_config = thinking_config
        self.max_concurrent_requests = int(max_concurrent_requests)
        self.extra_generation_config = dict(extra_generation_config or {})

        # Pull the shared client (lazy-initializes default config if needed).
        self._client = _shared.get_client()

        # Register self so the first-call safety prompt can list every Gemini
        # model the process will hit. No-op once the prompt has already
        # confirmed or been skipped.
        _shared.register_engine(self.model_name)

        self._warned_thinking_unsupported = False

    # ------------------------------------------------------------------ public

    def chat(
        self,
        *,
        prompt: str,
        images: Optional[List[Union[Image.Image, Any]]] = None,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        stop: Optional[List[str]] = None,
        seed: Optional[int] = None,
        thinking_config: Optional[Any] = None,
    ) -> GeminiChatResult:
        return self._chat_once(
            prompt=prompt,
            images=images,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            stop=stop,
            seed=seed,
            thinking_config=thinking_config,
        )

    def chat_many(
        self,
        requests: List[GeminiChatRequest],
        *,
        max_concurrency: Optional[int] = None,
    ) -> List[GeminiChatResult]:
        """Run a batch of chats concurrently via a ThreadPoolExecutor.

        Returns results in the same order as ``requests``. The google-genai
        client is thread-safe; the server's continuous batching turns these
        into a true parallel decode on the Gemini side. For collect-then-submit
        with on-disk caching (and ~50% lower per-token cost), use
        ``GeminiBatchEngine`` instead.
        """
        if not requests:
            return []
        cap = self.max_concurrent_requests if max_concurrency is None else int(max_concurrency)
        if cap < 1:
            raise ValueError(f"max_concurrency must be >= 1, got {cap}")

        workers = min(cap, len(requests))
        results: List[Optional[GeminiChatResult]] = [None] * len(requests)
        failures: List[Tuple[int, BaseException]] = []
        with ThreadPoolExecutor(
            max_workers=workers,
            thread_name_prefix="gemini_chat_many",
        ) as ex:
            futures = [
                ex.submit(
                    self._chat_once,
                    prompt=req.prompt,
                    images=req.images,
                    max_new_tokens=req.max_new_tokens,
                    temperature=req.temperature,
                    top_p=req.top_p,
                    stop=req.stop,
                    seed=req.seed,
                    thinking_config=req.thinking_config,
                )
                for req in requests
            ]
            for i, fut in enumerate(futures):
                try:
                    results[i] = fut.result()
                except BaseException as e:  # noqa: BLE001 - aggregate, re-raise below
                    failures.append((i, e))
        if failures:
            err = ChatManyError(failures, total=len(requests))
            err.__cause__ = failures[0][1]
            raise err
        return results  # type: ignore[return-value]

    # ---- pass-through accessors for callers that want process-wide totals ----

    @staticmethod
    def get_cumulative_usage() -> TokenUsage:
        return _shared.get_cumulative_usage()

    @staticmethod
    def get_cumulative_usage_by_model() -> Dict[str, TokenUsage]:
        return _shared.get_cumulative_usage_by_model()

    @staticmethod
    def get_request_count() -> int:
        return _shared.get_request_count()

    # --------------------------------------------------------------- internals

    def _chat_once(
        self,
        *,
        prompt: str,
        images: Optional[List[Union[Image.Image, Any]]],
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        stop: Optional[List[str]],
        seed: Optional[int],
        thinking_config: Optional[Any],
    ) -> GeminiChatResult:
        # Safety prompt before any billed Gemini call. Fires at most once
        # per process; auto-skips for env-var / non-TTY / pre-configured cases.
        _shared.maybe_confirm_first_call()

        contents = self._build_contents(prompt, images)
        config = self._build_generation_config(
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            stop=stop,
            seed=seed,
            thinking_config=thinking_config,
        )
        response = self._client.models.generate_content(
            model=self.model_name,
            contents=contents,
            config=config,
        )
        text, reasoning, finish_reason, usage = self._parse_response(response)
        _shared.record_usage(self.model_name, usage)
        return GeminiChatResult(
            text=text,
            reasoning=reasoning,
            finish_reason=finish_reason,
            usage=usage,
        )

    def _build_contents(
        self,
        prompt: str,
        images: Optional[List[Union[Image.Image, Any]]],
    ) -> List[Any]:
        from google.genai import types as genai_types

        parts: List[Any] = []
        if images:
            for im in images:
                if isinstance(im, Image.Image):
                    parts.append(pil_to_part(im))
                else:
                    parts.append(im)  # assume already a Part
        parts.append(genai_types.Part.from_text(text=prompt))
        return [genai_types.Content(role="user", parts=parts)]

    def _build_generation_config(
        self,
        *,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        stop: Optional[List[str]],
        seed: Optional[int],
        thinking_config: Optional[Any],
    ) -> Any:
        from google.genai import types as genai_types

        effective_thinking = thinking_config if thinking_config is not None else self.default_thinking_config
        if effective_thinking is not None and self.model_name not in _THINKING_CAPABLE_MODELS:
            if not self._warned_thinking_unsupported:
                warnings.warn(
                    f"thinking_config supplied but model {self.model_name!r} "
                    f"does not support thinking; dropping it. "
                    f"Thinking-capable models: {sorted(_THINKING_CAPABLE_MODELS)}.",
                    stacklevel=3,
                )
                self._warned_thinking_unsupported = True
            effective_thinking = None

        kwargs: Dict[str, Any] = dict(self.extra_generation_config)
        kwargs["max_output_tokens"] = max_new_tokens
        kwargs["temperature"] = temperature
        kwargs["top_p"] = top_p
        if stop is not None:
            kwargs["stop_sequences"] = stop
        if seed is not None:
            kwargs["seed"] = seed
        if effective_thinking is not None:
            kwargs["thinking_config"] = effective_thinking
        return genai_types.GenerateContentConfig(**kwargs)

    def _parse_response(self, response: Any) -> Tuple[str, Optional[str], Optional[str], TokenUsage]:
        text_chunks: List[str] = []
        thought_chunks: List[str] = []
        finish_reason: Optional[str] = None

        candidates = getattr(response, "candidates", None) or []
        if candidates:
            cand = candidates[0]
            fr = getattr(cand, "finish_reason", None)
            if fr is not None:
                # Enum -> readable string (e.g. FinishReason.STOP -> "STOP")
                finish_reason = getattr(fr, "name", None) or str(fr)
            content = getattr(cand, "content", None)
            parts = getattr(content, "parts", None) or [] if content else []
            for p in parts:
                txt = getattr(p, "text", None)
                if not txt:
                    continue
                if getattr(p, "thought", False):
                    thought_chunks.append(txt)
                else:
                    text_chunks.append(txt)

        text = "".join(text_chunks).strip()
        if not text:
            # Fall back to the SDK's convenience accessor if part-walking found
            # nothing (e.g. some content types).
            fallback = getattr(response, "text", None)
            if fallback:
                text = fallback.strip()
        reasoning = "".join(thought_chunks).strip() or None
        usage = _usage_from_metadata(getattr(response, "usage_metadata", None))
        return text, reasoning, finish_reason, usage


def _usage_from_metadata(meta: Any) -> TokenUsage:
    if meta is None:
        return TokenUsage()
    prompt = int(getattr(meta, "prompt_token_count", 0) or 0)
    completion = int(getattr(meta, "candidates_token_count", 0) or 0)
    thinking = int(getattr(meta, "thoughts_token_count", 0) or 0)
    total = int(getattr(meta, "total_token_count", 0) or 0)
    if total == 0:
        total = prompt + completion + thinking
    return TokenUsage(
        prompt_tokens=prompt,
        completion_tokens=completion,
        thinking_tokens=thinking,
        total_tokens=total,
    )

"""
Shared Gemma 4 chat engine: a single class used by both the LLM querier,
VLM querier, scene captioner, and object describer to talk to a Gemma 4
model in either vLLM-server mode or local Transformers mode.

Returns (main_text, reasoning) from `chat(...)` — the caller is responsible
for placing reasoning into a DebugPayload (the engine does not know the
naming convention or per-call context).

Thinking mode is enabled per-call via `enable_thinking` set at ctor time:
- vLLM mode passes `chat_template_kwargs={"enable_thinking": ...}` in
  `extra_body`; reasoning arrives on `message.reasoning`. Requires the
  server to be launched with `--reasoning-parser gemma4` (see
  start_gemma_vllm_server.sh's --thinking flag).
- Local mode passes `enable_thinking` to `apply_chat_template`, decodes
  with `skip_special_tokens=False`, and uses `processor.parse_response`
  to split thinking from content (per Gemma's official docs).
"""
from __future__ import annotations

import base64
import io
import re
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from PIL import Image

from dataset_generation.llm_persistence_querying.utils.llm_querying_utils import (
    derive_served_model_name,
    resolve_served_model_name,
    resolve_torch_dtype,
)
from utils.threading_utils import ForwardSerializeLock


def pil_to_data_url_jpeg(image: Image.Image) -> str:
    """Encode a PIL image as a JPEG data URL.

    Public so callers (e.g. describers) can pre-encode an image once and pass
    the resulting data URL into ``chat()``/``chat_many()`` repeatedly,
    avoiding redundant JPEG+base64 work on every request.
    """
    buf = io.BytesIO()
    image.save(buf, format="JPEG", quality=95)
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{b64}"


# Fallback regex for Gemma 4 thinking delimiters when processor.parse_response
# is unavailable. Per the model card the thinking content is wrapped as
# `<|channel>thought ... <channel|>`.
_GEMMA_THOUGHT_RE = re.compile(
    r"<\|channel\|?>\s*thought\s*(.*?)\s*<\|?channel\|?>",
    flags=re.DOTALL | re.IGNORECASE,
)


@dataclass
class GemmaChatResult:
    text: str
    reasoning: Optional[str] = None
    finish_reason: Optional[str] = None  # vLLM only; None for local mode


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


@dataclass
class GemmaChatRequest:
    """One unit of work for chat_many: mirrors the kwargs of chat().

    Required fields: prompt, max_new_tokens, temperature, top_p.

    Optional fields (leave unset / None unless the caller specifically needs
    them — they're pass-through to the vLLM OpenAI-compatible API):

    - ``images``: list of PIL images and/or pre-encoded JPEG data-URL strings
      (see ``pil_to_data_url_jpeg``). Strings let callers encode once and
      reuse across many requests; only supported in vLLM-server mode.

    - ``stop``: list of token sequences that end generation as soon as any of
      them appears in the output. The stop sequence itself is not included in
      the returned text. Useful for forcing termination at a structural
      boundary (e.g. ``["\\n\\n", "User:"]``). vLLM only — ignored by
      ``_chat_local``.

    - ``seed``: integer seed for sampling. With ``temperature > 0`` the server
      attempts deterministic output across runs with the same seed; vLLM
      documents this as best-effort and not guaranteed under concurrent load.
      At ``temperature=0`` the seed has essentially no effect (greedy decoding
      is already deterministic up to GPU FP nondeterminism). vLLM only —
      ignored by ``_chat_local``.
    """
    prompt: str
    max_new_tokens: int
    temperature: float
    top_p: float
    images: Optional[List[Union[Image.Image, str]]] = None
    stop: Optional[List[str]] = None
    seed: Optional[int] = None


class GemmaChatEngine:
    def __init__(
        self,
        *,
        model_name: str = "google/gemma-4-31B-it",
        use_vllm_server: bool = True,
        enable_thinking: bool = False,
        # local (Transformers)
        device_map: Union[str, Dict[str, Any]] = "auto",
        torch_dtype: Union[torch.dtype, str] = "auto",
        # vLLM server
        vllm_url_root: str = "http://localhost",
        vllm_port: int = 9008,
        vllm_url_suffix: str = "/v1",
        vllm_api_key: str = "EMPTY",
        vllm_request_timeout_s: float = 240.0,
        vllm_extra_request_params: Optional[Dict[str, Any]] = None,
    ):
        self.model_name = model_name
        self.use_vllm_server = use_vllm_server
        self.enable_thinking = bool(enable_thinking)
        self.vllm_extra_request_params = dict(vllm_extra_request_params or {})

        self.model = None
        self.processor = None
        self.vllm_client = None
        self.vllm_base_url = ""
        self.vllm_served_model_name: Optional[str] = None
        self._name_resolved = False
        # Guards the lazy fallback path of _resolve_served_name. Eager
        # resolution in __init__ avoids the lock entirely on the happy path.
        self._name_resolve_lock = threading.Lock()
        # Defense-in-depth: serialize concurrent forward passes on the local
        # PyTorch model. PyTorch models are not safe under concurrent calls
        # from multiple host threads. In normal use (vLLM-server mode) this
        # lock is never taken; in local-Transformers mode it makes the path
        # safe-but-serialized if ever called from multiple threads, with a
        # one-shot warning so the operator notices the configuration is
        # suboptimal rather than failing silently.
        self._local_forward_lock = ForwardSerializeLock(
            label=f"{type(self).__name__}._chat_local",
            advice="use vLLM-server mode for real concurrency",
        )

        if self.use_vllm_server:
            self.vllm_served_model_name = derive_served_model_name(
                self.model_name, torch_dtype
            )
            base_url = f"{vllm_url_root}:{vllm_port}{vllm_url_suffix}".rstrip("/")
            self.vllm_base_url = base_url
            from openai import OpenAI
            # The OpenAI sync client is built on httpx, which is documented
            # thread-safe; sharing one instance across threads is the
            # recommended pattern (better connection pooling than per-thread).
            # See https://github.com/encode/httpx/discussions/1633
            self.vllm_client = OpenAI(
                base_url=base_url,
                api_key=vllm_api_key,
                timeout=vllm_request_timeout_s,
            )
            # Eagerly resolve the served model name so misconfiguration fails
            # at startup rather than on the first chat call. If the server
            # isn't reachable yet (started after this engine), fall back to
            # the lazy + locked path on first chat() use.
            try:
                self._resolve_served_name()
            except Exception as e:
                print(
                    f"[GemmaChatEngine] eager served-name resolution failed "
                    f"({type(e).__name__}: {e}); will retry lazily on first "
                    f"chat() call. base_url={base_url!r}"
                )
        else:
            from transformers import AutoModelForCausalLM, AutoProcessor
            self.processor = AutoProcessor.from_pretrained(self.model_name)
            resolved_dtype = resolve_torch_dtype(torch_dtype)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=resolved_dtype,
                device_map=device_map,
            )
            self.model.eval()

    def chat(
        self,
        *,
        prompt: str,
        images: Optional[List[Union[Image.Image, str]]] = None,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        stop: Optional[List[str]] = None,
        seed: Optional[int] = None,
    ) -> GemmaChatResult:
        if self.use_vllm_server:
            return self._chat_vllm(
                prompt=prompt,
                images=images,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                stop=stop,
                seed=seed,
            )
        return self._chat_local(
            prompt=prompt,
            images=images,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
        )

    def chat_many(
        self,
        requests: List[GemmaChatRequest],
        *,
        max_concurrency: int = 16,
    ) -> List[GemmaChatResult]:
        """Run a batch of chats; in vLLM mode they run concurrently via a
        ThreadPoolExecutor. Returns results in the same order as ``requests``.

        - vLLM mode: requests fan out across ``min(max_concurrency, len)`` worker
          threads, all sharing this engine's single ``OpenAI`` client (httpx is
          documented thread-safe; see https://github.com/encode/httpx/discussions/1633).
          The vLLM server's continuous batching turns these into a true parallel
          decode at the server.
        - Local-Transformers mode: requests run serially; concurrent calls would
          just contend on one GPU. A warning is emitted once per engine if
          chat_many is invoked with more than one request in local mode so the
          operator notices the path is not parallelized.
        """
        if not requests:
            return []
        if max_concurrency < 1:
            raise ValueError(f"max_concurrency must be >= 1, got {max_concurrency}")

        if not self.use_vllm_server:
            if len(requests) > 1 and not getattr(self, "_warned_local_chat_many", False):
                warnings.warn(
                    f"{type(self).__name__}.chat_many called with "
                    f"{len(requests)} requests in local-Transformers mode; "
                    f"this path is serialized (one GPU). Use vLLM-server mode "
                    f"for true concurrency.",
                    stacklevel=2,
                )
                self._warned_local_chat_many = True
            return [
                self._chat_local(
                    prompt=req.prompt,
                    images=req.images,
                    max_new_tokens=req.max_new_tokens,
                    temperature=req.temperature,
                    top_p=req.top_p,
                )
                for req in requests
            ]

        # vLLM path: parallel fan-out via threads. Submission order = input
        # order, so we keep a plain list of futures and walk it by index when
        # collecting results. Failures don't short-circuit; we drain the batch,
        # collect every exception, and raise a single ChatManyError at the end
        # so callers see *all* failures rather than just the first.
        workers = min(max_concurrency, len(requests))
        results: List[Optional[GemmaChatResult]] = [None] * len(requests)
        failures: List[Tuple[int, BaseException]] = []
        with ThreadPoolExecutor(
            max_workers=workers,
            thread_name_prefix="gemma_chat_many",
        ) as ex:
            futures = [
                ex.submit(
                    self._chat_vllm,
                    prompt=req.prompt,
                    images=req.images,
                    max_new_tokens=req.max_new_tokens,
                    temperature=req.temperature,
                    top_p=req.top_p,
                    stop=req.stop,
                    seed=req.seed,
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

    def _resolve_served_name(self) -> None:
        # Fast path: already resolved (set either by eager resolution in
        # __init__, or by a prior lazy resolution). No lock needed because
        # _name_resolved only flips False -> True once and the read is atomic.
        if self._name_resolved:
            return
        assert self.vllm_client is not None
        assert self.vllm_served_model_name is not None
        # Lazy fallback: only reached if eager resolution failed (e.g., server
        # not up at construction time). Lock so concurrent first callers don't
        # both make the /v1/models request.
        with self._name_resolve_lock:
            if self._name_resolved:
                return
            self.vllm_served_model_name = resolve_served_model_name(
                self.vllm_client,
                model_name=self.model_name,
                derived_name=self.vllm_served_model_name,
                base_url=self.vllm_base_url,
            )
            self._name_resolved = True

    def _build_user_content(
        self,
        prompt: str,
        images: Optional[List[Union[Image.Image, str]]],
    ) -> Union[str, List[Dict[str, Any]]]:
        """Build the OpenAI-style user message content.

        Each entry in ``images`` may be either a ``PIL.Image.Image`` (encoded
        to a JPEG data URL here) or an already-encoded data-URL string (used
        as-is, so callers can encode once and reuse across many requests).
        """
        if not images:
            return prompt
        content: List[Dict[str, Any]] = []
        for im in images:
            if isinstance(im, str):
                url = im
            else:
                url = pil_to_data_url_jpeg(im)
            content.append({
                "type": "image_url",
                "image_url": {"url": url},
            })
        content.append({"type": "text", "text": prompt})
        return content

    def _merged_extra_body(self) -> Dict[str, Any]:
        extra: Dict[str, Any] = dict(self.vllm_extra_request_params)
        chat_kwargs = dict(extra.get("chat_template_kwargs") or {})
        chat_kwargs.setdefault("enable_thinking", self.enable_thinking)
        extra["chat_template_kwargs"] = chat_kwargs
        return extra

    def _chat_vllm(
        self,
        *,
        prompt: str,
        images: Optional[List[Union[Image.Image, str]]],
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        stop: Optional[List[str]],
        seed: Optional[int],
    ) -> GemmaChatResult:
        self._resolve_served_name()
        assert self.vllm_client is not None
        content = self._build_user_content(prompt, images)
        params: Dict[str, Any] = {
            "max_tokens": max_new_tokens,
            "temperature": temperature,
            "top_p": top_p,
        }
        if stop is not None:
            params["stop"] = stop
        if seed is not None:
            params["seed"] = seed
        resp = self.vllm_client.chat.completions.create(
            model=self.vllm_served_model_name,
            messages=[{"role": "user", "content": content}],
            extra_body=self._merged_extra_body(),
            **params,
        )
        choice = resp.choices[0]
        msg = choice.message
        main = (msg.content or "").strip()
        raw_reasoning = (
            getattr(msg, "reasoning", None)
            or getattr(msg, "reasoning_content", None)
        )
        reasoning = (raw_reasoning or "").strip() or None
        return GemmaChatResult(
            text=main,
            reasoning=reasoning,
            finish_reason=choice.finish_reason,
        )

    @torch.no_grad()
    def _chat_local(
        self,
        *,
        prompt: str,
        images: Optional[List[Union[Image.Image, str]]],
        max_new_tokens: int,
        temperature: float,
        top_p: float,
    ) -> GemmaChatResult:
        assert self.model is not None
        assert self.processor is not None

        if images:
            for im in images:
                if isinstance(im, str):
                    raise TypeError(
                        "Local-Transformers mode requires PIL.Image.Image inputs; "
                        "got a pre-encoded data-URL string. Pre-encoding is only "
                        "supported in vLLM-server mode."
                    )

        user_content: List[Dict[str, Any]] = []
        if images:
            for im in images:
                user_content.append({"type": "image", "image": im})
        user_content.append({"type": "text", "text": prompt})
        messages = [{"role": "user", "content": user_content}]

        chat_template_kwargs: Dict[str, Any] = {}
        try:
            text = self.processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=self.enable_thinking,
            )
        except TypeError:
            text = self.processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            chat_template_kwargs["enable_thinking"] = self.enable_thinking

        proc_kwargs: Dict[str, Any] = {
            "text": text,
            "return_tensors": "pt",
        }
        if images:
            proc_kwargs["images"] = images

        with self._local_forward_lock.acquire():
            inputs = self.processor(**proc_kwargs).to(self.model.device)

            do_sample = temperature is not None and temperature > 0
            gen_kwargs: Dict[str, Any] = {
                "max_new_tokens": max_new_tokens,
                "do_sample": do_sample,
            }
            if do_sample:
                gen_kwargs["temperature"] = temperature
                gen_kwargs["top_p"] = top_p

            generated_ids = self.model.generate(**inputs, **gen_kwargs)
            generated_trimmed = [
                out_ids[len(in_ids):]
                for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            decoded = self.processor.batch_decode(
                generated_trimmed,
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            )[0]

        main, reasoning = self._split_local_response(decoded)
        return GemmaChatResult(text=main, reasoning=reasoning, finish_reason=None)

    def _split_local_response(self, decoded: str) -> Tuple[str, Optional[str]]:
        parse_response = getattr(self.processor, "parse_response", None)
        if callable(parse_response):
            try:
                parsed = parse_response(decoded)
                if isinstance(parsed, dict):
                    main = (parsed.get("content") or "").strip()
                    thinking = (parsed.get("thinking") or "").strip() or None
                    return main, thinking
            except Exception:
                pass

        match = _GEMMA_THOUGHT_RE.search(decoded)
        if match:
            reasoning = match.group(1).strip() or None
            main = (decoded[: match.start()] + decoded[match.end():]).strip()
            return main, reasoning

        return decoded.strip(), None

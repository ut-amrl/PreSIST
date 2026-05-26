"""
Collect-then-submit batch wrapper around ``GeminiChatEngine`` with on-disk
caching. Use this when you can stage many requests up front and don't need
the answers immediately: batch jobs are ~50% the per-token cost of the sync
API and the cache means re-running the same workload doesn't re-bill.

Workflow:
1. ``add(request)`` to stage one ``GeminiChatRequest``. Returns a stable
   content-hash request_id you can use to correlate.
2. ``submit_and_wait()`` looks up each staged request in ``cache_dir``;
   cache hits skip the API entirely. Misses are submitted as one inline batch
   job, polled to completion, then their results are written to the cache.
   Results come back in original ``add()`` order.

Cache key: sha256 over model name, prompt, image JPEG bytes, generation
config, and thinking config. Changing temperature / model / thinking budget
naturally invalidates entries. Cache files are written atomically (tmp +
``os.replace``), so a killed process won't leave half-written entries.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from PIL import Image

from dataset_generation.llm_persistence_querying.gemini import _cache, _shared
from dataset_generation.llm_persistence_querying.gemini.gemini_chat_engine import (
    ChatManyError,
    GeminiChatEngine,
    GeminiChatRequest,
    GeminiChatResult,
    pil_to_jpeg_bytes,
)


# Terminal states that stop the polling loop.
_TERMINAL_STATES = {
    "JOB_STATE_SUCCEEDED",
    "JOB_STATE_FAILED",
    "JOB_STATE_CANCELLED",
    "JOB_STATE_EXPIRED",
    "JOB_STATE_PARTIALLY_SUCCEEDED",
}


@dataclass
class _StagedRequest:
    request_id: str
    request: GeminiChatRequest
    fingerprint: Dict[str, Any]
    image_bytes: List[bytes] = field(default_factory=list)  # cached JPEG bytes for the API call


class GeminiBatchEngine:
    def __init__(
        self,
        *,
        chat_engine: GeminiChatEngine,
        cache_dir: Union[str, Path],
        batch_display_name_prefix: str = "spf-batch",
        poll_interval_s: float = 30.0,
        max_wait_s: float = 24 * 3600,
        request_id_fn: Optional[Callable[[GeminiChatRequest], str]] = None,
    ):
        self.chat_engine = chat_engine
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.batch_display_name_prefix = batch_display_name_prefix
        self.poll_interval_s = float(poll_interval_s)
        self.max_wait_s = float(max_wait_s)
        self.request_id_fn = request_id_fn

        self._staged: List[_StagedRequest] = []

    # ------------------------------------------------------------------ public

    def add(self, request: GeminiChatRequest) -> str:
        image_bytes = self._image_bytes_list(request)
        fp = _cache.fingerprint(
            model_name=self.chat_engine.model_name,
            prompt=request.prompt,
            image_bytes_list=image_bytes,
            max_new_tokens=request.max_new_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            stop=request.stop,
            seed=request.seed,
            extra_generation_config=self.chat_engine.extra_generation_config,
            thinking_config=request.thinking_config or self.chat_engine.default_thinking_config,
        )
        request_id = (
            self.request_id_fn(request)
            if self.request_id_fn is not None
            else _cache.hash_fingerprint(fp)
        )
        self._staged.append(_StagedRequest(
            request_id=request_id,
            request=request,
            fingerprint=fp,
            image_bytes=image_bytes,
        ))
        return request_id

    def cached_count(self) -> int:
        return sum(
            1 for s in self._staged
            if _cache.cache_path(self.cache_dir, s.request_id).exists()
        )

    def staged_count(self) -> int:
        return len(self._staged)

    def reset(self) -> None:
        """Clear staged requests. Does not touch the cache directory."""
        self._staged = []

    def submit_and_wait(self) -> List[GeminiChatResult]:
        """Resolve all staged requests, submitting cache misses as one batch.

        Returns results in the order they were ``add``ed. Raises
        ``ChatManyError`` if any cached or batch entry came back as an error.
        """
        if not self._staged:
            return []

        # Partition staged requests into cache hits and misses.
        hits: Dict[int, Union[GeminiChatResult, BaseException]] = {}
        misses: List[Tuple[int, _StagedRequest]] = []
        for i, staged in enumerate(self._staged):
            cached = _cache.load(self.cache_dir, staged.request_id)
            if cached is None:
                misses.append((i, staged))
            else:
                hits[i] = cached

        if misses:
            miss_results = self._run_batch([s for _, s in misses])
            for (i, staged), result_or_err in zip(misses, miss_results):
                # Cache regardless of success so reruns don't repeat doomed
                # requests; caller can delete entries to retry.
                _cache.write(
                    self.cache_dir,
                    staged.request_id,
                    model_name=self.chat_engine.model_name,
                    request_fingerprint=staged.fingerprint,
                    result_or_err=result_or_err,
                )
                hits[i] = result_or_err

        # Reassemble in input order, collecting any errors.
        results: List[GeminiChatResult] = []
        failures: List[Tuple[int, BaseException]] = []
        for i in range(len(self._staged)):
            entry = hits[i]
            if isinstance(entry, BaseException):
                failures.append((i, entry))
                results.append(GeminiChatResult(text=""))  # placeholder
            else:
                results.append(entry)

        if failures:
            err = ChatManyError(failures, total=len(self._staged))
            err.__cause__ = failures[0][1]
            raise err
        return results

    # --------------------------------------------------------- batch internals

    def _run_batch(
        self, misses: List[_StagedRequest]
    ) -> List[Union[GeminiChatResult, BaseException]]:
        from google.genai import types as genai_types

        client = self.chat_engine._client  # shared SDK client
        model = self.chat_engine.model_name

        # Build inline requests. Use metadata to carry our request_id so we can
        # match responses back even if the API doesn't preserve order.
        inlined: List[Any] = []
        for staged in misses:
            req = staged.request
            contents = self.chat_engine._build_contents(
                req.prompt,
                self._materialize_images(staged),
            )
            config = self.chat_engine._build_generation_config(
                max_new_tokens=req.max_new_tokens,
                temperature=req.temperature,
                top_p=req.top_p,
                stop=req.stop,
                seed=req.seed,
                thinking_config=req.thinking_config,
            )
            inlined.append(genai_types.InlinedRequest(
                model=model,
                contents=contents,
                config=config,
                metadata={"request_id": staged.request_id},
            ))

        display_name = f"{self.batch_display_name_prefix}-{int(time.time())}"
        print(f"[GeminiBatchEngine] submitting batch '{display_name}' with {len(inlined)} requests")
        job = client.batches.create(
            model=model,
            src=inlined,
            config=genai_types.CreateBatchJobConfig(display_name=display_name),
        )

        job = self._poll_until_terminal(job)
        final_state = _state_name(job.state)
        print(f"[GeminiBatchEngine] batch '{display_name}' finished in state {final_state}")

        return self._collect_responses(job, misses)

    def _poll_until_terminal(self, job: Any) -> Any:
        client = self.chat_engine._client
        deadline = time.monotonic() + self.max_wait_s
        while _state_name(job.state) not in _TERMINAL_STATES:
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    f"Batch job {job.name} did not reach a terminal state "
                    f"within {self.max_wait_s}s (last state {_state_name(job.state)})"
                )
            time.sleep(self.poll_interval_s)
            job = client.batches.get(name=job.name)
        return job

    def _collect_responses(
        self,
        job: Any,
        misses: List[_StagedRequest],
    ) -> List[Union[GeminiChatResult, BaseException]]:
        dest = getattr(job, "dest", None)
        responses = getattr(dest, "inlined_responses", None) if dest else None

        if not responses:
            # No per-request output — usually whole-job failure.
            err = getattr(job, "error", None)
            msg = f"Batch job {job.name} produced no inlined_responses; job.error={err!r}"
            return [RuntimeError(msg) for _ in misses]

        # Build a lookup by request_id from metadata so order surprises are
        # tolerated. Fall back to positional matching if metadata is missing.
        by_id: Dict[str, Any] = {}
        positional: List[Any] = []
        for resp in responses:
            md = getattr(resp, "metadata", None) or {}
            rid = None
            if isinstance(md, dict):
                rid = md.get("request_id")
            if rid:
                by_id[rid] = resp
            else:
                positional.append(resp)

        out: List[Union[GeminiChatResult, BaseException]] = []
        pos_idx = 0
        for staged in misses:
            resp = by_id.get(staged.request_id)
            if resp is None and pos_idx < len(positional):
                resp = positional[pos_idx]
                pos_idx += 1
            if resp is None:
                out.append(RuntimeError(
                    f"No batch response for request_id {staged.request_id}"
                ))
                continue

            err = getattr(resp, "error", None)
            if err:
                out.append(RuntimeError(f"batch entry error: {err}"))
                continue

            inner = getattr(resp, "response", None)
            if inner is None:
                out.append(RuntimeError("batch entry has neither error nor response"))
                continue

            text, reasoning, finish_reason, usage = self.chat_engine._parse_response(inner)
            _shared.record_usage(self.chat_engine.model_name, usage)
            out.append(GeminiChatResult(
                text=text,
                reasoning=reasoning,
                finish_reason=finish_reason,
                usage=usage,
            ))

        return out

    # --------------------------------------------------------- cache internals

    def _cache_path(self, request_id: str) -> Path:
        return self.cache_dir / f"{request_id}.json"

    def _load_cache(self, request_id: str) -> Optional[GeminiChatResult]:
        path = self._cache_path(request_id)
        if not path.exists():
            return None
        with path.open("r") as f:
            blob = json.load(f)
        if "error" in blob:
            # Surface as an exception so submit_and_wait raises ChatManyError;
            # delete the cache file to retry.
            return RuntimeError(f"cached error for {request_id}: {blob['error']}")  # type: ignore[return-value]
        r = blob.get("result") or {}
        usage = r.get("usage") or {}
        return GeminiChatResult(
            text=r.get("text", ""),
            reasoning=r.get("reasoning"),
            finish_reason=r.get("finish_reason"),
            usage=TokenUsage(**usage),
        )

    def _write_cache(
        self,
        staged: _StagedRequest,
        result_or_err: Union[GeminiChatResult, BaseException],
    ) -> None:
        path = self._cache_path(staged.request_id)
        if isinstance(result_or_err, BaseException):
            blob: Dict[str, Any] = {
                "request_id": staged.request_id,
                "model_name": self.chat_engine.model_name,
                "request_fingerprint": staged.fingerprint,
                "error": f"{type(result_or_err).__name__}: {result_or_err}",
                "submitted_at": _now_iso(),
            }
        else:
            blob = {
                "request_id": staged.request_id,
                "model_name": self.chat_engine.model_name,
                "request_fingerprint": staged.fingerprint,
                "result": {
                    "text": result_or_err.text,
                    "reasoning": result_or_err.reasoning,
                    "finish_reason": result_or_err.finish_reason,
                    "usage": asdict(result_or_err.usage),
                },
                "submitted_at": _now_iso(),
            }
        tmp = path.with_suffix(".json.tmp")
        with tmp.open("w") as f:
            json.dump(blob, f, indent=2, sort_keys=True)
        os.replace(tmp, path)

    # ------------------------------------------------- fingerprint + hashing

    def _image_bytes_list(self, request: GeminiChatRequest) -> List[bytes]:
        """JPEG-encode any PIL images now so the same bytes feed both the
        cache key and the API call. Pre-encoded ``Part`` objects are passed
        through unchanged at API-call time; we hash their string form for
        cache stability.
        """
        if not request.images:
            return []
        encoded: List[bytes] = []
        for im in request.images:
            if isinstance(im, Image.Image):
                encoded.append(pil_to_jpeg_bytes(im))
            else:
                # Pre-encoded Part — best effort: hash its repr so cache key
                # remains stable for the same Part object. Callers who want
                # cross-process cache reuse for pre-encoded images should pass
                # PIL.Image.Image instead.
                encoded.append(repr(im).encode("utf-8"))
        return encoded

    def _materialize_images(self, staged: _StagedRequest) -> Optional[List[Any]]:
        """Return the list of (PIL or Part) inputs for the API call. For PIL
        inputs we use the already-encoded JPEG bytes via pil_to_part-style
        Part to avoid re-encoding.
        """
        from google.genai import types as genai_types

        if not staged.request.images:
            return None
        materialized: List[Any] = []
        for im, raw in zip(staged.request.images, staged.image_bytes):
            if isinstance(im, Image.Image):
                materialized.append(
                    genai_types.Part.from_bytes(data=raw, mime_type="image/jpeg")
                )
            else:
                materialized.append(im)
        return materialized

    def _fingerprint(
        self, request: GeminiChatRequest, image_bytes: List[bytes]
    ) -> Dict[str, Any]:
        thinking = request.thinking_config or self.chat_engine.default_thinking_config
        return {
            "model_name": self.chat_engine.model_name,
            "prompt": request.prompt,
            "image_sha256s": [hashlib.sha256(b).hexdigest() for b in image_bytes],
            "generation_config": {
                "max_new_tokens": request.max_new_tokens,
                "temperature": request.temperature,
                "top_p": request.top_p,
                "stop": request.stop,
                "seed": request.seed,
                "extra": self.chat_engine.extra_generation_config,
            },
            "thinking_config": _thinking_repr(thinking),
        }


def _hash_fingerprint(fp: Dict[str, Any]) -> str:
    payload = json.dumps(fp, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _thinking_repr(tc: Any) -> Optional[Dict[str, Any]]:
    if tc is None:
        return None
    out: Dict[str, Any] = {}
    for k in ("include_thoughts", "thinking_budget"):
        v = getattr(tc, k, None)
        if v is not None:
            out[k] = v
    return out


def _state_name(state: Any) -> str:
    return getattr(state, "name", None) or str(state)


def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

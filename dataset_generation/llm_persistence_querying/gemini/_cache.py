"""
Shared on-disk cache for Gemini chat results. Used by both ``GeminiBatchEngine``
(to skip already-resolved requests in a batch) and ``GeminiSceneLevelCaptioner``
(to avoid re-querying the same scene+prompt+config).

Cache key is a content sha256 over: model name, prompt, image JPEG bytes,
generation config (max_new_tokens / temperature / top_p / stop / seed / any
extra_generation_config), and thinking config. Changing any of those naturally
invalidates the entry. Cache files are written atomically (tmp + os.replace)
so a killed process won't leave half-written entries.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from dataset_generation.llm_persistence_querying.gemini.gemini_chat_engine import (
    GeminiChatResult,
    TokenUsage,
)


def fingerprint(
    *,
    model_name: str,
    prompt: str,
    image_bytes_list: List[bytes],
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    stop: Optional[List[str]],
    seed: Optional[int],
    extra_generation_config: Optional[Dict[str, Any]],
    thinking_config: Any,
) -> Dict[str, Any]:
    """Build the canonical fingerprint dict for a request. Stored alongside the
    cached result for debuggability so a stale entry can be inspected without
    re-running anything.
    """
    return {
        "model_name": model_name,
        "prompt": prompt,
        "image_sha256s": [hashlib.sha256(b).hexdigest() for b in image_bytes_list],
        "generation_config": {
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "stop": stop,
            "seed": seed,
            "extra": extra_generation_config or {},
        },
        "thinking_config": _thinking_repr(thinking_config),
    }


def hash_fingerprint(fp: Dict[str, Any]) -> str:
    payload = json.dumps(fp, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def cache_path(cache_dir: Union[str, Path], request_id: str) -> Path:
    return Path(cache_dir) / f"{request_id}.json"


def load(
    cache_dir: Union[str, Path], request_id: str
) -> Optional[Union[GeminiChatResult, BaseException]]:
    """Return a cached result, an exception sentinel (for cached errors), or
    None if the entry doesn't exist.
    """
    path = cache_path(cache_dir, request_id)
    if not path.exists():
        return None
    with path.open("r") as f:
        blob = json.load(f)
    if "error" in blob:
        return RuntimeError(f"cached error for {request_id}: {blob['error']}")
    r = blob.get("result") or {}
    usage = r.get("usage") or {}
    return GeminiChatResult(
        text=r.get("text", ""),
        reasoning=r.get("reasoning"),
        finish_reason=r.get("finish_reason"),
        usage=TokenUsage(**usage),
    )


def write(
    cache_dir: Union[str, Path],
    request_id: str,
    *,
    model_name: str,
    request_fingerprint: Dict[str, Any],
    result_or_err: Union[GeminiChatResult, BaseException],
) -> None:
    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    path = cache_path(cache_dir, request_id)
    if isinstance(result_or_err, BaseException):
        blob: Dict[str, Any] = {
            "request_id": request_id,
            "model_name": model_name,
            "request_fingerprint": request_fingerprint,
            "error": f"{type(result_or_err).__name__}: {result_or_err}",
            "submitted_at": _now_iso(),
        }
    else:
        blob = {
            "request_id": request_id,
            "model_name": model_name,
            "request_fingerprint": request_fingerprint,
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


def _thinking_repr(tc: Any) -> Optional[Dict[str, Any]]:
    if tc is None:
        return None
    out: Dict[str, Any] = {}
    for k in ("include_thoughts", "thinking_budget"):
        v = getattr(tc, k, None)
        if v is not None:
            out[k] = v
    return out


def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

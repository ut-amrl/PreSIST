"""
Process-wide shared state for Gemini engines: one google-genai client, one
cumulative token-usage tracker, and one usage-snapshot file. ``GeminiChatEngine``
instances stay lightweight per-(model, thinking_config) shims and all funnel
into this single tracker, so a run with multiple Gemini-backed components
(scene captioner, object describer, persistence querier, …) produces one
unified usage report instead of one per component.

Lifecycle:

1. Process startup: call ``configure_gemini_shared(...)`` once, ideally before
   any ``GeminiChatEngine`` is constructed. ``usage_log_file`` configured here
   is the durable path written to on every periodic-log fire and on shutdown.
2. Engine construction: each ``GeminiChatEngine(...)`` reaches into this module
   for the client and the tracker; no per-engine API key / log-file setup.
3. Process shutdown: an ``atexit`` hook flushes one final snapshot.

If ``configure_gemini_shared`` is never called explicitly, the first engine
construction triggers a lazy default-configuration (API key from
``GEMINI_API_KEY`` / ``GOOGLE_API_KEY``, no log file, periodic logs every 50
requests). That keeps non-pipeline usage (notebooks, smoke tests) painless.
"""
from __future__ import annotations

import atexit
import json
import os
import sys
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union


# Re-exported under this module's namespace to avoid an import cycle with
# gemini_chat_engine (which imports from here).
@dataclass
class TokenUsage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    thinking_tokens: int = 0
    total_tokens: int = 0

    def __iadd__(self, other: "TokenUsage") -> "TokenUsage":
        self.prompt_tokens += other.prompt_tokens
        self.completion_tokens += other.completion_tokens
        self.thinking_tokens += other.thinking_tokens
        self.total_tokens += other.total_tokens
        return self

    def copy(self) -> "TokenUsage":
        return TokenUsage(
            prompt_tokens=self.prompt_tokens,
            completion_tokens=self.completion_tokens,
            thinking_tokens=self.thinking_tokens,
            total_tokens=self.total_tokens,
        )


# Callback shape: (overall_usage, per_model_usage, total_request_count) -> None
UsageLogCallback = Callable[[TokenUsage, Dict[str, TokenUsage], int], None]


@dataclass
class _SharedState:
    """Holds the process-wide Gemini resources. Treat fields as opaque outside
    this module — use the exported accessors below.
    """
    client: Any = None  # genai.Client
    api_key: Optional[str] = None
    vertexai: bool = False
    vertex_project: Optional[str] = None
    vertex_location: Optional[str] = None
    request_timeout_s: float = 240.0

    cumulative_usage: TokenUsage = field(default_factory=TokenUsage)
    cumulative_usage_by_model: Dict[str, TokenUsage] = field(default_factory=dict)
    request_count: int = 0
    request_count_by_model: Dict[str, int] = field(default_factory=dict)

    usage_log_every_n_requests: int = 50
    usage_log_callback: Optional[UsageLogCallback] = None
    usage_log_file: Optional[Path] = None


_state = _SharedState()
_state_lock = threading.Lock()
_file_lock = threading.Lock()
_configured = False
_atexit_registered = False

# Safety-prompt machinery — see ``maybe_confirm_first_call`` for the full
# behaviour spec. Engines register themselves at construction time so the
# prompt can list every Gemini model the process will hit before the first
# billed API call goes out.
_registered_engines: List[Dict[str, Any]] = []
_confirm_lock = threading.Lock()
_confirm_state: str = "needed"  # "needed" | "confirmed" | "skipped"


def configure_gemini_shared(
    *,
    api_key: Optional[str] = None,
    vertexai: bool = False,
    vertex_project: Optional[str] = None,
    vertex_location: Optional[str] = None,
    request_timeout_s: float = 240.0,
    usage_log_every_n_requests: int = 50,
    usage_log_callback: Optional[UsageLogCallback] = None,
    usage_log_file: Optional[Union[str, Path]] = None,
    confirm_before_first_call: bool = True,
    overwrite: bool = False,
) -> None:
    """Configure the process-wide Gemini state. Idempotent on its first call;
    subsequent calls log a warning unless ``overwrite=True``.

    Call this once at process startup (e.g. from the eval entry point) before
    any ``GeminiChatEngine`` is constructed. If skipped, the first engine
    construction lazily applies env-var defaults.
    """
    global _configured
    with _state_lock:
        if _configured and not overwrite:
            print(
                "[gemini.shared] configure_gemini_shared called again with "
                "overwrite=False; ignoring. Pass overwrite=True to reset."
            )
            return

        # Wipe usage if we're reconfiguring on top of an existing setup.
        if _configured and overwrite:
            _state.cumulative_usage = TokenUsage()
            _state.cumulative_usage_by_model = {}
            _state.request_count = 0
            _state.request_count_by_model = {}

        resolved_key = (
            api_key
            or os.environ.get("GEMINI_API_KEY")
            or os.environ.get("GOOGLE_API_KEY")
        )
        if not vertexai and not resolved_key:
            raise RuntimeError(
                "configure_gemini_shared: no Gemini API key found. Pass "
                "api_key=..., or set GEMINI_API_KEY (preferred) or "
                "GOOGLE_API_KEY in the environment. See GEMINI.md."
            )

        from google import genai
        from google.genai import types as genai_types

        if vertexai:
            client = genai.Client(
                vertexai=True,
                project=vertex_project,
                location=vertex_location,
                http_options=genai_types.HttpOptions(timeout=int(request_timeout_s * 1000)),
            )
        else:
            client = genai.Client(
                api_key=resolved_key,
                http_options=genai_types.HttpOptions(timeout=int(request_timeout_s * 1000)),
            )

        _state.client = client
        _state.api_key = resolved_key
        _state.vertexai = vertexai
        _state.vertex_project = vertex_project
        _state.vertex_location = vertex_location
        _state.request_timeout_s = request_timeout_s
        _state.usage_log_every_n_requests = int(usage_log_every_n_requests)
        _state.usage_log_callback = usage_log_callback
        _state.usage_log_file = Path(usage_log_file) if usage_log_file else None

        _configured = True
        _register_atexit_snapshot()

    # Apply the confirmation override outside the state lock to avoid lock-
    # ordering issues with _confirm_lock.
    if not confirm_before_first_call:
        skip_confirmation()


def set_usage_log_file(path: Union[str, Path]) -> None:
    """Configure the snapshot file path on the running singleton. Safe to
    call before or after engines have done work; writes an immediate snapshot
    of the current totals so the file appears right away. No-op (apart from
    storing the path) if no usage has been recorded yet.

    This is the hook the inference runner uses to pin the snapshot file
    under the per-run output directory once the run_dir is known.
    """
    _state.usage_log_file = Path(path)
    _register_atexit_snapshot()
    _write_usage_snapshot()


def get_client() -> Any:
    """Return the shared genai client. Triggers lazy default-config if needed."""
    if not _configured:
        configure_gemini_shared()
    return _state.client


def get_cumulative_usage() -> TokenUsage:
    with _state_lock:
        return _state.cumulative_usage.copy()


def get_cumulative_usage_by_model() -> Dict[str, TokenUsage]:
    with _state_lock:
        return {k: v.copy() for k, v in _state.cumulative_usage_by_model.items()}


def get_request_count() -> int:
    with _state_lock:
        return _state.request_count


def reset_usage() -> None:
    """Wipe cumulative usage. Intended for tests; doesn't touch the snapshot
    file or the client.
    """
    with _state_lock:
        _state.cumulative_usage = TokenUsage()
        _state.cumulative_usage_by_model = {}
        _state.request_count = 0
        _state.request_count_by_model = {}


def record_usage(model_name: str, usage: TokenUsage) -> None:
    """Add one request's usage to the cumulative tracker. Triggers the
    periodic log + snapshot write when the request count crosses an N-boundary.
    Engines call this after every successful chat call.
    """
    should_log = False
    overall: TokenUsage = TokenUsage()
    per_model_snapshot: Dict[str, TokenUsage] = {}
    per_model_counts: Dict[str, int] = {}
    total_count = 0

    with _state_lock:
        _state.cumulative_usage += usage
        bucket = _state.cumulative_usage_by_model.setdefault(model_name, TokenUsage())
        bucket += usage  # in-place; dict already holds the same object
        _state.request_count += 1
        _state.request_count_by_model[model_name] = (
            _state.request_count_by_model.get(model_name, 0) + 1
        )
        n = _state.usage_log_every_n_requests
        if n > 0 and _state.request_count % n == 0:
            should_log = True
            overall = _state.cumulative_usage.copy()
            per_model_snapshot = {
                k: v.copy() for k, v in _state.cumulative_usage_by_model.items()
            }
            per_model_counts = dict(_state.request_count_by_model)
            total_count = _state.request_count

    if should_log:
        cb = _state.usage_log_callback
        if cb is not None:
            cb(overall, per_model_snapshot, total_count)
        else:
            _default_usage_log(overall, per_model_snapshot, per_model_counts, total_count)
        _write_usage_snapshot()


# ---------- safety prompt before first billed Gemini API call ----------

def register_engine(model_name: str) -> None:
    """Engines call this from ``__init__`` so the safety prompt can list
    every Gemini model the process will hit before the first API call.

    Roles aren't tracked here — the engine only knows its model name.
    Multi-engine pipelines simply show one line per registered engine.
    """
    with _confirm_lock:
        _registered_engines.append({"model_name": str(model_name)})


def skip_confirmation() -> None:
    """Mark the singleton as not needing the first-call confirmation prompt.
    Idempotent. Used by ``configure_gemini_shared(confirm_before_first_call=False)``
    and by tests that don't want the interactive prompt.
    """
    global _confirm_state
    with _confirm_lock:
        _confirm_state = "skipped"


def maybe_confirm_first_call() -> None:
    """Run the first-call safety prompt if it hasn't run yet. Safe to call
    from any number of threads — the prompt fires at most once per process,
    behind a lock.

    Auto-skip conditions (in priority order):
      1. State is already "confirmed" or "skipped".
      2. ``GEMINI_SKIP_CONFIRM`` env var is truthy.
      3. stdin is not a TTY — we print the projection banner but auto-proceed
         (so CI / batch / piped runs work).

    On user refusal (interactive only): calls ``sys.exit(0)``.
    """
    global _confirm_state

    # Fast path — already decided, no lock needed for the read because the
    # state only ever transitions away from "needed" once.
    if _confirm_state != "needed":
        return

    with _confirm_lock:
        if _confirm_state != "needed":
            return

        if os.environ.get("GEMINI_SKIP_CONFIRM", "").lower() in ("1", "true", "yes"):
            _confirm_state = "skipped"
            return

        from dataset_generation.llm_persistence_querying.gemini.pricing import (
            format_projection_from_engines,
        )
        banner = "=" * 70
        print(banner)
        print(" GEMINI-BACKED RUN — please confirm before charges accrue")
        print(banner)
        print(format_projection_from_engines(
            [e["model_name"] for e in _registered_engines],
        ))
        print(banner)

        if not sys.stdin.isatty():
            print(
                "[gemini.shared] stdin is not a TTY — cannot prompt for "
                "confirmation. PROCEEDING. Pass "
                "configure_gemini_shared(confirm_before_first_call=False) or "
                "set GEMINI_SKIP_CONFIRM=1 in the environment to silence this "
                "banner for known-safe automated runs.",
            )
            _confirm_state = "skipped"
            return

        try:
            reply = input("Proceed with this run? [y/N]: ").strip().lower()
        except EOFError:
            reply = ""
        if reply not in ("y", "yes"):
            print("Aborted by user.")
            sys.exit(0)
        _confirm_state = "confirmed"


def _register_atexit_snapshot() -> None:
    global _atexit_registered
    if _atexit_registered:
        return
    atexit.register(_write_usage_snapshot)
    _atexit_registered = True


def _write_usage_snapshot() -> None:
    """Atomic overwrite of ``usage_log_file`` with the current totals.
    No-op when no file path is configured. Errors are logged, not raised.
    """
    if _state.usage_log_file is None:
        return
    with _state_lock:
        payload = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "request_count": _state.request_count,
            "request_count_by_model": dict(_state.request_count_by_model),
            "cumulative_usage": asdict(_state.cumulative_usage),
            "cumulative_usage_by_model": {
                k: asdict(v) for k, v in _state.cumulative_usage_by_model.items()
            },
        }
    path = _state.usage_log_file
    with _file_lock:
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(path.suffix + ".tmp")
            with tmp.open("w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2, sort_keys=True)
            os.replace(tmp, path)
        except OSError as e:
            print(f"[gemini.shared] failed to write usage snapshot to {path}: {e}")


def _default_usage_log(
    overall: TokenUsage,
    per_model: Dict[str, TokenUsage],
    per_model_counts: Dict[str, int],
    total_count: int,
) -> None:
    lines = [f"[gemini.shared] usage after {total_count} requests:"]
    lines.append(
        f"  overall: prompt={overall.prompt_tokens} "
        f"completion={overall.completion_tokens} "
        f"thinking={overall.thinking_tokens} "
        f"total={overall.total_tokens}"
    )
    for model, usage in sorted(per_model.items()):
        count = per_model_counts.get(model, 0)
        lines.append(
            f"  {model} ({count}): prompt={usage.prompt_tokens} "
            f"completion={usage.completion_tokens} "
            f"thinking={usage.thinking_tokens} "
            f"total={usage.total_tokens}"
        )
    print("\n".join(lines))

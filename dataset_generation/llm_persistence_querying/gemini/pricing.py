"""
Gemini API pricing and cost-projection helpers. Shared by ``collect_gemini_usage.py``
(which sums actual post-hoc usage) and the run-time confirmation prompt in
``eval_persistence.py`` (which projects cost *before* a run starts).

All rates are USD per 1M tokens, sourced from
https://ai.google.dev/gemini-api/docs/pricing
Lower-tier (≤200K input) where the model has tiered pricing; text/image/video
rate where the model distinguishes modalities. Output rate already includes
thinking tokens (Google bills thoughts as output).

Per-call token estimates are derived from the 5-anchor all-Robotics-ER
smoketest (avg 4.4K prompt / 0.24K output per call across captioner + describer
+ querier on 1306×1306 HD-EPIC images). They are intentionally approximate;
real costs vary with K (tracks/anchor), prompt template, image resolution, and
cache hit rate. The projection therefore reports a *range* rather than a point
estimate.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


DEFAULT_PRICING: Dict[str, Dict[str, float]] = {
    "gemini-2.5-pro":                 {"input_per_million": 1.25, "output_per_million": 10.00},
    "gemini-2.5-flash":               {"input_per_million": 0.30, "output_per_million":  2.50},
    "gemini-3.5-flash":               {"input_per_million": 1.50, "output_per_million":  9.00},
    "gemini-3.1-flash-lite":          {"input_per_million": 0.25, "output_per_million":  1.50},
    "gemini-3.1-pro-preview":         {"input_per_million": 2.00, "output_per_million": 12.00},
    "gemini-robotics-er-1.6-preview": {"input_per_million": 1.00, "output_per_million":  5.00},
}


# Per-call token budgets (measured average + observed max from the all-ER
# smoketest on 1306×1306 HD-EPIC images with thinking off). These are
# rough — switch to a different prompt template / image resolution / thinking-on
# and the numbers drift, but they're a defensible starting point for projection.
@dataclass(frozen=True)
class _CallProfile:
    role: str
    prompt_tokens_per_call: int
    output_tokens_per_call: int

# Note: describer calls scale with K (one per object); the prompt token count
# stays roughly constant per object because the same overlay images are sent.
CAPTIONER_PROFILE = _CallProfile("captioner", prompt_tokens_per_call=4_000, output_tokens_per_call=300)
DESCRIBER_PROFILE = _CallProfile("describer", prompt_tokens_per_call=5_000, output_tokens_per_call=400)
QUERIER_PROFILE  = _CallProfile("querier",   prompt_tokens_per_call=3_000, output_tokens_per_call=400)
QUERIER_MAX_OBJS_PER_CALL = 20   # mirrors ChainedPromptQueryPersistenceQuerier default


def call_cost_usd(model: str, prompt_tokens: int, output_tokens: int) -> Optional[float]:
    """Single-call cost in USD. Returns None if the model isn't priced."""
    rates = DEFAULT_PRICING.get(model)
    if rates is None:
        return None
    return (
        prompt_tokens  * rates["input_per_million"]  / 1_000_000.0
        + output_tokens * rates["output_per_million"] / 1_000_000.0
    )


@dataclass(frozen=True)
class GeminiComponent:
    """One Gemini-backed component identified in a Hydra inferencer config."""
    role: str        # "captioner" | "describer" | "querier" | "unknown"
    target: str      # Python target path (for error messages)
    model: str       # gemini-2.5-pro / gemini-robotics-er-1.6-preview / ...


def project_run_cost(
    components: List[GeminiComponent],
    num_anchors: Optional[int],
    *,
    k_low: int = 1,
    k_high: int = 3,
    max_objs_per_query_call: int = QUERIER_MAX_OBJS_PER_CALL,
) -> Dict:
    """Cost projection for a planned eval run.

    Returns a dict with: ``per_anchor_low_usd``, ``per_anchor_high_usd``,
    ``total_low_usd``, ``total_high_usd`` (None for total when num_anchors
    is None), and a per-component breakdown showing the model used and
    whether it's priced. ``low`` uses K=k_low, ``high`` uses K=k_high.

    All numbers are *list* prices (sync API). Apply a 50% discount manually
    if you plan to route through the Batch API.
    """
    role_to_component = {c.role: c for c in components}

    def _phase_cost_per_anchor(role: str, k: int) -> Tuple[Optional[float], int]:
        """Return (usd_per_anchor, calls_per_anchor) for one phase."""
        comp = role_to_component.get(role)
        if comp is None:
            return 0.0, 0
        if role == "captioner":
            profile = CAPTIONER_PROFILE
            calls_per_anchor = 1
        elif role == "describer":
            profile = DESCRIBER_PROFILE
            calls_per_anchor = k
        elif role == "querier":
            profile = QUERIER_PROFILE
            calls_per_anchor = max(1, math.ceil(k / max_objs_per_query_call))
        else:
            return None, 0
        single_call = call_cost_usd(comp.model, profile.prompt_tokens_per_call, profile.output_tokens_per_call)
        if single_call is None:
            return None, calls_per_anchor
        return single_call * calls_per_anchor, calls_per_anchor

    def _total_per_anchor(k: int) -> Tuple[Optional[float], int]:
        total, calls = 0.0, 0
        any_unpriced = False
        for role in ("captioner", "describer", "querier"):
            cost, n = _phase_cost_per_anchor(role, k)
            if cost is None:
                any_unpriced = True
            else:
                total += cost
            calls += n
        return (None if any_unpriced else total), calls

    per_anchor_low, calls_low   = _total_per_anchor(k_low)
    per_anchor_high, calls_high = _total_per_anchor(k_high)

    total_low  = per_anchor_low  * num_anchors if (per_anchor_low  is not None and num_anchors is not None) else None
    total_high = per_anchor_high * num_anchors if (per_anchor_high is not None and num_anchors is not None) else None

    return {
        "components": [
            {"role": c.role, "model": c.model, "priced": c.model in DEFAULT_PRICING}
            for c in components
        ],
        "num_anchors": num_anchors,
        "k_low": k_low,
        "k_high": k_high,
        "per_anchor_low_usd":  per_anchor_low,
        "per_anchor_high_usd": per_anchor_high,
        "calls_per_anchor_low":  calls_low,
        "calls_per_anchor_high": calls_high,
        "total_low_usd":  total_low,
        "total_high_usd": total_high,
    }


def format_projection_from_engines(model_names: List[str]) -> str:
    """Light-weight projection from just a list of model names — used by the
    first-call safety prompt, which fires from inside the engine and has no
    access to the inferencer cfg or anchor count. Reports per-call cost per
    distinct model and an aggregated per-engine-call total, leaving total
    run cost to the user (they have anchor count + K).

    For the full inferencer-aware projection (with run-total extrapolation),
    use ``project_run_cost(...)`` + ``format_projection(...)`` from a context
    that has the Hydra cfg.
    """
    if not model_names:
        return "(no Gemini engines registered)"

    # Group by model so duplicates collapse but with a count.
    counts: Dict[str, int] = {}
    for m in model_names:
        counts[m] = counts.get(m, 0) + 1

    lines = ["Gemini engines registered for this run:"]
    for model, n in sorted(counts.items()):
        cost = call_cost_usd(
            model,
            CAPTIONER_PROFILE.prompt_tokens_per_call,
            CAPTIONER_PROFILE.output_tokens_per_call,
        )
        if cost is None:
            cost_str = "—  (UNPRICED model)"
        else:
            cost_str = f"~${cost:.4f}/call (rough; per-call tokens vary)"
        suffix = f" × {n}" if n > 1 else ""
        lines.append(f"  - {model}{suffix}    {cost_str}")

    lines.append("")
    lines.append(
        "Cost depends on number of anchors and tracks-per-anchor (K), which "
        "this prompt cannot see from inside the engine. From the all-Robotics-ER "
        "smoketest measurement: ~$0.017/anchor at K=1, ~$0.032/anchor at K=3. "
        "After the run, ``python3 collect_gemini_usage.py`` produces an "
        "authoritative cost report from the per-run snapshot files."
    )
    return "\n".join(lines)


def format_projection(p: Dict) -> str:
    """Render a projection dict as a human-readable multi-line string."""
    lines = ["Gemini components detected in this run:"]
    for c in p["components"]:
        pr = "" if c["priced"] else "  (UNPRICED — cost cannot be projected)"
        lines.append(f"  - {c['role']:<10}{c['model']}{pr}")

    lines.append("")
    n = p["num_anchors"]
    if n is None:
        lines.append("Number of anchors: UNKNOWN (dataset does not advertise an upper bound)")
    else:
        lines.append(f"Number of anchors planned: {n:,}")

    def _fmt(v):
        return f"${v:,.4f}" if v is not None else "—"

    lines.append("")
    lines.append(f"Per-anchor cost (sync API list price, thinking off):")
    lines.append(f"  K={p['k_low']} (low):  {_fmt(p['per_anchor_low_usd'])}   ({p['calls_per_anchor_low']} API call(s)/anchor)")
    lines.append(f"  K={p['k_high']} (high): {_fmt(p['per_anchor_high_usd'])}   ({p['calls_per_anchor_high']} API call(s)/anchor)")

    if n is not None:
        lines.append("")
        lines.append(f"Estimated total cost range: {_fmt(p['total_low_usd'])}  to  {_fmt(p['total_high_usd'])}")
        if p["total_low_usd"] is not None:
            lines.append(f"  (Batch API would halve this: {_fmt(p['total_low_usd'] / 2)} to {_fmt(p['total_high_usd'] / 2)})")
    lines.append("")
    lines.append("Note: estimates are based on the 5-anchor all-Robotics-ER smoketest")
    lines.append("(avg ~4.4K prompt / ~0.24K output tokens per call on 1306² images).")
    lines.append("Real cost will vary with K, prompt size, image resolution, cache hits.")
    return "\n".join(lines)

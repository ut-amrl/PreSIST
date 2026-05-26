#!/usr/bin/env python3
"""
Walk an output tree for every ``gemini_token_usage.json`` snapshot left by
``GeminiChatEngine`` runs, aggregate token usage, and report estimated cost.

Default behaviour: write the full JSON report to
``/workspace/data/out/gemini_costs.json`` (overwriting), and print a compact
human-readable summary table to stdout.

Usage:
    python collect_gemini_usage.py                                  # default root + default output file
    python collect_gemini_usage.py --root /some/other/output_root
    python collect_gemini_usage.py --pricing pricing.json           # override rates
    python collect_gemini_usage.py --output /tmp/elsewhere.json     # different file
    python collect_gemini_usage.py --no-save                        # skip the file write

The JSON document has these top-level keys:
    ``totals``    - aggregate request count, per-token-class totals,
                    estimated USD cost. ``totals.by_model`` breaks the same
                    fields down per model.
    ``runs``      - one object per snapshot file, in chronological order by
                    snapshot timestamp (fallback: file mtime).
    ``pricing``   - the rate table actually used (effective after any
                    --pricing override).
    ``warnings``  - human-readable issues (unknown models, unparseable
                    snapshots, etc).

Pricing notes:
    Built-in rates cover Gemini 2.5 Pro / 2.5 Flash from the AI Studio
    "<= 200K input" tier. Thinking tokens are billed as output tokens
    (Google's documented behaviour). Any model not in the table is reported
    in tokens only with ``estimated_cost_usd: null`` and an entry in
    ``warnings``. Override or extend via a JSON file:
        {
          "gemini-2.5-flash-lite": {"input_per_million": 0.10, "output_per_million": 0.40}
        }
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


DEFAULT_ROOT = Path("/workspace/data/output/persistence_eval_outputs")
DEFAULT_OUTPUT = Path("/workspace/data/out/gemini_costs.json")

# Pricing table is shared with the run-time confirmation prompt in
# eval_persistence.py. Edit the table at the source if you need to bump rates.
# Supply --pricing to override or extend at runtime.
from dataset_generation.llm_persistence_querying.gemini.pricing import (  # noqa: E402
    DEFAULT_PRICING,
)


@dataclass
class _ModelAgg:
    request_count: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    thinking_tokens: int = 0
    total_tokens: int = 0

    def add(self, other: Dict[str, Any], requests: int) -> None:
        self.request_count += int(requests)
        self.prompt_tokens += int(other.get("prompt_tokens", 0))
        self.completion_tokens += int(other.get("completion_tokens", 0))
        self.thinking_tokens += int(other.get("thinking_tokens", 0))
        self.total_tokens += int(other.get("total_tokens", 0))


def _cost_usd(usage: _ModelAgg, rates: Dict[str, float]) -> float:
    """Thinking tokens bill as output tokens per Google's docs."""
    input_m = usage.prompt_tokens / 1_000_000.0
    output_m = (usage.completion_tokens + usage.thinking_tokens) / 1_000_000.0
    return input_m * rates["input_per_million"] + output_m * rates["output_per_million"]


def _model_block(
    agg: _ModelAgg, pricing: Dict[str, Dict[str, float]], model: str,
) -> Dict[str, Any]:
    rates = pricing.get(model)
    return {
        "request_count": agg.request_count,
        "prompt_tokens": agg.prompt_tokens,
        "completion_tokens": agg.completion_tokens,
        "thinking_tokens": agg.thinking_tokens,
        "total_tokens": agg.total_tokens,
        "estimated_cost_usd": round(_cost_usd(agg, rates), 6) if rates else None,
        "priced": rates is not None,
    }


def _extract_run_metadata(snapshot_path: Path, root: Path) -> Dict[str, Optional[str]]:
    """Best-effort dataset / inferencer / timestamp from the snapshot path.

    Expected layout:
        <root>/<dataset>/<inferencer>/<benchmark>/inference_runs/<timestamp>/gemini_token_usage.json
    Anything that doesn't fit returns None for the missing field.
    """
    try:
        rel = snapshot_path.parent.relative_to(root)
    except ValueError:
        return {"dataset": None, "inferencer": None, "run_timestamp": None}
    parts = rel.parts
    # parts[-2] == "inference_runs", parts[-1] == <timestamp>
    run_timestamp = parts[-1] if len(parts) >= 1 else None
    dataset = parts[0] if len(parts) >= 1 else None
    inferencer = parts[1] if len(parts) >= 2 else None
    return {
        "dataset": dataset,
        "inferencer": inferencer,
        "run_timestamp": run_timestamp,
    }


def _load_pricing_override(path: Path) -> Dict[str, Dict[str, float]]:
    with path.open("r", encoding="utf-8") as f:
        loaded = json.load(f)
    if not isinstance(loaded, dict):
        raise ValueError(f"pricing override {path} must be a JSON object")
    for model, rates in loaded.items():
        if not isinstance(rates, dict):
            raise ValueError(f"pricing entry for {model!r} must be an object")
        for k in ("input_per_million", "output_per_million"):
            if k not in rates:
                raise ValueError(f"pricing entry for {model!r} missing {k!r}")
    return loaded


def collect(
    root: Path,
    *,
    pricing: Dict[str, Dict[str, float]],
) -> Dict[str, Any]:
    snapshots = sorted(root.rglob("gemini_token_usage.json"))
    runs: List[Dict[str, Any]] = []
    grand_total = _ModelAgg()
    by_model: Dict[str, _ModelAgg] = {}
    unknown_model_files: Dict[str, List[str]] = {}
    warnings: List[str] = []

    for snap in snapshots:
        try:
            with snap.open("r", encoding="utf-8") as f:
                blob = json.load(f)
        except (OSError, json.JSONDecodeError) as e:
            warnings.append(f"failed to read {snap}: {e}")
            continue

        per_model_blob = blob.get("cumulative_usage_by_model") or {}
        per_model_counts = blob.get("request_count_by_model") or {}
        if not per_model_blob:
            # Snapshot exists but no requests recorded (e.g. all-cache-hit run).
            # Still emit a row so the user sees the run was visited.
            runs.append({
                "path": str(snap.parent),
                "snapshot_timestamp": blob.get("timestamp"),
                **_extract_run_metadata(snap, root),
                "request_count": 0,
                "by_model": {},
                "estimated_cost_usd": 0.0,
            })
            continue

        run_by_model: Dict[str, Dict[str, Any]] = {}
        run_total_cost = 0.0
        run_request_count = 0
        for model, usage_blob in per_model_blob.items():
            req_count = int(per_model_counts.get(model, 0))
            run_request_count += req_count
            agg = _ModelAgg()
            agg.add(usage_blob, req_count)
            block = _model_block(agg, pricing, model)
            run_by_model[model] = block
            if block["estimated_cost_usd"] is not None:
                run_total_cost += block["estimated_cost_usd"]
            else:
                unknown_model_files.setdefault(model, []).append(str(snap))

            grand_total.add(usage_blob, req_count)
            by_model.setdefault(model, _ModelAgg()).add(usage_blob, req_count)

        runs.append({
            "path": str(snap.parent),
            "snapshot_timestamp": blob.get("timestamp"),
            **_extract_run_metadata(snap, root),
            "request_count": run_request_count,
            "by_model": run_by_model,
            "estimated_cost_usd": round(run_total_cost, 6),
        })

    # Sort runs by snapshot_timestamp (fallback: path) so the JSON is stable
    # across invocations.
    runs.sort(key=lambda r: (r.get("snapshot_timestamp") or "", r["path"]))

    totals_block = {
        "request_count": grand_total.request_count,
        "prompt_tokens": grand_total.prompt_tokens,
        "completion_tokens": grand_total.completion_tokens,
        "thinking_tokens": grand_total.thinking_tokens,
        "total_tokens": grand_total.total_tokens,
        "estimated_cost_usd": round(
            sum(
                _cost_usd(agg, pricing[m])
                for m, agg in by_model.items()
                if m in pricing
            ),
            6,
        ),
        "by_model": {
            model: _model_block(agg, pricing, model)
            for model, agg in sorted(by_model.items())
        },
    }

    for model, files in sorted(unknown_model_files.items()):
        warnings.append(
            f"unknown model {model!r} in {len(files)} snapshot(s); "
            f"reported in tokens only (override via --pricing). "
            f"Example: {files[0]}"
        )

    return {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "root": str(root),
        "pricing": pricing,
        "totals": totals_block,
        "runs": runs,
        "warnings": warnings,
    }


def _fmt_int(n: int) -> str:
    return f"{n:,}"


def _fmt_cost(c: Optional[float]) -> str:
    return f"${c:,.4f}" if c is not None else "—"


def _truncate(s: Optional[str], width: int) -> str:
    if s is None:
        return ""
    return s if len(s) <= width else s[: max(1, width - 1)] + "…"


def _print_summary(report: Dict[str, Any]) -> None:
    """Compact human-readable rollup. Mirrors the JSON file content with
    less precision and truncated paths so it fits in a terminal.
    """
    totals = report["totals"]
    runs = report["runs"]
    warnings = report["warnings"]

    print(f"Gemini token-usage report  (generated {report['generated_at']})")
    print(f"  scanned root: {report['root']}")
    print(f"  runs found:   {len(runs)}")
    print()

    print("GRAND TOTAL")
    print(f"  requests:        {_fmt_int(totals['request_count'])}")
    print(f"  prompt tokens:   {_fmt_int(totals['prompt_tokens'])}")
    print(f"  completion:      {_fmt_int(totals['completion_tokens'])}")
    print(f"  thinking:        {_fmt_int(totals['thinking_tokens'])}")
    print(f"  total:           {_fmt_int(totals['total_tokens'])}")
    print(f"  estimated cost:  {_fmt_cost(totals['estimated_cost_usd'])}")
    print()

    by_model = totals.get("by_model") or {}
    if by_model:
        header = f"  {'model':<28} {'reqs':>6} {'prompt':>10} {'compl.':>9} {'think':>9} {'total':>10} {'cost':>10}  priced"
        print("BY MODEL")
        print(header)
        print("  " + "-" * (len(header) - 2))
        for model, m in sorted(by_model.items()):
            print(
                f"  {_truncate(model, 28):<28} "
                f"{_fmt_int(m['request_count']):>6} "
                f"{_fmt_int(m['prompt_tokens']):>10} "
                f"{_fmt_int(m['completion_tokens']):>9} "
                f"{_fmt_int(m['thinking_tokens']):>9} "
                f"{_fmt_int(m['total_tokens']):>10} "
                f"{_fmt_cost(m['estimated_cost_usd']):>10}  "
                f"{'yes' if m['priced'] else 'NO '}"
            )
        print()

    if runs:
        header = f"  {'run timestamp':<17} {'dataset':<32} {'inferencer':<40} {'reqs':>6} {'cost':>10}"
        print("BY RUN")
        print(header)
        print("  " + "-" * (len(header) - 2))
        for r in runs:
            print(
                f"  {_truncate(r.get('run_timestamp'), 17):<17} "
                f"{_truncate(r.get('dataset'), 32):<32} "
                f"{_truncate(r.get('inferencer'), 40):<40} "
                f"{_fmt_int(r['request_count']):>6} "
                f"{_fmt_cost(r['estimated_cost_usd']):>10}"
            )
        print()

    if warnings:
        print("WARNINGS")
        for w in warnings:
            print(f"  - {w}")
        print()


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    p.add_argument(
        "--root",
        type=Path,
        default=DEFAULT_ROOT,
        help=f"Directory to scan recursively. Default: {DEFAULT_ROOT}",
    )
    p.add_argument(
        "--pricing",
        type=Path,
        default=None,
        help="JSON file with rate overrides. Merged on top of the built-in table.",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"JSON output file (overwritten). Default: {DEFAULT_OUTPUT}",
    )
    p.add_argument(
        "--no-save",
        action="store_true",
        help="Skip writing the JSON file; only print the stdout summary.",
    )
    args = p.parse_args()

    if not args.root.exists():
        print(f"ERROR: root does not exist: {args.root}", file=sys.stderr)
        return 2

    pricing = dict(DEFAULT_PRICING)
    if args.pricing is not None:
        try:
            override = _load_pricing_override(args.pricing)
        except (OSError, ValueError, json.JSONDecodeError) as e:
            print(f"ERROR: bad --pricing file: {e}", file=sys.stderr)
            return 2
        pricing.update(override)

    report = collect(args.root, pricing=pricing)

    if not args.no_save:
        payload = json.dumps(report, indent=2, sort_keys=False)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n", encoding="utf-8")
        report["_saved_to"] = str(args.output)

    _print_summary(report)
    if not args.no_save:
        print(f"full JSON report written to {args.output}")

    return 0


if __name__ == "__main__":
    sys.exit(main())

"""
Cost estimator: takes the same Hydra args as ``eval_persistence.py`` and
prints a Gemini cost projection for the resolved config — without
constructing any engines or making any API calls.

Usage (same flags as eval_persistence.py):
    python3 estimate_gemini_cost.py --config-name=eval_config_llm_pipeline_allGemini_cache_smoketest
    python3 estimate_gemini_cost.py --config-name=...  benchmark_dataset.max_anchor_observations=500
    python3 estimate_gemini_cost.py --config-name=...  inferencer/sceneCaptioner=gemini_cap_2_5_pro_proc_long7

Reports:
  - Every Gemini component detected in the resolved cfg (role + model)
  - Per-anchor cost at K=1 (low) and K=3 (high) — passes from the all-ER smoketest
  - Total run cost range if the dataset advertises ``max_anchor_observations``
  - 50% Batch API discount preview

"""
from __future__ import annotations

from typing import List, Optional

import hydra
from omegaconf import DictConfig

import persistence_eval.utils.hydra_resolvers  # noqa: F401  - registers ${dot_to_p:...} etc.

from dataset_generation.llm_persistence_querying.gemini.pricing import (
    GeminiComponent,
    format_projection,
    project_run_cost,
)


# Parent-key → semantic role for the Hydra inferencer subtree. Anything not
# in this map gets role="unknown" (still detected, but cost projection skips
# its phase-specific contribution).
_GEMINI_ROLE_BY_PARENT = {
    "sceneCaptioner":  "captioner",
    "objectDescriber": "describer",
    "llmQuerier":      "querier",
}


def _detect_gemini_components(cfg: DictConfig) -> List[GeminiComponent]:
    """Walk ``cfg.inferencer`` for any ``_target_`` paths under the
    ``dataset_generation....gemini...`` Python package and return a list
    describing the role, target class, and model name for each.
    """
    if not isinstance(cfg, DictConfig) or "inferencer" not in cfg:
        return []
    found: List[GeminiComponent] = []

    def visit(node, parent_key: Optional[str] = None) -> None:
        if not hasattr(node, "items"):
            return
        target = node.get("_target_") if hasattr(node, "get") else None
        if isinstance(target, str) and ".gemini" in target.lower():
            role = _GEMINI_ROLE_BY_PARENT.get(parent_key or "", "unknown")
            model = node.get("model_name") if hasattr(node, "get") else None
            found.append(GeminiComponent(
                role=role, target=target, model=str(model or "unknown"),
            ))
        for k, v in node.items():
            visit(v, parent_key=k)

    visit(cfg.inferencer)
    return found


@hydra.main(config_path="configs/train_confs", config_name="eval_config", version_base=None)
def main(cfg: DictConfig) -> None:
    components = _detect_gemini_components(cfg)
    if not components:
        print("No Gemini components detected in this config. Nothing to estimate.")
        return

    # Best-effort anchor count from the resolved benchmark_dataset config
    num_anchors: Optional[int] = None
    try:
        n = cfg.benchmark_dataset.get("max_anchor_observations")
        if n is not None:
            num_anchors = int(n)
    except Exception:
        pass

    projection = project_run_cost(components, num_anchors=num_anchors)

    banner = "=" * 70
    print(banner)
    print(" GEMINI COST PROJECTION (no API calls made)")
    print(banner)
    print(format_projection(projection))
    print(banner)


if __name__ == "__main__":
    main()

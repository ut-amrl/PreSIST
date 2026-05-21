"""Replay only the LLM querier for a single failing anchor.

Reads the per-anchor debug JSON produced by LLMPipelineInferencer
(captioning + per-object descriptions are already captured there) and
re-issues just the querier call against the live LLM server, so we can
inspect the raw response without rerunning captioning/description.

Usage from inside the spf_plus_datasets container:
    python scripts/debug_replay_querier_for_anchor.py \\
        --debug-json /workspace/data/output/persistence_eval_outputs/.../debug_vis/debug_1353440383.000000.json \\
        --config-name pucpr_eval_config_llm_split_50_20_30_gemma_v3iqrBehav4_mO20_percentile_balanced
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import hydra
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate

import persistence_eval.utils.hydra_resolvers  # noqa: F401  # registers resolvers


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--debug-json", required=True, type=Path)
    ap.add_argument(
        "--config-name",
        default="pucpr_eval_config_llm_split_50_20_30_gemma_v3iqrBehav4_mO20_percentile_balanced",
    )
    ap.add_argument(
        "--config-dir",
        default=str(Path(__file__).resolve().parent.parent / "configs" / "train_confs"),
    )
    ap.add_argument(
        "--single-batch",
        action="store_true",
        help="Force maxObjsPerQuery=None so all objects go in one prompt "
        "(avoids needing a real SegmentationResults for spatial splitting).",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=None,
        help=(
            "File to dump replay result JSON. If omitted, writes to "
            "/workspace/data/output/debug_replays/replay_<anchor_time>.json "
            "(visible on the host at "
            "/scratch/aaadkins/data/semantic-persistence-filter/output/debug_replays/...)."
        ),
    )
    args = ap.parse_args()

    with args.debug_json.open() as f:
        debug = json.load(f)

    if args.out is None:
        anchor_tag = f"{debug['anchor_time']:.6f}" if isinstance(debug.get("anchor_time"), (int, float)) else "unknown"
        args.out = Path(f"/workspace/data/output/debug_replays/replay_{anchor_tag}.json")

    scene_caption = debug["scene_caption"]
    # JSON keys are strings; the querier expects int seg-ids when used in the
    # spatial-grouping path, but for the LLM-only path keys are stringified
    # again inside divideIntoObjectGroups, so int keys round-trip fine.
    object_descriptions = {int(k): v for k, v in debug["object_descriptions"].items()}
    image_id = debug.get("image_path") or f"anchor_t{debug['anchor_time']}"

    with initialize_config_dir(config_dir=args.config_dir, version_base=None):
        cfg = compose(config_name=args.config_name)

    querier_cfg = cfg.inferencer.llmPersistenceQuerier
    if args.single_batch:
        # Override so we don't need real instances for spatial splitting.
        from omegaconf import OmegaConf, open_dict
        OmegaConf.set_struct(querier_cfg, False)
        with open_dict(querier_cfg):
            querier_cfg.maxObjsPerQuery = None
            querier_cfg.useSpatialGrouping = False

    querier = instantiate(querier_cfg)

    # SegmentationResults is unused when maxObjsPerQuery=None or
    # len(object_descriptions) <= maxObjsPerQuery. For the failing anchor
    # (30 objs, maxObjsPerQuery=20) we DO need real instances unless
    # --single-batch is passed. We don't have them here, so require the flag.
    if not args.single_batch and len(object_descriptions) > int(querier_cfg.get("maxObjsPerQuery", 0) or 0):
        print(
            "ERROR: this anchor has more objects than maxObjsPerQuery and the "
            "querier needs real SegmentationResults for spatial splitting. "
            "Re-run with --single-batch to merge everything into one prompt.",
            file=sys.stderr,
        )
        sys.exit(2)

    prompts, response, raw_estimates, debug_payload = querier.estimatePersistenceParameters(
        scene_caption,
        object_descriptions,
        {},
        None,
        None,  # image — unused on the LLM-only path
        None,  # instances — unused when single-batch path is taken
        image_id,
    )

    print("=" * 80)
    print(f"image_id: {image_id}")
    print(f"n_objects: {len(object_descriptions)}")
    print(f"n_raw_estimates: {len(raw_estimates)}")
    print("-" * 80)
    print("RAW LLM RESPONSE(S):")
    print(response if isinstance(response, str) else "\n\n---\n\n".join(response))
    print("-" * 80)
    print("PARSED raw_estimates:")
    print(json.dumps(raw_estimates, indent=2, default=str))
    print("=" * 80)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(
            {
                "image_id": image_id,
                "prompts": list(prompts),
                "response": response,
                "raw_estimates": raw_estimates,
            },
            indent=2,
            default=str,
        )
    )
    print(f"Wrote replay result to {args.out}")


if __name__ == "__main__":
    main()

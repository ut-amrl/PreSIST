"""
One-time script to query a VLM (Qwen3-VL) for persistence parameters for each semantic class
in a closed set. Results are saved as a SemanticClassPersistenceParamLookup JSON file.

The prompt template file must contain a {class_name} placeholder.
The VLM response is expected to be (or contain) a JSON object with the keys listed in --param_keys.

Usage example:
    python -m persistence_eval.methods.inferencers.closed_set_semantics.semantic_class_persistence_precomputer \
        --class_list_file path/to/classes.txt \
        --output_file path/to/persistence_params.json \
        --prompt_template_file prompt_files/persistence_precomputer/class_persistence_query.txt \
        --param_keys p50_persistence_hours p90_persistence_hours \
        --output_param_keys t50 t90 \
        --model_name Qwen/Qwen3-VL-8B-Thinking \
        --use_vllm \
        --vllm_port 9001
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from typing import Optional
from tqdm import tqdm

from persistence_eval.core.domain import PersistenceParameterEstimate
from persistence_eval.methods.inferencers.closed_set_semantics.closed_set_semantics import ClosedSetSemanticClassList
from persistence_eval.methods.inferencers.closed_set_semantics.semantic_class_persistence_param_lookup import SemanticClassPersistenceParamLookup


def _strip_think_block(text: str) -> str:
    """Returns text after the last </think> tag, or the full text if none present."""
    idx = text.rfind("</think>")
    if idx != -1:
        return text[idx + len("</think>"):].strip()
    return text


def _extract_json_object(text: str) -> dict:
    """Extract the first JSON object from text. Raises ValueError if none found."""
    # Find the outermost {...} block
    match = re.search(r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)?\}", text, re.DOTALL)
    if match is None:
        # Try a greedier approach for nested objects
        start = text.find("{")
        if start == -1:
            raise ValueError("No JSON object found in response.")
        depth = 0
        for i, ch in enumerate(text[start:], start=start):
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return json.loads(text[start:i + 1])
        raise ValueError("Unbalanced braces in response.")
    return json.loads(match.group())


def _parse_response(
    response: str,
    param_keys: list[str],
    output_param_keys: list[str],
) -> dict[str, float]:
    """Parse LLM response into a dict of output_key → float (stored as-is, in hours)."""
    print("Base LLM Response: ")
    print(response)
    text = _strip_think_block(response)
    print("Raw response: ")
    print(text)
    parsed = _extract_json_object(text)
    print("Raw parsed: ")
    print(parsed)

    params: dict[str, float] = {}
    for src_key, dst_key in zip(param_keys, output_param_keys):
        if src_key not in parsed:
            raise KeyError(f"Expected key '{src_key}' not found in LLM response JSON.")
        params[dst_key] = float(parsed[src_key])
    return params


def run_precomputer(
    class_list_file: str,
    output_file: str,
    prompt_template_file: str,
    param_keys: list[str],
    output_param_keys: list[str],
    model_name: str,
    use_vllm: bool,
    max_tokens: int,
    vllm_port: int,
    force_recompute: bool = False,
    save_every: int = 10,
) -> None:
    from dataset_generation.llm_persistence_querying.llm_querying.qwen_llm_querier import QwenLLMQuerier
    import os

    class_list = ClosedSetSemanticClassList(class_text_file=class_list_file)
    classes = class_list.semantic_classes

    with open(prompt_template_file, "r") as f:
        prompt_template = f.read()

    lookup = SemanticClassPersistenceParamLookup()
    if not force_recompute and os.path.exists(output_file):
        lookup.load_params_from_file(output_file)
        already_done = lookup.id_to_params.keys()
        classes = [c for c in classes if class_list.class_to_id[c] not in already_done]
        print(f"Resuming: {len(classes)} class(es) remaining (skipping {len(already_done)} already computed).")
    else:
        print(f"Querying all {len(classes)} classes.")

    if not classes:
        print("Nothing to do.")
        return

    querier = QwenLLMQuerier(
        name="persistence_precomputer",
        model_name=model_name,
        use_vllm_server=use_vllm,
        vllm_port=vllm_port,
        max_new_tokens=max_tokens,
    )

    failures: list[str] = []
    queries_since_save = 0

    for class_name in tqdm(classes, desc="Processing classes"):
        prompt = prompt_template.format(class_name=class_name)
        print("\n"*7)
        print("Prompt for class: ", class_name)
        print(prompt)
        print("\n"*3)

        try:
            response, _ = querier.queryLLM(prompt)
            params = _parse_response(response, param_keys, output_param_keys)
            print("Params: ")
            print(params)
            estimate = PersistenceParameterEstimate(params=params)
            class_id = class_list.class_to_id[class_name]
            lookup.set_params_for_id(class_id, estimate)
            print(f"OK  [{class_id}] {params}")
            queries_since_save += 1
            if queries_since_save >= save_every:
                lookup.save_params_to_file(output_file)
                print(f"  [checkpoint] {len(lookup.id_to_params)} entries written to {output_file}")
                queries_since_save = 0
        except Exception as e:
            print(f"FAILED: {e}")
            failures.append(class_name)

    lookup.save_params_to_file(output_file)
    print(f"\nSaved {len(lookup.id_to_params)} entries to {output_file}")

    if failures:
        print(f"WARNING: {len(failures)} class(es) failed: {failures}")
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Query Qwen3-VL to precompute persistence parameters for a closed set of semantic classes."
    )
    parser.add_argument("--class_list_file", required=True, help="Text file with one class name per line.")
    parser.add_argument("--output_file", required=True, help="Output JSON file for SemanticClassPersistenceParamLookup.")
    parser.add_argument("--prompt_template_file", required=True, help="Text file with prompt template (use {class_name} placeholder).")
    parser.add_argument("--param_keys", required=True, nargs="+", help="Keys to extract from VLM JSON response (e.g. t50_hours t90_hours).")
    parser.add_argument("--output_param_keys", nargs="+", default=None, help="Rename keys in stored params (defaults to same as --param_keys).")
    parser.add_argument("--model_name", default="Qwen/Qwen3-VL-8B-Thinking")
    parser.add_argument("--use_vllm", action="store_true")
    parser.add_argument("--max_tokens", type=int, default=8192, help="Max tokens for LLM/VLM")
    parser.add_argument("--vllm_port", type=int, default=9001)
    parser.add_argument("--force_recompute", action="store_true", help="Re-query all classes even if output file already exists.")
    parser.add_argument("--save_every", type=int, default=10, help="Write results to disk after every N successful queries.")
    args = parser.parse_args()

    output_param_keys = args.output_param_keys if args.output_param_keys is not None else args.param_keys
    if len(output_param_keys) != len(args.param_keys):
        parser.error("--output_param_keys must have the same number of entries as --param_keys.")

    run_precomputer(
        class_list_file=args.class_list_file,
        output_file=args.output_file,
        prompt_template_file=args.prompt_template_file,
        param_keys=args.param_keys,
        output_param_keys=output_param_keys,
        model_name=args.model_name,
        use_vllm=args.use_vllm,
        vllm_port=args.vllm_port,
        max_tokens=args.max_tokens,
        force_recompute=args.force_recompute,
        save_every=args.save_every,
    )

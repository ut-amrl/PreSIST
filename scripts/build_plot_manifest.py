"""Build a `label,path` manifest CSV for plot_horizon_metrics.py from the
sweep_metrics.csv produced by sweep_train_eval.sh.

Filter cells by --variant / --grouping / --fraction (each repeatable). The
emitted `label` is "<variant>_<grouping>_frac<F>" by default; override with
--label-template using {variant}, {grouping}, {fraction}, {seed} placeholders.

Examples:

  # All training fractions of single_lognormal_class, labelled by fraction:
  python scripts/build_plot_manifest.py \
    --csv /workspace/data/output/perpetua/sweep_metrics.csv \
    --variant single_lognormal --grouping class \
    --label-template 'frac={fraction}' \
    --out /tmp/lognormal_class_by_fraction.csv

  # All four variants at fraction=1.0:
  python scripts/build_plot_manifest.py \
    --csv /workspace/data/output/perpetua/sweep_metrics.csv \
    --fraction 1.0 \
    --label-template '{variant}_{grouping}' \
    --out /tmp/all_variants_at_full.csv

Then feed to the plotter:
  python plot_horizon_metrics.py --manifest /tmp/lognormal_class_by_fraction.csv \
    --x-units hours --show-base-rate
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--csv", type=Path, required=True,
                   help="sweep_metrics.csv path")
    p.add_argument("--variant", action="append", default=[],
                   help="Filter by variant (repeatable). Match is exact.")
    p.add_argument("--grouping", action="append", default=[],
                   help="Filter by grouping (class|location, repeatable).")
    p.add_argument("--fraction", action="append", default=[],
                   help="Filter by fraction (repeatable). Compared as float.")
    p.add_argument("--seed", action="append", default=[],
                   help="Filter by seed (repeatable). Compared as int.")
    p.add_argument("--label-template", default="{variant}_{grouping}_frac{fraction}",
                   help="Format string with {variant}/{grouping}/{fraction}/{seed} "
                        "placeholders. Default: '{variant}_{grouping}_frac{fraction}'.")
    p.add_argument("--out", type=Path, default=None,
                   help="Output manifest path. Default: stdout.")
    p.add_argument("--sort-by", action="append", default=[],
                   choices=["variant", "grouping", "fraction", "seed", "label"],
                   help="Sort manifest rows by these fields (repeatable, in order).")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if not args.csv.is_file():
        print(f"ERROR: {args.csv} not found", file=sys.stderr)
        return 2

    variant_filter = set(args.variant)
    grouping_filter = set(args.grouping)
    fraction_filter = {float(f) for f in args.fraction}
    seed_filter = {int(s) for s in args.seed}

    rows_out: list[dict[str, str]] = []
    with args.csv.open() as f:
        reader = csv.DictReader(f)
        for r in reader:
            if variant_filter and r["variant"] not in variant_filter:
                continue
            if grouping_filter and r["grouping"] not in grouping_filter:
                continue
            if fraction_filter and float(r["fraction"]) not in fraction_filter:
                continue
            if seed_filter and int(r["seed"]) not in seed_filter:
                continue
            eval_dir_metric = r.get("eval_dir", "")
            if not eval_dir_metric or eval_dir_metric == "MISSING":
                print(f"WARN: skipping {r['variant']}/{r['grouping']}/frac={r['fraction']}: "
                      f"missing eval_dir", file=sys.stderr)
                continue
            # eval_dir in the sweep CSV is the path to metric_results.json;
            # plot_horizon_metrics.py wants the *directory* containing
            # prediction_records.json (same dir).
            eval_dir = str(Path(eval_dir_metric).parent)
            label = args.label_template.format(
                variant=r["variant"],
                grouping=r["grouping"],
                fraction=r["fraction"],
                seed=r["seed"],
            )
            rows_out.append({"label": label, "path": eval_dir,
                             "_variant": r["variant"], "_grouping": r["grouping"],
                             "_fraction": float(r["fraction"]), "_seed": int(r["seed"])})

    if args.sort_by:
        keymap = {"variant": "_variant", "grouping": "_grouping",
                  "fraction": "_fraction", "seed": "_seed", "label": "label"}
        rows_out.sort(key=lambda r: tuple(r[keymap[k]] for k in args.sort_by))
    if not rows_out:
        print("ERROR: no rows matched filters", file=sys.stderr)
        return 1

    sink = args.out.open("w", newline="") if args.out else sys.stdout
    try:
        writer = csv.writer(sink)
        writer.writerow(["label", "path"])
        for r in rows_out:
            writer.writerow([r["label"], r["path"]])
    finally:
        if args.out:
            sink.close()

    if args.out:
        print(f"Wrote {len(rows_out)} rows to {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())

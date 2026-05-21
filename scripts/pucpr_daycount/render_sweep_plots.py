"""Generic day-count sweep plotter.

Renders bAcc / Accuracy / F1 / MAE PDFs for one (scenario, grouping-mode)
combination. Driven by a CSV manifest specifying the perpetua CSV and the
vision+LLM CSV for the chosen scenario.

Usage:
  python render_daycount_sweep.py \
    --perp-csv <path> --vl-csv <path> \
    --grouping {both,class,location} \
    --out-dir <dir>

Plot styling:
  - PDF output
  - No title
  - Legend: 2-column at the bottom of the figure
  - Linear x-scale (training day count)
  - Method labels: Exponential, Log-Normal, Exponential Mixture,
    Log-Normal Mixture, Vision only, PerSIST-Lang
  - One marker per method; class -> solid, location -> dashed; all lines
    differently coloured.
  - In "both" mode the per-method curves are suffixed " (Global)" for class
    and " (Location)" for location; in single-grouping mode no suffix.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker


METHODS = [
    # (csv_variant_token, label, marker)
    ("single_exp",        "Exponential",          "o"),
    ("single_lognormal",  "Log-Normal",           "s"),
    ("mixture_exp",       "Exponential Mixture",  "^"),
    ("mixture_lognormal", "Log-Normal Mixture",   "D"),
]
GROUPING_LINESTYLE = {"class": "-", "location": "--"}

# Distinct colour per (method, grouping). Tab10-ish, with a paired light variant
# for the secondary grouping so visually-related methods stay close in hue.
COLORS = {
    # Per-method colour pairs: "class" gets the standard tab10-style hue,
    # "location" gets a BRIGHTER (more saturated / vivid) variant of the same
    # hue family — same colour identity, more punch on the page.
    ("single_exp",        "class"):    "#8c564b",  # brown
    ("single_exp",        "location"): "#deb887",  # tan
    ("single_lognormal",  "class"):    "#2ca02c",  # green
    ("single_lognormal",  "location"): "#3ddc3d",  # bright (but less highlighter-y) green
    ("mixture_exp",       "class"):    "#d62728",  # red
    ("mixture_exp",       "location"): "#ff5252",  # bright coral
    ("mixture_lognormal", "class"):    "#9467bd",  # purple
    ("mixture_lognormal", "location"): "#c266ff",  # bright violet
}
VISION_COLOR = "#ff9800"   # orange
LLM_COLOR    = "#17becf"   # teal
VISION_MARKER = "*"
LLM_MARKER    = "X"

METRICS = [
    # (csv field, y label, output basename, lower_is_better)
    ("balanced_accuracy@0.5", "Balanced Accuracy @ 0.5", "bacc",     False),
    ("accuracy@0.5",          "Accuracy @ 0.5",          "accuracy", False),
    ("f1@0.5",                "F1 @ 0.5",                "f1",       False),
    ("existence_mae",         "MAE",                     "mae",      True),
]
# Day column order for the LaTeX table; intersected with whatever the CSVs hold.
DAYS_ORDER = [0.5, 1, 2, 3, 4, 5, 7, 10, 14, 21, 28, 35]


def load_perp(path: Path, metric: str) -> dict[tuple[str, str], list[tuple[float, float]]]:
    out: dict[tuple[str, str], dict[float, float]] = {}
    if not path.exists():
        return {}
    with open(path) as f:
        for r in csv.DictReader(f):
            try:
                d = float(r["days"]); v = float(r[metric])
            except (ValueError, KeyError):
                continue
            out.setdefault((r["variant"], r["grouping"]), {})[d] = v
    return {k: sorted(v.items()) for k, v in out.items()}


def load_vl(path: Path, metric: str):
    vision: dict[float, float] = {}
    llm: float | None = None
    if not path.exists():
        return [], None
    with open(path) as f:
        for r in csv.DictReader(f):
            try:
                v = float(r[metric])
            except (ValueError, KeyError):
                continue
            if r["trainer_kind"] == "vision":
                vision[float(r["days"])] = v
            elif r["trainer_kind"] == "llm":
                llm = v
    return sorted(vision.items()), llm


def _fmt_day(d: float) -> str:
    return f"{d:g}"   # "0.5" / "1" / "35" — no trailing zeros


def _fmt_val(v: float | None) -> str:
    return "--" if v is None else f"{v:.3f}"


def write_latex_table(
    perp_csv: Path, vl_csv: Path, grouping_mode: str, out_dir: Path,
    metric: str, ylabel: str, out_basename: str, lower_is_better: bool,
) -> None:
    """Per-metric LaTeX table: methods as rows, days as columns. PerSIST-Lang
    spans all day columns via \\multicolumn. Best per column is \\bfseries."""
    perp = load_perp(perp_csv, metric)
    vision, llm = load_vl(vl_csv, metric)

    # Determine which days actually appear across all method rows so the
    # header reflects available data (and the multicolumn span is correct).
    days_present: set[float] = set()
    for series in perp.values():
        days_present.update(d for d, _ in series)
    days_present.update(d for d, _ in vision)
    days = [d for d in DAYS_ORDER if d in days_present]
    if not days:
        return

    if grouping_mode == "both":
        groupings = ("class", "location"); suffix = {"class": " (Global)", "location": " (Location)"}
    elif grouping_mode == "class":
        groupings = ("class",); suffix = {"class": ""}
    else:
        groupings = ("location",); suffix = {"location": ""}

    # Assemble row data so we can bold the best in each column.
    rows: list[tuple[str, dict[float, float]]] = []
    for variant_token, label, _ in METHODS:
        for grouping in groupings:
            series = dict(perp.get((variant_token, grouping), []))
            if not series:
                continue
            rows.append((f"{label}{suffix[grouping]}", series))
    if vision:
        rows.append(("Vision only", dict(vision)))

    # Best per column among the per-D rows; PerSIST-Lang competes too via its
    # single value being valid for every D.
    best_per_d: dict[float, float] = {}
    for d in days:
        col = [s[d] for _, s in rows if d in s]
        if llm is not None:
            col.append(llm)
        if not col:
            continue
        best_per_d[d] = min(col) if lower_is_better else max(col)

    def render_cell(v: float | None, d: float) -> str:
        if v is None:
            return "--"
        is_best = d in best_per_d and abs(v - best_per_d[d]) < 1e-9
        s = f"{v:.3f}"
        return f"\\bfseries {s}" if is_best else s

    n_days = len(days)
    col_spec = "l|" + "c" * n_days
    lines: list[str] = []
    lines.append(f"% {ylabel} | scenario CSVs: {perp_csv.name}, {vl_csv.name}")
    lines.append(f"% grouping_mode={grouping_mode}; bold = best in column "
                 f"({'min' if lower_is_better else 'max'}); PerSIST-Lang spans all D.")
    lines.append("\\begin{tabular}{" + col_spec + "}")
    lines.append("\\hline")
    header = "Method & " + " & ".join(f"D={_fmt_day(d)}" for d in days) + " \\\\"
    lines.append(header)
    lines.append("\\hline")
    for label, series in rows:
        cells = [render_cell(series.get(d), d) for d in days]
        lines.append(f"{label} & " + " & ".join(cells) + " \\\\")
    if llm is not None:
        llm_is_best_anywhere = any(
            d in best_per_d and abs(llm - best_per_d[d]) < 1e-9 for d in days
        )
        llm_str = f"{llm:.3f}"
        if llm_is_best_anywhere:
            llm_str = f"\\bfseries {llm_str}"
        lines.append(
            "PerSIST-Lang & "
            f"\\multicolumn{{{n_days}}}{{c}}{{{llm_str}}} \\\\"
        )
    lines.append("\\hline")
    lines.append("\\end{tabular}")

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{out_basename}.tex"
    out_path.write_text("\n".join(lines) + "\n")
    print(f"wrote {out_path}")


def render_one(
    perp_csv: Path, vl_csv: Path, grouping_mode: str, out_dir: Path,
    metric: str, ylabel: str, out_name: str, lower_is_better: bool,
) -> None:
    perp = load_perp(perp_csv, metric)
    vision, llm = load_vl(vl_csv, metric)

    if grouping_mode == "both":
        groupings = ("class", "location")
        suffix = {"class": " (G)", "location": " (L)"}
    elif grouping_mode == "class":
        groupings = ("class",)
        suffix = {"class": ""}
    elif grouping_mode == "location":
        groupings = ("location",)
        suffix = {"location": ""}
    else:
        raise ValueError(f"grouping_mode must be both/class/location; got {grouping_mode!r}")

    fig, ax = plt.subplots(figsize=(10.5, 5.5))
    # Track plotted (handle, label) entries grouped by which legend column they
    # belong to: column 1 = single-component methods (Exponential, Log-Normal),
    # column 2 = multi-component (Exponential Mixture, Log-Normal Mixture),
    # column 3 = Vision only + PerSIST-Lang.
    SINGLE_TOKENS = {"single_exp", "single_lognormal"}
    MIXTURE_TOKENS = {"mixture_exp", "mixture_lognormal"}
    single_handles: list[tuple] = []
    mixture_handles: list[tuple] = []
    for variant_token, label, marker in METHODS:
        for grouping in groupings:
            pts = perp.get((variant_token, grouping), [])
            if not pts:
                continue
            xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
            (line,) = ax.plot(
                xs, ys,
                color=COLORS[(variant_token, grouping)],
                linestyle=GROUPING_LINESTYLE[grouping],
                marker=marker, markersize=6, linewidth=1.4,
                label=f"{label}{suffix[grouping]}",
                alpha=0.95,
            )
            entry = (line, f"{label}{suffix[grouping]}")
            if variant_token in SINGLE_TOKENS:
                single_handles.append(entry)
            elif variant_token in MIXTURE_TOKENS:
                mixture_handles.append(entry)

    extra_handles: list[tuple] = []
    if vision:
        (line,) = ax.plot(
            [p[0] for p in vision], [p[1] for p in vision],
            color=VISION_COLOR, marker=VISION_MARKER, linestyle="-",
            markersize=9, linewidth=1.5,
            label="Vision only",
        )
        extra_handles.append((line, "Vision only"))
    if llm is not None:
        hline = ax.axhline(
            llm, color=LLM_COLOR, linestyle="-.", linewidth=1.5,
            label=f"PerSIST-Lang ({llm:.3f})",
        )
        extra_handles.append((hline, f"PerSIST-Lang ({llm:.3f})"))

    ticks = [0.5, 1, 2, 3, 4, 5, 7, 10, 14, 21, 28, 35]
    ax.set_xticks(ticks)
    ax.set_xlim(0, 36)
    ax.get_xaxis().set_major_formatter(mticker.ScalarFormatter())
    # Rotate labels so the dense low-D cluster (0.5/1/2/3/4/5) doesn't overlap
    # at the linear scale we use here.
    for lbl in ax.get_xticklabels():
        lbl.set_rotation(45)
        lbl.set_ha("right")
        lbl.set_rotation_mode("anchor")
        lbl.set_fontsize(8)
    for lbl in ax.get_yticklabels():
        lbl.set_fontsize(8)
    ax.set_xlabel("Training Data Amount (Days)", fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.grid(True, which="both", alpha=0.3)

    # 3-column legend below the axes, column-major fill:
    #   col 1 — single-component (Exponential, Log-Normal) × groupings
    #   col 2 — multi-component  (Exponential Mixture, Log-Normal Mixture) × groupings
    #   col 3 — Vision only + PerSIST-Lang
    columns: list[list[tuple]] = []
    if single_handles:  columns.append(single_handles)
    if mixture_handles: columns.append(mixture_handles)
    if extra_handles:   columns.append(extra_handles)
    # Matplotlib fills column-major in row-major-of-the-flattened-list order;
    # to lay out N columns of (possibly unequal) length L_max, we pad shorter
    # columns with invisible (handle, label) entries so each column has the
    # same height. The legend then renders with ncol=len(columns).
    if columns:
        ncol = len(columns)
        max_len = max(len(c) for c in columns)
        # Matplotlib's legend fills column-major when given `ncol`, so to get
        # column c at visual position c we just concatenate columns in order
        # and pad shorter ones with blanks.
        blank = (plt.Line2D([], [], linestyle="none", marker=""), "")
        flat_handles: list = []
        flat_labels: list[str] = []
        for col in columns:
            for h, l in col:
                flat_handles.append(h); flat_labels.append(l)
            for _ in range(max_len - len(col)):
                flat_handles.append(blank[0]); flat_labels.append(blank[1])
        ax.legend(
            flat_handles, flat_labels,
            loc="upper center", bbox_to_anchor=(0.5, -0.12),
            ncol=ncol, fontsize=9, framealpha=0.92,
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{out_name}.pdf"
    fig.savefig(out_path, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--perp-csv", type=Path, required=True)
    ap.add_argument("--vl-csv",   type=Path, required=True)
    ap.add_argument("--grouping", choices=("both", "class", "location"), required=True)
    ap.add_argument("--out-dir",  type=Path, required=True)
    args = ap.parse_args()

    for metric, ylabel, out_basename, lower_is_better in METRICS:
        render_one(
            args.perp_csv, args.vl_csv, args.grouping, args.out_dir,
            metric, ylabel, out_basename, lower_is_better,
        )
        write_latex_table(
            args.perp_csv, args.vl_csv, args.grouping, args.out_dir,
            metric, ylabel, out_basename, lower_is_better,
        )


if __name__ == "__main__":
    main()

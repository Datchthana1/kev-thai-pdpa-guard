#!/usr/bin/env python
"""Compare evaluation results and write results/comparison.md plus charts.

  .venv/Scripts/python.exe scripts/compare.py                       # all results/eval_*.json
  .venv/Scripts/python.exe scripts/compare.py --systems kev-base,kev-pdpa,openthai

Outputs (in --results-dir, default results/):
  comparison.md, bar_headline.png, bar_by_genre.png,
  risk_confusion_<system>.png, reliability_<system>.png

Works with 1..3 systems and with genres missing from some systems (shown as "—").
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
CANON_ORDER = ["kev-base", "kev-pdpa", "openthai"]
QIDS_ORDER = [
    "is_sensitive",
    "risk_level",
    "primary_category",
    "has_name",
    "has_contact",
    "has_address",
    "has_national_id",
    "has_financial",
    "has_credential",
    "has_health_or_special",
    "has_other_identifier",
]

# (label, path inside result["metrics"], higher_is_better)
HEADLINE_ROWS = [
    ("is_sensitive precision", ("is_sensitive", "precision"), True),
    ("is_sensitive recall", ("is_sensitive", "recall"), True),
    ("is_sensitive F1", ("is_sensitive", "f1"), True),
    ("has_* macro-F1 (8 nouls)", ("has", "macro_f1"), True),
    ("primary_category accuracy", ("primary_category", "accuracy"), True),
    ("primary_category macro-F1", ("primary_category", "macro_f1"), True),
    ("risk_level exact accuracy", ("risk_level", "exact_acc"), True),
    ("risk_level +-1 accuracy", ("risk_level", "within1_acc"), True),
    ("risk_level MAE (argmax)", ("risk_level", "mae_argmax"), False),
    ("risk_level MAE (expected)", ("risk_level", "mae_expected"), False),
    ("Brier (all questions)", ("calibration", "overall", "brier"), False),
    ("ECE (all questions, 15 bins)", ("calibration", "overall", "ece"), False),
    ("latency mean (s)", ("latency", "mean_s"), False),
    ("latency p50 (s)", ("latency", "p50_s"), False),
    ("latency p95 (s)", ("latency", "p95_s"), False),
]
CHART_HEADLINE = [
    ("is_sensitive F1", ("is_sensitive", "f1")),
    ("has_* macro-F1", ("has", "macro_f1")),
    ("primary acc", ("primary_category", "accuracy")),
    ("primary macro-F1", ("primary_category", "macro_f1")),
    ("risk exact", ("risk_level", "exact_acc")),
    ("risk +-1", ("risk_level", "within1_acc")),
]


# --------------------------------------------------------------------------- loading
def load_results(results_dir: Path, names: list[str] | None) -> dict[str, dict]:
    import json

    if names:
        paths = [(n, results_dir / f"eval_{n}.json") for n in names]
    else:
        paths = [(p.stem[len("eval_"):], p) for p in results_dir.glob("eval_*.json")]
    paths.sort(key=lambda t: (CANON_ORDER.index(t[0]) if t[0] in CANON_ORDER else len(CANON_ORDER), t[0]))
    out: dict[str, dict] = {}
    for name, p in paths:
        if not p.exists():
            print(f"warning: {p} not found, skipping", file=sys.stderr)
            continue
        with open(p, encoding="utf-8") as f:
            out[name] = json.load(f)
    return out


def get(d, path):
    cur = d
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            return None
        cur = cur[key]
    return cur


def metric(result: dict, path):
    v = get(result.get("metrics", {}), path)
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return None
    return float(v)


# --------------------------------------------------------------------------- markdown helpers
def fmt(v) -> str:
    return "—" if v is None else f"{v:.3f}"


def best_indices(values: list, higher_is_better: bool) -> set[int]:
    vals = [(i, v) for i, v in enumerate(values) if v is not None]
    if not vals:
        return set()
    target = max(v for _, v in vals) if higher_is_better else min(v for _, v in vals)
    return {i for i, v in vals if abs(v - target) < 1e-12}


def md_table(headers: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    for r in rows:
        lines.append("| " + " | ".join(r) + " |")
    return "\n".join(lines)


def comparison_row(label: str, values: list, higher_is_better: bool | None, formatter=fmt) -> list[str]:
    comparable = sum(v is not None for v in values) >= 2  # with one system there is nothing to be best against
    best = best_indices(values, higher_is_better) if (higher_is_better is not None and comparable) else set()
    cells = []
    for i, v in enumerate(values):
        s = formatter(v)
        if i in best and v is not None:
            s = f"**{s}**"
        cells.append(s)
    return [label] + cells


def sections_md(results: dict[str, dict]) -> str:
    names = list(results)
    out = []
    out.append("# PDPA guard evaluation: comparison\n")
    out.append("Systems: " + ", ".join(f"`{n}`" for n in names) + "\n")
    out.append(
        "Records per system: "
        + ", ".join(f"{n}={results[n].get('n_records')}" for n in names)
        + ". Bold = best per row (higher is better for F1/accuracy/precision/recall; lower for MAE, Brier, ECE, latency). "
        "Brier is the multi-class sum over classes, averaged over questions; ECE uses 15 equal-width bins.\n"
    )

    # headline
    out.append("## Headline metrics\n")
    rows = [["n_records"] + [str(results[n].get("n_records", "—")) for n in names]]
    for label, path, hib in HEADLINE_ROWS:
        rows.append(comparison_row(label, [metric(results[n], path) for n in names], hib))
    out.append(md_table(["metric"] + names, rows) + "\n")

    # per-question accuracy
    qids = []
    for n in names:
        for q in get(results[n], ("metrics", "accuracy_by_qid")) or {}:
            if q not in qids:
                qids.append(q)
    qids.sort(key=lambda q: QIDS_ORDER.index(q) if q in QIDS_ORDER else len(QIDS_ORDER))
    out.append("## Per-question accuracy (every qid)\n")
    rows = []
    for q in qids:
        vals = [get(results[n], ("metrics", "accuracy_by_qid", q, "accuracy")) for n in names]
        vals = [None if v is None else float(v) for v in vals]
        rows.append(comparison_row(q, vals, True))
    out.append(md_table(["qid"] + names, rows) + "\n")

    # genre
    genres = sorted({g for n in names for g in (get(results[n], ("metrics", "by_genre")) or {})})
    out.append("## By genre\n")
    if genres:
        for title, key in [("is_sensitive F1", "is_sensitive_f1"), ("risk_level exact accuracy", "risk_exact_acc")]:
            rows = []
            for g in genres:
                vals = [get(results[n], ("metrics", "by_genre", g, key)) for n in names]
                vals = [None if v is None else float(v) for v in vals]
                ns = [get(results[n], ("metrics", "by_genre", g, "n")) for n in names]
                n_txt = next((str(x) for x in ns if x is not None), "—")
                rows.append(comparison_row(f"{g} (n={n_txt})", vals, True))
            out.append(f"### {title}\n")
            out.append(md_table(["genre"] + names, rows) + "\n")
    else:
        out.append("_No genre information in these results._\n")

    # hard negatives
    out.append("### Hard negatives vs not\n")
    rows = []
    for flag in ("true", "false"):
        for key, label in [("is_sensitive_f1", "is_sensitive F1"), ("risk_exact_acc", "risk exact acc")]:
            vals = [get(results[n], ("metrics", "by_hard_negative", flag, key)) for n in names]
            vals = [None if v is None else float(v) for v in vals]
            rows.append(comparison_row(f"hard_negative={flag}: {label}", vals, True))
    out.append(md_table(["slice"] + names, rows) + "\n")

    # gold risk level
    levels = sorted({int(l) for n in names for l in (get(results[n], ("metrics", "by_gold_risk")) or {})})
    out.append("## By gold risk level\n")
    if levels:
        rows = []
        for lvl in levels:
            cells = [str(lvl)]
            for n in names:
                blk = get(results[n], ("metrics", "by_gold_risk", str(lvl))) or {}
                acc, mae, cnt = blk.get("risk_exact_acc"), blk.get("risk_mae_argmax"), blk.get("n")
                cells.append(f"acc {fmt(acc)} / MAE {fmt(mae)} (n={cnt if cnt is not None else '—'})")
            rows.append(cells)
        out.append(md_table(["gold risk"] + names, rows) + "\n")
    else:
        out.append("_No gold-risk information in these results._\n")

    # calibration
    out.append("## Calibration by question type\n")
    rows = []
    for t in ("noul", "score", "choice", "overall"):
        for kind, label in [("brier", "Brier"), ("ece", "ECE")]:
            vals = []
            for n in names:
                if t == "overall":
                    v = get(results[n], ("metrics", "calibration", "overall", kind))
                else:
                    v = get(results[n], ("metrics", "calibration", "by_type", t, kind))
                vals.append(None if v is None else float(v))
            rows.append(comparison_row(f"{t} {label}", vals, False))
    out.append(md_table(["type / metric"] + names, rows) + "\n")

    out.append("Brier and ECE per question are in `results/eval_<system>.json` under `metrics.calibration.by_qid`.\n")
    return "\n".join(out)


# --------------------------------------------------------------------------- charts
def _colors(n: int):
    cmap = plt.get_cmap("tab10")
    return [cmap(i % 10) for i in range(n)]


def chart_headline(results: dict[str, dict], out: Path) -> None:
    names = list(results)
    labels = [l for l, _ in CHART_HEADLINE]
    x = np.arange(len(labels))
    width = 0.8 / max(len(names), 1)
    fig, ax = plt.subplots(figsize=(max(7, 1.6 * len(labels)), 4.2))
    for i, (name, col) in enumerate(zip(names, _colors(len(names)))):
        vals = [metric(results[name], p) for _, p in CHART_HEADLINE]
        heights = [0.0 if v is None else v for v in vals]
        bars = ax.bar(x + (i - (len(names) - 1) / 2) * width, heights, width, label=name, color=col)
        for b, v in zip(bars, vals):
            if v is None:
                ax.text(b.get_x() + b.get_width() / 2, 0.01, "n/a", ha="center", fontsize=7, rotation=90)
            else:
                ax.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.2f}", ha="center", fontsize=7)
    ax.set_xticks(x, labels, rotation=15, ha="right")
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("score (higher is better)")
    ax.set_title("Headline metrics on the test split")
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


def chart_by_genre(results: dict[str, dict], out: Path) -> None:
    names = list(results)
    genres = sorted({g for n in names for g in (get(results[n], ("metrics", "by_genre")) or {})})
    fig, ax = plt.subplots(figsize=(max(7, 1.2 * max(len(genres), 1) + 2), 4.2))
    if not genres:
        ax.text(0.5, 0.5, "no genre information", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
    else:
        x = np.arange(len(genres))
        width = 0.8 / max(len(names), 1)
        for i, (name, col) in enumerate(zip(names, _colors(len(names)))):
            vals = []
            for g in genres:
                v = get(results[name], ("metrics", "by_genre", g, "is_sensitive_f1"))
                vals.append(np.nan if v is None else float(v))
            ax.bar(x + (i - (len(names) - 1) / 2) * width, vals, width, label=name, color=col)
        ax.set_xticks(x, genres, rotation=30, ha="right", fontsize=8)
        ax.set_ylim(0, 1.1)
        ax.set_ylabel("is_sensitive F1")
        ax.set_title("is_sensitive F1 by genre (missing bars = genre absent for that system)")
        ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


def chart_risk_confusion(name: str, result: dict, out: Path) -> bool:
    conf = get(result, ("metrics", "risk_level", "confusion"))
    if conf is None:
        return False
    m = np.asarray(conf, dtype=float)
    fig, ax = plt.subplots(figsize=(5, 4.5))
    im = ax.imshow(m, cmap="Blues")
    for i in range(m.shape[0]):
        for j in range(m.shape[1]):
            ax.text(j, i, f"{int(m[i, j])}", ha="center", va="center",
                    color="white" if m[i, j] > m.max() / 2 and m.max() > 0 else "black", fontsize=9)
    ax.set_xticks(range(m.shape[1]), [str(i) for i in range(m.shape[1])])
    ax.set_yticks(range(m.shape[0]), [str(i) for i in range(m.shape[0])])
    ax.set_xlabel("predicted risk (argmax)")
    ax.set_ylabel("gold risk")
    ax.set_title(f"risk_level confusion: {name}")
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return True


def chart_reliability(name: str, result: dict, out: Path) -> bool:
    bins = get(result, ("metrics", "calibration", "overall", "bins"))
    if not bins:
        return False
    used = [b for b in bins if b.get("n")]
    fig, ax = plt.subplots(figsize=(5, 4.5))
    ax.plot([0, 1], [0, 1], "--", color="grey", lw=1, label="perfect")
    if used:
        conf = np.array([b["conf"] for b in used], dtype=float)
        acc = np.array([b["acc"] for b in used], dtype=float)
        sizes = np.array([b["n"] for b in used], dtype=float)
        sizes = 20 + 180 * sizes / sizes.max()
        ax.scatter(conf, acc, s=sizes, color="tab:blue", alpha=0.75, edgecolor="k", label="bin (size = count)")
    ece = get(result, ("metrics", "calibration", "overall", "ece"))
    ax.set_xlim(0, 1.02)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("mean confidence (max prob)")
    ax.set_ylabel("accuracy")
    ax.set_title(f"reliability: {name} (ECE={fmt(ece)})")
    ax.legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return True


# --------------------------------------------------------------------------- main
def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results-dir", default=str(REPO / "results"))
    p.add_argument("--systems", default=None, help="comma-separated names, e.g. kev-base,kev-pdpa")
    p.add_argument("--out-md", default=None, help="default: <results-dir>/comparison.md")
    args = p.parse_args(argv)

    results_dir = Path(args.results_dir)
    names = [s.strip() for s in args.systems.split(",") if s.strip()] if args.systems else None
    results = load_results(results_dir, names)
    if not results:
        print(f"no eval results found in {results_dir}", file=sys.stderr)
        return 1
    if len(results) > 3:
        print("warning: more than 3 systems; tables and charts will be crowded", file=sys.stderr)

    results_dir.mkdir(parents=True, exist_ok=True)
    md_path = Path(args.out_md) if args.out_md else results_dir / "comparison.md"
    md_path.write_text(sections_md(results), encoding="utf-8")
    print(f"wrote {md_path}")

    chart_headline(results, results_dir / "bar_headline.png")
    chart_by_genre(results, results_dir / "bar_by_genre.png")
    written = [str(results_dir / "bar_headline.png"), str(results_dir / "bar_by_genre.png")]
    for name, res in results.items():
        if chart_risk_confusion(name, res, results_dir / f"risk_confusion_{name}.png"):
            written.append(str(results_dir / f"risk_confusion_{name}.png"))
        if chart_reliability(name, res, results_dir / f"reliability_{name}.png"):
            written.append(str(results_dir / f"reliability_{name}.png"))
    for w in written:
        print(f"wrote {w}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

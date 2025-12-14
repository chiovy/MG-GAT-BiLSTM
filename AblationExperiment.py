#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Open-source friendly ablation study evaluation script.

This script summarizes and compares ablation experiments based on
pre-computed evaluation_summary.json files produced by the ablation
evaluation scripts.
"""
import sys
from pathlib import Path
from typing import Dict, List
import argparse
import json

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except AttributeError:
        import codecs

        sys.stdout = codecs.getwriter("utf-8")(sys.stdout.buffer, "strict")
        sys.stderr = codecs.getwriter("utf-8")(sys.stderr.buffer, "strict")


plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


SCRIPT_DIR = Path(__file__).parent


DEFAULT_ABLATION_CONFIGS: Dict[str, Dict[str, str]] = {
    "No Attention": {
        "folder": "Ablation_no_attention",
        "results_pattern": "evaluation_no_attention_*",
    },
    "No Spearman": {
        "folder": "Ablation_no_spearman",
        "results_pattern": "evaluation_no_spearman_*",
    },
    "No dCor": {
        "folder": "Ablation-no_dCor",
        "results_pattern": "evaluation_no_dcor_*",
    },
    "No MIC": {
        "folder": "Ablation-no_mic",
        "results_pattern": "evaluation_no_mic_*",
    },
}


def _extract_flight_metrics(summary: Dict) -> List[Dict]:
    if "flight_metrics" in summary and isinstance(summary["flight_metrics"], dict):
        return [m for m in summary["flight_metrics"].values() if isinstance(m, dict)]

    flights: List[Dict] = []
    for value in summary.values():
        if isinstance(value, dict) and "precision" in value and "f1_score" in value:
            flights.append(value)
    return flights


def aggregate_from_summary(path: Path) -> Dict[str, float]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    flight_metrics = _extract_flight_metrics(data)
    if not flight_metrics:
        raise ValueError(f"No flight metrics found in {path}")

    precision_list: List[float] = []
    recall_list: List[float] = []
    f1_list: List[float] = []
    fpr_list: List[float] = []
    roc_list: List[float] = []

    for m in flight_metrics:
        if "precision" in m:
            precision_list.append(float(m["precision"]))
        if "recall" in m:
            recall_list.append(float(m["recall"]))
        if "f1_score" in m:
            f1_list.append(float(m["f1_score"]))
        if "fpr" in m:
            fpr_list.append(float(m["fpr"]))
        if "roc_auc" in m and m["roc_auc"] is not None:
            roc_list.append(float(m["roc_auc"]))

    def _mean_std(xs: List[float]) -> (float, float):
        if not xs:
            return float("nan"), float("nan")
        arr = np.asarray(xs, dtype=float)
        return float(arr.mean()), float(arr.std())

    precision_mean, precision_std = _mean_std(precision_list)
    recall_mean, recall_std = _mean_std(recall_list)
    f1_mean, f1_std = _mean_std(f1_list)
    fpr_mean, fpr_std = _mean_std(fpr_list)
    roc_mean, roc_std = _mean_std(roc_list)

    return {
        "precision": precision_mean,
        "precision_std": precision_std,
        "recall": recall_mean,
        "recall_std": recall_std,
        "f1_score": f1_mean,
        "f1_std": f1_std,
        "fpr": fpr_mean,
        "fpr_std": fpr_std,
        "roc_auc": roc_mean,
        "roc_auc_std": roc_std,
        "num_flights": len(flight_metrics),
    }


def load_latest_ablation_results(
    base_dir: Path, ablation_configs: Dict[str, Dict[str, str]]
) -> Dict[str, Dict[str, float]]:
    results: Dict[str, Dict[str, float]] = {}

    for name, config in ablation_configs.items():
        results_dir = base_dir / config["folder"] / "results"
        if not results_dir.exists():
            print(f"Warning: results directory not found: {results_dir}")
            continue

        eval_dirs = sorted(
            results_dir.glob(config["results_pattern"]), reverse=True
        )
        if not eval_dirs:
            print(f"Warning: no evaluation results for {name}")
            continue

        latest_eval_dir = eval_dirs[0]
        summary_file = latest_eval_dir / "evaluation_summary.json"
        if not summary_file.exists():
            print(f"Warning: summary file not found: {summary_file}")
            continue

        try:
            stats = aggregate_from_summary(summary_file)
            results[name] = stats
            print(
                f"[Loaded] {name}: {stats['num_flights']} flights "
                f"(F1 = {stats['f1_score']:.4f})"
            )
        except Exception as e:
            print(f"Error loading {name} from {summary_file}: {e}")

    return results


def results_to_dataframe(results: Dict[str, Dict[str, float]]) -> pd.DataFrame:
    rows: List[Dict[str, float]] = []
    for name, stats in results.items():
        rows.append(
            {
                "experiment": name,
                "precision": stats["precision"],
                "precision_std": stats["precision_std"],
                "recall": stats["recall"],
                "recall_std": stats["recall_std"],
                "f1_score": stats["f1_score"],
                "f1_std": stats["f1_std"],
                "fpr": stats["fpr"],
                "fpr_std": stats["fpr_std"],
                "roc_auc": stats["roc_auc"],
                "roc_auc_std": stats["roc_auc_std"],
                "num_flights": stats["num_flights"],
            }
        )

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("f1_score", ascending=False).reset_index(drop=True)
    return df


def print_summary(df: pd.DataFrame) -> None:
    if df.empty:
        print("No ablation results to summarize.")
        return

    print("=" * 80)
    print("Ablation study summary (averaged over flights)")
    print("=" * 80)
    print()
    header = (
        f"{'Experiment':<20}"
        f"{'Precision':<12}"
        f"{'Recall':<12}"
        f"{'F1-Score':<12}"
        f"{'ROC-AUC':<12}"
    )
    print(header)
    print("-" * len(header))

    for _, row in df.iterrows():
        name = str(row["experiment"])
        precision = float(row["precision"])
        recall = float(row["recall"])
        f1 = float(row["f1_score"])
        roc = float(row["roc_auc"]) if np.isfinite(row["roc_auc"]) else float("nan")
        roc_str = f"{roc:.4f}" if np.isfinite(roc) else "nan"
        print(
            f"{name:<20}"
            f"{precision:<12.4f}"
            f"{recall:<12.4f}"
            f"{f1:<12.4f}"
            f"{roc_str:<12}"
        )

    best_idx = int(df["f1_score"].idxmax())
    best_row = df.loc[best_idx]
    print()
    print(
        f"Best F1-Score: {best_row['experiment']} "
        f"({best_row['f1_score']:.4f})"
    )
    print("=" * 80)


def save_table(df: pd.DataFrame, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "ablation_results.csv"
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")


def generate_comparison_chart(df: pd.DataFrame, output_dir: Path) -> None:
    if df.empty:
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    names = df["experiment"].tolist()
    precision = df["precision"].to_numpy(dtype=float)
    recall = df["recall"].to_numpy(dtype=float)
    f1 = df["f1_score"].to_numpy(dtype=float)

    x = np.arange(len(names))
    width = 0.25

    plt.style.use("seaborn-v0_8-darkgrid")
    fig, ax = plt.subplots(figsize=(14, 8))

    bars1 = ax.bar(
        x - width,
        precision,
        width,
        label="Precision",
        color="#3498db",
        alpha=0.8,
        edgecolor="black",
        linewidth=1.5,
    )
    bars2 = ax.bar(
        x,
        recall,
        width,
        label="Recall",
        color="#e74c3c",
        alpha=0.8,
        edgecolor="black",
        linewidth=1.5,
    )
    bars3 = ax.bar(
        x + width,
        f1,
        width,
        label="F1-Score",
        color="#2ecc71",
        alpha=0.8,
        edgecolor="black",
        linewidth=1.5,
    )

    for bars in (bars1, bars2, bars3):
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                height + 0.01,
                f"{height:.3f}",
                ha="center",
                va="bottom",
                fontsize=10,
                fontweight="bold",
            )

    ax.set_xlabel("Ablation Experiment", fontsize=14, fontweight="bold")
    ax.set_ylabel("Score", fontsize=14, fontweight="bold")
    ax.set_title(
        "Ablation Study Performance Comparison",
        fontsize=16,
        fontweight="bold",
        pad=16,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=0, ha="center", fontsize=12)
    ax.legend(fontsize=12, loc="upper right")
    ax.grid(True, linestyle="--", alpha=0.3, axis="y")
    ax.set_ylim(0.0, 1.1)

    fig.tight_layout()
    chart_path = output_dir / "ablation_comparison_chart.png"
    fig.savefig(chart_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def generate_markdown_table(df: pd.DataFrame, output_dir: Path) -> None:
    if df.empty:
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    lines: List[str] = []
    lines.append("# Ablation Study Comparison")
    lines.append("")
    lines.append("Averaged performance metrics over all flights.")
    lines.append("")
    lines.append("| Rank | Ablation Experiment | Precision | Recall | F1-Score | ROC-AUC |")
    lines.append("|------|---------------------|-----------|--------|----------|---------|")

    precision_vals = df["precision"].to_numpy(dtype=float)
    recall_vals = df["recall"].to_numpy(dtype=float)
    f1_vals = df["f1_score"].to_numpy(dtype=float)

    best_p_idx = int(precision_vals.argmax())
    best_r_idx = int(recall_vals.argmax())
    best_f_idx = int(f1_vals.argmax())

    for rank, (_, row) in enumerate(
        df.sort_values("f1_score", ascending=False).iterrows(), start=1
    ):
        idx = int(row.name)
        name = str(row["experiment"])
        p = float(row["precision"])
        r = float(row["recall"])
        f = float(row["f1_score"])
        roc = float(row["roc_auc"]) if np.isfinite(row["roc_auc"]) else float("nan")
        roc_str = f"{roc:.4f}" if np.isfinite(roc) else "nan"

        p_mark = "⭐" if idx == best_p_idx else ""
        r_mark = "⭐" if idx == best_r_idx else ""
        f_mark = "⭐" if idx == best_f_idx else ""

        lines.append(
            f"| {rank} | {name} | "
            f"{p:.4f} {p_mark} | "
            f"{r:.4f} {r_mark} | "
            f"{f:.4f} {f_mark} | "
            f"{roc_str} |"
        )

    lines.append("")
    lines.append("⭐ marks the best value in each metric.")
    lines.append("")

    md_path = output_dir / "ablation_comparison_table.md"
    md_path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize ablation experiments from evaluation_summary.json files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Typical usage:

  cd Ablation
  python ablation_evaluation_open_source.py

By default, this script searches for the latest evaluation_summary.json in:
  - Ablation_no_attention/results/evaluation_no_attention_*/
  - Ablation_no_spearman/results/evaluation_no_spearman_*/
  - Ablation-no_dCor/results/evaluation_no_dcor_*/
  - Ablation-no_mic/results/evaluation_no_mic_*/
""",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory to store comparison tables and figures (default: script directory).",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.output_dir is None:
        output_dir = SCRIPT_DIR
    else:
        output_dir = Path(args.output_dir)

    print("=" * 80)
    print("Loading ablation evaluation summaries")
    print("=" * 80)
    print()

    results = load_latest_ablation_results(SCRIPT_DIR, DEFAULT_ABLATION_CONFIGS)
    if not results:
        print("No ablation evaluation results found.")
        return

    df = results_to_dataframe(results)

    print()
    print_summary(df)
    save_table(df, output_dir)
    generate_comparison_chart(df, output_dir)
    generate_markdown_table(df, output_dir)

    print()
    print(f"Outputs saved to: {output_dir.resolve()}")


if __name__ == "__main__":
    main()


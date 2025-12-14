#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Open-source friendly threshold evaluation script.

This script compares multiple unsupervised thresholding methods on
pre-computed anomaly scores:

- POT-EVT (Peaks Over Threshold - Extreme Value Theory)
- MAD (Median Absolute Deviation)
- IQR (Interquartile Range)
- 3-Sigma (Z-Score)

Thresholds are always estimated on validation scores without using labels.
The same fixed thresholds are then applied to test flights to report
Precision/Recall/F1 and ROC-AUC, which matches common paper and
benchmarking practice.
"""
import sys
from pathlib import Path
from typing import Dict, List
import argparse
import json

import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except AttributeError:
        import codecs

        sys.stdout = codecs.getwriter("utf-8")(sys.stdout.buffer, "strict")
        sys.stderr = codecs.getwriter("utf-8")(sys.stderr.buffer, "strict")


def _ensure_1d_scores(scores: np.ndarray) -> np.ndarray:
    scores = np.asarray(scores, dtype=float)
    if scores.ndim > 1:
        scores = scores.reshape(-1)
    scores = scores[np.isfinite(scores)]
    return scores


def calculate_threshold_pot(
    scores: np.ndarray, u_q: float = 0.95, tail_prob: float = 0.005, fallback_q: float = 0.995
) -> float:
    """
    POT-EVT threshold on validation scores.

    Implementation follows the standard Peaks Over Threshold procedure:
    1) choose a high quantile u_q as the base threshold;
    2) fit a Generalized Pareto Distribution (GPD) on the tail;
    3) compute the tail_prob quantile of the fitted GPD.
    """
    from scipy.stats import genpareto as gpd

    scores = _ensure_1d_scores(scores)
    if len(scores) == 0:
        return float("inf")

    try:
        u = np.quantile(scores, u_q)
        tail = scores[scores > u] - u
        if len(tail) < 50:
            raise RuntimeError("tail too short")
        c, loc, scale = gpd.fit(tail, floc=0)
        thr = u + gpd.ppf(1 - tail_prob, c, loc=0, scale=scale)
        return float(thr)
    except Exception:
        return float(np.quantile(scores, fallback_q))


def calculate_threshold_mad(scores: np.ndarray, k: float = 3.0) -> float:
    """
    MAD-based threshold on validation scores.

    threshold = median(scores) + k * MAD(scores)
    where MAD is median(|x - median(x)|).
    """
    scores = _ensure_1d_scores(scores)
    if len(scores) == 0:
        return float("inf")
    median = float(np.median(scores))
    mad = float(np.median(np.abs(scores - median)) + 1e-8)
    threshold = median + k * mad
    return float(threshold)


def calculate_threshold_iqr(scores: np.ndarray, k: float = 1.5) -> float:
    """
    IQR-based threshold on validation scores.

    threshold = Q3 + k * IQR, where IQR = Q3 - Q1.
    """
    scores = _ensure_1d_scores(scores)
    if len(scores) == 0:
        return float("inf")
    q1 = float(np.percentile(scores, 25))
    q3 = float(np.percentile(scores, 75))
    iqr = q3 - q1
    threshold = q3 + k * iqr
    return float(threshold)


def calculate_threshold_3sigma(scores: np.ndarray, k: float = 3.0) -> float:
    """
    3-Sigma (Z-Score) threshold on validation scores.

    threshold = mean(scores) + k * std(scores)
    """
    scores = _ensure_1d_scores(scores)
    if len(scores) == 0:
        return float("inf")
    mean = float(np.mean(scores))
    std = float(np.std(scores) + 1e-8)
    threshold = mean + k * std
    return float(threshold)


class ThresholdEvaluator:
    """
    Threshold comparison on pre-computed anomaly scores.

    Expected data layout:
    - Validation scores: a single NumPy array, shape (N,) or (N, 1)
      provided via --val_scores.
    - Test scores and labels: stored in a directory passed via
      --test_scores_dir with the following naming convention:
        scores_flight_XXX.npy
        labels_flight_XXX.npy
      where XXX is the flight identifier (any suffix).
    """

    def __init__(
        self,
        val_scores_path: Path,
        test_scores_dir: Path,
        output_dir: Path,
    ):
        self.val_scores_path = val_scores_path
        self.test_scores_dir = test_scores_dir
        self.output_root = output_dir

        timestamp = None
        if timestamp is None:
            from datetime import datetime

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        self.output_dir = self.output_root / f"threshold_comparison_{timestamp}"
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def load_validation_scores(self) -> np.ndarray:
        """Load validation anomaly scores from a .npy file."""
        if not self.val_scores_path.exists():
            raise FileNotFoundError(f"Validation scores not found: {self.val_scores_path}")
        scores = np.load(self.val_scores_path)
        scores = _ensure_1d_scores(scores)
        return scores

    def load_test_flights(self) -> Dict[str, Dict[str, np.ndarray]]:
        """
        Load test flights' scores and labels.

        For each scores_flight_XXX.npy file, the script expects a
        labels_flight_XXX.npy file in the same directory.
        """
        if not self.test_scores_dir.exists():
            raise FileNotFoundError(f"Test scores directory not found: {self.test_scores_dir}")

        flights: Dict[str, Dict[str, np.ndarray]] = {}
        pattern = "scores_flight_*.npy"
        score_files = sorted(self.test_scores_dir.glob(pattern))

        if not score_files:
            raise FileNotFoundError(
                f"No test score files found in {self.test_scores_dir} "
                f"(expected pattern: {pattern})"
            )

        for score_file in score_files:
            suffix = score_file.stem.replace("scores_", "")
            label_file = score_file.with_name(f"labels_{suffix}.npy")
            if not label_file.exists():
                raise FileNotFoundError(f"Missing label file for {score_file.name}: {label_file}")

            scores = _ensure_1d_scores(np.load(score_file))
            labels = np.load(label_file)
            labels = np.asarray(labels).reshape(-1).astype(int)

            if len(scores) != len(labels):
                raise ValueError(
                    f"Length mismatch for {suffix}: scores={len(scores)}, labels={len(labels)}"
                )

            flights[suffix] = {"scores": scores, "labels": labels}

        return flights

    def calculate_thresholds(self, validation_scores: np.ndarray) -> Dict[str, float]:
        """Compute thresholds on validation scores using multiple methods."""
        thresholds: Dict[str, float] = {}

        thresholds["POT-EVT"] = calculate_threshold_pot(validation_scores)
        thresholds["MAD"] = calculate_threshold_mad(validation_scores, k=3.0)
        thresholds["IQR"] = calculate_threshold_iqr(validation_scores, k=1.5)
        thresholds["3-Sigma"] = calculate_threshold_3sigma(validation_scores, k=3.0)

        return thresholds

    def evaluate_single_flight(
        self,
        scores: np.ndarray,
        labels: np.ndarray,
        thresholds: Dict[str, float],
    ) -> Dict[str, Dict[str, float]]:
        """Evaluate a single flight under multiple thresholding methods."""
        results: Dict[str, Dict[str, float]] = {}

        for method, thr in thresholds.items():
            preds = (scores > thr).astype(int)

            precision = precision_score(labels, preds, zero_division=0)
            recall = recall_score(labels, preds, zero_division=0)
            f1 = f1_score(labels, preds, zero_division=0)

            try:
                if len(np.unique(labels)) > 1:
                    roc_auc = roc_auc_score(labels, scores)
                else:
                    roc_auc = float("nan")
            except ValueError:
                roc_auc = float("nan")

            results[method] = {
                "precision": float(precision),
                "recall": float(recall),
                "f1_score": float(f1),
                "roc_auc": float(roc_auc) if np.isfinite(roc_auc) else None,
                "threshold": float(thr),
                "num_anomalies_predicted": int(np.sum(preds == 1)),
                "num_anomalies_true": int(np.sum(labels == 1)),

            }

        return results
    def _generate_summary(
        self,
        all_results: Dict[str, Dict[str, Dict[str, float]]],
        thresholds: Dict[str, float],
    ) -> None:
        """Aggregate per-flight results into method-level summary statistics."""
        methods = list(thresholds.keys())
        summary_rows: List[Dict[str, float]] = []

        for method in methods:
            precisions: List[float] = []
            recalls: List[float] = []
            f1_scores: List[float] = []
            roc_aucs: List[float] = []

            for flight_id, flight_results in all_results.items():
                if method not in flight_results:
                    continue
                res = flight_results[method]
                precisions.append(res["precision"])
                recalls.append(res["recall"])
                f1_scores.append(res["f1_score"])
                if res["roc_auc"] is not None:
                    roc_aucs.append(res["roc_auc"])

            if not precisions:
                continue

            summary_rows.append(
                {
                    "Method": method,
                    "Threshold": thresholds[method],
                    "Avg_Precision": float(np.mean(precisions)),
                    "Std_Precision": float(np.std(precisions)),
                    "Avg_Recall": float(np.mean(recalls)),
                    "Std_Recall": float(np.std(recalls)),
                    "Avg_F1": float(np.mean(f1_scores)),
                    "Std_F1": float(np.std(f1_scores)),
                    "Avg_ROC_AUC": float(np.mean(roc_aucs)) if roc_aucs else float("nan"),
                    "Std_ROC_AUC": float(np.std(roc_aucs)) if roc_aucs else float("nan"),
                }
            )

        df = pd.DataFrame(summary_rows)
        csv_path = self.output_dir / "summary_statistics.csv"
        df.to_csv(csv_path, index=False, encoding="utf-8-sig")

        json_path = self.output_dir / "summary_statistics.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(summary_rows, f, indent=2, ensure_ascii=False)

    def _plot_comparison(
        self,
        all_results: Dict[str, Dict[str, Dict[str, float]]],
        thresholds: Dict[str, float],
    ) -> None:
        """Visual comparison of methods: metrics and thresholds."""
        methods = list(thresholds.keys())
        flights = sorted(all_results.keys())

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        metrics = ["precision", "recall", "f1_score"]
        metric_names = ["Precision", "Recall", "F1-Score"]

        for idx, (metric, metric_name) in enumerate(zip(metrics, metric_names)):
            ax = axes[idx // 2, idx % 2]

            method_means: List[float] = []
            method_stds: List[float] = []

            for method in methods:
                values = [
                    all_results[f][method][metric]
                    for f in flights
                    if method in all_results[f]
                ]
                if not values:
                    method_means.append(0.0)
                    method_stds.append(0.0)
                    continue
                method_means.append(float(np.mean(values)))
                method_stds.append(float(np.std(values)))

            x = np.arange(len(methods))
            bars = ax.bar(
                x,
                method_means,
                yerr=method_stds,
                capsize=5,
                alpha=0.8,
                edgecolor="black",
                linewidth=1.5,
            )

            for bar, mean, std in zip(bars, method_means, method_stds):
                height = bar.get_height()
                ax.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    height + std + 0.01,
                    f"{mean:.3f}",
                    ha="center",
                    va="bottom",
                    fontsize=10,
                    fontweight="bold",
                )

            ax.set_xlabel("Threshold Method", fontsize=12, fontweight="bold")
            ax.set_ylabel(metric_name, fontsize=12, fontweight="bold")
            ax.set_title(f"{metric_name} Comparison", fontsize=14, fontweight="bold")
            ax.set_xticks(x)
            ax.set_xticklabels(methods, rotation=45, ha="right")
            ax.set_ylim(0, 1.1)
            ax.grid(True, linestyle="--", alpha=0.3, axis="y")

        ax = axes[1, 1]
        threshold_values = [thresholds[m] for m in methods]
        bars = ax.bar(
            methods,
            threshold_values,
            alpha=0.8,
            edgecolor="black",
            linewidth=1.5,
            color=["#3498db", "#e74c3c", "#2ecc71", "#f39c12"],
        )

        for bar, val in zip(bars, threshold_values):
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                height + abs(height) * 0.02,
                f"{val:.4f}",
                ha="center",
                va="bottom",
                fontsize=10,
                fontweight="bold",
            )

        ax.set_xlabel("Threshold Method", fontsize=12, fontweight="bold")
        ax.set_ylabel("Threshold", fontsize=12, fontweight="bold")
        ax.set_title("Threshold Value Comparison", fontsize=14, fontweight="bold")
        ax.tick_params(axis="x", rotation=45)
        ax.grid(True, linestyle="--", alpha=0.3, axis="y")

        plt.tight_layout()
        fig_path = self.output_dir / "threshold_methods_comparison.png"
        plt.savefig(fig_path, dpi=300, bbox_inches="tight")
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(16, 8))
        x = np.arange(len(flights))
        width = 0.2

        for i, method in enumerate(methods):
            f1_values = [
                all_results[f][method]["f1_score"]
                for f in flights
                if method in all_results[f]
            ]
            offset = (i - len(methods) / 2.0 + 0.5) * width
            ax.bar(
                x + offset,
                f1_values,
                width,
                label=method,
                alpha=0.8,
                edgecolor="black",
            )

        ax.set_xlabel("Flight", fontsize=12, fontweight="bold")
        ax.set_ylabel("F1-Score", fontsize=12, fontweight="bold")
        ax.set_title(
            "Per-flight F1-Score across threshold methods", fontsize=14, fontweight="bold"
        )
        ax.set_xticks(x)
        ax.set_xticklabels(flights, rotation=45, ha="right")
        ax.legend(fontsize=10)
        ax.set_ylim(0, 1.1)
        ax.grid(True, linestyle="--", alpha=0.3, axis="y")

        plt.tight_layout()
        f1_path = self.output_dir / "f1_score_by_flight.png"
        plt.savefig(f1_path, dpi=300, bbox_inches="tight")
        plt.close(fig)

    def run(self) -> None:
        """Run the full evaluation pipeline."""
        val_scores = self.load_validation_scores()
        thresholds = self.calculate_thresholds(val_scores)

        thresholds_path = self.output_dir / "thresholds.json"
        with open(thresholds_path, "w", encoding="utf-8") as f:
            json.dump(thresholds, f, indent=2, ensure_ascii=False)

        flights = self.load_test_flights()
        all_results: Dict[str, Dict[str, Dict[str, float]]] = {}

        for flight_id, data in flights.items():
            scores = data["scores"]
            labels = data["labels"]
            all_results[flight_id] = self.evaluate_single_flight(scores, labels, thresholds)

        detailed_path = self.output_dir / "detailed_results.json"
        with open(detailed_path, "w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)

        self._generate_summary(all_results, thresholds)
        self._plot_comparison(all_results, thresholds)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate multiple thresholding methods on anomaly scores",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Typical usage:

  python ThresholdCompare.py \\
      --val_scores path/to/val_scores.npy \\
      --test_scores_dir path/to/test_scores \\
      --output_dir threshold

Data layout:
  - val_scores.npy: validation anomaly scores (1D or 2D array)
  - test_scores_dir/
      scores_flight_XX.npy
      labels_flight_XX.npy
""",
    )

    parser.add_argument(
        "--val_scores",
        type=str,
        required=True,
        help="Path to validation anomaly scores (.npy).",
    )
    parser.add_argument(
        "--test_scores_dir",
        type=str,
        required=True,
        help="Directory containing test scores and labels.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="threshold",
        help="Root output directory for evaluation results.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    evaluator = ThresholdEvaluator(
        val_scores_path=Path(args.val_scores),
        test_scores_dir=Path(args.test_scores_dir),
        output_dir=Path(args.output_dir),
    )
    evaluator.run()


if __name__ == "__main__":
    main()


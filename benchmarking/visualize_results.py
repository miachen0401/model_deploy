"""
Visualization module for benchmarking results.
Creates plots and charts for model performance analysis.
"""
import logging
import argparse
import json
from pathlib import Path
from typing import Dict
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class BenchmarkVisualizer:
    """Visualizer for model evaluation results."""

    def __init__(self, output_dir: str = "benchmarking/visualizations"):
        """
        Initialize visualizer.

        Args:
            output_dir: Directory to save visualizations
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Set style
        sns.set_style("whitegrid")
        plt.rcParams["figure.figsize"] = (12, 8)
        plt.rcParams["font.size"] = 10

    def plot_confusion_matrix(
        self,
        confusion_matrix: np.ndarray,
        categories: list,
        filename: str = "confusion_matrix.png",
        normalize: bool = True,
    ):
        """
        Plot confusion matrix heatmap.

        Args:
            confusion_matrix: Confusion matrix array
            categories: List of category names
            filename: Output filename
            normalize: Whether to normalize by row
        """
        logger.info("Generating confusion matrix plot...")

        cm = np.array(confusion_matrix)

        if normalize:
            # Normalize by row (true labels)
            cm_norm = cm.astype("float") / (cm.sum(axis=1)[:, np.newaxis] + 1e-10)
        else:
            cm_norm = cm

        # Create figure
        fig, ax = plt.subplots(figsize=(16, 14))

        # Plot heatmap
        sns.heatmap(
            cm_norm,
            annot=True,
            fmt=".2f" if normalize else "d",
            cmap="Blues",
            xticklabels=categories,
            yticklabels=categories,
            ax=ax,
            cbar_kws={"label": "Proportion" if normalize else "Count"},
        )

        ax.set_xlabel("Predicted Category", fontsize=12)
        ax.set_ylabel("True Category", fontsize=12)
        ax.set_title(
            f"Confusion Matrix ({'Normalized' if normalize else 'Raw Counts'})",
            fontsize=14,
            pad=20,
        )

        # Rotate labels
        plt.xticks(rotation=45, ha="right")
        plt.yticks(rotation=0)

        plt.tight_layout()

        filepath = self.output_dir / filename
        plt.savefig(filepath, dpi=300, bbox_inches="tight")
        plt.close()

        logger.info(f"Confusion matrix saved to {filepath}")

    def plot_per_class_metrics(
        self,
        per_class_metrics: Dict,
        filename: str = "per_class_metrics.png",
    ):
        """
        Plot per-class precision, recall, and F1 scores.

        Args:
            per_class_metrics: Dictionary of per-class metrics
            filename: Output filename
        """
        logger.info("Generating per-class metrics plot...")

        # Filter classes with support > 0
        valid_classes = {
            k: v for k, v in per_class_metrics.items() if v["support"] > 0
        }

        if not valid_classes:
            logger.warning("No classes with support > 0")
            return

        # Prepare data
        categories = list(valid_classes.keys())
        precision = [valid_classes[c]["precision"] for c in categories]
        recall = [valid_classes[c]["recall"] for c in categories]
        f1 = [valid_classes[c]["f1"] for c in categories]
        support = [valid_classes[c]["support"] for c in categories]

        # Create figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))

        # Plot 1: Bar chart for precision, recall, F1
        x = np.arange(len(categories))
        width = 0.25

        ax1.bar(x - width, precision, width, label="Precision", alpha=0.8)
        ax1.bar(x, recall, width, label="Recall", alpha=0.8)
        ax1.bar(x + width, f1, width, label="F1 Score", alpha=0.8)

        ax1.set_xlabel("Category", fontsize=12)
        ax1.set_ylabel("Score", fontsize=12)
        ax1.set_title("Per-Class Performance Metrics", fontsize=14)
        ax1.set_xticks(x)
        ax1.set_xticklabels(categories, rotation=45, ha="right")
        ax1.legend()
        ax1.grid(axis="y", alpha=0.3)
        ax1.set_ylim([0, 1.0])

        # Plot 2: Support (sample counts)
        ax2.barh(categories, support, alpha=0.8, color="steelblue")
        ax2.set_xlabel("Number of Samples", fontsize=12)
        ax2.set_ylabel("Category", fontsize=12)
        ax2.set_title("Class Distribution (Support)", fontsize=14)
        ax2.grid(axis="x", alpha=0.3)

        plt.tight_layout()

        filepath = self.output_dir / filename
        plt.savefig(filepath, dpi=300, bbox_inches="tight")
        plt.close()

        logger.info(f"Per-class metrics plot saved to {filepath}")

    def plot_roc_auc_comparison(
        self,
        per_class_metrics: Dict,
        filename: str = "roc_auc_comparison.png",
    ):
        """
        Plot ROC AUC scores for each class.

        Args:
            per_class_metrics: Dictionary of per-class metrics
            filename: Output filename
        """
        logger.info("Generating ROC AUC comparison plot...")

        # Filter classes with support > 0
        valid_classes = {
            k: v for k, v in per_class_metrics.items() if v["support"] > 0
        }

        if not valid_classes:
            logger.warning("No classes with support > 0")
            return

        categories = list(valid_classes.keys())
        roc_aucs = [valid_classes[c].get("roc_auc", 0.5) for c in categories]

        # Sort by ROC AUC
        sorted_indices = np.argsort(roc_aucs)[::-1]
        categories = [categories[i] for i in sorted_indices]
        roc_aucs = [roc_aucs[i] for i in sorted_indices]

        # Create figure
        fig, ax = plt.subplots(figsize=(12, max(8, len(categories) * 0.4)))

        colors = ["green" if auc >= 0.8 else "orange" if auc >= 0.6 else "red" for auc in roc_aucs]

        bars = ax.barh(categories, roc_aucs, color=colors, alpha=0.7)

        # Add value labels
        for i, (bar, auc) in enumerate(zip(bars, roc_aucs)):
            ax.text(auc + 0.02, i, f"{auc:.3f}", va="center", fontsize=9)

        # Add reference line at 0.5 (random baseline)
        ax.axvline(x=0.5, color="red", linestyle="--", alpha=0.5, label="Random (0.5)")
        ax.axvline(x=0.7, color="orange", linestyle="--", alpha=0.3, label="Fair (0.7)")
        ax.axvline(x=0.9, color="green", linestyle="--", alpha=0.3, label="Excellent (0.9)")

        ax.set_xlabel("ROC AUC Score", fontsize=12)
        ax.set_ylabel("Category", fontsize=12)
        ax.set_title("ROC AUC Scores by Category", fontsize=14)
        ax.set_xlim([0, 1.05])
        ax.legend(loc="lower right")
        ax.grid(axis="x", alpha=0.3)

        plt.tight_layout()

        filepath = self.output_dir / filename
        plt.savefig(filepath, dpi=300, bbox_inches="tight")
        plt.close()

        logger.info(f"ROC AUC comparison plot saved to {filepath}")

    def plot_metrics_summary(
        self,
        overall_metrics: Dict,
        filename: str = "metrics_summary.png",
    ):
        """
        Plot summary of overall metrics.

        Args:
            overall_metrics: Dictionary of overall metrics
            filename: Output filename
        """
        logger.info("Generating metrics summary plot...")

        # Prepare data
        metric_names = [
            "Accuracy",
            "Precision\n(Macro)",
            "Recall\n(Macro)",
            "F1\n(Macro)",
            "F1\n(Weighted)",
        ]
        metric_values = [
            overall_metrics["accuracy"],
            overall_metrics["precision_macro"],
            overall_metrics["recall_macro"],
            overall_metrics["f1_macro"],
            overall_metrics["f1_weighted"],
        ]

        # Create figure
        fig, ax = plt.subplots(figsize=(10, 6))

        colors = ["steelblue" if v >= 0.7 else "orange" if v >= 0.5 else "red" for v in metric_values]
        bars = ax.bar(metric_names, metric_values, color=colors, alpha=0.7)

        # Add value labels
        for bar, value in zip(bars, metric_values):
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                height + 0.02,
                f"{value:.4f}",
                ha="center",
                va="bottom",
                fontsize=11,
                fontweight="bold",
            )

        # Add reference lines
        ax.axhline(y=0.7, color="orange", linestyle="--", alpha=0.3, label="Good (0.7)")
        ax.axhline(y=0.8, color="green", linestyle="--", alpha=0.3, label="Excellent (0.8)")

        ax.set_ylabel("Score", fontsize=12)
        ax.set_title("Overall Model Performance Metrics", fontsize=14, pad=20)
        ax.set_ylim([0, 1.0])
        ax.legend(loc="lower right")
        ax.grid(axis="y", alpha=0.3)

        plt.tight_layout()

        filepath = self.output_dir / filename
        plt.savefig(filepath, dpi=300, bbox_inches="tight")
        plt.close()

        logger.info(f"Metrics summary plot saved to {filepath}")

    def plot_confidence_distribution(
        self,
        predictions_df: pd.DataFrame,
        filename: str = "confidence_distribution.png",
    ):
        """
        Plot distribution of prediction confidences.

        Args:
            predictions_df: DataFrame with predictions and confidences
            filename: Output filename
        """
        logger.info("Generating confidence distribution plot...")

        if "prediction_confidence" not in predictions_df.columns:
            logger.warning("No confidence column found")
            return

        confidences = predictions_df["prediction_confidence"].values
        correct = (
            predictions_df["category"] == predictions_df["predicted_category"]
        ).values

        # Create figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        # Plot 1: Overall distribution
        ax1.hist(confidences, bins=30, alpha=0.7, color="steelblue", edgecolor="black")
        ax1.axvline(
            np.mean(confidences),
            color="red",
            linestyle="--",
            label=f"Mean: {np.mean(confidences):.3f}",
        )
        ax1.axvline(
            np.median(confidences),
            color="green",
            linestyle="--",
            label=f"Median: {np.median(confidences):.3f}",
        )
        ax1.set_xlabel("Prediction Confidence", fontsize=12)
        ax1.set_ylabel("Frequency", fontsize=12)
        ax1.set_title("Overall Confidence Distribution", fontsize=14)
        ax1.legend()
        ax1.grid(axis="y", alpha=0.3)

        # Plot 2: Correct vs Incorrect
        ax2.hist(
            confidences[correct],
            bins=20,
            alpha=0.7,
            color="green",
            label="Correct",
            edgecolor="black",
        )
        ax2.hist(
            confidences[~correct],
            bins=20,
            alpha=0.7,
            color="red",
            label="Incorrect",
            edgecolor="black",
        )
        ax2.set_xlabel("Prediction Confidence", fontsize=12)
        ax2.set_ylabel("Frequency", fontsize=12)
        ax2.set_title("Confidence: Correct vs Incorrect Predictions", fontsize=14)
        ax2.legend()
        ax2.grid(axis="y", alpha=0.3)

        plt.tight_layout()

        filepath = self.output_dir / filename
        plt.savefig(filepath, dpi=300, bbox_inches="tight")
        plt.close()

        logger.info(f"Confidence distribution plot saved to {filepath}")

    def plot_calibration_curve(
        self,
        predictions_df: pd.DataFrame,
        filename: str = "calibration_curve.png",
        n_bins: int = 10,
    ):
        """
        Plot calibration curve (reliability diagram) with Expected Calibration Error.

        Args:
            predictions_df: DataFrame with predictions and confidences
            filename: Output filename
            n_bins: Number of confidence bins
        """
        logger.info("Generating calibration curve...")

        if "prediction_confidence" not in predictions_df.columns:
            logger.warning("No confidence column found")
            return

        confidences = predictions_df["prediction_confidence"].values
        correct = (
            predictions_df["category"] == predictions_df["predicted_category"]
        ).values

        # Create bins
        bin_edges = np.linspace(0, 1, n_bins + 1)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

        # Calculate empirical accuracy per bin
        bin_accuracies = []
        bin_confidences = []
        bin_counts = []

        for i in range(n_bins):
            mask = (confidences >= bin_edges[i]) & (confidences < bin_edges[i + 1])
            if i == n_bins - 1:  # Include right edge in last bin
                mask = (confidences >= bin_edges[i]) & (confidences <= bin_edges[i + 1])

            if mask.sum() > 0:
                bin_acc = correct[mask].mean()
                bin_conf = confidences[mask].mean()
                bin_accuracies.append(bin_acc)
                bin_confidences.append(bin_conf)
                bin_counts.append(mask.sum())
            else:
                bin_accuracies.append(None)
                bin_confidences.append(bin_centers[i])
                bin_counts.append(0)

        # Calculate Expected Calibration Error (ECE)
        ece = 0.0
        total_samples = len(confidences)
        for i, count in enumerate(bin_counts):
            if count > 0 and bin_accuracies[i] is not None:
                ece += (count / total_samples) * abs(bin_confidences[i] - bin_accuracies[i])

        # Create figure
        fig, ax = plt.subplots(figsize=(10, 10))

        # Plot perfect calibration line
        ax.plot([0, 1], [0, 1], "k--", label="Perfect Calibration", linewidth=2)

        # Plot empirical calibration
        valid_indices = [i for i, acc in enumerate(bin_accuracies) if acc is not None]
        valid_confs = [bin_confidences[i] for i in valid_indices]
        valid_accs = [bin_accuracies[i] for i in valid_indices]
        valid_counts = [bin_counts[i] for i in valid_indices]

        if valid_confs:
            # Plot line
            ax.plot(valid_confs, valid_accs, "o-", linewidth=2, markersize=8,
                   color="steelblue", label="Model Calibration")

            # Add bar chart showing sample distribution
            ax2 = ax.twinx()
            ax2.bar(bin_centers, bin_counts, width=(bin_edges[1] - bin_edges[0]),
                   alpha=0.3, color="gray", label="Sample Count")
            ax2.set_ylabel("Number of Samples", fontsize=12)
            ax2.legend(loc="upper left")

        ax.set_xlabel("Predicted Confidence", fontsize=12)
        ax.set_ylabel("Empirical Accuracy", fontsize=12)
        ax.set_title(
            f"Calibration Curve (Reliability Diagram)\nECE: {ece:.4f}",
            fontsize=14,
            pad=20,
        )
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        ax.legend(loc="upper right")
        ax.grid(alpha=0.3)

        plt.tight_layout()

        filepath = self.output_dir / filename
        plt.savefig(filepath, dpi=300, bbox_inches="tight")
        plt.close()

        logger.info(f"Calibration curve saved to {filepath} (ECE: {ece:.4f})")

    def plot_confidence_correctness(
        self,
        predictions_df: pd.DataFrame,
        filename: str = "confidence_correctness.png",
    ):
        """
        Plot confidence distributions for correct vs incorrect predictions.

        Args:
            predictions_df: DataFrame with predictions and confidences
            filename: Output filename
        """
        logger.info("Generating confidence vs correctness analysis...")

        if "prediction_confidence" not in predictions_df.columns:
            logger.warning("No confidence column found")
            return

        confidences = predictions_df["prediction_confidence"].values
        correct = (
            predictions_df["category"] == predictions_df["predicted_category"]
        ).values

        correct_conf = confidences[correct]
        incorrect_conf = confidences[~correct]

        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

        # Plot 1: Overlapping histograms
        ax1.hist(correct_conf, bins=30, alpha=0.6, color="green",
                label=f"Correct (n={len(correct_conf)})", edgecolor="black")
        ax1.hist(incorrect_conf, bins=30, alpha=0.6, color="red",
                label=f"Incorrect (n={len(incorrect_conf)})", edgecolor="black")

        ax1.axvline(np.mean(correct_conf), color="green", linestyle="--",
                   linewidth=2, label=f"Correct Mean: {np.mean(correct_conf):.3f}")
        ax1.axvline(np.mean(incorrect_conf), color="red", linestyle="--",
                   linewidth=2, label=f"Incorrect Mean: {np.mean(incorrect_conf):.3f}")

        ax1.set_xlabel("Prediction Confidence", fontsize=12)
        ax1.set_ylabel("Frequency", fontsize=12)
        ax1.set_title("Confidence Distribution by Correctness", fontsize=14)
        ax1.legend()
        ax1.grid(axis="y", alpha=0.3)

        # Plot 2: Box plots
        box_data = [correct_conf, incorrect_conf]
        bp = ax2.boxplot(box_data, labels=["Correct", "Incorrect"],
                        patch_artist=True, widths=0.6)

        # Color boxes
        bp["boxes"][0].set_facecolor("lightgreen")
        bp["boxes"][1].set_facecolor("lightcoral")

        # Add statistics text
        stats_text = (
            f"Correct:\n"
            f"  Mean: {np.mean(correct_conf):.3f}\n"
            f"  Median: {np.median(correct_conf):.3f}\n"
            f"  Std: {np.std(correct_conf):.3f}\n\n"
            f"Incorrect:\n"
            f"  Mean: {np.mean(incorrect_conf):.3f}\n"
            f"  Median: {np.median(incorrect_conf):.3f}\n"
            f"  Std: {np.std(incorrect_conf):.3f}"
        )
        ax2.text(0.02, 0.98, stats_text, transform=ax2.transAxes,
                fontsize=10, verticalalignment="top",
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

        ax2.set_ylabel("Prediction Confidence", fontsize=12)
        ax2.set_title("Confidence Statistics by Correctness", fontsize=14)
        ax2.grid(axis="y", alpha=0.3)

        plt.tight_layout()

        filepath = self.output_dir / filename
        plt.savefig(filepath, dpi=300, bbox_inches="tight")
        plt.close()

        logger.info(f"Confidence vs correctness plot saved to {filepath}")

    def plot_latency_distribution(
        self,
        metrics: Dict,
        filename: str = "latency_distribution.png",
    ):
        """
        Plot inference latency distribution and statistics.

        Args:
            metrics: Metrics dictionary containing latency data
            filename: Output filename
        """
        logger.info("Generating latency distribution plot...")

        latency_data = metrics.get("latency", {})
        if not latency_data or "per_batch" not in latency_data:
            logger.warning("No latency data found in metrics")
            return

        latencies = latency_data["per_batch"]

        if not latencies:
            logger.warning("Empty latency data")
            return

        # Calculate statistics
        p50 = np.percentile(latencies, 50)
        p90 = np.percentile(latencies, 90)
        p99 = np.percentile(latencies, 99)
        mean_lat = np.mean(latencies)
        std_lat = np.std(latencies)

        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

        # Plot 1: Histogram with density
        ax1.hist(latencies, bins=30, alpha=0.7, color="steelblue",
                edgecolor="black", density=True)

        # Add KDE
        from scipy import stats as scipy_stats
        kde = scipy_stats.gaussian_kde(latencies)
        x_range = np.linspace(min(latencies), max(latencies), 200)
        ax1.plot(x_range, kde(x_range), 'r-', linewidth=2, label="KDE")

        # Add percentile lines
        ax1.axvline(p50, color="green", linestyle="--", linewidth=2,
                   label=f"p50: {p50:.3f}s")
        ax1.axvline(p90, color="orange", linestyle="--", linewidth=2,
                   label=f"p90: {p90:.3f}s")
        ax1.axvline(p99, color="red", linestyle="--", linewidth=2,
                   label=f"p99: {p99:.3f}s")

        ax1.set_xlabel("Latency (seconds)", fontsize=12)
        ax1.set_ylabel("Density", fontsize=12)
        ax1.set_title("Inference Latency Distribution", fontsize=14)
        ax1.legend()
        ax1.grid(axis="y", alpha=0.3)

        # Plot 2: Box plot and statistics
        bp = ax2.boxplot([latencies], vert=True, patch_artist=True, widths=0.5)
        bp["boxes"][0].set_facecolor("lightblue")

        # Add statistics table
        stats_text = (
            f"Latency Statistics:\n"
            f"{'─' * 30}\n"
            f"Mean:     {mean_lat:.4f} s\n"
            f"Std Dev:  {std_lat:.4f} s\n"
            f"Min:      {min(latencies):.4f} s\n"
            f"Max:      {max(latencies):.4f} s\n"
            f"{'─' * 30}\n"
            f"p50:      {p50:.4f} s\n"
            f"p90:      {p90:.4f} s\n"
            f"p99:      {p99:.4f} s\n"
        )

        # Add tokens/second if available
        if "tokens_per_second" in latency_data:
            tps = latency_data["tokens_per_second"]
            stats_text += f"{'─' * 30}\n"
            stats_text += f"Tokens/sec: {tps:.2f}"

        # Add throughput if available
        if "total_samples" in latency_data and "total_time" in latency_data:
            throughput = latency_data["total_samples"] / latency_data["total_time"]
            stats_text += f"\nThroughput: {throughput:.2f} samples/s"

        ax2.text(1.5, 0.5, stats_text, transform=ax2.transData,
                fontsize=11, verticalalignment="center",
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
                family="monospace")

        ax2.set_ylabel("Latency (seconds)", fontsize=12)
        ax2.set_title("Latency Statistics", fontsize=14)
        ax2.set_xticklabels(["Inference"])
        ax2.grid(axis="y", alpha=0.3)

        plt.tight_layout()

        filepath = self.output_dir / filename
        plt.savefig(filepath, dpi=300, bbox_inches="tight")
        plt.close()

        logger.info(f"Latency distribution plot saved to {filepath}")
        logger.info(f"Latency p50: {p50:.3f}s, p90: {p90:.3f}s, p99: {p99:.3f}s")

    def generate_all_plots(
        self,
        metrics: Dict,
        predictions_df: pd.DataFrame = None,
    ):
        """
        Generate all visualization plots.

        Args:
            metrics: Metrics dictionary from evaluation
            predictions_df: Optional predictions DataFrame
        """
        logger.info("Generating all visualization plots...")

        # Metrics summary
        self.plot_metrics_summary(metrics["overall"])

        # Confusion matrix
        categories = list(metrics["per_class"].keys())
        self.plot_confusion_matrix(
            metrics["confusion_matrix"],
            categories,
            normalize=True,
        )
        self.plot_confusion_matrix(
            metrics["confusion_matrix"],
            categories,
            filename="confusion_matrix_raw.png",
            normalize=False,
        )

        # Per-class metrics
        self.plot_per_class_metrics(metrics["per_class"])

        # ROC AUC comparison
        self.plot_roc_auc_comparison(metrics["per_class"])

        # Confidence distribution (if predictions available)
        if predictions_df is not None:
            self.plot_confidence_distribution(predictions_df)

            # New visualizations
            self.plot_calibration_curve(predictions_df)
            self.plot_confidence_correctness(predictions_df)

        # Latency distribution (if latency data available)
        if "latency" in metrics:
            self.plot_latency_distribution(metrics)

        logger.info(f"All plots saved to {self.output_dir}")


def main():
    """Main visualization workflow."""
    parser = argparse.ArgumentParser(description="Visualize benchmarking results")
    parser.add_argument(
        "--metrics-file",
        type=str,
        required=True,
        help="Path to metrics JSON file",
    )
    parser.add_argument(
        "--predictions-file",
        type=str,
        help="Path to predictions CSV file (optional)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="benchmarking/visualizations",
        help="Output directory for plots",
    )

    args = parser.parse_args()

    try:
        # Load metrics
        logger.info(f"Loading metrics from {args.metrics_file}...")
        with open(args.metrics_file, "r") as f:
            metrics = json.load(f)

        # Load predictions if provided
        predictions_df = None
        if args.predictions_file:
            logger.info(f"Loading predictions from {args.predictions_file}...")
            predictions_df = pd.read_csv(args.predictions_file)

        # Create visualizations
        visualizer = BenchmarkVisualizer(output_dir=args.output_dir)
        visualizer.generate_all_plots(metrics, predictions_df)

        logger.info(f"\nVisualization complete!")
        logger.info(f"Plots saved in: {visualizer.output_dir}")

    except Exception as e:
        logger.error(f"Visualization failed: {str(e)}")
        raise


if __name__ == "__main__":
    main()

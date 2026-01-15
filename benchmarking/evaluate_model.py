"""
Main benchmarking script for evaluating deployed model performance.
"""
import logging
import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, List
import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report,
    roc_auc_score,
)
from sklearn.preprocessing import label_binarize
import json

from benchmarking.data_fetcher.supabase_client import SupabaseNewsClient
from benchmarking.model_inference import NewsClassifier

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class ModelEvaluator:
    """Evaluator for model performance assessment."""

    def __init__(self, results_dir: str = "benchmarking/results"):
        """
        Initialize evaluator.

        Args:
            results_dir: Directory to save results
        """
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)

        self.all_categories = NewsClassifier.VALID_CATEGORIES

    def calculate_metrics(
        self,
        y_true: List[str],
        y_pred: List[str],
        confidences: List[float] = None,
    ) -> Dict:
        """
        Calculate comprehensive evaluation metrics.

        Args:
            y_true: Ground truth labels
            y_pred: Predicted labels
            confidences: Prediction confidences (optional)

        Returns:
            Dictionary of metrics
        """
        logger.info("Calculating evaluation metrics...")

        # Basic metrics
        accuracy = accuracy_score(y_true, y_pred)

        # Per-class metrics
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true, y_pred, average=None, labels=self.all_categories, zero_division=0
        )

        # Macro/Micro averages
        precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
            y_true, y_pred, average="macro", zero_division=0
        )

        precision_micro, recall_micro, f1_micro, _ = precision_recall_fscore_support(
            y_true, y_pred, average="micro", zero_division=0
        )

        # Weighted averages (accounts for class imbalance)
        precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
            y_true, y_pred, average="weighted", zero_division=0
        )

        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred, labels=self.all_categories)

        # ROC AUC (one-vs-rest for multiclass)
        try:
            # Binarize labels
            y_true_bin = label_binarize(y_true, classes=self.all_categories)
            y_pred_bin = label_binarize(y_pred, classes=self.all_categories)

            # Calculate ROC AUC for each class
            roc_auc_ovr = {}
            for i, category in enumerate(self.all_categories):
                if y_true_bin[:, i].sum() > 0:  # Only if class exists in true labels
                    try:
                        roc_auc_ovr[category] = roc_auc_score(
                            y_true_bin[:, i], y_pred_bin[:, i]
                        )
                    except ValueError:
                        roc_auc_ovr[category] = 0.5  # Default for undefined

            roc_auc_macro = np.mean(list(roc_auc_ovr.values()))

        except Exception as e:
            logger.warning(f"Could not calculate ROC AUC: {e}")
            roc_auc_ovr = {}
            roc_auc_macro = None

        # Compile metrics
        metrics = {
            "overall": {
                "accuracy": float(accuracy),
                "precision_macro": float(precision_macro),
                "recall_macro": float(recall_macro),
                "f1_macro": float(f1_macro),
                "precision_micro": float(precision_micro),
                "recall_micro": float(recall_micro),
                "f1_micro": float(f1_micro),
                "precision_weighted": float(precision_weighted),
                "recall_weighted": float(recall_weighted),
                "f1_weighted": float(f1_weighted),
                "roc_auc_macro": float(roc_auc_macro) if roc_auc_macro else None,
            },
            "per_class": {
                category: {
                    "precision": float(precision[i]),
                    "recall": float(recall[i]),
                    "f1": float(f1[i]),
                    "support": int(support[i]),
                    "roc_auc": float(roc_auc_ovr.get(category, 0.5)),
                }
                for i, category in enumerate(self.all_categories)
            },
            "confusion_matrix": cm.tolist(),
        }

        if confidences:
            metrics["overall"]["mean_confidence"] = float(np.mean(confidences))
            metrics["overall"]["std_confidence"] = float(np.std(confidences))

        return metrics

    def save_metrics(self, metrics: Dict, filename: str):
        """Save metrics to JSON file."""
        filepath = self.results_dir / filename
        with open(filepath, "w") as f:
            json.dump(metrics, f, indent=2)
        logger.info(f"Metrics saved to {filepath}")

    def save_predictions(self, df: pd.DataFrame, filename: str):
        """Save predictions DataFrame to CSV."""
        filepath = self.results_dir / filename
        df.to_csv(filepath, index=False)
        logger.info(f"Predictions saved to {filepath}")

    def print_summary(self, metrics: Dict):
        """Print formatted summary of metrics."""
        logger.info("\n" + "=" * 80)
        logger.info("MODEL EVALUATION SUMMARY")
        logger.info("=" * 80)

        overall = metrics["overall"]
        logger.info(f"\nOverall Performance:")
        logger.info(f"  Accuracy:           {overall['accuracy']:.4f}")
        logger.info(f"  Precision (macro):  {overall['precision_macro']:.4f}")
        logger.info(f"  Recall (macro):     {overall['recall_macro']:.4f}")
        logger.info(f"  F1 Score (macro):   {overall['f1_macro']:.4f}")
        logger.info(f"  F1 Score (weighted):{overall['f1_weighted']:.4f}")

        if overall.get("roc_auc_macro"):
            logger.info(f"  ROC AUC (macro):    {overall['roc_auc_macro']:.4f}")

        if overall.get("mean_confidence"):
            logger.info(f"  Mean Confidence:    {overall['mean_confidence']:.4f}")

        logger.info(f"\nPer-Class Performance (Top 5 by F1):")
        per_class = metrics["per_class"]

        # Sort by F1 score
        sorted_classes = sorted(
            per_class.items(),
            key=lambda x: x[1]["f1"],
            reverse=True,
        )[:5]

        logger.info(f"{'Category':<25} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Support':>10}")
        logger.info("-" * 75)
        for category, scores in sorted_classes:
            if scores["support"] > 0:  # Only show classes with data
                logger.info(
                    f"{category:<25} "
                    f"{scores['precision']:>10.4f} "
                    f"{scores['recall']:>10.4f} "
                    f"{scores['f1']:>10.4f} "
                    f"{scores['support']:>10d}"
                )

        logger.info("=" * 80 + "\n")


def main():
    """Main benchmarking workflow."""
    parser = argparse.ArgumentParser(description="Evaluate deployed model performance")
    parser.add_argument(
        "--limit",
        type=int,
        default=100,
        help="Maximum number of samples to evaluate (default: 100)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=3,
        help="Batch size for inference (default: 3)",
    )
    parser.add_argument(
        "--api-url",
        type=str,
        default="http://localhost:8000/v1",
        help="vLLM API URL",
    )
    parser.add_argument(
        "--skip-inference",
        action="store_true",
        help="Skip inference (use existing predictions file)",
    )
    parser.add_argument(
        "--predictions-file",
        type=str,
        help="Path to predictions CSV file (for --skip-inference)",
    )

    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    try:
        # Step 1: Fetch data from Supabase
        logger.info("Step 1: Fetching labeled data from Supabase...")
        supabase_client = SupabaseNewsClient()
        df = supabase_client.fetch_labeled_data(limit=args.limit)

        if df.empty:
            logger.error("No labeled data found in Supabase")
            return

        logger.info(f"Fetched {len(df)} labeled samples")

        # Step 2: Run inference (or load existing predictions)
        if args.skip_inference and args.predictions_file:
            logger.info(f"Step 2: Loading predictions from {args.predictions_file}...")
            df_with_predictions = pd.read_csv(args.predictions_file)
        else:
            logger.info("Step 2: Running model inference...")
            classifier = NewsClassifier(api_base_url=args.api_url)

            def progress_callback(current, total):
                logger.info(f"Progress: {current}/{total} samples classified")

            df_with_predictions = classifier.classify_dataframe(
                df,
                batch_size=args.batch_size,
                progress_callback=progress_callback,
            )

            # Save predictions
            pred_file = f"predictions_{timestamp}.csv"
            evaluator = ModelEvaluator()
            evaluator.save_predictions(df_with_predictions, pred_file)

        # Step 3: Calculate metrics
        logger.info("Step 3: Calculating evaluation metrics...")
        evaluator = ModelEvaluator()

        y_true = df_with_predictions["category"].tolist()
        y_pred = df_with_predictions["predicted_category"].tolist()
        confidences = df_with_predictions["prediction_confidence"].tolist()

        metrics = evaluator.calculate_metrics(y_true, y_pred, confidences)

        # Step 4: Save results
        logger.info("Step 4: Saving results...")
        metrics_file = f"metrics_{timestamp}.json"
        evaluator.save_metrics(metrics, metrics_file)

        # Step 5: Print summary
        evaluator.print_summary(metrics)

        logger.info(f"\nBenchmarking complete!")
        logger.info(f"Results saved in: {evaluator.results_dir}")
        logger.info(f"  - Predictions: predictions_{timestamp}.csv")
        logger.info(f"  - Metrics: metrics_{timestamp}.json")

    except Exception as e:
        logger.error(f"Benchmarking failed: {str(e)}")
        raise


if __name__ == "__main__":
    main()

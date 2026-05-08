"""Model evaluation utilities: confusion matrices, ROC curves, precision/recall."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

logger = logging.getLogger(__name__)


class ModelEvaluator:
    """Comprehensive evaluation suite for medical diagnostic models."""

    CONDITION_NAMES = ["diabetes", "heart_disease", "liver_condition"]

    def __init__(
        self,
        output_dir: str | Path = "./evaluation",
        dpi: int = 150,
    ) -> None:
        """
        Initialize evaluator.

        Args:
            output_dir: Directory to save evaluation artifacts.
            dpi: Figure resolution.
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.dpi = dpi

    def compute_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_prob: np.ndarray,
    ) -> Dict[str, Dict[str, float]]:
        """
        Compute per-condition metrics.

        Returns:
            Dict mapping condition name to metric dict.
        """
        if y_true.shape != y_pred.shape:
            raise ValueError(f"Shape mismatch: y_true {y_true.shape} vs y_pred {y_pred.shape}")
        if y_true.shape != y_prob.shape:
            raise ValueError(f"Shape mismatch: y_true {y_true.shape} vs y_prob {y_prob.shape}")

        results: Dict[str, Dict[str, float]] = {}
        for i, name in enumerate(self.CONDITION_NAMES):
            if i >= y_true.shape[1]:
                break
            yt, yp, ypr = y_true[:, i], y_pred[:, i], y_prob[:, i]

            from sklearn.metrics import (
                accuracy_score,
                f1_score,
                precision_score,
                recall_score,
            )

            results[name] = {
                "accuracy": accuracy_score(yt, yp),
                "precision": precision_score(yt, yp, zero_division=0),
                "recall": recall_score(yt, yp, zero_division=0),
                "f1": f1_score(yt, yp, zero_division=0),
                "roc_auc": roc_auc_score(yt, ypr),
                "average_precision": average_precision_score(yt, ypr),
            }

        logger.info("Computed metrics for %d conditions", len(results))
        return results

    def plot_confusion_matrices(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        filename: str = "confusion_matrices.png",
    ) -> Path:
        """
        Plot confusion matrices for all conditions in a single figure.

        Returns:
            Path to saved figure.
        """
        n = min(y_true.shape[1], len(self.CONDITION_NAMES))
        fig, axes = plt.subplots(1, n, figsize=(4.5 * n, 4))
        if n == 1:
            axes = [axes]

        for i, ax in enumerate(axes):
            if i >= n:
                ax.set_visible(False)
                continue
            cm = confusion_matrix(y_true[:, i], y_pred[:, i])
            self._draw_confusion_matrix(ax, cm, self.CONDITION_NAMES[i])

        plt.tight_layout()
        path = self.output_dir / filename
        fig.savefig(path, dpi=self.dpi, bbox_inches="tight")
        plt.close(fig)
        logger.info("Saved confusion matrices to %s", path)
        return path

    @staticmethod
    def _draw_confusion_matrix(ax: Any, cm: np.ndarray, title: str) -> None:
        """Draw a single confusion matrix on an axis."""
        im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
        ax.set_title(f"{title.replace('_', ' ').title()}")
        ax.figure.colorbar(im, ax=ax)
        tick_marks = np.arange(2)
        ax.set_xticks(tick_marks)
        ax.set_yticks(tick_marks)
        ax.set_xticklabels(["Negative", "Positive"])
        ax.set_yticklabels(["Negative", "Positive"])
        ax.set_ylabel("True Label")
        ax.set_xlabel("Predicted Label")

        # Annotate cells
        thresh = cm.max() / 2.0
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(
                    j, i, format(cm[i, j], "d"),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black",
                    fontsize=12,
                )

    def plot_roc_curves(
        self,
        y_true: np.ndarray,
        y_prob: np.ndarray,
        filename: str = "roc_curves.png",
    ) -> Path:
        """
        Plot ROC curves for all conditions.

        Returns:
            Path to saved figure.
        """
        n = min(y_true.shape[1], len(self.CONDITION_NAMES))
        fig, axes = plt.subplots(1, n, figsize=(5 * n, 4.5))
        if n == 1:
            axes = [axes]

        for i, ax in enumerate(axes):
            if i >= n:
                ax.set_visible(False)
                continue
            fpr, tpr, _ = roc_curve(y_true[:, i], y_prob[:, i])
            auc = roc_auc_score(y_true[:, i], y_prob[:, i])
            ax.plot(fpr, tpr, lw=2, label=f"AUC = {auc:.3f}")
            ax.plot([0, 1], [0, 1], "k--", lw=1)
            ax.set_xlim([0.0, 1.0])
            ax.set_ylim([0.0, 1.05])
            ax.set_xlabel("False Positive Rate")
            ax.set_ylabel("True Positive Rate")
            ax.set_title(f"ROC: {self.CONDITION_NAMES[i].replace('_', ' ').title()}")
            ax.legend(loc="lower right")
            ax.grid(alpha=0.3)

        plt.tight_layout()
        path = self.output_dir / filename
        fig.savefig(path, dpi=self.dpi, bbox_inches="tight")
        plt.close(fig)
        logger.info("Saved ROC curves to %s", path)
        return path

    def plot_precision_recall_curves(
        self,
        y_true: np.ndarray,
        y_prob: np.ndarray,
        filename: str = "precision_recall_curves.png",
    ) -> Path:
        """
        Plot precision-recall curves for all conditions.

        Returns:
            Path to saved figure.
        """
        n = min(y_true.shape[1], len(self.CONDITION_NAMES))
        fig, axes = plt.subplots(1, n, figsize=(5 * n, 4.5))
        if n == 1:
            axes = [axes]

        for i, ax in enumerate(axes):
            if i >= n:
                ax.set_visible(False)
                continue
            precision, recall, _ = precision_recall_curve(y_true[:, i], y_prob[:, i])
            ap = average_precision_score(y_true[:, i], y_prob[:, i])
            ax.plot(recall, precision, lw=2, label=f"AP = {ap:.3f}")
            ax.set_xlabel("Recall")
            ax.set_ylabel("Precision")
            ax.set_title(f"P-R: {self.CONDITION_NAMES[i].replace('_', ' ').title()}")
            ax.legend(loc="lower left")
            ax.grid(alpha=0.3)
            ax.set_xlim([0.0, 1.0])
            ax.set_ylim([0.0, 1.05])

        plt.tight_layout()
        path = self.output_dir / filename
        fig.savefig(path, dpi=self.dpi, bbox_inches="tight")
        plt.close(fig)
        logger.info("Saved precision-recall curves to %s", path)
        return path

    def plot_risk_distribution(
        self,
        y_prob: np.ndarray,
        y_true: np.ndarray,
        filename: str = "risk_distribution.png",
    ) -> Path:
        """
        Plot histogram of predicted probabilities stratified by true label.

        Returns:
            Path to saved figure.
        """
        n = min(y_true.shape[1], len(self.CONDITION_NAMES))
        fig, axes = plt.subplots(1, n, figsize=(5 * n, 4))
        if n == 1:
            axes = [axes]

        for i, ax in enumerate(axes):
            if i >= n:
                ax.set_visible(False)
                continue

            probs_neg = y_prob[y_true[:, i] == 0, i]
            probs_pos = y_prob[y_true[:, i] == 1, i]

            ax.hist(probs_neg, bins=30, alpha=0.6, label="Negative", color="green")
            ax.hist(probs_pos, bins=30, alpha=0.6, label="Positive", color="red")
            ax.set_xlabel("Predicted Probability")
            ax.set_ylabel("Count")
            ax.set_title(f"Risk Dist: {self.CONDITION_NAMES[i].replace('_', ' ').title()}")
            ax.legend()
            ax.grid(alpha=0.3)

        plt.tight_layout()
        path = self.output_dir / filename
        fig.savefig(path, dpi=self.dpi, bbox_inches="tight")
        plt.close(fig)
        logger.info("Saved risk distribution to %s", path)
        return path

    def generate_report(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_prob: np.ndarray,
    ) -> Dict[str, Any]:
        """
        Full evaluation: compute all metrics and generate all plots.

        Returns:
            Dict with metrics, paths to saved figures, and classification report strings.
        """
        metrics = self.compute_metrics(y_true, y_pred, y_prob)

        cm_path = self.plot_confusion_matrices(y_true, y_pred)
        roc_path = self.plot_roc_curves(y_true, y_prob)
        pr_path = self.plot_precision_recall_curves(y_true, y_prob)
        risk_path = self.plot_risk_distribution(y_prob, y_true)

        reports: Dict[str, str] = {}
        for i, name in enumerate(self.CONDITION_NAMES):
            if i >= y_true.shape[1]:
                break
            reports[name] = classification_report(
                y_true[:, i], y_pred[:, i], target_names=["Neg", "Pos"], zero_division=0
            )

        return {
            "metrics": metrics,
            "figures": {
                "confusion_matrices": str(cm_path),
                "roc_curves": str(roc_path),
                "precision_recall": str(pr_path),
                "risk_distribution": str(risk_path),
            },
            "classification_reports": reports,
        }

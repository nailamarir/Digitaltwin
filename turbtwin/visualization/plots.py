"""Visualization utilities for TurbTwin."""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple
from pathlib import Path


class ResultsPlotter:
    """
    Plotting utilities for TurbTwin results visualization.

    Creates publication-quality figures for:
    - Prediction vs actual scatter plots
    - Model comparison bar charts
    - Uncertainty quantification plots
    - Feature importance visualizations
    """

    def __init__(
        self,
        output_dir: str = "results",
        figsize: Tuple[int, int] = (10, 8),
        dpi: int = 300,
        style: str = "seaborn-v0_8-whitegrid"
    ):
        """
        Initialize plotter.

        Args:
            output_dir: Directory to save figures
            figsize: Default figure size
            dpi: Resolution for saved figures
            style: Matplotlib style
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.figsize = figsize
        self.dpi = dpi

        try:
            plt.style.use(style)
        except OSError:
            plt.style.use("seaborn-whitegrid")

    def plot_predictions(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_std: Optional[np.ndarray] = None,
        title: str = "TurbTwin Predictions vs Actual",
        save_name: Optional[str] = None
    ) -> plt.Figure:
        """
        Create scatter plot of predictions vs actual values.

        Args:
            y_true: Ground truth values
            y_pred: Predicted values
            y_std: Optional uncertainty estimates
            title: Plot title
            save_name: Filename to save figure

        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=self.figsize)

        # Perfect prediction line
        min_val = min(y_true.min(), y_pred.min())
        max_val = max(y_true.max(), y_pred.max())
        ax.plot(
            [min_val, max_val], [min_val, max_val],
            "k--", lw=2, label="Perfect Prediction"
        )

        # Scatter plot with optional error bars
        if y_std is not None:
            ax.errorbar(
                y_true, y_pred, yerr=1.96 * y_std,
                fmt="o", alpha=0.6, capsize=2,
                label="Predictions ± 95% CI"
            )
        else:
            ax.scatter(
                y_true, y_pred, alpha=0.6,
                c="steelblue", edgecolors="white", s=50,
                label="Predictions"
            )

        ax.set_xlabel("Actual Temperature (K)", fontsize=12)
        ax.set_ylabel("Predicted Temperature (K)", fontsize=12)
        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.legend(loc="upper left")
        ax.grid(True, alpha=0.3)

        # Add R² annotation
        from sklearn.metrics import r2_score
        r2 = r2_score(y_true, y_pred)
        ax.annotate(
            f"R² = {r2:.4f}",
            xy=(0.95, 0.05), xycoords="axes fraction",
            fontsize=12, ha="right",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5)
        )

        plt.tight_layout()

        if save_name:
            fig.savefig(self.output_dir / save_name, dpi=self.dpi, bbox_inches="tight")

        return fig

    def plot_model_comparison(
        self,
        metrics: Dict[str, Dict[str, float]],
        metric_name: str = "rmse",
        title: str = "Model Comparison",
        save_name: Optional[str] = None
    ) -> plt.Figure:
        """
        Create bar chart comparing model performance.

        Args:
            metrics: Dictionary of model metrics
            metric_name: Which metric to plot
            title: Plot title
            save_name: Filename to save figure

        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=(10, 6))

        models = list(metrics.keys())
        values = [metrics[m][metric_name] for m in models]

        colors = sns.color_palette("husl", len(models))
        bars = ax.bar(models, values, color=colors, edgecolor="black", linewidth=1.2)

        # Add value labels on bars
        for bar, val in zip(bars, values):
            height = bar.get_height()
            ax.annotate(
                f"{val:.4f}",
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3), textcoords="offset points",
                ha="center", va="bottom", fontsize=10
            )

        ax.set_xlabel("Model", fontsize=12)
        ax.set_ylabel(metric_name.upper(), fontsize=12)
        ax.set_title(title, fontsize=14, fontweight="bold")
        plt.xticks(rotation=45, ha="right")
        ax.grid(True, axis="y", alpha=0.3)

        plt.tight_layout()

        if save_name:
            fig.savefig(self.output_dir / save_name, dpi=self.dpi, bbox_inches="tight")

        return fig

    def plot_feature_importance(
        self,
        importance: Dict[str, np.ndarray],
        feature_names: List[str],
        top_k: int = 10,
        save_name: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot feature importance from multiple models.

        Args:
            importance: Dictionary of importance arrays
            feature_names: Names of features
            top_k: Number of top features to show
            save_name: Filename to save figure

        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=(10, 8))

        # Use average importance
        if "average" in importance:
            avg_imp = importance["average"]
        else:
            avg_imp = np.mean([imp for imp in importance.values()], axis=0)

        # Sort by importance
        indices = np.argsort(avg_imp)[-top_k:]
        top_features = [feature_names[i] for i in indices]
        top_importance = avg_imp[indices]

        colors = sns.color_palette("viridis", top_k)
        ax.barh(top_features, top_importance, color=colors, edgecolor="black")

        ax.set_xlabel("Importance", fontsize=12)
        ax.set_title("Feature Importance (Average Across Models)", fontsize=14, fontweight="bold")
        ax.grid(True, axis="x", alpha=0.3)

        plt.tight_layout()

        if save_name:
            fig.savefig(self.output_dir / save_name, dpi=self.dpi, bbox_inches="tight")

        return fig

    def plot_uncertainty_coverage(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_std: np.ndarray,
        save_name: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot prediction intervals and coverage analysis.

        Args:
            y_true: Ground truth values
            y_pred: Predicted values
            y_std: Uncertainty estimates
            save_name: Filename to save figure

        Returns:
            Matplotlib figure
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # Sort by predicted value for better visualization
        sort_idx = np.argsort(y_pred)
        y_true_sorted = y_true[sort_idx]
        y_pred_sorted = y_pred[sort_idx]
        y_std_sorted = y_std[sort_idx]

        x = np.arange(len(y_true_sorted))

        # Left plot: Predictions with intervals
        ax1 = axes[0]
        ax1.fill_between(
            x,
            y_pred_sorted - 1.96 * y_std_sorted,
            y_pred_sorted + 1.96 * y_std_sorted,
            alpha=0.3, color="steelblue", label="95% CI"
        )
        ax1.plot(x, y_pred_sorted, "b-", lw=1.5, label="Prediction")
        ax1.scatter(x, y_true_sorted, c="red", s=10, alpha=0.6, label="Actual")

        ax1.set_xlabel("Sample Index (sorted)", fontsize=12)
        ax1.set_ylabel("Temperature (K)", fontsize=12)
        ax1.set_title("Predictions with 95% Confidence Interval", fontsize=12)
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Right plot: Coverage histogram
        ax2 = axes[1]
        z_scores = (y_true - y_pred) / (y_std + 1e-8)

        ax2.hist(z_scores, bins=30, density=True, alpha=0.7, color="steelblue", edgecolor="black")

        # Overlay standard normal
        x_norm = np.linspace(-4, 4, 100)
        from scipy.stats import norm
        ax2.plot(x_norm, norm.pdf(x_norm), "r-", lw=2, label="Standard Normal")

        ax2.set_xlabel("Standardized Residual", fontsize=12)
        ax2.set_ylabel("Density", fontsize=12)
        ax2.set_title("Residual Distribution (should match N(0,1))", fontsize=12)
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # Calculate coverage
        lower = y_pred - 1.96 * y_std
        upper = y_pred + 1.96 * y_std
        coverage = np.mean((y_true >= lower) & (y_true <= upper))

        fig.suptitle(f"Uncertainty Quantification (95% Coverage: {coverage*100:.1f}%)", fontsize=14, fontweight="bold")

        plt.tight_layout()

        if save_name:
            fig.savefig(self.output_dir / save_name, dpi=self.dpi, bbox_inches="tight")

        return fig

    def create_summary_figure(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_std: np.ndarray,
        metrics: Dict[str, Dict[str, float]],
        save_name: str = "turbtwin_summary.png"
    ) -> plt.Figure:
        """
        Create comprehensive summary figure with multiple panels.

        Args:
            y_true: Ground truth values
            y_pred: Predicted values
            y_std: Uncertainty estimates
            metrics: Model metrics dictionary
            save_name: Filename to save figure

        Returns:
            Matplotlib figure
        """
        fig = plt.figure(figsize=(16, 12))

        # Panel 1: Predictions vs Actual
        ax1 = fig.add_subplot(2, 2, 1)
        min_val, max_val = min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())
        ax1.plot([min_val, max_val], [min_val, max_val], "k--", lw=2)
        ax1.scatter(y_true, y_pred, alpha=0.6, c="steelblue", s=30)
        ax1.set_xlabel("Actual (K)")
        ax1.set_ylabel("Predicted (K)")
        ax1.set_title("Predictions vs Actual")

        # Panel 2: Model Comparison (RMSE)
        ax2 = fig.add_subplot(2, 2, 2)
        models = list(metrics.keys())
        rmse_values = [metrics[m]["rmse"] for m in models]
        colors = sns.color_palette("husl", len(models))
        ax2.bar(models, rmse_values, color=colors)
        ax2.set_xlabel("Model")
        ax2.set_ylabel("RMSE (K)")
        ax2.set_title("Model RMSE Comparison")
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha="right")

        # Panel 3: Model Comparison (R²)
        ax3 = fig.add_subplot(2, 2, 3)
        r2_values = [metrics[m]["r2"] for m in models]
        ax3.bar(models, r2_values, color=colors)
        ax3.set_xlabel("Model")
        ax3.set_ylabel("R² Score")
        ax3.set_title("Model R² Comparison")
        plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45, ha="right")

        # Panel 4: Residual Distribution
        ax4 = fig.add_subplot(2, 2, 4)
        residuals = y_true - y_pred
        ax4.hist(residuals, bins=30, density=True, alpha=0.7, color="steelblue", edgecolor="black")
        ax4.axvline(x=0, color="red", linestyle="--", lw=2)
        ax4.set_xlabel("Residual (K)")
        ax4.set_ylabel("Density")
        ax4.set_title(f"Residual Distribution (mean={np.mean(residuals):.4f})")

        plt.suptitle("TurbTwin Ensemble Results", fontsize=16, fontweight="bold")
        plt.tight_layout()

        fig.savefig(self.output_dir / save_name, dpi=self.dpi, bbox_inches="tight")

        return fig

"""Metrics calculation utilities."""

import numpy as np
from typing import Dict, Tuple, Optional
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def calculate_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_pred_std: Optional[np.ndarray] = None
) -> Dict[str, float]:
    """
    Calculate regression metrics.

    Args:
        y_true: Ground truth values
        y_pred: Predicted values
        y_pred_std: Optional standard deviation of predictions

    Returns:
        Dictionary containing RMSE, MAE, R², and optionally coverage metrics
    """
    metrics = {
        "rmse": np.sqrt(mean_squared_error(y_true, y_pred)),
        "mae": mean_absolute_error(y_true, y_pred),
        "r2": r2_score(y_true, y_pred),
        "mape": np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100,
    }

    # Add uncertainty metrics if std is provided
    if y_pred_std is not None:
        lower = y_pred - 1.96 * y_pred_std
        upper = y_pred + 1.96 * y_pred_std
        coverage = np.mean((y_true >= lower) & (y_true <= upper))
        metrics["coverage_95"] = coverage
        metrics["mean_interval_width"] = np.mean(upper - lower)

    return metrics


def print_metrics(metrics: Dict[str, float], model_name: str = "Model") -> None:
    """Print metrics in a formatted way."""
    print(f"\n{'='*50}")
    print(f"{model_name} Performance Metrics")
    print(f"{'='*50}")
    print(f"RMSE:  {metrics['rmse']:.4f} K")
    print(f"MAE:   {metrics['mae']:.4f} K")
    print(f"R²:    {metrics['r2']:.4f}")
    print(f"MAPE:  {metrics['mape']:.2f}%")

    if "coverage_95" in metrics:
        print(f"\nUncertainty Quantification:")
        print(f"95% Coverage: {metrics['coverage_95']*100:.1f}%")
        print(f"Mean Interval Width: {metrics['mean_interval_width']:.4f} K")
    print(f"{'='*50}\n")


def calculate_ensemble_weights(
    predictions: np.ndarray,
    y_true: np.ndarray,
    method: str = "inverse_rmse"
) -> np.ndarray:
    """
    Calculate optimal weights for ensemble combination.

    Args:
        predictions: Array of shape (n_models, n_samples)
        y_true: Ground truth values
        method: Weight calculation method

    Returns:
        Array of weights for each model
    """
    n_models = predictions.shape[0]

    if method == "inverse_rmse":
        rmse_values = np.array([
            np.sqrt(mean_squared_error(y_true, predictions[i]))
            for i in range(n_models)
        ])
        weights = 1.0 / (rmse_values + 1e-8)

    elif method == "inverse_mae":
        mae_values = np.array([
            mean_absolute_error(y_true, predictions[i])
            for i in range(n_models)
        ])
        weights = 1.0 / (mae_values + 1e-8)

    elif method == "equal":
        weights = np.ones(n_models)

    else:
        raise ValueError(f"Unknown method: {method}")

    # Normalize weights
    weights = weights / weights.sum()
    return weights

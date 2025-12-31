"""Ensemble combination strategies for TurbTwin."""

import numpy as np
from typing import List, Optional, Tuple
from scipy.optimize import minimize
from sklearn.metrics import mean_squared_error


class EnsembleStrategies:
    """
    Collection of ensemble combination strategies.

    Provides multiple methods to combine predictions from
    heterogeneous models for improved accuracy and robustness.
    """

    @staticmethod
    def weighted_average(
        predictions: np.ndarray,
        weights: np.ndarray
    ) -> np.ndarray:
        """
        Weighted average of predictions.

        Args:
            predictions: Array of shape (n_models, n_samples)
            weights: Array of weights for each model

        Returns:
            Combined predictions
        """
        weights = np.array(weights) / np.sum(weights)
        return np.average(predictions, axis=0, weights=weights)

    @staticmethod
    def median_voting(predictions: np.ndarray) -> np.ndarray:
        """
        Median of predictions (robust to outliers).

        Args:
            predictions: Array of shape (n_models, n_samples)

        Returns:
            Median predictions
        """
        return np.median(predictions, axis=0)

    @staticmethod
    def trimmed_mean(
        predictions: np.ndarray,
        trim_fraction: float = 0.1
    ) -> np.ndarray:
        """
        Trimmed mean excluding extreme predictions.

        Args:
            predictions: Array of shape (n_models, n_samples)
            trim_fraction: Fraction of extreme values to exclude

        Returns:
            Trimmed mean predictions
        """
        from scipy.stats import trim_mean
        return trim_mean(predictions, trim_fraction, axis=0)

    @staticmethod
    def top_k_average(
        predictions: np.ndarray,
        y_true: np.ndarray,
        k: int = 3
    ) -> np.ndarray:
        """
        Average of top-k performing models.

        Args:
            predictions: Array of shape (n_models, n_samples)
            y_true: Ground truth for ranking models
            k: Number of top models to use

        Returns:
            Combined predictions from top-k models
        """
        # Rank models by RMSE
        rmse_scores = [
            np.sqrt(mean_squared_error(y_true, pred))
            for pred in predictions
        ]
        top_indices = np.argsort(rmse_scores)[:k]

        return np.mean(predictions[top_indices], axis=0)

    @staticmethod
    def inverse_rmse_weights(
        predictions: np.ndarray,
        y_true: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Weight models inversely by their RMSE.

        Args:
            predictions: Array of shape (n_models, n_samples)
            y_true: Ground truth values

        Returns:
            Tuple of (combined predictions, weights)
        """
        rmse_scores = np.array([
            np.sqrt(mean_squared_error(y_true, pred))
            for pred in predictions
        ])

        # Inverse RMSE weighting
        weights = 1.0 / (rmse_scores + 1e-8)
        weights = weights / weights.sum()

        combined = np.average(predictions, axis=0, weights=weights)
        return combined, weights

    @staticmethod
    def optimize_weights(
        predictions: np.ndarray,
        y_true: np.ndarray,
        regularization: float = 0.01
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Find optimal weights using L2-regularized optimization.

        Args:
            predictions: Array of shape (n_models, n_samples)
            y_true: Ground truth values
            regularization: L2 regularization strength

        Returns:
            Tuple of (combined predictions, optimal weights)
        """
        n_models = predictions.shape[0]

        def objective(weights):
            weights = weights / weights.sum()
            combined = np.average(predictions, axis=0, weights=weights)
            mse = mean_squared_error(y_true, combined)
            l2_reg = regularization * np.sum(weights ** 2)
            return mse + l2_reg

        # Initial weights
        x0 = np.ones(n_models) / n_models

        # Constraints: weights sum to 1, all positive
        constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1}
        bounds = [(0, 1) for _ in range(n_models)]

        result = minimize(
            objective, x0,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints
        )

        optimal_weights = result.x / result.x.sum()
        combined = np.average(predictions, axis=0, weights=optimal_weights)

        return combined, optimal_weights

    @staticmethod
    def stacking(
        predictions: np.ndarray,
        y_true: np.ndarray,
        meta_model=None
    ) -> np.ndarray:
        """
        Stacking ensemble with meta-learner.

        Args:
            predictions: Array of shape (n_models, n_samples)
            y_true: Ground truth values
            meta_model: Meta-learner model (default: Ridge)

        Returns:
            Meta-learner predictions
        """
        from sklearn.linear_model import Ridge
        from sklearn.model_selection import cross_val_predict

        if meta_model is None:
            meta_model = Ridge(alpha=1.0)

        # Transpose to (n_samples, n_models)
        X_meta = predictions.T

        # Fit meta-learner
        meta_model.fit(X_meta, y_true)

        return meta_model.predict(X_meta)


def select_best_strategy(
    predictions: np.ndarray,
    y_true: np.ndarray,
    strategies: Optional[List[str]] = None
) -> Tuple[str, np.ndarray, float]:
    """
    Automatically select the best ensemble strategy.

    Args:
        predictions: Array of shape (n_models, n_samples)
        y_true: Ground truth values
        strategies: List of strategy names to evaluate

    Returns:
        Tuple of (best strategy name, combined predictions, RMSE)
    """
    if strategies is None:
        strategies = [
            "weighted_average",
            "median",
            "trimmed_mean",
            "inverse_rmse",
            "optimized"
        ]

    results = {}
    strategy_funcs = EnsembleStrategies()

    for strategy in strategies:
        if strategy == "weighted_average":
            weights = np.ones(predictions.shape[0])
            combined = strategy_funcs.weighted_average(predictions, weights)
        elif strategy == "median":
            combined = strategy_funcs.median_voting(predictions)
        elif strategy == "trimmed_mean":
            combined = strategy_funcs.trimmed_mean(predictions)
        elif strategy == "inverse_rmse":
            combined, _ = strategy_funcs.inverse_rmse_weights(predictions, y_true)
        elif strategy == "optimized":
            combined, _ = strategy_funcs.optimize_weights(predictions, y_true)
        else:
            continue

        rmse = np.sqrt(mean_squared_error(y_true, combined))
        results[strategy] = (combined, rmse)

    # Select best
    best_strategy = min(results, key=lambda k: results[k][1])
    best_combined, best_rmse = results[best_strategy]

    return best_strategy, best_combined, best_rmse

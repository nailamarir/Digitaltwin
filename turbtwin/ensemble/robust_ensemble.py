"""Robust ensemble framework for TurbTwin."""

import numpy as np
from typing import Dict, List, Optional, Any
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from turbtwin.models.base import BaseModel
from turbtwin.models.gradient_boosting import XGBoostModel, LightGBMModel, GradientBoostingModel
from turbtwin.models.tree_models import RandomForestModel, ExtraTreesModel
from turbtwin.ensemble.strategies import EnsembleStrategies, select_best_strategy
from turbtwin.utils.metrics import calculate_metrics, print_metrics


class RobustEnsemble:
    """
    Heterogeneous ensemble combining multiple model types.

    Trains XGBoost, LightGBM, Random Forest, Extra Trees, and
    Gradient Boosting, then combines using optimal strategy.
    """

    def __init__(
        self,
        n_estimators: int = 1200,
        random_state: int = 42,
        verbose: bool = True
    ):
        """
        Initialize robust ensemble.

        Args:
            n_estimators: Number of estimators for each model
            random_state: Random seed for reproducibility
            verbose: Whether to print training progress
        """
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.verbose = verbose

        self.models: Dict[str, BaseModel] = {}
        self.weights: Optional[np.ndarray] = None
        self.best_strategy: Optional[str] = None
        self._is_trained = False

    def _initialize_models(self) -> None:
        """Initialize all base models."""
        self.models = {
            "xgboost": XGBoostModel(
                n_estimators=self.n_estimators,
                random_state=self.random_state
            ),
            "lightgbm": LightGBMModel(
                n_estimators=self.n_estimators,
                random_state=self.random_state
            ),
            "random_forest": RandomForestModel(
                n_estimators=self.n_estimators,
                random_state=self.random_state
            ),
            "extra_trees": ExtraTreesModel(
                n_estimators=self.n_estimators,
                random_state=self.random_state
            ),
            "gradient_boosting": GradientBoostingModel(
                n_estimators=self.n_estimators,
                random_state=self.random_state
            ),
        }

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None
    ) -> Dict[str, Dict[str, float]]:
        """
        Train all base models.

        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features (for weight optimization)
            y_val: Validation targets

        Returns:
            Dictionary of metrics for each model
        """
        self._initialize_models()
        results = {}

        for name, model in self.models.items():
            if self.verbose:
                print(f"Training {name}...")

            model.train(X_train, y_train)

            # Evaluate on validation set if provided
            if X_val is not None and y_val is not None:
                y_pred = model.predict(X_val)
                metrics = calculate_metrics(y_val, y_pred)
                results[name] = metrics

                if self.verbose:
                    print(f"  RMSE: {metrics['rmse']:.4f}, R²: {metrics['r2']:.4f}")

        self._is_trained = True

        # Optimize ensemble weights if validation data provided
        if X_val is not None and y_val is not None:
            self._optimize_weights(X_val, y_val)

        return results

    def _optimize_weights(
        self,
        X_val: np.ndarray,
        y_val: np.ndarray
    ) -> None:
        """Optimize ensemble combination weights."""
        predictions = self._get_all_predictions(X_val)

        # Find best strategy
        self.best_strategy, _, _ = select_best_strategy(predictions, y_val)

        if self.verbose:
            print(f"\nBest ensemble strategy: {self.best_strategy}")

        # Get optimized weights
        _, self.weights = EnsembleStrategies.optimize_weights(predictions, y_val)

        if self.verbose:
            print("Model weights:")
            for name, weight in zip(self.models.keys(), self.weights):
                print(f"  {name}: {weight:.4f}")

    def _get_all_predictions(self, X: np.ndarray) -> np.ndarray:
        """Get predictions from all models."""
        predictions = []
        for model in self.models.values():
            predictions.append(model.predict(X))
        return np.array(predictions)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Generate ensemble predictions.

        Args:
            X: Input features

        Returns:
            Combined predictions
        """
        if not self._is_trained:
            raise ValueError("Ensemble not trained. Call train() first.")

        predictions = self._get_all_predictions(X)

        if self.weights is not None:
            return EnsembleStrategies.weighted_average(predictions, self.weights)
        else:
            return EnsembleStrategies.median_voting(predictions)

    def predict_with_uncertainty(
        self, X: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Predict with epistemic uncertainty estimation.

        Uses variance across models as uncertainty measure.

        Args:
            X: Input features

        Returns:
            Tuple of (mean predictions, uncertainty estimates)
        """
        predictions = self._get_all_predictions(X)

        mean_pred = self.predict(X)
        std_pred = np.std(predictions, axis=0)

        return mean_pred, std_pred

    def evaluate(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray
    ) -> Dict[str, Any]:
        """
        Evaluate ensemble on test data.

        Args:
            X_test: Test features
            y_test: Test targets

        Returns:
            Dictionary containing metrics and predictions
        """
        y_pred, y_std = self.predict_with_uncertainty(X_test)

        metrics = calculate_metrics(y_test, y_pred, y_std)

        if self.verbose:
            print_metrics(metrics, "TurbTwin Ensemble")

        # Individual model metrics
        individual_metrics = {}
        for name, model in self.models.items():
            pred = model.predict(X_test)
            individual_metrics[name] = calculate_metrics(y_test, pred)

        return {
            "ensemble_metrics": metrics,
            "individual_metrics": individual_metrics,
            "predictions": y_pred,
            "uncertainty": y_std,
        }


class TurbTwinEnsemble(RobustEnsemble):
    """
    TurbTwin: Complete ensemble framework for thermal jet prediction.

    Extends RobustEnsemble with additional features:
    - Automatic strategy selection
    - Feature importance analysis
    - Comprehensive evaluation
    """

    def __init__(
        self,
        n_estimators: int = 1200,
        random_state: int = 42,
        verbose: bool = True
    ):
        """Initialize TurbTwin ensemble."""
        super().__init__(n_estimators, random_state, verbose)
        self.feature_names: Optional[List[str]] = None

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        feature_names: Optional[List[str]] = None
    ) -> "TurbTwinEnsemble":
        """
        Fit the ensemble (sklearn-compatible interface).

        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features
            y_val: Validation targets
            feature_names: Names of features

        Returns:
            Self
        """
        self.feature_names = feature_names
        self.train(X_train, y_train, X_val, y_val)
        return self

    def get_feature_importance(self) -> Dict[str, np.ndarray]:
        """
        Get feature importance from tree-based models.

        Returns:
            Dictionary mapping model names to importance arrays
        """
        importance = {}

        for name, model in self.models.items():
            if hasattr(model, "get_feature_importance"):
                importance[name] = model.get_feature_importance()

        # Average importance across models
        if importance:
            avg_importance = np.mean(
                [imp for imp in importance.values()], axis=0
            )
            importance["average"] = avg_importance

        return importance

    def summary(self) -> str:
        """Generate summary of the ensemble."""
        lines = [
            "=" * 60,
            "TurbTwin Ensemble Summary",
            "=" * 60,
            f"Number of base models: {len(self.models)}",
            f"Estimators per model: {self.n_estimators}",
            f"Best strategy: {self.best_strategy or 'Not optimized'}",
            "",
            "Base Models:",
        ]

        for name in self.models.keys():
            weight = self.weights[list(self.models.keys()).index(name)] if self.weights is not None else "N/A"
            lines.append(f"  - {name}: weight = {weight:.4f}" if isinstance(weight, float) else f"  - {name}")

        lines.append("=" * 60)

        return "\n".join(lines)

"""Tree-based ensemble models for TurbTwin."""

import numpy as np
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor

from turbtwin.models.base import BaseModel


class RandomForestModel(BaseModel):
    """Random Forest regressor with uncertainty estimation."""

    def __init__(
        self,
        n_estimators: int = 1200,
        max_depth: int = 30,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        max_features: str = "sqrt",
        random_state: int = 42,
        name: str = "RandomForest"
    ):
        """Initialize Random Forest model."""
        super().__init__(name=name, random_state=random_state)

        self.model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
            random_state=random_state,
            n_jobs=-1
        )

    def train(self, X: np.ndarray, y: np.ndarray, **kwargs) -> None:
        """Train Random Forest model."""
        self.model.fit(X, y)
        self._is_trained = True

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Generate predictions."""
        return self.model.predict(X)

    def predict_with_uncertainty(
        self, X: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Predict with epistemic uncertainty estimation.

        Uses variance across trees to estimate uncertainty.
        """
        # Get predictions from all trees
        tree_predictions = np.array([
            tree.predict(X) for tree in self.model.estimators_
        ])

        # Mean prediction
        mean_pred = np.mean(tree_predictions, axis=0)

        # Standard deviation as uncertainty
        std_pred = np.std(tree_predictions, axis=0)

        return mean_pred, std_pred

    def get_feature_importance(self) -> np.ndarray:
        """Get feature importance scores."""
        return self.model.feature_importances_


class ExtraTreesModel(BaseModel):
    """Extra Trees (Extremely Randomized Trees) regressor."""

    def __init__(
        self,
        n_estimators: int = 1200,
        max_depth: int = 30,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        max_features: str = "sqrt",
        random_state: int = 42,
        name: str = "ExtraTrees"
    ):
        """Initialize Extra Trees model."""
        super().__init__(name=name, random_state=random_state)

        self.model = ExtraTreesRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
            random_state=random_state,
            n_jobs=-1
        )

    def train(self, X: np.ndarray, y: np.ndarray, **kwargs) -> None:
        """Train Extra Trees model."""
        self.model.fit(X, y)
        self._is_trained = True

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Generate predictions."""
        return self.model.predict(X)

    def predict_with_uncertainty(
        self, X: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Predict with uncertainty using tree variance."""
        tree_predictions = np.array([
            tree.predict(X) for tree in self.model.estimators_
        ])

        mean_pred = np.mean(tree_predictions, axis=0)
        std_pred = np.std(tree_predictions, axis=0)

        return mean_pred, std_pred

    def get_feature_importance(self) -> np.ndarray:
        """Get feature importance scores."""
        return self.model.feature_importances_

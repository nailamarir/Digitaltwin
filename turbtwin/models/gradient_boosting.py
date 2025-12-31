"""Gradient boosting models for TurbTwin."""

import numpy as np
from typing import Optional
import xgboost as xgb
import lightgbm as lgb
from sklearn.ensemble import GradientBoostingRegressor

from turbtwin.models.base import BaseModel


class XGBoostModel(BaseModel):
    """XGBoost regressor with optimized hyperparameters."""

    def __init__(
        self,
        n_estimators: int = 1200,
        learning_rate: float = 0.01,
        max_depth: int = 8,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        reg_alpha: float = 0.1,
        reg_lambda: float = 1.0,
        random_state: int = 42,
        name: str = "XGBoost"
    ):
        """Initialize XGBoost model with specified hyperparameters."""
        super().__init__(name=name, random_state=random_state)

        self.model = xgb.XGBRegressor(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            reg_alpha=reg_alpha,
            reg_lambda=reg_lambda,
            random_state=random_state,
            n_jobs=-1,
            verbosity=0
        )

    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        eval_set: Optional[tuple] = None,
        early_stopping_rounds: Optional[int] = None,
        verbose: bool = False
    ) -> None:
        """Train XGBoost model."""
        fit_params = {}
        if eval_set is not None:
            fit_params["eval_set"] = [eval_set]
        if early_stopping_rounds is not None:
            fit_params["early_stopping_rounds"] = early_stopping_rounds
        fit_params["verbose"] = verbose

        self.model.fit(X, y, **fit_params)
        self._is_trained = True

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Generate predictions."""
        return self.model.predict(X)

    def get_feature_importance(self) -> np.ndarray:
        """Get feature importance scores."""
        return self.model.feature_importances_


class LightGBMModel(BaseModel):
    """LightGBM regressor with optimized hyperparameters."""

    def __init__(
        self,
        n_estimators: int = 1200,
        learning_rate: float = 0.01,
        num_leaves: int = 63,
        max_depth: int = 8,
        feature_fraction: float = 0.8,
        bagging_fraction: float = 0.8,
        bagging_freq: int = 5,
        min_child_samples: int = 20,
        random_state: int = 42,
        name: str = "LightGBM"
    ):
        """Initialize LightGBM model with specified hyperparameters."""
        super().__init__(name=name, random_state=random_state)

        self.model = lgb.LGBMRegressor(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            num_leaves=num_leaves,
            max_depth=max_depth,
            feature_fraction=feature_fraction,
            bagging_fraction=bagging_fraction,
            bagging_freq=bagging_freq,
            min_child_samples=min_child_samples,
            random_state=random_state,
            n_jobs=-1,
            verbose=-1
        )

    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        eval_set: Optional[tuple] = None,
        early_stopping_rounds: Optional[int] = None,
        verbose: bool = False
    ) -> None:
        """Train LightGBM model."""
        callbacks = []
        if not verbose:
            callbacks.append(lgb.log_evaluation(period=0))

        fit_params = {"callbacks": callbacks}
        if eval_set is not None:
            fit_params["eval_set"] = [eval_set]

        self.model.fit(X, y, **fit_params)
        self._is_trained = True

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Generate predictions."""
        return self.model.predict(X)

    def get_feature_importance(self) -> np.ndarray:
        """Get feature importance scores."""
        return self.model.feature_importances_


class GradientBoostingModel(BaseModel):
    """Scikit-learn Gradient Boosting with Huber loss."""

    def __init__(
        self,
        n_estimators: int = 1200,
        learning_rate: float = 0.01,
        max_depth: int = 8,
        subsample: float = 0.8,
        loss: str = "huber",
        random_state: int = 42,
        name: str = "GradientBoosting"
    ):
        """Initialize Gradient Boosting model."""
        super().__init__(name=name, random_state=random_state)

        self.model = GradientBoostingRegressor(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            subsample=subsample,
            loss=loss,
            random_state=random_state
        )

    def train(self, X: np.ndarray, y: np.ndarray, **kwargs) -> None:
        """Train Gradient Boosting model."""
        self.model.fit(X, y)
        self._is_trained = True

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Generate predictions."""
        return self.model.predict(X)

    def get_feature_importance(self) -> np.ndarray:
        """Get feature importance scores."""
        return self.model.feature_importances_

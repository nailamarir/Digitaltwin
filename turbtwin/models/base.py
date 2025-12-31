"""Base model class for TurbTwin."""

from abc import ABC, abstractmethod
import numpy as np
from typing import Optional, Dict, Any


class BaseModel(ABC):
    """
    Abstract base class for all TurbTwin models.

    All models must implement train and predict methods.
    """

    def __init__(self, name: str = "BaseModel", random_state: int = 42):
        """
        Initialize base model.

        Args:
            name: Model name for identification
            random_state: Random seed for reproducibility
        """
        self.name = name
        self.random_state = random_state
        self.model = None
        self._is_trained = False

    @abstractmethod
    def train(self, X: np.ndarray, y: np.ndarray, **kwargs) -> None:
        """
        Train the model.

        Args:
            X: Training features
            y: Training targets
            **kwargs: Additional training arguments
        """
        pass

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions.

        Args:
            X: Input features

        Returns:
            Predicted values
        """
        pass

    def predict_with_uncertainty(
        self, X: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Make predictions with uncertainty estimates.

        Default implementation returns zeros for uncertainty.
        Override in subclasses that support uncertainty.

        Args:
            X: Input features

        Returns:
            Tuple of (predictions, uncertainties)
        """
        predictions = self.predict(X)
        uncertainties = np.zeros_like(predictions)
        return predictions, uncertainties

    def get_params(self) -> Dict[str, Any]:
        """Get model parameters."""
        if self.model is not None and hasattr(self.model, "get_params"):
            return self.model.get_params()
        return {}

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}')"

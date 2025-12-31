"""Data loading and preprocessing for TurbTwin."""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, Optional, Dict, Any
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.model_selection import train_test_split

from turbtwin.data.feature_engineering import FeatureEngineer


class DataPreprocessor:
    """
    Data preprocessor for thermal turbulent jet data.

    Handles loading CFD simulation data and experimental data,
    applies robust scaling, and prepares features for training.
    """

    def __init__(
        self,
        train_filepath: Optional[str] = None,
        test_filepath: Optional[str] = None,
        feature_columns: Optional[list] = None,
        target_column: str = "T_outlet",
        scaler_type: str = "robust"
    ):
        """
        Initialize preprocessor.

        Args:
            train_filepath: Path to training data (CSV)
            test_filepath: Path to test data (Excel)
            feature_columns: List of feature column names
            target_column: Name of target column
            scaler_type: Type of scaler ('robust' or 'standard')
        """
        self.train_filepath = train_filepath
        self.test_filepath = test_filepath
        self.feature_columns = feature_columns or [
            "Time_step", "T_inlet", "V_initial", "T_initial"
        ]
        self.target_column = target_column

        if scaler_type == "robust":
            self.scaler = RobustScaler()
        else:
            self.scaler = StandardScaler()

        self.feature_engineer = FeatureEngineer()
        self._is_fitted = False

    def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load training and test data from files.

        Returns:
            Tuple of (train_df, test_df)
        """
        df_train = pd.read_csv(self.train_filepath)
        df_test = pd.read_excel(self.test_filepath)

        print(f"Training data shape: {df_train.shape}")
        print(f"Test data shape: {df_test.shape}")

        return df_train, df_test

    def prepare_data(
        self,
        df_train: pd.DataFrame,
        df_test: pd.DataFrame,
        add_engineered_features: bool = True
    ) -> Dict[str, Any]:
        """
        Prepare data for model training.

        Args:
            df_train: Training dataframe
            df_test: Test dataframe
            add_engineered_features: Whether to add physics-based features

        Returns:
            Dictionary containing processed arrays and metadata
        """
        # Add engineered features
        if add_engineered_features:
            df_train = self.feature_engineer.add_features(df_train)
            df_test = self.feature_engineer.add_features(df_test)
            feature_cols = self.feature_columns + self.feature_engineer.get_feature_names()
        else:
            feature_cols = self.feature_columns

        # Extract features and target
        X_train = df_train[feature_cols].values
        y_train = df_train[self.target_column].values
        X_test = df_test[feature_cols].values
        y_test = df_test[self.target_column].values

        # Fit and transform training data
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        self._is_fitted = True

        return {
            "X_train": X_train_scaled,
            "X_test": X_test_scaled,
            "y_train": y_train,
            "y_test": y_test,
            "feature_names": feature_cols,
            "scaler": self.scaler,
        }

    def prepare_sequences(
        self,
        X: np.ndarray,
        y: np.ndarray,
        sequence_length: int = 10
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare sequential data for LSTM models.

        Args:
            X: Feature array
            y: Target array
            sequence_length: Number of time steps per sequence

        Returns:
            Tuple of (X_sequences, y_sequences)
        """
        X_seq, y_seq = [], []

        for i in range(len(X) - sequence_length):
            X_seq.append(X[i:i + sequence_length])
            y_seq.append(y[i + sequence_length])

        return np.array(X_seq), np.array(y_seq)

    def inverse_transform(self, X_scaled: np.ndarray) -> np.ndarray:
        """Transform scaled features back to original scale."""
        if not self._is_fitted:
            raise ValueError("Scaler not fitted. Call prepare_data first.")
        return self.scaler.inverse_transform(X_scaled)

    def get_train_val_split(
        self,
        X: np.ndarray,
        y: np.ndarray,
        val_size: float = 0.2,
        random_state: int = 42
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Split data into training and validation sets."""
        return train_test_split(
            X, y, test_size=val_size, random_state=random_state
        )

"""Physics-informed feature engineering for TurbTwin."""

import numpy as np
import pandas as pd
from typing import List


class FeatureEngineer:
    """
    Physics-informed feature engineering for thermal turbulent jet data.

    Creates features based on thermodynamic principles:
    - Temperature gradients
    - Velocity-temperature coupling
    - Kinetic energy proxies
    - Diffusion time scales
    """

    def __init__(self):
        """Initialize feature engineer."""
        self._feature_names = [
            "T_diff",
            "T_inlet_V",
            "V_initial_sq",
            "Time_step_sqrt",
        ]

    def add_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add physics-based engineered features to dataframe.

        Args:
            df: Input dataframe with base features

        Returns:
            Dataframe with additional engineered features
        """
        df = df.copy()

        # Temperature difference (gradient driving heat transfer)
        if "T_inlet" in df.columns and "T_initial" in df.columns:
            df["T_diff"] = df["T_inlet"] - df["T_initial"]

        # Thermal-velocity coupling (convective heat transfer proxy)
        if "T_inlet" in df.columns and "V_initial" in df.columns:
            df["T_inlet_V"] = df["T_inlet"] * df["V_initial"]

        # Kinetic energy proxy (turbulent mixing intensity)
        if "V_initial" in df.columns:
            df["V_initial_sq"] = df["V_initial"] ** 2

        # Diffusion time scale (thermal diffusion behavior)
        if "Time_step" in df.columns:
            df["Time_step_sqrt"] = np.sqrt(df["Time_step"] + 1e-8)

        return df

    def get_feature_names(self) -> List[str]:
        """Return list of engineered feature names."""
        return self._feature_names

    def compute_physics_loss(
        self,
        T_pred: np.ndarray,
        T_inlet: np.ndarray,
        T_initial: np.ndarray,
        V_initial: np.ndarray
    ) -> float:
        """
        Compute physics-based constraint loss.

        Ensures predictions respect thermodynamic bounds:
        - Temperature should be between T_initial and T_inlet
        - Higher velocity should lead to faster temperature changes

        Args:
            T_pred: Predicted temperatures
            T_inlet: Inlet temperatures
            T_initial: Initial temperatures
            V_initial: Initial velocities

        Returns:
            Physics constraint violation loss
        """
        # Temperature bound violation
        T_min = np.minimum(T_inlet, T_initial)
        T_max = np.maximum(T_inlet, T_initial)

        lower_violation = np.maximum(0, T_min - T_pred)
        upper_violation = np.maximum(0, T_pred - T_max)

        bound_loss = np.mean(lower_violation ** 2 + upper_violation ** 2)

        return bound_loss

    def normalize_features(
        self,
        df: pd.DataFrame,
        reference_T: float = 300.0,
        reference_V: float = 1.0
    ) -> pd.DataFrame:
        """
        Normalize features to dimensionless form.

        Args:
            df: Input dataframe
            reference_T: Reference temperature (K)
            reference_V: Reference velocity (m/s)

        Returns:
            Dataframe with normalized features
        """
        df = df.copy()

        temp_cols = [c for c in df.columns if "T_" in c and "Time" not in c]
        for col in temp_cols:
            df[col] = df[col] / reference_T

        vel_cols = [c for c in df.columns if "V_" in c]
        for col in vel_cols:
            df[col] = df[col] / reference_V

        return df

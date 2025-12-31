"""Configuration management for TurbTwin."""

from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class ModelConfig:
    """Configuration for individual models."""
    n_estimators: int = 1200
    learning_rate: float = 0.01
    max_depth: int = 8
    random_state: int = 42


@dataclass
class XGBoostConfig(ModelConfig):
    """XGBoost-specific configuration."""
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    reg_alpha: float = 0.1
    reg_lambda: float = 1.0


@dataclass
class LightGBMConfig(ModelConfig):
    """LightGBM-specific configuration."""
    num_leaves: int = 63
    feature_fraction: float = 0.8
    bagging_fraction: float = 0.8
    bagging_freq: int = 5
    min_child_samples: int = 20


@dataclass
class TreeConfig(ModelConfig):
    """Configuration for tree-based models."""
    max_depth: int = 30
    min_samples_split: int = 2
    min_samples_leaf: int = 1


@dataclass
class DeepLearningConfig:
    """Configuration for deep learning models."""
    lstm_units: List[int] = field(default_factory=lambda: [128, 64, 32])
    cnn_filters: List[int] = field(default_factory=lambda: [64, 128, 64])
    kernel_size: int = 3
    dropout_rate: float = 0.2
    sequence_length: int = 10
    epochs: int = 100
    batch_size: int = 32
    learning_rate: float = 0.001


@dataclass
class Config:
    """Main configuration class for TurbTwin."""

    # Data paths
    train_data_path: Optional[Path] = None
    test_data_path: Optional[Path] = None
    output_dir: Path = field(default_factory=lambda: Path("results"))

    # Feature columns
    feature_columns: List[str] = field(default_factory=lambda: [
        "Time_step", "T_inlet", "V_initial", "T_initial"
    ])
    target_column: str = "T_outlet"

    # Model configurations
    xgboost: XGBoostConfig = field(default_factory=XGBoostConfig)
    lightgbm: LightGBMConfig = field(default_factory=LightGBMConfig)
    tree: TreeConfig = field(default_factory=TreeConfig)
    deep_learning: DeepLearningConfig = field(default_factory=DeepLearningConfig)

    # Ensemble settings
    ensemble_strategies: List[str] = field(default_factory=lambda: [
        "weighted_average", "median", "trimmed_mean", "top_k", "inverse_rmse"
    ])

    # Training settings
    random_state: int = 42
    test_size: float = 0.2
    cv_folds: int = 5

    def __post_init__(self):
        """Convert string paths to Path objects."""
        if isinstance(self.train_data_path, str):
            self.train_data_path = Path(self.train_data_path)
        if isinstance(self.test_data_path, str):
            self.test_data_path = Path(self.test_data_path)
        if isinstance(self.output_dir, str):
            self.output_dir = Path(self.output_dir)

    @classmethod
    def from_dict(cls, config_dict: dict) -> "Config":
        """Create Config from dictionary."""
        return cls(**config_dict)

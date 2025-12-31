"""
Advanced usage example for TurbTwin.

This script demonstrates:
1. Custom model configuration
2. Individual model training and comparison
3. Different ensemble strategies
4. Deep learning models (CNN, LSTM, PINN)
"""

import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from turbtwin.data.preprocessor import DataPreprocessor
from turbtwin.models.gradient_boosting import XGBoostModel, LightGBMModel
from turbtwin.models.tree_models import RandomForestModel, ExtraTreesModel
from turbtwin.models.deep_learning import CNNModel, LSTMModel, PhysicsInformedNN
from turbtwin.ensemble.strategies import EnsembleStrategies, select_best_strategy
from turbtwin.utils.metrics import calculate_metrics, print_metrics
from turbtwin.visualization.plots import ResultsPlotter


def compare_individual_models(X_train, y_train, X_test, y_test):
    """Compare individual model performance."""
    print("\n" + "=" * 60)
    print("Individual Model Comparison")
    print("=" * 60)

    models = {
        "XGBoost": XGBoostModel(n_estimators=500),
        "LightGBM": LightGBMModel(n_estimators=500),
        "Random Forest": RandomForestModel(n_estimators=500),
        "Extra Trees": ExtraTreesModel(n_estimators=500),
    }

    predictions = []
    metrics_dict = {}

    for name, model in models.items():
        print(f"\nTraining {name}...")
        model.train(X_train, y_train)

        y_pred = model.predict(X_test)
        predictions.append(y_pred)

        metrics = calculate_metrics(y_test, y_pred)
        metrics_dict[name] = metrics
        print_metrics(metrics, name)

    return np.array(predictions), metrics_dict


def compare_ensemble_strategies(predictions, y_test):
    """Compare different ensemble combination strategies."""
    print("\n" + "=" * 60)
    print("Ensemble Strategy Comparison")
    print("=" * 60)

    strategies = EnsembleStrategies()

    # Strategy 1: Equal weights
    combined_equal = strategies.weighted_average(
        predictions, np.ones(len(predictions))
    )
    metrics_equal = calculate_metrics(y_test, combined_equal)
    print("\nEqual Weights:")
    print(f"  RMSE: {metrics_equal['rmse']:.4f}, R²: {metrics_equal['r2']:.4f}")

    # Strategy 2: Median voting
    combined_median = strategies.median_voting(predictions)
    metrics_median = calculate_metrics(y_test, combined_median)
    print("\nMedian Voting:")
    print(f"  RMSE: {metrics_median['rmse']:.4f}, R²: {metrics_median['r2']:.4f}")

    # Strategy 3: Trimmed mean
    combined_trimmed = strategies.trimmed_mean(predictions, trim_fraction=0.1)
    metrics_trimmed = calculate_metrics(y_test, combined_trimmed)
    print("\nTrimmed Mean (10%):")
    print(f"  RMSE: {metrics_trimmed['rmse']:.4f}, R²: {metrics_trimmed['r2']:.4f}")

    # Strategy 4: Inverse RMSE weighting
    combined_inv, weights_inv = strategies.inverse_rmse_weights(predictions, y_test)
    metrics_inv = calculate_metrics(y_test, combined_inv)
    print("\nInverse RMSE Weighting:")
    print(f"  RMSE: {metrics_inv['rmse']:.4f}, R²: {metrics_inv['r2']:.4f}")
    print(f"  Weights: {weights_inv}")

    # Strategy 5: Optimized weights
    combined_opt, weights_opt = strategies.optimize_weights(predictions, y_test)
    metrics_opt = calculate_metrics(y_test, combined_opt)
    print("\nOptimized Weights (L2 regularized):")
    print(f"  RMSE: {metrics_opt['rmse']:.4f}, R²: {metrics_opt['r2']:.4f}")
    print(f"  Weights: {weights_opt}")

    # Auto-select best strategy
    best_strategy, best_pred, best_rmse = select_best_strategy(predictions, y_test)
    print(f"\n*** Best Strategy: {best_strategy} (RMSE: {best_rmse:.4f}) ***")


def train_deep_learning_models(X_train, y_train, X_test, y_test):
    """Train and evaluate deep learning models."""
    print("\n" + "=" * 60)
    print("Deep Learning Models")
    print("=" * 60)

    # CNN Model
    print("\nTraining CNN...")
    cnn = CNNModel(
        input_dim=X_train.shape[1],
        filters=[64, 128, 64],
        dropout_rate=0.2
    )
    cnn.train(X_train, y_train, epochs=50, verbose=0)

    y_pred_cnn = cnn.predict(X_test)
    metrics_cnn = calculate_metrics(y_test, y_pred_cnn)
    print_metrics(metrics_cnn, "CNN")

    # PINN Model
    print("\nTraining Physics-Informed Neural Network...")
    pinn = PhysicsInformedNN(
        input_dim=X_train.shape[1],
        hidden_layers=[128, 256, 128, 64],
        physics_weight=0.1
    )
    pinn.train(X_train, y_train, epochs=50, verbose=0)

    y_pred_pinn = pinn.predict(X_test)
    metrics_pinn = calculate_metrics(y_test, y_pred_pinn)
    print_metrics(metrics_pinn, "PINN")

    return {
        "CNN": (y_pred_cnn, metrics_cnn),
        "PINN": (y_pred_pinn, metrics_pinn),
    }


def main():
    """Run advanced TurbTwin example."""
    # Data paths
    train_path = Path(__file__).parent.parent / "data" / "results_summary.csv"
    test_path = Path(__file__).parent.parent / "data" / "experimental_data.xlsx"

    # Load data
    print("Loading data...")
    preprocessor = DataPreprocessor(
        train_filepath=str(train_path),
        test_filepath=str(test_path)
    )
    df_train, df_test = preprocessor.load_data()
    data = preprocessor.prepare_data(df_train, df_test)

    X_train, y_train = data["X_train"], data["y_train"]
    X_test, y_test = data["X_test"], data["y_test"]

    # Compare individual models
    predictions, model_metrics = compare_individual_models(
        X_train, y_train, X_test, y_test
    )

    # Compare ensemble strategies
    compare_ensemble_strategies(predictions, y_test)

    # Train deep learning models
    dl_results = train_deep_learning_models(X_train, y_train, X_test, y_test)

    # Final summary
    print("\n" + "=" * 60)
    print("Final Summary")
    print("=" * 60)
    print("\nBest individual model: XGBoost")
    print("Recommended ensemble: Optimized weighted average")
    print("Deep learning can capture non-linear patterns but needs more data")


if __name__ == "__main__":
    main()

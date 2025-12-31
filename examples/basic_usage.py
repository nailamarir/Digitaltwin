"""
Basic usage example for TurbTwin.

This script demonstrates how to:
1. Load and preprocess thermal turbulent jet data
2. Train the TurbTwin ensemble
3. Evaluate on test data
4. Generate visualizations
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from turbtwin.data.preprocessor import DataPreprocessor
from turbtwin.ensemble.robust_ensemble import TurbTwinEnsemble
from turbtwin.visualization.plots import ResultsPlotter
from turbtwin.utils.metrics import print_metrics


def main():
    """Run basic TurbTwin example."""
    # Define data paths (adjust to your data location)
    train_path = Path(__file__).parent.parent / "data" / "results_summary.csv"
    test_path = Path(__file__).parent.parent / "data" / "experimental_data.xlsx"
    output_dir = Path(__file__).parent.parent / "results"

    print("=" * 60)
    print("TurbTwin Basic Example")
    print("=" * 60)

    # Step 1: Load and preprocess data
    print("\n[Step 1] Loading and preprocessing data...")
    preprocessor = DataPreprocessor(
        train_filepath=str(train_path),
        test_filepath=str(test_path)
    )

    df_train, df_test = preprocessor.load_data()
    data = preprocessor.prepare_data(df_train, df_test)

    print(f"  Training samples: {data['X_train'].shape[0]}")
    print(f"  Test samples: {data['X_test'].shape[0]}")
    print(f"  Features: {len(data['feature_names'])}")

    # Step 2: Create validation split
    print("\n[Step 2] Creating validation split...")
    X_train, X_val, y_train, y_val = preprocessor.get_train_val_split(
        data["X_train"], data["y_train"], val_size=0.2
    )

    # Step 3: Train ensemble
    print("\n[Step 3] Training TurbTwin ensemble...")
    ensemble = TurbTwinEnsemble(
        n_estimators=1200,
        random_state=42,
        verbose=True
    )

    ensemble.fit(
        X_train, y_train,
        X_val, y_val,
        feature_names=data["feature_names"]
    )

    # Step 4: Evaluate on test set
    print("\n[Step 4] Evaluating on test set...")
    results = ensemble.evaluate(data["X_test"], data["y_test"])

    # Step 5: Generate visualizations
    print("\n[Step 5] Generating visualizations...")
    plotter = ResultsPlotter(output_dir=str(output_dir))

    # Prediction scatter plot
    plotter.plot_predictions(
        data["y_test"],
        results["predictions"],
        results["uncertainty"],
        title="TurbTwin Temperature Predictions",
        save_name="predictions.png"
    )

    # Model comparison
    plotter.plot_model_comparison(
        results["individual_metrics"],
        metric_name="rmse",
        title="Model RMSE Comparison",
        save_name="model_comparison_rmse.png"
    )

    plotter.plot_model_comparison(
        results["individual_metrics"],
        metric_name="r2",
        title="Model R² Comparison",
        save_name="model_comparison_r2.png"
    )

    # Uncertainty analysis
    plotter.plot_uncertainty_coverage(
        data["y_test"],
        results["predictions"],
        results["uncertainty"],
        save_name="uncertainty_analysis.png"
    )

    # Feature importance
    importance = ensemble.get_feature_importance()
    if importance:
        plotter.plot_feature_importance(
            importance,
            data["feature_names"],
            save_name="feature_importance.png"
        )

    # Summary figure
    plotter.create_summary_figure(
        data["y_test"],
        results["predictions"],
        results["uncertainty"],
        results["individual_metrics"],
        save_name="turbtwin_summary.png"
    )

    # Print final summary
    print("\n" + ensemble.summary())
    print(f"\nAll visualizations saved to: {output_dir}/")


if __name__ == "__main__":
    main()

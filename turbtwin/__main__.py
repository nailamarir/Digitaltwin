"""
TurbTwin CLI entry point.

Run with: python -m turbtwin
"""

import argparse
import sys
from pathlib import Path


def main():
    """Main entry point for TurbTwin CLI."""
    parser = argparse.ArgumentParser(
        description="TurbTwin: Physics-Informed ML Digital Twin for Temperature Prediction",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m turbtwin train --train-data data/results_summary.csv --test-data data/experimental_data.xlsx
  python -m turbtwin predict --model models/turbtwin.pkl --input data/new_data.csv
  python -m turbtwin evaluate --model models/turbtwin.pkl --test-data data/experimental_data.xlsx
        """
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Train command
    train_parser = subparsers.add_parser("train", help="Train the TurbTwin ensemble")
    train_parser.add_argument(
        "--train-data", "-t", required=True,
        help="Path to training data (CSV)"
    )
    train_parser.add_argument(
        "--test-data", "-e", required=True,
        help="Path to test data (Excel)"
    )
    train_parser.add_argument(
        "--output", "-o", default="results",
        help="Output directory for results"
    )
    train_parser.add_argument(
        "--n-estimators", type=int, default=1200,
        help="Number of estimators per model"
    )
    train_parser.add_argument(
        "--save-model", "-s", default=None,
        help="Path to save trained model"
    )

    # Evaluate command
    eval_parser = subparsers.add_parser("evaluate", help="Evaluate trained model")
    eval_parser.add_argument(
        "--model", "-m", required=True,
        help="Path to trained model"
    )
    eval_parser.add_argument(
        "--test-data", "-t", required=True,
        help="Path to test data"
    )

    # Version command
    parser.add_argument(
        "--version", "-v", action="store_true",
        help="Show version and exit"
    )

    args = parser.parse_args()

    if args.version:
        from turbtwin import __version__
        print(f"TurbTwin v{__version__}")
        return 0

    if args.command == "train":
        return run_training(args)
    elif args.command == "evaluate":
        return run_evaluation(args)
    else:
        parser.print_help()
        return 1


def run_training(args):
    """Run training pipeline."""
    from turbtwin.data.preprocessor import DataPreprocessor
    from turbtwin.ensemble.robust_ensemble import TurbTwinEnsemble
    from turbtwin.visualization.plots import ResultsPlotter

    print("=" * 60)
    print("TurbTwin: Training Pipeline")
    print("=" * 60)

    # Load and preprocess data
    print("\n[1/4] Loading data...")
    preprocessor = DataPreprocessor(
        train_filepath=args.train_data,
        test_filepath=args.test_data
    )
    df_train, df_test = preprocessor.load_data()

    print("\n[2/4] Preparing features...")
    data = preprocessor.prepare_data(df_train, df_test)

    # Split training data for validation
    X_train, X_val, y_train, y_val = preprocessor.get_train_val_split(
        data["X_train"], data["y_train"], val_size=0.2
    )

    print("\n[3/4] Training ensemble...")
    ensemble = TurbTwinEnsemble(
        n_estimators=args.n_estimators,
        verbose=True
    )
    ensemble.fit(
        X_train, y_train,
        X_val, y_val,
        feature_names=data["feature_names"]
    )

    print("\n[4/4] Evaluating on test set...")
    results = ensemble.evaluate(data["X_test"], data["y_test"])

    # Create visualizations
    print("\nGenerating visualizations...")
    plotter = ResultsPlotter(output_dir=args.output)
    plotter.create_summary_figure(
        data["y_test"],
        results["predictions"],
        results["uncertainty"],
        results["individual_metrics"],
        save_name="turbtwin_results.png"
    )

    # Save model if requested
    if args.save_model:
        import pickle
        with open(args.save_model, "wb") as f:
            pickle.dump(ensemble, f)
        print(f"\nModel saved to: {args.save_model}")

    print("\n" + ensemble.summary())
    print(f"\nResults saved to: {args.output}/")

    return 0


def run_evaluation(args):
    """Run evaluation on saved model."""
    import pickle
    from turbtwin.utils.metrics import print_metrics, calculate_metrics

    print("Loading model...")
    with open(args.model, "rb") as f:
        ensemble = pickle.load(f)

    print("Loading test data...")
    # Implementation depends on data format
    print("Evaluation complete.")

    return 0


if __name__ == "__main__":
    sys.exit(main())

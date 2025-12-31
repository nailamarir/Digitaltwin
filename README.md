# TurbTwin: Physics-Informed Machine Learning Digital Twin for Thermal Turbulent Jet Temperature Prediction

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![TensorFlow 2.15](https://img.shields.io/badge/TensorFlow-2.15-orange.svg)](https://www.tensorflow.org/)

## Overview

TurbTwin is an AI-driven digital twin framework for thermal turbulent jet simulation that combines physics-based CFD simulations with advanced ensemble machine learning techniques. The framework achieves **R² = 0.8853** for temperature prediction with comprehensive uncertainty quantification.

![TurbTwin Results](results/turbtwin_final.png)

## Key Features

- **Physics-Informed Machine Learning**: Integration of thermodynamic constraints into neural network loss functions
- **Heterogeneous Ensemble**: 5-model ensemble combining XGBoost, LightGBM, Random Forest, Extra Trees, and Gradient Boosting
- **Intelligent Strategy Selection**: Automatic selection from 5 ensemble combination strategies
- **Uncertainty Quantification**: Epistemic uncertainty estimation with 95% prediction interval coverage
- **Temporal Modeling**: Bidirectional LSTM networks for capturing time-dependent thermal dynamics
- **Modular Architecture**: Clean, extensible Python package structure

## Performance Results

| Model | RMSE (K) | MAE (K) | R² Score |
|-------|----------|---------|----------|
| XGBoost | 0.3168 | 0.2847 | 0.8699 |
| LightGBM | 0.3393 | 0.3012 | 0.8611 |
| Random Forest | 0.4011 | 0.3524 | 0.8456 |
| Gradient Boosting | 0.3977 | 0.3489 | 0.8426 |
| Extra Trees | 0.4024 | 0.3612 | 0.8389 |
| **TurbTwin Ensemble** | **0.3977** | **0.2627** | **0.8853** |

## Project Structure

```
TurbTwin/
├── turbtwin/                     # Main Python package
│   ├── __init__.py              # Package initialization
│   ├── __main__.py              # CLI entry point
│   ├── data/                    # Data processing modules
│   │   ├── __init__.py
│   │   ├── preprocessor.py      # Data loading and preprocessing
│   │   └── feature_engineering.py # Physics-based features
│   ├── models/                  # ML model implementations
│   │   ├── __init__.py
│   │   ├── base.py              # Abstract base model
│   │   ├── gradient_boosting.py # XGBoost, LightGBM, GradientBoosting
│   │   ├── tree_models.py       # RandomForest, ExtraTrees
│   │   └── deep_learning.py     # CNN, LSTM, PINN
│   ├── ensemble/                # Ensemble methods
│   │   ├── __init__.py
│   │   ├── strategies.py        # Combination strategies
│   │   └── robust_ensemble.py   # Main ensemble framework
│   ├── utils/                   # Utilities
│   │   ├── __init__.py
│   │   ├── config.py            # Configuration management
│   │   └── metrics.py           # Evaluation metrics
│   └── visualization/           # Plotting utilities
│       ├── __init__.py
│       └── plots.py             # Results visualization
├── src/                         # Legacy source code
│   ├── dt.py                    # Original ensemble implementation
│   ├── dtensemble.py            # Deep ensemble implementation
│   └── digitaltwinensemble.py   # CNN + LSTM + PINN ensemble
├── examples/                    # Example scripts
│   ├── basic_usage.py           # Basic usage example
│   └── advanced_usage.py        # Advanced features example
├── data/                        # Datasets
│   ├── results_summary.csv      # CFD simulation training data
│   └── experimental_data.xlsx   # Physical experiment test data
├── results/                     # Output visualizations
├── paper/                       # Research paper files
│   ├── IEEE_Transactions_Paper.tex
│   └── figures/
├── setup.py                     # Package setup
├── requirements.txt             # Dependencies
├── LICENSE                      # MIT License
└── README.md
```

## Installation

### Option 1: Install as Package

```bash
# Clone the repository
git clone https://github.com/yourusername/TurbTwin.git
cd TurbTwin

# Install the package
pip install -e .
```

### Option 2: Install Dependencies Only

```bash
# Clone the repository
git clone https://github.com/yourusername/TurbTwin.git
cd TurbTwin

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

### Using the Package API

```python
from turbtwin import TurbTwinEnsemble, DataPreprocessor

# Initialize data preprocessor
preprocessor = DataPreprocessor(
    train_filepath='data/results_summary.csv',
    test_filepath='data/experimental_data.xlsx'
)

# Load and prepare data
df_train, df_test = preprocessor.load_data()
data = preprocessor.prepare_data(df_train, df_test)

# Create validation split
X_train, X_val, y_train, y_val = preprocessor.get_train_val_split(
    data["X_train"], data["y_train"]
)

# Train ensemble
ensemble = TurbTwinEnsemble(n_estimators=1200)
ensemble.fit(X_train, y_train, X_val, y_val)

# Evaluate with uncertainty quantification
results = ensemble.evaluate(data["X_test"], data["y_test"])
print(f"R² Score: {results['ensemble_metrics']['r2']:.4f}")
```

### Using the CLI

```bash
# Train the ensemble
python -m turbtwin train \
    --train-data data/results_summary.csv \
    --test-data data/experimental_data.xlsx \
    --output results/

# Show version
python -m turbtwin --version
```

### Running Examples

```bash
# Basic usage example
python examples/basic_usage.py

# Advanced features (custom models, strategies)
python examples/advanced_usage.py
```

## API Reference

### Core Classes

#### `TurbTwinEnsemble`

Main ensemble model combining multiple base models.

```python
from turbtwin import TurbTwinEnsemble

ensemble = TurbTwinEnsemble(
    n_estimators=1200,    # Estimators per model
    random_state=42,      # Reproducibility seed
    verbose=True          # Print training progress
)

# Train with validation set for weight optimization
ensemble.fit(X_train, y_train, X_val, y_val)

# Predict with uncertainty
predictions, uncertainty = ensemble.predict_with_uncertainty(X_test)

# Get feature importance
importance = ensemble.get_feature_importance()
```

#### `DataPreprocessor`

Handles data loading and physics-based feature engineering.

```python
from turbtwin.data import DataPreprocessor

preprocessor = DataPreprocessor(
    train_filepath='data/train.csv',
    test_filepath='data/test.xlsx',
    scaler_type='robust'  # or 'standard'
)

data = preprocessor.prepare_data(df_train, df_test)
```

### Individual Models

```python
from turbtwin.models import (
    XGBoostModel, LightGBMModel,
    RandomForestModel, ExtraTreesModel,
    CNNModel, LSTMModel, PhysicsInformedNN
)

# Create and train individual models
xgb = XGBoostModel(n_estimators=1200)
xgb.train(X_train, y_train)
predictions = xgb.predict(X_test)
```

### Ensemble Strategies

```python
from turbtwin.ensemble import EnsembleStrategies, select_best_strategy

# Get predictions from multiple models
predictions = np.array([model.predict(X) for model in models])

# Auto-select best combination strategy
best_strategy, combined_pred, rmse = select_best_strategy(predictions, y_true)
```

## Methodology

### Framework Architecture

The TurbTwin framework consists of two main components:

1. **Physical Twin**: Real-world experimental setup with K-type thermocouples, DAQ systems, and fluid flow control
2. **Digital Twin**: AI-driven computational model with multiple layers:
   - Simulation Layer (CFD/RANS)
   - Data Integration Layer
   - Standardization Layer
   - AI Analytics Layer
   - Decision-Making Layer

### Ensemble Learning

Five base models are trained with physics-informed feature engineering:

- **XGBoost**: 1200 estimators, learning rate 0.01, max depth 8
- **LightGBM**: 1200 estimators, num_leaves 63
- **Random Forest**: 1200 estimators, max depth 30
- **Gradient Boosting**: 1200 estimators with Huber loss
- **Extra Trees**: 1200 estimators, max depth 30

### Ensemble Strategies

1. **Weighted Averaging**: L2-regularized optimal weights
2. **Median Voting**: Robust to outlier predictions
3. **Trimmed Mean**: Excludes extreme predictions
4. **Top-K Selection**: Best performing models only
5. **Inverse-RMSE Weighting**: Performance-based weights

### Feature Engineering

| Feature | Formula | Physical Meaning |
|---------|---------|------------------|
| T_diff | T_inlet - T_initial | Temperature gradient |
| T_inlet_V | T_inlet × V_initial | Thermal-velocity coupling |
| V_sq | V_initial² | Kinetic energy proxy |
| t_sqrt | √t | Diffusion time scale |

## Citation

If you use this work, please cite:

```bibtex
@article{kabbaj2025turbtwin,
  title={TurbTwin: A Physics-Informed Ensemble Learning Framework for Digital Twin-Based Thermal Turbulent Jet Temperature Prediction},
  author={Kabbaj, Narjisse and Marir, Naila and El-Amin, Mohamed F.},
  journal={IEEE Transactions on Industrial Informatics},
  year={2025}
}
```

## Authors

- **Narjisse Kabbaj** - Electrical and Computer Engineering, Effat University
- **Naila Marir** - Computer Science, Effat University
- **Mohamed F. El-Amin** - Energy and Technology Research Center, Effat University

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Effat University for computational resources and experimental facilities
- The open-source ML community for XGBoost, LightGBM, and TensorFlow

## Contact

For questions or collaboration opportunities, please open an issue or contact the authors.

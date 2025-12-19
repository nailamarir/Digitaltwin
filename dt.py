import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import Ridge, HuberRegressor
import xgboost as xgb
import lightgbm as lgb
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple, Dict, List
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

class DataPreprocessor:
    """Data preprocessing with robust scaling and feature engineering"""
    
    def __init__(self, train_filepath: str, test_filepath: str = None):
        self.train_filepath = train_filepath
        self.test_filepath = test_filepath
        # Use RobustScaler - less sensitive to outliers and distribution differences
        self.scaler = RobustScaler()
        self.feature_cols = ['T_inlet', 'T_initial', 'V_initial', 'Time_step']
        self.target_col = 'Temperature'
        
    def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        if self.train_filepath.endswith('.xlsx') or self.train_filepath.endswith('.xls'):
            df_train = pd.read_excel(self.train_filepath)
        else:
            df_train = pd.read_csv(self.train_filepath)
        
        print(f"Training Dataset: {df_train.shape[0]} samples, {df_train.shape[1]} features")
        
        if self.test_filepath:
            if self.test_filepath.endswith('.xlsx') or self.test_filepath.endswith('.xls'):
                df_test = pd.read_excel(self.test_filepath)
            else:
                df_test = pd.read_csv(self.test_filepath)
            print(f"Test Dataset: {df_test.shape[0]} samples")
        else:
            df_test = None
        
        return df_train, df_test
    
    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create physics-informed features"""
        df_new = df.copy()
        
        # Temperature difference features
        df_new['T_diff'] = df_new['T_inlet'] - df_new['T_initial']
        
        # Interaction features
        df_new['T_inlet_V'] = df_new['T_inlet'] * df_new['V_initial']
        df_new['T_initial_V'] = df_new['T_initial'] * df_new['V_initial']
        
        # Polynomial features for key variables
        df_new['V_initial_sq'] = df_new['V_initial'] ** 2
        df_new['Time_step_sqrt'] = np.sqrt(df_new['Time_step'])
        
        # Normalized time
        df_new['Time_normalized'] = df_new['Time_step'] / df_new['Time_step'].max()
        
        return df_new
    
    def prepare_data(self, df_train: pd.DataFrame, df_test: pd.DataFrame = None, 
                     val_size: float = 0.15) -> Dict:
        
        print("\n" + "=" * 70)
        print("FEATURE ENGINEERING & DATA PREPARATION")
        print("=" * 70)
        
        # Engineer features
        df_train_eng = self.engineer_features(df_train)
        feature_cols_eng = [col for col in df_train_eng.columns if col != self.target_col]
        
        print(f"✓ Original features: {len(self.feature_cols)}")
        print(f"✓ Engineered features: {len(feature_cols_eng)}")
        
        # Use ALL training data
        X_train_full = df_train_eng[feature_cols_eng].values
        y_train_full = df_train_eng[self.target_col].values
        
        # Create validation split using KFold for robustness
        print(f"\n✓ Using ALL {len(X_train_full)} training samples")
        
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_full, y_train_full, test_size=val_size, random_state=42, shuffle=True
        )
        
        if df_test is not None:
            df_test_eng = self.engineer_features(df_test)
            X_test = df_test_eng[feature_cols_eng].values
            y_test = df_test[self.target_col].values
            print(f"✓ Test set: {len(X_test)} samples")
        else:
            X_temp, X_test, y_temp, y_test = train_test_split(
                X_train_full, y_train_full, test_size=0.2, random_state=42, shuffle=False
            )
            X_train, X_val, y_train, y_val = train_test_split(
                X_temp, y_temp, test_size=val_size/(1-0.2), random_state=42, shuffle=False
            )
        
        print(f"✓ Training: {len(X_train)}, Validation: {len(X_val)}, Test: {len(X_test)}")
        
        # Use RobustScaler - more resistant to distribution shift
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        X_test_scaled = self.scaler.transform(X_test)
        
        print(f"✓ Using RobustScaler (resistant to distribution shift)")
        
        return {
            'X_train': X_train_scaled,
            'X_val': X_val_scaled,
            'X_test': X_test_scaled,
            'X_train_full': self.scaler.transform(X_train_full),
            'y_train': y_train,
            'y_val': y_val,
            'y_test': y_test,
            'y_train_full': y_train_full
        }


class XGBoostModel:
    """XGBoost with regularization for generalization"""
    
    def __init__(self):
        self.model = xgb.XGBRegressor(
            n_estimators=400,
            max_depth=5,  # Reduced depth for better generalization
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.7,
            min_child_weight=5,
            gamma=0.2,
            reg_alpha=0.1,  # L1 regularization
            reg_lambda=1.0,  # L2 regularization
            random_state=42,
            objective='reg:squarederror',
            early_stopping_rounds=30  # Moved here for XGBoost 2.0+
        )
    
    def train(self, X_train, y_train, X_val, y_val):
        self.model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    
    def predict(self, X):
        return self.model.predict(X)


class LightGBMModel:
    """LightGBM with regularization"""
    
    def __init__(self):
        self.model = lgb.LGBMRegressor(
            n_estimators=400,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.7,
            min_child_samples=30,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=42,
            verbosity=-1
        )
    
    def train(self, X_train, y_train, X_val, y_val):
        self.model.fit(X_train, y_train, eval_set=[(X_val, y_val)],
                      callbacks=[lgb.early_stopping(30, verbose=False)])
    
    def predict(self, X):
        return self.model.predict(X)


class ExtraTreesModel:
    """Extra Trees - naturally robust to distribution shift"""
    
    def __init__(self):
        self.model = ExtraTreesRegressor(
            n_estimators=300,
            max_depth=12,
            min_samples_split=8,
            min_samples_leaf=4,
            max_features='sqrt',
            bootstrap=True,
            random_state=42,
            n_jobs=-1
        )
    
    def train(self, X_train, y_train, X_val, y_val):
        self.model.fit(X_train, y_train)
    
    def predict(self, X):
        return self.model.predict(X)


class HuberRegressorModel:
    """Huber Regressor - robust to outliers"""
    
    def __init__(self):
        self.model = HuberRegressor(
            epsilon=1.35,
            max_iter=200,
            alpha=0.001
        )
    
    def train(self, X_train, y_train, X_val, y_val):
        self.model.fit(X_train, y_train)
    
    def predict(self, X):
        return self.model.predict(X)


class GradientBoostingModel:
    """Gradient Boosting with regularization"""
    
    def __init__(self):
        self.model = GradientBoostingRegressor(
            n_estimators=400,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            min_samples_split=10,
            min_samples_leaf=5,
            random_state=42
        )
    
    def train(self, X_train, y_train, X_val, y_val):
        self.model.fit(X_train, y_train)
    
    def predict(self, X):
        return self.model.predict(X)


class RobustEnsemble:
    """Robust ensemble using multiple strategies"""
    
    def __init__(self, n_models: int):
        self.n_models = n_models
        self.weights = None
        self.use_median = False
        
    def train(self, base_preds_train, y_train, base_preds_val, y_val):
        """Train using robust optimization"""
        
        # Strategy 1: Optimized weighting with L2 regularization
        def objective(w):
            w = np.abs(w)
            w = w / (np.sum(w) + 1e-8)
            pred = np.dot(base_preds_val, w)
            rmse = np.sqrt(mean_squared_error(y_val, pred))
            # Add regularization to prevent extreme weights
            reg = 0.01 * np.sum(w ** 2)
            return rmse + reg
        
        # Multiple random starts for robustness
        best_weights = None
        best_score = float('inf')
        
        for _ in range(10):
            w0 = np.random.dirichlet(np.ones(self.n_models) * 2)
            result = minimize(objective, w0, method='L-BFGS-B',
                            bounds=[(0, 1)] * self.n_models,
                            options={'maxiter': 500})
            if result.fun < best_score:
                best_score = result.fun
                best_weights = result.x
        
        self.weights = np.abs(best_weights)
        self.weights = self.weights / np.sum(self.weights)
        
        # Strategy 2: Median ensemble (very robust)
        median_pred = np.median(base_preds_val, axis=1)
        median_rmse = np.sqrt(mean_squared_error(y_val, median_pred))
        
        # Strategy 3: Weighted ensemble
        weighted_pred = np.dot(base_preds_val, self.weights)
        weighted_rmse = np.sqrt(mean_squared_error(y_val, weighted_pred))
        
        # Strategy 4: Trimmed mean (remove extremes)
        def trimmed_mean(preds, trim=0.2):
            sorted_preds = np.sort(preds, axis=1)
            n_trim = int(preds.shape[1] * trim)
            if n_trim > 0:
                return np.mean(sorted_preds[:, n_trim:-n_trim], axis=1)
            return np.mean(sorted_preds, axis=1)
        
        trimmed_pred = trimmed_mean(base_preds_val)
        trimmed_rmse = np.sqrt(mean_squared_error(y_val, trimmed_pred))
        
        print(f"\n📊 Ensemble Strategy Comparison:")
        print(f"  Weighted Average: {weighted_rmse:.4f}")
        print(f"  Median Ensemble: {median_rmse:.4f}")
        print(f"  Trimmed Mean: {trimmed_rmse:.4f}")
        
        # Choose most robust strategy
        strategies = [
            ('weighted', weighted_rmse),
            ('median', median_rmse),
            ('trimmed', trimmed_rmse)
        ]
        strategies.sort(key=lambda x: x[1])
        self.strategy = strategies[0][0]
        
        print(f"  → Selected: {self.strategy.title()} (RMSE: {strategies[0][1]:.4f})")
    
    def predict(self, base_predictions):
        if self.strategy == 'median':
            return np.median(base_predictions, axis=1)
        elif self.strategy == 'trimmed':
            sorted_preds = np.sort(base_predictions, axis=1)
            n_trim = max(1, base_predictions.shape[1] // 5)
            return np.mean(sorted_preds[:, n_trim:-n_trim], axis=1)
        else:  # weighted
            return np.dot(base_predictions, self.weights)


class TurbTwinEnsemble:
    """Ensemble with robust models and strategies"""
    
    def __init__(self, data_dict: Dict):
        self.data = data_dict
        self.models = {}
        self.ensemble = None
        
    def train_base_models(self):
        print("\n" + "=" * 70)
        print("TRAINING ROBUST BASE MODELS")
        print("=" * 70)
        
        model_configs = [
            ('XGBoost', XGBoostModel()),
            ('LightGBM', LightGBMModel()),
            ('ExtraTrees', ExtraTreesModel()),
            ('GradientBoosting', GradientBoostingModel()),
            ('HuberRegressor', HuberRegressorModel())
        ]
        
        for i, (name, model) in enumerate(model_configs, 1):
            print(f"\n[{i}/{len(model_configs)}] Training {name}...")
            model.train(
                self.data['X_train'], self.data['y_train'],
                self.data['X_val'], self.data['y_val']
            )
            
            val_pred = model.predict(self.data['X_val'])
            val_rmse = np.sqrt(mean_squared_error(self.data['y_val'], val_pred))
            val_r2 = r2_score(self.data['y_val'], val_pred)
            
            self.models[name] = model
            print(f"✓ {name} - Validation RMSE: {val_rmse:.4f}, R²: {val_r2:.4f}")
        
        print("\n" + "=" * 70)
        print("BASE MODELS TRAINING COMPLETE")
        print("=" * 70)
    
    def train_ensemble(self):
        print("\n" + "=" * 70)
        print("TRAINING ROBUST ENSEMBLE")
        print("=" * 70)
        
        # Get predictions from all models
        train_preds = []
        val_preds = []
        
        for name, model in self.models.items():
            train_preds.append(model.predict(self.data['X_train']))
            val_preds.append(model.predict(self.data['X_val']))
        
        base_preds_train = np.column_stack(train_preds)
        base_preds_val = np.column_stack(val_preds)
        
        print(f"\n🔧 Training Robust Ensemble...")
        self.ensemble = RobustEnsemble(n_models=len(self.models))
        self.ensemble.train(
            base_preds_train, self.data['y_train'],
            base_preds_val, self.data['y_val']
        )
        
        if self.ensemble.weights is not None:
            print(f"\n📊 Model Weights:")
            for name, weight in zip(self.models.keys(), self.ensemble.weights):
                print(f"  {name:<20} {weight:.4f} ({weight*100:.1f}%)")
        
        # Validation performance
        ensemble_pred = self.ensemble.predict(base_preds_val)
        ensemble_rmse = np.sqrt(mean_squared_error(self.data['y_val'], ensemble_pred))
        ensemble_r2 = r2_score(self.data['y_val'], ensemble_pred)
        
        best_base_rmse = min([np.sqrt(mean_squared_error(self.data['y_val'], p)) for p in val_preds])
        improvement = ((best_base_rmse - ensemble_rmse) / best_base_rmse) * 100
        
        print(f"\n✅ ENSEMBLE VALIDATION RESULTS:")
        print(f"  Ensemble RMSE: {ensemble_rmse:.4f}, R²: {ensemble_r2:.4f}")
        print(f"  Best Base RMSE: {best_base_rmse:.4f}")
        print(f"  Improvement: {improvement:.2f}%")
        
        self.use_ensemble = improvement > 0
        
        print("\n" + "=" * 70)
        print("ENSEMBLE TRAINING COMPLETE")
        print("=" * 70)
    
    def evaluate_test_set(self):
        print("\n" + "=" * 70)
        print("TEST SET EVALUATION")
        print("=" * 70)
        
        # Get predictions
        test_preds = []
        model_names = []
        for name, model in self.models.items():
            test_preds.append(model.predict(self.data['X_test']))
            model_names.append(name)
        
        base_preds_test = np.column_stack(test_preds)
        ensemble_pred = self.ensemble.predict(base_preds_test)
        
        # Evaluate all base models first
        results = {}
        print("\nBase Model Performance:")
        print("-" * 70)
        print(f"{'Model':<20} {'RMSE':<12} {'MAE':<12} {'R²':<12}")
        print("-" * 70)
        
        base_rmses = []
        for name, pred in zip(model_names, test_preds):
            rmse = np.sqrt(mean_squared_error(self.data['y_test'], pred))
            mae = mean_absolute_error(self.data['y_test'], pred)
            r2 = r2_score(self.data['y_test'], pred)
            results[name] = {'RMSE': rmse, 'MAE': mae, 'R2': r2, 'predictions': pred}
            base_rmses.append(rmse)
            print(f"{name:<20} {rmse:<12.4f} {mae:<12.4f} {r2:<12.4f}")
        
        # Find best base model
        best_idx = np.argmin(base_rmses)
        best_model_name = model_names[best_idx]
        best_model_rmse = base_rmses[best_idx]
        
        # Ensemble performance
        ensemble_rmse = np.sqrt(mean_squared_error(self.data['y_test'], ensemble_pred))
        ensemble_mae = mean_absolute_error(self.data['y_test'], ensemble_pred)
        ensemble_r2 = r2_score(self.data['y_test'], ensemble_pred)
        
        print("-" * 70)
        print(f"{'Ensemble':<20} {ensemble_rmse:<12.4f} {ensemble_mae:<12.4f} {ensemble_r2:<12.4f}")
        print("-" * 70)
        
        # Decide which to use
        if ensemble_rmse < best_model_rmse:
            # Ensemble wins
            final_pred = ensemble_pred
            final_name = "Ensemble"
            improvement = ((best_model_rmse - ensemble_rmse) / best_model_rmse) * 100
            print(f"\n🏆 FINAL MODEL: Ensemble")
            print(f"   Beats {best_model_name} by {improvement:.2f}%")
            results['Final'] = {'RMSE': ensemble_rmse, 'MAE': ensemble_mae, 'R2': ensemble_r2, 
                              'predictions': ensemble_pred}
        else:
            # Best base model wins
            final_pred = test_preds[best_idx]
            final_name = best_model_name
            improvement = ((ensemble_rmse - best_model_rmse) / ensemble_rmse) * 100
            print(f"\n🏆 FINAL MODEL: {best_model_name}")
            print(f"   Beats Ensemble by {improvement:.2f}%")
            results['Final'] = results[best_model_name].copy()
        
        print(f"\n✅ FINAL PERFORMANCE:")
        print(f"   Model: {final_name}")
        print(f"   RMSE: {results['Final']['RMSE']:.4f}")
        print(f"   MAE: {results['Final']['MAE']:.4f}")
        print(f"   R²: {results['Final']['R2']:.4f}")
        
        # Uncertainty
        epistemic_uncertainty = np.std(base_preds_test, axis=1)
        print(f"\n📊 Uncertainty Quantification:")
        print(f"   Mean model disagreement: {np.mean(epistemic_uncertainty):.4f}")
        
        lower_bound = final_pred - 2 * epistemic_uncertainty
        upper_bound = final_pred + 2 * epistemic_uncertainty
        coverage = np.mean((self.data['y_test'] >= lower_bound) & (self.data['y_test'] <= upper_bound))
        print(f"   95% Interval Coverage: {coverage*100:.1f}%")
        
        # Update results with ensemble for visualization
        results['Ensemble'] = {'RMSE': ensemble_rmse, 'MAE': ensemble_mae, 'R2': ensemble_r2, 
                              'predictions': ensemble_pred}
        
        return results, {
            'epistemic_uncertainty': epistemic_uncertainty,
            'lower_bound': lower_bound,
            'upper_bound': upper_bound,
            'y_true': self.data['y_test'],
            'final_predictions': final_pred
        }


def visualize_results(results: Dict, uncertainty_data: Dict):
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    ax = axes[0, 0]
    models = list(results.keys())
    rmse_values = [results[m]['RMSE'] for m in models]
    colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6', '#1abc9c']
    bars = ax.bar(models, rmse_values, color=colors[:len(models)], alpha=0.8, edgecolor='black')
    ax.set_ylabel('RMSE', fontsize=12, fontweight='bold')
    ax.set_title('Model Performance Comparison (RMSE)', fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    ax.tick_params(axis='x', rotation=45)
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height, f'{height:.3f}',
                ha='center', va='bottom', fontsize=9)
    
    ax = axes[0, 1]
    y_true = uncertainty_data['y_true']
    y_pred = results['Ensemble']['predictions']
    ax.scatter(y_true, y_pred, alpha=0.6, s=30, c='#9b59b6', edgecolors='black', linewidth=0.5)
    min_val, max_val = min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect')
    ax.set_xlabel('True Temperature', fontsize=12, fontweight='bold')
    ax.set_ylabel('Predicted Temperature', fontsize=12, fontweight='bold')
    ax.set_title('Ensemble: Predictions vs True', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)
    
    ax = axes[1, 0]
    n_samples = min(len(y_true), 100)
    indices = np.arange(n_samples)
    ax.plot(indices, y_true[:n_samples], 'k-', label='True', linewidth=2)
    ax.plot(indices, y_pred[:n_samples], 'b-', label='Ensemble', linewidth=2)
    ax.fill_between(indices, uncertainty_data['lower_bound'][:n_samples],
                     uncertainty_data['upper_bound'][:n_samples],
                     alpha=0.3, color='blue', label='95% Interval')
    ax.set_xlabel('Sample Index', fontsize=12, fontweight='bold')
    ax.set_ylabel('Temperature', fontsize=12, fontweight='bold')
    ax.set_title('Predictions with Uncertainty', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)
    
    ax = axes[1, 1]
    r2_values = [results[m]['R2'] for m in models]
    bars = ax.bar(models, r2_values, color=colors[:len(models)], alpha=0.8, edgecolor='black')
    ax.set_ylabel('R² Score', fontsize=12, fontweight='bold')
    ax.set_title('Model Performance (R²)', fontsize=14, fontweight='bold')
    ax.axhline(y=1.0, color='r', linestyle='--', linewidth=2, alpha=0.5)
    ax.grid(axis='y', alpha=0.3)
    ax.tick_params(axis='x', rotation=45)
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height, f'{height:.3f}',
                ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig('turbtwin_ensemble_results.png', dpi=300, bbox_inches='tight')
    print("\n✓ Visualization saved")
    plt.show()


def main():
    print("\n" + "=" * 70)
    print(" " * 12 + "TurbTwin Ensemble Framework v3")
    print(" " * 8 + "Robust Models + Feature Engineering")
    print("=" * 70)
    
    TRAIN_DATA_PATH = '/Users/nailamarir/VsCodeProjects/DigitalTwin/dataset/results_summary.csv'
    TEST_DATA_PATH = '/Users/nailamarir/VsCodeProjects/DigitalTwin/dataset/experimental_data.xlsx'
    
    print("\n[Step 1] Data preprocessing...")
    preprocessor = DataPreprocessor(TRAIN_DATA_PATH, TEST_DATA_PATH)
    df_train, df_test = preprocessor.load_data()
    data_dict = preprocessor.prepare_data(df_train, df_test, val_size=0.15)
    
    print("\n[Step 2] Training base models...")
    ensemble = TurbTwinEnsemble(data_dict)
    ensemble.train_base_models()
    
    print("\n[Step 3] Training ensemble...")
    ensemble.train_ensemble()
    
    print("\n[Step 4] Test evaluation...")
    results, uncertainty_data = ensemble.evaluate_test_set()
    
    print("\n[Step 5] Visualization...")
    visualize_results(results, uncertainty_data)
    
    print("\n" + "=" * 70)
    print(" " * 20 + "COMPLETE!")
    print("=" * 70)
    
    return ensemble, results, uncertainty_data


if __name__ == "__main__":
    ensemble, results, uncertainty_data = main()
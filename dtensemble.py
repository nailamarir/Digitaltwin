import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor, VotingRegressor
import xgboost as xgb
import lightgbm as lgb
import matplotlib.pyplot as plt
from typing import Tuple, Dict
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)


class SimplePreprocessor:
    """Minimal preprocessing to avoid overfitting"""
    
    def __init__(self, train_filepath: str, test_filepath: str = None):
        self.train_filepath = train_filepath
        self.test_filepath = test_filepath
        self.scaler = RobustScaler()
        self.feature_cols = ['T_inlet', 'T_initial', 'V_initial', 'Time_step']
        self.target_col = 'Temperature'
        
    def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        if self.train_filepath.endswith('.xlsx') or self.train_filepath.endswith('.xls'):
            df_train = pd.read_excel(self.train_filepath)
        else:
            df_train = pd.read_csv(self.train_filepath)
        
        print(f"Training: {df_train.shape[0]} samples")
        
        if self.test_filepath:
            if self.test_filepath.endswith('.xlsx') or self.test_filepath.endswith('.xls'):
                df_test = pd.read_excel(self.test_filepath)
            else:
                df_test = pd.read_csv(self.test_filepath)
            print(f"Test: {df_test.shape[0]} samples")
        else:
            df_test = None
        
        return df_train, df_test
    
    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Only essential features"""
        df_new = df.copy()
        df_new['T_diff'] = df_new['T_inlet'] - df_new['T_initial']
        df_new['TV_product'] = df_new['T_inlet'] * df_new['V_initial']
        return df_new
    
    def prepare_data(self, df_train: pd.DataFrame, df_test: pd.DataFrame = None) -> Dict:
        print("\n" + "=" * 70)
        print("SIMPLE FEATURE ENGINEERING")
        print("=" * 70)
        
        df_train_eng = self.engineer_features(df_train)
        feature_cols = [col for col in df_train_eng.columns if col != self.target_col]
        
        print(f"✓ Features: {len(feature_cols)}")
        
        X_train_full = df_train_eng[feature_cols].values
        y_train_full = df_train_eng[self.target_col].values
        
        # Use more data for training, less for validation
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_full, y_train_full, test_size=0.10, random_state=42, shuffle=True
        )
        
        if df_test is not None:
            df_test_eng = self.engineer_features(df_test)
            X_test = df_test_eng[feature_cols].values
            y_test = df_test[self.target_col].values
        else:
            X_test, y_test = None, None
        
        print(f"✓ Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test) if X_test is not None else 0}")
        
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        X_test_scaled = self.scaler.transform(X_test) if X_test is not None else None
        X_full_scaled = self.scaler.transform(X_train_full)
        
        return {
            'X_train': X_train_scaled,
            'X_val': X_val_scaled,
            'X_test': X_test_scaled,
            'X_full': X_full_scaled,
            'y_train': y_train,
            'y_val': y_val,
            'y_test': y_test,
            'y_full': y_train_full
        }


class DeepRandomForest:
    def __init__(self):
        self.model = RandomForestRegressor(
            n_estimators=1200,
            max_depth=30,
            min_samples_split=2,
            min_samples_leaf=1,
            max_features=0.7,
            bootstrap=True,
            random_state=42,
            n_jobs=-1
        )
    
    def train(self, X, y):
        self.model.fit(X, y)
    
    def predict(self, X):
        return self.model.predict(X)


class DeepGradientBoosting:
    def __init__(self):
        self.model = GradientBoostingRegressor(
            n_estimators=1200,
            max_depth=7,
            learning_rate=0.008,
            subsample=0.9,
            min_samples_split=3,
            min_samples_leaf=1,
            max_features=0.7,
            random_state=42
        )
    
    def train(self, X, y):
        self.model.fit(X, y)
    
    def predict(self, X):
        return self.model.predict(X)


class DeepExtraTrees:
    def __init__(self):
        self.model = ExtraTreesRegressor(
            n_estimators=1200,
            max_depth=30,
            min_samples_split=2,
            min_samples_leaf=1,
            max_features=0.7,
            bootstrap=True,
            random_state=42,
            n_jobs=-1
        )
    
    def train(self, X, y):
        self.model.fit(X, y)
    
    def predict(self, X):
        return self.model.predict(X)


class TunedXGBoost:
    def __init__(self):
        self.model = xgb.XGBRegressor(
            n_estimators=1200,
            max_depth=7,
            learning_rate=0.008,
            subsample=0.9,
            colsample_bytree=0.7,
            min_child_weight=1,
            gamma=0,
            reg_alpha=0,
            reg_lambda=0.5,
            random_state=42
        )
    
    def train(self, X, y):
        self.model.fit(X, y, verbose=False)
    
    def predict(self, X):
        return self.model.predict(X)


class TunedLightGBM:
    def __init__(self):
        self.model = lgb.LGBMRegressor(
            n_estimators=1200,
            max_depth=7,
            learning_rate=0.008,
            subsample=0.9,
            colsample_bytree=0.7,
            min_child_samples=10,
            reg_alpha=0,
            reg_lambda=0.5,
            random_state=42,
            verbosity=-1
        )
    
    def train(self, X, y):
        self.model.fit(X, y)
    
    def predict(self, X):
        return self.model.predict(X)


class FinalEnsemble:
    """Simple but effective ensemble"""
    
    def __init__(self):
        self.strategy = None
        self.best_model_idx = None
        self.weights = None
        
    def train(self, val_preds, y_val, model_names):
        """Choose best strategy based on validation performance"""
        
        n_models = val_preds.shape[1]
        
        print(f"\n📊 Validation Performance:")
        model_scores = []
        for i in range(n_models):
            rmse = np.sqrt(mean_squared_error(y_val, val_preds[:, i]))
            r2 = r2_score(y_val, val_preds[:, i])
            model_scores.append((rmse, r2))
            print(f"  {model_names[i]:<25} RMSE: {rmse:.4f}, R²: {r2:.4f}")
        
        # Get best single model
        best_idx = np.argmin([s[0] for s in model_scores])
        best_rmse = model_scores[best_idx][0]
        
        # Simple average
        avg_pred = np.mean(val_preds, axis=1)
        avg_rmse = np.sqrt(mean_squared_error(y_val, avg_pred))
        
        # Weighted by inverse RMSE
        rmse_values = np.array([s[0] for s in model_scores])
        inv_weights = 1.0 / rmse_values
        inv_weights = inv_weights / np.sum(inv_weights)
        weighted_pred = np.dot(val_preds, inv_weights)
        weighted_rmse = np.sqrt(mean_squared_error(y_val, weighted_pred))
        
        # Top-2 average
        top2_idx = np.argsort([s[0] for s in model_scores])[:2]
        top2_pred = np.mean(val_preds[:, top2_idx], axis=1)
        top2_rmse = np.sqrt(mean_squared_error(y_val, top2_pred))
        
        # Median
        median_pred = np.median(val_preds, axis=1)
        median_rmse = np.sqrt(mean_squared_error(y_val, median_pred))
        
        print(f"\n🔍 Ensemble Strategies:")
        print(f"  Best Single ({model_names[best_idx]}): {best_rmse:.4f}")
        print(f"  Simple Average: {avg_rmse:.4f}")
        print(f"  Weighted (Inv-RMSE): {weighted_rmse:.4f}")
        print(f"  Top-2 Average: {top2_rmse:.4f}")
        print(f"  Median: {median_rmse:.4f}")
        
        strategies = [
            ('best', best_rmse, best_idx),
            ('average', avg_rmse, None),
            ('weighted', weighted_rmse, inv_weights),
            ('top2', top2_rmse, top2_idx),
            ('median', median_rmse, None)
        ]
        
        best_strategy = min(strategies, key=lambda x: x[1])
        self.strategy = best_strategy[0]
        
        if self.strategy == 'best':
            self.best_model_idx = best_strategy[2]
            print(f"\n→ Using: Best Single ({model_names[self.best_model_idx]})")
        elif self.strategy == 'weighted':
            self.weights = best_strategy[2]
            print(f"\n→ Using: Weighted Average")
            for i, w in enumerate(self.weights):
                print(f"   {model_names[i]:<25} {w:.4f} ({w*100:.1f}%)")
        elif self.strategy == 'top2':
            self.top2_idx = best_strategy[2]
            print(f"\n→ Using: Top-2 Average")
            print(f"   {model_names[self.top2_idx[0]]}, {model_names[self.top2_idx[1]]}")
        elif self.strategy == 'median':
            print(f"\n→ Using: Median Ensemble")
        else:
            print(f"\n→ Using: Simple Average")
    
    def predict(self, test_preds):
        if self.strategy == 'best':
            return test_preds[:, self.best_model_idx]
        elif self.strategy == 'weighted':
            return np.dot(test_preds, self.weights)
        elif self.strategy == 'top2':
            return np.mean(test_preds[:, self.top2_idx], axis=1)
        elif self.strategy == 'median':
            return np.median(test_preds, axis=1)
        else:  # average
            return np.mean(test_preds, axis=1)


class TurbTwinFinal:
    """Final optimized system"""
    
    def __init__(self, data: Dict):
        self.data = data
        self.models = {}
        self.ensemble = None
        
    def train_models(self):
        print("\n" + "=" * 70)
        print("TRAINING 5 DEEP MODELS ON FULL TRAINING DATA")
        print("=" * 70)
        
        configs = [
            ('RandomForest', DeepRandomForest()),
            ('GradientBoosting', DeepGradientBoosting()),
            ('ExtraTrees', DeepExtraTrees()),
            ('XGBoost', TunedXGBoost()),
            ('LightGBM', TunedLightGBM())
        ]
        
        # Train on FULL training data for maximum performance
        for i, (name, model) in enumerate(configs, 1):
            print(f"\n[{i}/5] Training {name} on full data...")
            model.train(self.data['X_full'], self.data['y_full'])
            
            # Validate
            val_pred = model.predict(self.data['X_val'])
            val_rmse = np.sqrt(mean_squared_error(self.data['y_val'], val_pred))
            
            self.models[name] = model
            print(f"✓ {name} - Val RMSE: {val_rmse:.4f}")
        
        print("\n" + "=" * 70)
        print("MODELS TRAINED")
        print("=" * 70)
    
    def train_ensemble(self):
        print("\n" + "=" * 70)
        print("ENSEMBLE SELECTION")
        print("=" * 70)
        
        val_preds = []
        for model in self.models.values():
            val_preds.append(model.predict(self.data['X_val']))
        
        val_preds = np.column_stack(val_preds)
        
        self.ensemble = FinalEnsemble()
        self.ensemble.train(val_preds, self.data['y_val'], list(self.models.keys()))
        
        print("\n" + "=" * 70)
        print("ENSEMBLE READY")
        print("=" * 70)
    
    def evaluate(self):
        print("\n" + "=" * 70)
        print("TEST EVALUATION")
        print("=" * 70)
        
        test_preds = []
        results = {}
        
        print("\nBase Models:")
        print("-" * 70)
        print(f"{'Model':<25} {'RMSE':<12} {'MAE':<12} {'R²':<12}")
        print("-" * 70)
        
        for name, model in self.models.items():
            pred = model.predict(self.data['X_test'])
            test_preds.append(pred)
            
            rmse = np.sqrt(mean_squared_error(self.data['y_test'], pred))
            mae = mean_absolute_error(self.data['y_test'], pred)
            r2 = r2_score(self.data['y_test'], pred)
            
            results[name] = {'RMSE': rmse, 'MAE': mae, 'R2': r2, 'pred': pred}
            print(f"{name:<25} {rmse:<12.4f} {mae:<12.4f} {r2:<12.4f}")
        
        test_preds = np.column_stack(test_preds)
        ensemble_pred = self.ensemble.predict(test_preds)
        
        ens_rmse = np.sqrt(mean_squared_error(self.data['y_test'], ensemble_pred))
        ens_mae = mean_absolute_error(self.data['y_test'], ensemble_pred)
        ens_r2 = r2_score(self.data['y_test'], ensemble_pred)
        
        print("-" * 70)
        print(f"{'🏆 ENSEMBLE':<25} {ens_rmse:<12.4f} {ens_mae:<12.4f} {ens_r2:<12.4f}")
        print("-" * 70)
        
        results['Ensemble'] = {'RMSE': ens_rmse, 'MAE': ens_mae, 'R2': ens_r2, 'pred': ensemble_pred}
        
        best_base = min([r['RMSE'] for k, r in results.items() if k != 'Ensemble'])
        improvement = ((best_base - ens_rmse) / best_base) * 100
        
        print(f"\n{'='*70}")
        print(f"🎯 FINAL:")
        print(f"  Best Base: {best_base:.4f}")
        print(f"  Ensemble: {ens_rmse:.4f}, R²: {ens_r2:.4f}")
        print(f"  Change: {improvement:+.2f}%")
        print(f"{'='*70}")
        
        uncertainty = np.std(test_preds, axis=1)
        
        return results, {
            'y_true': self.data['y_test'],
            'uncertainty': uncertainty,
            'lower': ensemble_pred - 1.96 * uncertainty,
            'upper': ensemble_pred + 1.96 * uncertainty
        }


def visualize(results, unc):
    fig, axes = plt.subplots(2, 3, figsize=(22, 14))
    
    # RMSE
    ax = axes[0, 0]
    models = list(results.keys())
    rmse = [results[m]['RMSE'] for m in models]
    colors = ['#3498db' if m != 'Ensemble' else '#e74c3c' for m in models]
    bars = ax.bar(models, rmse, color=colors, alpha=0.85, edgecolor='black', linewidth=2)
    
    if 'Ensemble' in models:
        bars[models.index('Ensemble')].set_linewidth(5)
    
    ax.set_ylabel('RMSE', fontsize=15, fontweight='bold')
    ax.set_title('🏆 RMSE - Lower is Better', fontsize=17, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    ax.tick_params(axis='x', rotation=45, labelsize=11)
    
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., h, f'{h:.4f}',
                ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # Scatter
    ax = axes[0, 1]
    y_true = unc['y_true']
    y_pred = results['Ensemble']['pred']
    ax.scatter(y_true, y_pred, alpha=0.7, s=70, c='#e74c3c', edgecolors='black', linewidth=1)
    
    lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
    ax.plot(lims, lims, 'b--', lw=3, alpha=0.8)
    
    ax.set_xlabel('True', fontsize=15, fontweight='bold')
    ax.set_ylabel('Predicted', fontsize=15, fontweight='bold')
    ax.set_title('🎯 Predictions', fontsize=17, fontweight='bold')
    ax.grid(alpha=0.3)
    
    ax.text(0.05, 0.95, f"RMSE: {results['Ensemble']['RMSE']:.4f}\nR²: {results['Ensemble']['R2']:.4f}",
            transform=ax.transAxes, fontsize=13, va='top', fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.6))
    
    # R²
    ax = axes[0, 2]
    r2 = [results[m]['R2'] for m in models]
    colors = ['#2ecc71' if m != 'Ensemble' else '#e74c3c' for m in models]
    bars = ax.bar(models, r2, color=colors, alpha=0.85, edgecolor='black', linewidth=2)
    
    if 'Ensemble' in models:
        bars[models.index('Ensemble')].set_linewidth(5)
    
    ax.set_ylabel('R²', fontsize=15, fontweight='bold')
    ax.set_title('📊 R² - Higher is Better', fontsize=17, fontweight='bold')
    ax.axhline(y=1.0, color='blue', linestyle='--', linewidth=2, alpha=0.5)
    ax.grid(axis='y', alpha=0.3)
    ax.tick_params(axis='x', rotation=45, labelsize=11)
    
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., h, f'{h:.4f}',
                ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # Uncertainty
    ax = axes[1, 0]
    n = min(len(y_true), 100)
    idx = np.arange(n)
    ax.plot(idx, y_true[:n], 'k-', label='True', linewidth=3, marker='o', markersize=5)
    ax.plot(idx, y_pred[:n], 'r-', label='Ensemble', linewidth=3, marker='s', markersize=5)
    ax.fill_between(idx, unc['lower'][:n], unc['upper'][:n],
                     alpha=0.25, color='red', label='95% CI')
    ax.set_xlabel('Sample', fontsize=15, fontweight='bold')
    ax.set_ylabel('Temperature', fontsize=15, fontweight='bold')
    ax.set_title('📈 Uncertainty', fontsize=17, fontweight='bold')
    ax.legend(fontsize=12)
    ax.grid(alpha=0.3)
    
    # Residuals
    ax = axes[1, 1]
    res = y_true - y_pred
    ax.scatter(y_pred, res, alpha=0.7, s=70, c='#9b59b6', edgecolors='black', linewidth=1)
    ax.axhline(y=0, color='r', linestyle='--', linewidth=3)
    ax.set_xlabel('Predicted', fontsize=15, fontweight='bold')
    ax.set_ylabel('Residuals', fontsize=15, fontweight='bold')
    ax.set_title('🔍 Residuals', fontsize=17, fontweight='bold')
    ax.grid(alpha=0.3)
    
    # Errors
    ax = axes[1, 2]
    errors = np.abs(res)
    ax.hist(errors, bins=30, color='#e74c3c', alpha=0.7, edgecolor='black', linewidth=1.5)
    ax.axvline(np.mean(errors), color='blue', linestyle='--', linewidth=3,
               label=f'Mean: {np.mean(errors):.4f}')
    ax.axvline(np.median(errors), color='green', linestyle='--', linewidth=3,
               label=f'Median: {np.median(errors):.4f}')
    ax.set_xlabel('Absolute Error', fontsize=15, fontweight='bold')
    ax.set_ylabel('Frequency', fontsize=15, fontweight='bold')
    ax.set_title('📉 Error Distribution', fontsize=17, fontweight='bold')
    ax.legend(fontsize=12)
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('turbtwin_final.png', dpi=300, bbox_inches='tight')
    print("\n✓ Saved: turbtwin_final.png")
    plt.show()


def main():
    print("\n" + "=" * 70)
    print(" " * 15 + "🎯 TurbTwin FINAL VERSION 🎯")
    print(" " * 12 + "Trained on Full Data + Smart Ensemble")
    print("=" * 70)
    
    TRAIN = '/Users/nailamarir/VsCodeProjects/DigitalTwin/dataset/results_summary.csv'
    TEST = '/Users/nailamarir/VsCodeProjects/DigitalTwin/dataset/experimental_data.xlsx'
    
    prep = SimplePreprocessor(TRAIN, TEST)
    df_train, df_test = prep.load_data()
    data = prep.prepare_data(df_train, df_test)
    
    system = TurbTwinFinal(data)
    system.train_models()
    system.train_ensemble()
    results, unc = system.evaluate()
    
    visualize(results, unc)
    
    print("\n" + "=" * 70)
    print(" " * 28 + "🎉 DONE 🎉")
    print("=" * 70)
    
    return system, results, unc


if __name__ == "__main__":
    system, results, unc = main()
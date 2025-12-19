import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import minimize
from typing import Tuple, Dict, List
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

class DataPreprocessor:
    """Handles data loading and preprocessing with multiple scaling strategies"""
    
    def __init__(self, train_filepath: str, test_filepath: str = None):
        self.train_filepath = train_filepath
        self.test_filepath = test_filepath
        self.scaler_standard = StandardScaler()
        self.scaler_minmax = MinMaxScaler()
        self.feature_cols = ['T_inlet', 'T_initial', 'V_initial', 'Time_step']
        self.target_col = 'Temperature'
        
    def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load training and testing datasets from CSV or Excel"""
        # Load training data
        if self.train_filepath.endswith('.xlsx') or self.train_filepath.endswith('.xls'):
            df_train = pd.read_excel(self.train_filepath)
        else:
            df_train = pd.read_csv(self.train_filepath)
        
        print(f"Training Dataset loaded: {df_train.shape[0]} samples, {df_train.shape[1]} features")
        print(f"\nTraining Dataset Statistics:\n{df_train.describe()}")
        
        # Load test data
        if self.test_filepath:
            if self.test_filepath.endswith('.xlsx') or self.test_filepath.endswith('.xls'):
                df_test = pd.read_excel(self.test_filepath)
            else:
                df_test = pd.read_csv(self.test_filepath)
            
            print(f"\nTest Dataset loaded: {df_test.shape[0]} samples, {df_test.shape[1]} features")
            print(f"\nTest Dataset Statistics:\n{df_test.describe()}")
        else:
            df_test = None
            print("\nNo separate test dataset provided. Will split from training data.")
        
        return df_train, df_test
    
    def create_sequences(self, data: np.ndarray, seq_length: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        """Create sequences for LSTM (time-series windowing)"""
        X, y = [], []
        for i in range(len(data) - seq_length):
            X.append(data[i:i+seq_length, :-1])
            y.append(data[i+seq_length, -1])
        return np.array(X), np.array(y)
    
    def prepare_data(self, df_train: pd.DataFrame, df_test: pd.DataFrame = None, 
                     val_size: float = 0.1, seq_length: int = 10) -> Dict:
        """Prepare data for all models with different preprocessing strategies"""
        
        # Extract features and target from training data
        X_train_full = df_train[self.feature_cols].values
        y_train_full = df_train[self.target_col].values
        
        # Split training data into train and validation
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_full, y_train_full, test_size=val_size, random_state=42, shuffle=False
        )
        
        # Handle test data
        if df_test is not None:
            X_test = df_test[self.feature_cols].values
            y_test = df_test[self.target_col].values
            print(f"\n✓ Using separate test dataset: {len(X_test)} samples")
        else:
            X_temp, X_test, y_temp, y_test = train_test_split(
                X_train_full, y_train_full, test_size=0.2, random_state=42, shuffle=False
            )
            X_train, X_val, y_train, y_val = train_test_split(
                X_temp, y_temp, test_size=val_size/(1-0.2), random_state=42, shuffle=False
            )
            print(f"\n✓ Split training data - Test set: {len(X_test)} samples")
        
        print(f"✓ Training set: {len(X_train)} samples")
        print(f"✓ Validation set: {len(X_val)} samples")
        
        # Standard scaling for CNN and PINN
        X_train_std = self.scaler_standard.fit_transform(X_train)
        X_val_std = self.scaler_standard.transform(X_val)
        X_test_std = self.scaler_standard.transform(X_test)
        
        # MinMax scaling for XGBoost
        X_train_minmax = self.scaler_minmax.fit_transform(X_train)
        X_val_minmax = self.scaler_minmax.transform(X_val)
        X_test_minmax = self.scaler_minmax.transform(X_test)
        
        # Create sequences for LSTM
        train_data = np.column_stack([X_train_std, y_train])
        val_data = np.column_stack([X_val_std, y_val])
        test_data = np.column_stack([X_test_std, y_test])
        
        X_train_lstm, y_train_lstm = self.create_sequences(train_data, seq_length)
        X_val_lstm, y_val_lstm = self.create_sequences(val_data, seq_length)
        X_test_lstm, y_test_lstm = self.create_sequences(test_data, seq_length)
        
        return {
            'X_train_std': X_train_std,
            'X_val_std': X_val_std,
            'X_test_std': X_test_std,
            'y_train': y_train,
            'y_val': y_val,
            'y_test': y_test,
            'X_train_minmax': X_train_minmax,
            'X_val_minmax': X_val_minmax,
            'X_test_minmax': X_test_minmax,
            'X_train_lstm': X_train_lstm,
            'X_val_lstm': X_val_lstm,
            'X_test_lstm': X_test_lstm,
            'y_train_lstm': y_train_lstm,
            'y_val_lstm': y_val_lstm,
            'y_test_lstm': y_test_lstm,
            'seq_length': seq_length
        }


class CNNModel:
    """1D CNN for capturing spatial/feature patterns"""
    
    def __init__(self, input_dim: int):
        self.input_dim = input_dim
        self.model = self._build_model()
        
    def _build_model(self) -> Model:
        inputs = layers.Input(shape=(self.input_dim,))
        x = layers.Reshape((self.input_dim, 1))(inputs)
        x = layers.Conv1D(64, kernel_size=2, activation='relu', padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.2)(x)
        x = layers.Conv1D(128, kernel_size=2, activation='relu', padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling1D(pool_size=2)(x)
        x = layers.Dropout(0.2)(x)
        x = layers.Conv1D(64, kernel_size=2, activation='relu', padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.GlobalAveragePooling1D()(x)
        x = layers.Dense(128, activation='relu')(x)
        x = layers.Dropout(0.3)(x)
        x = layers.Dense(64, activation='relu')(x)
        x = layers.Dropout(0.2)(x)
        outputs = layers.Dense(1)(x)
        model = Model(inputs=inputs, outputs=outputs, name='CNN_Model')
        model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001),
                     loss='mse', metrics=['mae'])
        return model
    
    def train(self, X_train, y_train, X_val, y_val, epochs=100, batch_size=32):
        callbacks = [
            EarlyStopping(patience=15, restore_best_weights=True),
            ReduceLROnPlateau(factor=0.5, patience=5, min_lr=1e-6)
        ]
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=0
        )
        return history
    
    def predict(self, X):
        return self.model.predict(X, verbose=0).flatten()


class LSTMModel:
    """LSTM for temporal dynamics"""
    
    def __init__(self, seq_length: int, n_features: int):
        self.seq_length = seq_length
        self.n_features = n_features
        self.model = self._build_model()
        
    def _build_model(self) -> Model:
        inputs = layers.Input(shape=(self.seq_length, self.n_features))
        x = layers.Bidirectional(layers.LSTM(128, return_sequences=True))(inputs)
        x = layers.Dropout(0.3)(x)
        x = layers.Bidirectional(layers.LSTM(64, return_sequences=True))(x)
        x = layers.Dropout(0.2)(x)
        x = layers.LSTM(32, return_sequences=False)(x)
        x = layers.Dropout(0.2)(x)
        x = layers.Dense(64, activation='relu')(x)
        x = layers.Dropout(0.2)(x)
        x = layers.Dense(32, activation='relu')(x)
        outputs = layers.Dense(1)(x)
        model = Model(inputs=inputs, outputs=outputs, name='LSTM_Model')
        model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001),
                     loss='mse', metrics=['mae'])
        return model
    
    def train(self, X_train, y_train, X_val, y_val, epochs=100, batch_size=32):
        callbacks = [
            EarlyStopping(patience=15, restore_best_weights=True),
            ReduceLROnPlateau(factor=0.5, patience=5, min_lr=1e-6)
        ]
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=0
        )
        return history
    
    def predict(self, X):
        return self.model.predict(X, verbose=0).flatten()


class XGBoostModel:
    """XGBoost for parameter interactions"""
    
    def __init__(self):
        self.model = xgb.XGBRegressor(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            objective='reg:squarederror',
            early_stopping_rounds=20
        )
    
    def train(self, X_train, y_train, X_val, y_val):
        self.model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False
        )
    
    def predict(self, X):
        return self.model.predict(X)


class PhysicsInformedNN:
    """Physics-Informed Neural Network enforcing thermodynamic constraints"""
    
    def __init__(self, input_dim: int):
        self.input_dim = input_dim
        self.model = self._build_model()
        
    def _build_model(self) -> Model:
        inputs = layers.Input(shape=(self.input_dim,))
        x = layers.Dense(128, activation='tanh')(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.2)(x)
        x = layers.Dense(128, activation='tanh')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.2)(x)
        x = layers.Dense(64, activation='tanh')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.2)(x)
        x = layers.Dense(64, activation='tanh')(x)
        x = layers.Dropout(0.2)(x)
        outputs = layers.Dense(1)(x)
        model = Model(inputs=inputs, outputs=outputs, name='PINN_Model')
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss=self._physics_informed_loss,
            metrics=['mae']
        )
        return model
    
    def _physics_informed_loss(self, y_true, y_pred):
        data_loss = tf.reduce_mean(tf.square(y_true - y_pred))
        physics_loss_1 = tf.reduce_mean(tf.nn.relu(-y_pred))
        physics_loss_2 = tf.reduce_mean(tf.nn.relu(y_pred - 400))
        total_loss = data_loss + 0.1 * physics_loss_1 + 0.1 * physics_loss_2
        return total_loss
    
    def train(self, X_train, y_train, X_val, y_val, epochs=100, batch_size=32):
        callbacks = [
            EarlyStopping(patience=15, restore_best_weights=True),
            ReduceLROnPlateau(factor=0.5, patience=5, min_lr=1e-6)
        ]
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=0
        )
        return history
    
    def predict(self, X):
        return self.model.predict(X, verbose=0).flatten()


class PhysicsGuidedMetaLearner:
    """Simplified ensemble using optimal weighted averaging"""
    
    def __init__(self, n_base_models: int, n_features: int):
        self.n_base_models = n_base_models
        self.n_features = n_features
        self.weights = None
        self.use_simple_average = False
        
    def train(self, base_preds_train, X_train, y_train, 
             base_preds_val, X_val, y_val, epochs=100, batch_size=32):
        """Learn optimal weights using validation set"""
        
        def objective(w):
            w = np.abs(w)
            w = w / np.sum(w)
            pred = np.dot(base_preds_val, w)
            return np.sqrt(mean_squared_error(y_val, pred))
        
        w0 = np.ones(self.n_base_models) / self.n_base_models
        result = minimize(objective, w0, method='Nelder-Mead', 
                         options={'maxiter': 1000})
        
        if result.success:
            self.weights = np.abs(result.x)
            self.weights = self.weights / np.sum(self.weights)
            print(f"\nOptimized Ensemble Weights:")
            print(f"  CNN:     {self.weights[0]:.4f}")
            print(f"  LSTM:    {self.weights[1]:.4f}")
            print(f"  XGBoost: {self.weights[2]:.4f}")
            print(f"  PINN:    {self.weights[3]:.4f}")
        else:
            print("\n⚠️  Weight optimization failed, using simple average")
            self.weights = np.ones(self.n_base_models) / self.n_base_models
            self.use_simple_average = True
        
        return None
    
    def predict(self, base_predictions):
        if self.weights is None:
            return np.mean(base_predictions, axis=1)
        return np.dot(base_predictions, self.weights)


class TurbTwinEnsemble:
    """Main ensemble orchestrator"""
    
    def __init__(self, data_dict: Dict):
        self.data = data_dict
        self.cnn_model = None
        self.lstm_model = None
        self.xgb_model = None
        self.pinn_model = None
        self.meta_learner = None
        
    def train_base_models(self, epochs=100, batch_size=32):
        print("=" * 70)
        print("TRAINING BASE MODELS")
        print("=" * 70)
        
        print("\n[1/4] Training CNN Model...")
        self.cnn_model = CNNModel(input_dim=4)
        self.cnn_model.train(
            self.data['X_train_std'], self.data['y_train'],
            self.data['X_val_std'], self.data['y_val'],
            epochs=epochs, batch_size=batch_size
        )
        cnn_val_pred = self.cnn_model.predict(self.data['X_val_std'])
        cnn_val_rmse = np.sqrt(mean_squared_error(self.data['y_val'], cnn_val_pred))
        print(f"✓ CNN Validation RMSE: {cnn_val_rmse:.4f}")
        
        print("\n[2/4] Training LSTM Model...")
        self.lstm_model = LSTMModel(seq_length=self.data['seq_length'], n_features=4)
        self.lstm_model.train(
            self.data['X_train_lstm'], self.data['y_train_lstm'],
            self.data['X_val_lstm'], self.data['y_val_lstm'],
            epochs=epochs, batch_size=batch_size
        )
        lstm_val_pred = self.lstm_model.predict(self.data['X_val_lstm'])
        lstm_val_rmse = np.sqrt(mean_squared_error(self.data['y_val_lstm'], lstm_val_pred))
        print(f"✓ LSTM Validation RMSE: {lstm_val_rmse:.4f}")
        
        print("\n[3/4] Training XGBoost Model...")
        self.xgb_model = XGBoostModel()
        self.xgb_model.train(
            self.data['X_train_minmax'], self.data['y_train'],
            self.data['X_val_minmax'], self.data['y_val']
        )
        xgb_val_pred = self.xgb_model.predict(self.data['X_val_minmax'])
        xgb_val_rmse = np.sqrt(mean_squared_error(self.data['y_val'], xgb_val_pred))
        print(f"✓ XGBoost Validation RMSE: {xgb_val_rmse:.4f}")
        
        print("\n[4/4] Training Physics-Informed NN...")
        self.pinn_model = PhysicsInformedNN(input_dim=4)
        self.pinn_model.train(
            self.data['X_train_std'], self.data['y_train'],
            self.data['X_val_std'], self.data['y_val'],
            epochs=epochs, batch_size=batch_size
        )
        pinn_val_pred = self.pinn_model.predict(self.data['X_val_std'])
        pinn_val_rmse = np.sqrt(mean_squared_error(self.data['y_val'], pinn_val_pred))
        print(f"✓ PINN Validation RMSE: {pinn_val_rmse:.4f}")
        
        print("\n" + "=" * 70)
        print("BASE MODELS TRAINING COMPLETE")
        print("=" * 70)
    
    def train_meta_learner(self, epochs=100, batch_size=32):
        print("\n" + "=" * 70)
        print("TRAINING META-LEARNER")
        print("=" * 70)
        
        cnn_pred_val = self.cnn_model.predict(self.data['X_val_std'])
        xgb_pred_val = self.xgb_model.predict(self.data['X_val_minmax'])
        pinn_pred_val = self.pinn_model.predict(self.data['X_val_std'])
        lstm_pred_val = self.lstm_model.predict(self.data['X_val_lstm'])
        
        seq_length = self.data['seq_length']
        cnn_pred_val_aligned = cnn_pred_val[seq_length:]
        xgb_pred_val_aligned = xgb_pred_val[seq_length:]
        pinn_pred_val_aligned = pinn_pred_val[seq_length:]
        y_val_aligned = self.data['y_val'][seq_length:]
        
        min_length = min(len(cnn_pred_val_aligned), len(lstm_pred_val), 
                        len(xgb_pred_val_aligned), len(pinn_pred_val_aligned))
        
        base_preds_val = np.column_stack([
            cnn_pred_val_aligned[:min_length],
            lstm_pred_val[:min_length],
            xgb_pred_val_aligned[:min_length],
            pinn_pred_val_aligned[:min_length]
        ])
        y_val_aligned = y_val_aligned[:min_length]
        
        self.meta_learner = PhysicsGuidedMetaLearner(n_base_models=4, n_features=4)
        self.meta_learner.train(None, None, None, base_preds_val, None, y_val_aligned)
        
        meta_pred_val = self.meta_learner.predict(base_preds_val)
        meta_val_rmse = np.sqrt(mean_squared_error(y_val_aligned, meta_pred_val))
        
        simple_avg = np.mean(base_preds_val, axis=1)
        simple_avg_rmse = np.sqrt(mean_squared_error(y_val_aligned, simple_avg))
        
        best_base_rmse = min(
            np.sqrt(mean_squared_error(y_val_aligned, cnn_pred_val_aligned[:min_length])),
            np.sqrt(mean_squared_error(y_val_aligned, lstm_pred_val[:min_length])),
            np.sqrt(mean_squared_error(y_val_aligned, xgb_pred_val_aligned[:min_length])),
            np.sqrt(mean_squared_error(y_val_aligned, pinn_pred_val_aligned[:min_length]))
        )
        
        print(f"\n✓ Meta-Learner Validation RMSE: {meta_val_rmse:.4f}")
        print(f"✓ Simple Average RMSE: {simple_avg_rmse:.4f}")
        print(f"✓ Best Base Model RMSE: {best_base_rmse:.4f}")
        
        print("\n" + "=" * 70)
        print("META-LEARNER TRAINING COMPLETE")
        print("=" * 70)
    
    def evaluate_test_set(self):
        print("\n" + "=" * 70)
        print("TEST SET EVALUATION")
        print("=" * 70)
        
        cnn_pred = self.cnn_model.predict(self.data['X_test_std'])
        xgb_pred = self.xgb_model.predict(self.data['X_test_minmax'])
        pinn_pred = self.pinn_model.predict(self.data['X_test_std'])
        lstm_pred = self.lstm_model.predict(self.data['X_test_lstm'])
        
        seq_length = self.data['seq_length']
        cnn_pred_aligned = cnn_pred[seq_length:]
        xgb_pred_aligned = xgb_pred[seq_length:]
        pinn_pred_aligned = pinn_pred[seq_length:]
        y_test_aligned = self.data['y_test'][seq_length:]
        
        min_length = min(len(cnn_pred_aligned), len(lstm_pred),
                        len(xgb_pred_aligned), len(pinn_pred_aligned))
        
        cnn_pred_aligned = cnn_pred_aligned[:min_length]
        xgb_pred_aligned = xgb_pred_aligned[:min_length]
        pinn_pred_aligned = pinn_pred_aligned[:min_length]
        lstm_pred = lstm_pred[:min_length]
        y_test_aligned = y_test_aligned[:min_length]
        
        base_preds_test = np.column_stack([
            cnn_pred_aligned, lstm_pred, xgb_pred_aligned, pinn_pred_aligned
        ])
        
        ensemble_pred = self.meta_learner.predict(base_preds_test)
        
        models = {
            'CNN': cnn_pred_aligned,
            'LSTM': lstm_pred,
            'XGBoost': xgb_pred_aligned,
            'PINN': pinn_pred_aligned,
            'Ensemble': ensemble_pred
        }
        
        results = {}
        print("\nModel Performance Comparison:")
        print("-" * 70)
        print(f"{'Model':<15} {'RMSE':<12} {'MAE':<12} {'R²':<12}")
        print("-" * 70)
        
        for name, pred in models.items():
            rmse = np.sqrt(mean_squared_error(y_test_aligned, pred))
            mae = mean_absolute_error(y_test_aligned, pred)
            r2 = r2_score(y_test_aligned, pred)
            results[name] = {'RMSE': rmse, 'MAE': mae, 'R2': r2, 'predictions': pred}
            print(f"{name:<15} {rmse:<12.4f} {mae:<12.4f} {r2:<12.4f}")
        
        print("-" * 70)
        
        print("\n" + "=" * 70)
        print("UNCERTAINTY QUANTIFICATION")
        print("=" * 70)
        
        base_preds_array = np.column_stack([
            cnn_pred_aligned, lstm_pred, xgb_pred_aligned, pinn_pred_aligned
        ])
        epistemic_uncertainty = np.std(base_preds_array, axis=1)
        
        print(f"\nEpistemic Uncertainty (Model Disagreement):")
        print(f"  Mean: {np.mean(epistemic_uncertainty):.4f}")
        print(f"  Std:  {np.std(epistemic_uncertainty):.4f}")
        print(f"  Max:  {np.max(epistemic_uncertainty):.4f}")
        
        lower_bound = ensemble_pred - 2 * epistemic_uncertainty
        upper_bound = ensemble_pred + 2 * epistemic_uncertainty
        coverage = np.mean((y_test_aligned >= lower_bound) & (y_test_aligned <= upper_bound))
        
        print(f"\nPrediction Interval Coverage (95%): {coverage*100:.2f}%")
        
        return results, {
            'epistemic_uncertainty': epistemic_uncertainty,
            'lower_bound': lower_bound,
            'upper_bound': upper_bound,
            'y_true': y_test_aligned
        }


def visualize_results(results: Dict, uncertainty_data: Dict):
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    ax = axes[0, 0]
    models = list(results.keys())
    rmse_values = [results[m]['RMSE'] for m in models]
    colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6']
    bars = ax.bar(models, rmse_values, color=colors, alpha=0.7, edgecolor='black')
    ax.set_ylabel('RMSE', fontsize=12, fontweight='bold')
    ax.set_title('Model Performance Comparison (RMSE)', fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.4f}', ha='center', va='bottom', fontsize=10)
    
    ax = axes[0, 1]
    y_true = uncertainty_data['y_true']
    y_pred = results['Ensemble']['predictions']
    ax.scatter(y_true, y_pred, alpha=0.5, s=20, c='#9b59b6', edgecolors='black', linewidth=0.5)
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
    ax.set_xlabel('True Temperature', fontsize=12, fontweight='bold')
    ax.set_ylabel('Predicted Temperature', fontsize=12, fontweight='bold')
    ax.set_title('Ensemble: Predictions vs True Values', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)
    
    ax = axes[1, 0]
    sample_indices = np.arange(len(y_true))[:min(500, len(y_true))]
    ax.plot(sample_indices, y_true[:len(sample_indices)], 'k-', label='True', linewidth=2)
    ax.plot(sample_indices, y_pred[:len(sample_indices)], 'b-', label='Ensemble Prediction', linewidth=2)
    ax.fill_between(sample_indices,
                     uncertainty_data['lower_bound'][:len(sample_indices)],
                     uncertainty_data['upper_bound'][:len(sample_indices)],
                     alpha=0.3, color='blue', label='95% Prediction Interval')
    ax.set_xlabel('Sample Index', fontsize=12, fontweight='bold')
    ax.set_ylabel('Temperature', fontsize=12, fontweight='bold')
    ax.set_title('Ensemble Predictions with Uncertainty Bounds', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)
    
    ax = axes[1, 1]
    r2_values = [results[m]['R2'] for m in models]
    bars = ax.bar(models, r2_values, color=colors, alpha=0.7, edgecolor='black')
    ax.set_ylabel('R² Score', fontsize=12, fontweight='bold')
    ax.set_title('Model Performance Comparison (R²)', fontsize=14, fontweight='bold')
    ax.axhline(y=1.0, color='r', linestyle='--', linewidth=2, alpha=0.5, label='Perfect Score')
    ax.grid(axis='y', alpha=0.3)
    ax.legend()
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.4f}', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.savefig('turbtwin_ensemble_results.png', dpi=300, bbox_inches='tight')
    print("\n✓ Visualization saved as 'turbtwin_ensemble_results.png'")
    plt.show()


def main():
    print("\n" + "=" * 70)
    print(" " * 15 + "TurbTwin Ensemble Framework")
    print("=" * 70)
    
    TRAIN_DATA_PATH = '/Users/nailamarir/VsCodeProjects/DigitalTwin/dataset/results_summary.csv'
    TEST_DATA_PATH = '/Users/nailamarir/VsCodeProjects/DigitalTwin/dataset/experimental_data.xlsx'
    
    print("\n[Step 1] Loading and preprocessing data...")
    preprocessor = DataPreprocessor(train_filepath=TRAIN_DATA_PATH, 
                                   test_filepath=TEST_DATA_PATH)
    df_train, df_test = preprocessor.load_data()
    data_dict = preprocessor.prepare_data(df_train, df_test, val_size=0.1, seq_length=10)
    print("✓ Data preprocessing complete")
    
    print("\n[Step 2] Initializing ensemble framework...")
    ensemble = TurbTwinEnsemble(data_dict)
    
    print("\n[Step 3] Training base models...")
    ensemble.train_base_models(epochs=100, batch_size=32)
    
    print("\n[Step 4] Training meta-learner...")
    ensemble.train_meta_learner(epochs=100, batch_size=32)
    
    print("\n[Step 5] Evaluating on test set...")
    results, uncertainty_data = ensemble.evaluate_test_set()
    
    print("\n[Step 6] Generating visualizations...")
    visualize_results(results, uncertainty_data)
    
    print("\n" + "=" * 70)
    print(" " * 20 + "TRAINING COMPLETE!")
    print("=" * 70)
    
    return ensemble, results, uncertainty_data


if __name__ == "__main__":
    ensemble, results, uncertainty_data = main()
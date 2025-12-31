"""Deep learning models for TurbTwin."""

import numpy as np
from typing import Optional, List, Callable
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model

from turbtwin.models.base import BaseModel


class CNNModel(BaseModel):
    """1D Convolutional Neural Network for feature extraction."""

    def __init__(
        self,
        input_dim: int,
        filters: List[int] = [64, 128, 64],
        kernel_size: int = 3,
        dropout_rate: float = 0.2,
        random_state: int = 42,
        name: str = "CNN"
    ):
        """Initialize CNN model."""
        super().__init__(name=name, random_state=random_state)
        tf.random.set_seed(random_state)

        self.input_dim = input_dim
        self.filters = filters
        self.kernel_size = kernel_size
        self.dropout_rate = dropout_rate

        self._build_model()

    def _build_model(self) -> None:
        """Build the CNN architecture."""
        inputs = keras.Input(shape=(self.input_dim, 1))
        x = inputs

        # Convolutional blocks
        for n_filters in self.filters:
            x = layers.Conv1D(
                n_filters, self.kernel_size, padding="same", activation="relu"
            )(x)
            x = layers.BatchNormalization()(x)
            x = layers.Dropout(self.dropout_rate)(x)

        # Global pooling and dense layers
        x = layers.GlobalAveragePooling1D()(x)
        x = layers.Dense(64, activation="relu")(x)
        x = layers.Dropout(self.dropout_rate)(x)
        outputs = layers.Dense(1)(x)

        self.model = Model(inputs, outputs)
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss="mse",
            metrics=["mae"]
        )

    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        epochs: int = 100,
        batch_size: int = 32,
        validation_split: float = 0.2,
        verbose: int = 0
    ) -> None:
        """Train CNN model."""
        # Reshape for Conv1D: (samples, features, 1)
        X_reshaped = X.reshape(X.shape[0], X.shape[1], 1)

        callbacks = [
            keras.callbacks.EarlyStopping(
                patience=10, restore_best_weights=True
            ),
            keras.callbacks.ReduceLROnPlateau(
                factor=0.5, patience=5, min_lr=1e-6
            )
        ]

        self.model.fit(
            X_reshaped, y,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            callbacks=callbacks,
            verbose=verbose
        )
        self._is_trained = True

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Generate predictions."""
        X_reshaped = X.reshape(X.shape[0], X.shape[1], 1)
        return self.model.predict(X_reshaped, verbose=0).flatten()


class LSTMModel(BaseModel):
    """Bidirectional LSTM for temporal sequence modeling."""

    def __init__(
        self,
        input_shape: tuple,
        lstm_units: List[int] = [128, 64, 32],
        dropout_rate: float = 0.2,
        random_state: int = 42,
        name: str = "BiLSTM"
    ):
        """
        Initialize LSTM model.

        Args:
            input_shape: (sequence_length, n_features)
            lstm_units: List of units for each LSTM layer
            dropout_rate: Dropout rate
            random_state: Random seed
            name: Model name
        """
        super().__init__(name=name, random_state=random_state)
        tf.random.set_seed(random_state)

        self.input_shape = input_shape
        self.lstm_units = lstm_units
        self.dropout_rate = dropout_rate

        self._build_model()

    def _build_model(self) -> None:
        """Build bidirectional LSTM architecture."""
        inputs = keras.Input(shape=self.input_shape)
        x = inputs

        # Stacked Bidirectional LSTM layers
        for i, units in enumerate(self.lstm_units):
            return_sequences = i < len(self.lstm_units) - 1
            x = layers.Bidirectional(
                layers.LSTM(units, return_sequences=return_sequences)
            )(x)
            x = layers.Dropout(self.dropout_rate)(x)

        # Dense output
        x = layers.Dense(32, activation="relu")(x)
        outputs = layers.Dense(1)(x)

        self.model = Model(inputs, outputs)
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss="mse",
            metrics=["mae"]
        )

    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        epochs: int = 100,
        batch_size: int = 32,
        validation_split: float = 0.2,
        verbose: int = 0
    ) -> None:
        """Train LSTM model on sequential data."""
        callbacks = [
            keras.callbacks.EarlyStopping(
                patience=10, restore_best_weights=True
            ),
            keras.callbacks.ReduceLROnPlateau(
                factor=0.5, patience=5, min_lr=1e-6
            )
        ]

        self.model.fit(
            X, y,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            callbacks=callbacks,
            verbose=verbose
        )
        self._is_trained = True

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Generate predictions."""
        return self.model.predict(X, verbose=0).flatten()


class PhysicsInformedNN(BaseModel):
    """
    Physics-Informed Neural Network (PINN) for temperature prediction.

    Incorporates thermodynamic constraints into the loss function.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_layers: List[int] = [128, 256, 128, 64],
        physics_weight: float = 0.1,
        random_state: int = 42,
        name: str = "PINN"
    ):
        """
        Initialize PINN model.

        Args:
            input_dim: Number of input features
            hidden_layers: Units in each hidden layer
            physics_weight: Weight for physics loss term
            random_state: Random seed
            name: Model name
        """
        super().__init__(name=name, random_state=random_state)
        tf.random.set_seed(random_state)

        self.input_dim = input_dim
        self.hidden_layers = hidden_layers
        self.physics_weight = physics_weight

        self._build_model()

    def _build_model(self) -> None:
        """Build PINN architecture."""
        inputs = keras.Input(shape=(self.input_dim,))
        x = inputs

        # Hidden layers with residual connections
        for i, units in enumerate(self.hidden_layers):
            x_new = layers.Dense(units, activation="relu")(x)
            x_new = layers.BatchNormalization()(x_new)
            x_new = layers.Dropout(0.1)(x_new)

            # Residual connection if dimensions match
            if i > 0 and self.hidden_layers[i-1] == units:
                x = layers.Add()([x, x_new])
            else:
                x = x_new

        outputs = layers.Dense(1)(x)

        self.model = Model(inputs, outputs)

    def _physics_loss(
        self,
        y_true: tf.Tensor,
        y_pred: tf.Tensor,
        T_inlet: tf.Tensor,
        T_initial: tf.Tensor
    ) -> tf.Tensor:
        """
        Compute physics-informed loss.

        Enforces that predicted temperature stays within physical bounds.
        """
        # Data loss (MSE)
        data_loss = tf.reduce_mean(tf.square(y_true - y_pred))

        # Physics constraint: T_pred should be between T_initial and T_inlet
        T_min = tf.minimum(T_inlet, T_initial)
        T_max = tf.maximum(T_inlet, T_initial)

        lower_violation = tf.maximum(0.0, T_min - y_pred)
        upper_violation = tf.maximum(0.0, y_pred - T_max)

        physics_loss = tf.reduce_mean(
            tf.square(lower_violation) + tf.square(upper_violation)
        )

        return data_loss + self.physics_weight * physics_loss

    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        T_inlet_idx: int = 1,
        T_initial_idx: int = 3,
        epochs: int = 100,
        batch_size: int = 32,
        verbose: int = 0
    ) -> None:
        """
        Train PINN with physics constraints.

        Args:
            X: Input features
            y: Target values
            T_inlet_idx: Index of T_inlet in feature array
            T_initial_idx: Index of T_initial in feature array
            epochs: Number of epochs
            batch_size: Batch size
            verbose: Verbosity level
        """
        optimizer = keras.optimizers.Adam(learning_rate=0.001)

        # Training loop with custom loss
        dataset = tf.data.Dataset.from_tensor_slices((X, y))
        dataset = dataset.shuffle(1000).batch(batch_size)

        for epoch in range(epochs):
            epoch_loss = 0.0
            n_batches = 0

            for X_batch, y_batch in dataset:
                with tf.GradientTape() as tape:
                    y_pred = self.model(X_batch, training=True)

                    T_inlet = X_batch[:, T_inlet_idx:T_inlet_idx+1]
                    T_initial = X_batch[:, T_initial_idx:T_initial_idx+1]

                    loss = self._physics_loss(
                        y_batch[:, None], y_pred, T_inlet, T_initial
                    )

                gradients = tape.gradient(loss, self.model.trainable_variables)
                optimizer.apply_gradients(
                    zip(gradients, self.model.trainable_variables)
                )

                epoch_loss += loss.numpy()
                n_batches += 1

            if verbose and epoch % 10 == 0:
                print(f"Epoch {epoch}: Loss = {epoch_loss/n_batches:.4f}")

        self._is_trained = True

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Generate predictions."""
        return self.model.predict(X, verbose=0).flatten()

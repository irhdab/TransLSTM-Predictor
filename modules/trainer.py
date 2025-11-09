import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np
import sys
import os
import matplotlib.pyplot as plt

# Add the config directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'config'))
import config

class ModelTrainer:
    def __init__(self, model, config, scaler):
        """
        Initialize the ModelTrainer with model and configuration.
        
        Args:
            model: TensorFlow model to train
            config: Configuration object
            scaler: The MinMaxScaler object used for data normalization
        """
        self.model = model
        self.config = config
        self.scaler = scaler

    def compile_model(self):
        """
        Compile the model with specified optimizer, loss, and metrics.
        """
        print("Compiling model...")
        optimizer = tf.keras.optimizers.Adam(learning_rate=self.config.LEARNING_RATE)
        self.model.compile(
            optimizer=optimizer,
            loss=self.config.LOSS_FUNCTION,
            metrics=['mae']
        )
        print("✓ Model compiled successfully")
        self.model.summary()

    def setup_callbacks(self):
        """
        Set up training callbacks.
        
        Returns:
            list: List of callback objects
        """
        print("Setting up callbacks...")
        callbacks = []
        
        # Early stopping
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=self.config.EARLY_STOPPING_PATIENCE,
            restore_best_weights=True
        )
        callbacks.append(early_stopping)
        
        # Reduce learning rate on plateau
        reduce_lr = ReduceLROnPlateau(
            monitor='val_loss',
            factor=self.config.REDUCE_LR_FACTOR,
            patience=self.config.REDUCE_LR_PATIENCE,
            min_lr=self.config.MIN_LEARNING_RATE
        )
        callbacks.append(reduce_lr)
        
        print("✓ Callbacks set up successfully")
        return callbacks

    def train(self, X_train, y_train, X_val=None, y_val=None):
        """
        Train the model.
        
        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features (optional)
            y_val: Validation targets (optional)
            
        Returns:
            History object
        """
        print("Starting model training...")
        
        # If validation data not provided, use validation split
        validation_data = None
        if X_val is not None and y_val is not None:
            validation_data = (X_val, y_val)
        
        # Set up callbacks
        callbacks = self.setup_callbacks()
        
        # Train the model
        history = self.model.fit(
            X_train, y_train,
            batch_size=self.config.BATCH_SIZE,
            epochs=self.config.EPOCHS,
            validation_data=validation_data,
            validation_split=self.config.VALIDATION_SPLIT if validation_data is None else 0.0,
            callbacks=callbacks,
            verbose=1
        )
        
        print("✓ Model training completed")
        return history

    def evaluate(self, X_test, y_test, test_dates, future_predictions=None, future_dates=None):
        """
        Evaluate the model on test data and plot predictions.
        
        Args:
            X_test: Test features
            y_test: Test targets
            test_dates: Dates for the test set
            future_predictions: Rescaled future predictions
            future_dates: Dates for future predictions
            
        Returns:
            tuple: (mse, mae, mape)
        """
        print("Evaluating model...")
        y_pred_normalized = self.model.predict(X_test, verbose=0)
        
        # Inverse transform y_pred_normalized to match the scale of y_test
        # Create a dummy array with the same number of features as original_features
        # Assuming features.shape[1] is available or can be derived (e.g., from X_test.shape[2])
        num_features = X_test.shape[2] # Get num_features from X_test
        dummy_y_pred = np.zeros((len(y_pred_normalized), num_features))
        dummy_y_pred[:, 3] = y_pred_normalized.flatten() # Put predictions in the 'close' column
        y_pred_rescaled = self.scaler.inverse_transform(dummy_y_pred)[:, 3]

        # Calculate metrics using rescaled values
        mse = mean_squared_error(y_test, y_pred_rescaled)
        mae = mean_absolute_error(y_test, y_pred_rescaled)
        mape = np.mean(np.abs((y_test - y_pred_rescaled) / y_test)) * 100
        
        print(f"Evaluation results:")
        print(f"  - MSE: {mse:.6f}")
        print(f"  - MAE: {mae:.6f}")
        print(f"  - MAPE: {mape:.2f}%")

        print(f"\n--- Debugging trainer.py (evaluate) ---")
        print(f"y_test sample (first 5): {y_test[:5]}")
        print(f"y_pred_rescaled sample (first 5): {y_pred_rescaled[:5]}")
        print(f"test_dates sample (first 5): {test_dates[:5]}")
        if future_predictions is not None:
            print(f"future_predictions sample (first 5): {future_predictions[:5]}")
        if future_dates is not None:
            print(f"future_dates sample (first 5): {future_dates[:5]}")
        print(f"--- End Debugging trainer.py (evaluate) ---\n")
        
        self.plot_predictions(y_test, y_pred_rescaled, test_dates, future_predictions, future_dates)
        
        return mse, mae, mape

    def plot_predictions(self, y_true, y_pred, test_dates, future_predictions=None, future_dates=None):
        """
        Plot actual vs predicted values.
        """
        print(f"\n--- Debugging trainer.py (plot_predictions) ---")
        print(f"y_true sample (first 5): {y_true[:5]}")
        print(f"y_pred sample (first 5): {y_pred[:5].flatten()}")
        print(f"test_dates sample (first 5): {test_dates[:5]}")
        if future_predictions is not None:
            print(f"future_predictions sample (first 5): {future_predictions[:5]}")
        if future_dates is not None:
            print(f"future_dates sample (first 5): {future_dates[:5]}")
        print(f"--- End Debugging trainer.py (plot_predictions) ---\n")

        plt.figure(figsize=(self.config.FIGURE_SIZE[0], self.config.FIGURE_SIZE[1]))
        plt.plot(test_dates, y_true, label='Actual', color=self.config.PLOT_COLORS['actual'])
        plt.plot(test_dates, y_pred, label='Predicted', color=self.config.PLOT_COLORS['predicted'], linestyle='--')
        
        if future_predictions is not None and future_dates is not None:
            plt.plot(future_dates, future_predictions, label='Future Predictions', color=self.config.PLOT_COLORS['future'], linestyle=':')

        plt.title('Actual, Predicted, and Future Stock Prices')
        plt.xlabel('Date')
        plt.ylabel('Price') # Changed to 'Price'
        plt.legend()
        plt.grid(True, alpha=self.config.GRID_ALPHA)
        plt.tight_layout()
        
        plot_path = os.path.join(self.config.PLOTS_SAVE_PATH, 'predictions.png')
        plt.savefig(plot_path)
        print(f"Plot saved to {plot_path}")

    def save_model(self, filepath):
        """
        Save the trained model.
        
        Args:
            filepath (str): Path to save the model
        """
        print(f"Saving model to {filepath}...")
        self.model.save(filepath)
        print("✓ Model saved successfully")

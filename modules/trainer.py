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
    def __init__(self, model, config, scaler, csv_path):
        """
        Initialize the ModelTrainer with model and configuration.
        
        Args:
            model: TensorFlow model to train
            config: Configuration object
            scaler: The MinMaxScaler object used for data normalization
            csv_path: Path to the CSV file used for data
        """
        self.model = model
        self.config = config
        self.scaler = scaler
        self.csv_path = csv_path

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
        # self.model.summary() # Commented out to avoid console width error in some environments

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

    def evaluate(self, X_test, y_test, test_dates, future_predictions_rescaled=None, future_dates=None):
        """
        Evaluate the model on test data and plot predictions.
        
        Args:
            X_test (np.array): Test features.
            y_test (np.array): True target values for the test set (normalized, multi-step).
            test_dates (pd.Series): Dates corresponding to the test set.
            future_predictions_rescaled (np.array, optional): Rescaled future predictions. Defaults to None.
            future_dates (pd.Series, optional): Dates for future predictions. Defaults to None.
            
        Returns:
            tuple: (mse, mae, mape) for the multi-step predictions.
        """
        print("Evaluating model...")
        y_pred_normalized = self.model.predict(X_test, verbose=0) # Shape: [num_samples, FUTURE_DAYS]
        
        num_features = X_test.shape[2]
        
        # Rescale all predicted steps and true steps for evaluation
        y_pred_rescaled_all_steps = np.zeros_like(y_pred_normalized)
        y_test_rescaled_all_steps = np.zeros_like(y_test)

        # Create dummy arrays for inverse transformation
        # The scaler expects a full feature vector, so we fill the 'close' price and keep others as zeros
        for i in range(self.config.FUTURE_DAYS):
            # For predictions
            dummy_y_pred_step = np.zeros((len(y_pred_normalized), num_features))
            dummy_y_pred_step[:, self.config.CLOSE_COL_INDEX] = y_pred_normalized[:, i]
            y_pred_rescaled_all_steps[:, i] = self.scaler.inverse_transform(dummy_y_pred_step)[:, self.config.CLOSE_COL_INDEX]
            
            # For true values
            dummy_y_test_step = np.zeros((len(y_test), num_features))
            dummy_y_test_step[:, self.config.CLOSE_COL_INDEX] = y_test[:, i]
            y_test_rescaled_all_steps[:, i] = self.scaler.inverse_transform(dummy_y_test_step)[:, self.config.CLOSE_COL_INDEX]

        # Calculate metrics for all FUTURE_DAYS
        # Flatten arrays to calculate overall metrics across all steps and samples
        mse = mean_squared_error(y_test_rescaled_all_steps.flatten(), y_pred_rescaled_all_steps.flatten())
        mae = mean_absolute_error(y_test_rescaled_all_steps.flatten(), y_pred_rescaled_all_steps.flatten())
        
        # Avoid division by zero for MAPE
        # Filter out zero values from y_test_rescaled_all_steps to prevent division by zero
        non_zero_y_test = y_test_rescaled_all_steps.flatten()[y_test_rescaled_all_steps.flatten() != 0]
        if len(non_zero_y_test) > 0:
            mape = np.mean(np.abs((y_test_rescaled_all_steps.flatten()[y_test_rescaled_all_steps.flatten() != 0] - y_pred_rescaled_all_steps.flatten()[y_test_rescaled_all_steps.flatten() != 0]) / non_zero_y_test)) * 100
        else:
            mape = np.nan # Or handle as appropriate if all true values are zero

        print(f"Evaluation results (Multi-step prediction over {self.config.FUTURE_DAYS} days):")
        print(f"  - MSE: {mse:.6f}")
        print(f"  - MAE: {mae:.6f}")
        print(f"  - MAPE: {mape:.2f}%")
        
        # For plotting, we use the first day (Day 1) prediction as the "Predicted" value for historical comparison
        # We'll plot the actual vs predicted for the first day of each window for historical comparison
        self.plot_predictions(y_test_rescaled_all_steps[:, 0], y_pred_rescaled_all_steps[:, 0], test_dates, future_predictions_rescaled, future_dates)
        
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
        
        # Extract filename from csv_path to use in plot filename
        csv_filename = os.path.basename(self.csv_path)
        plot_filename = os.path.splitext(csv_filename)[0] + '.png'
        plot_path = os.path.join(self.config.PLOTS_SAVE_PATH, plot_filename)
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

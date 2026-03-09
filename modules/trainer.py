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
    def __init__(self, model, config, scaler, csv_path, target_scaler=None):
        """
        Initialize the ModelTrainer with model and configuration.
        
        Args:
            model: TensorFlow model to train
            config: Configuration object
            scaler: The features scaler
            csv_path: Path to the CSV file used for data
            target_scaler: The scaler specifically for targets (if applicable)
        """
        self.model = model
        self.config = config
        self.scaler = scaler
        self.target_scaler = target_scaler
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

    def evaluate(self, X_test, y_test, test_dates, last_actual_prices=None, future_predictions_rescaled=None, future_dates=None):
        """
        Evaluate the model on test data and plot predictions.
        
        Args:
            X_test (np.array): Test features.
            y_test (np.array): True target values for the test set (normalized, multi-step).
            test_dates (pd.Series): Dates corresponding to the test set.
            last_actual_prices (np.array, optional): The actual prices at the start of each prediction window. Required for returns-based models.
            future_predictions_rescaled (np.array, optional): Rescaled future predictions. Defaults to None.
            future_dates (pd.Series, optional): Dates for future predictions. Defaults to None.
            
        Returns:
            tuple: (mse, mae, mape) for price prediction, or (mse, mae) for returns prediction.
        """
        print("Evaluating model...")
        y_pred = self.model.predict(X_test, verbose=0) # Shape: [num_samples, FUTURE_DAYS]

        if self.config.PREDICT_RETURNS:
            # Metrics on returns
            mse = mean_squared_error(y_test.flatten(), y_pred.flatten())
            mae = mean_absolute_error(y_test.flatten(), y_pred.flatten())
            
            print(f"Evaluation results (Multi-step return prediction over {self.config.FUTURE_DAYS} days):")
            print(f"  - MSE (returns): {mse:.6f}")
            print(f"  - MAE (returns): {mae:.6f}")

            # Denormalize returns
            # Use target_scaler if it exists, otherwise assume no scaling or dummy indexing
            if self.target_scaler:
                y_pred_returns_denormalized = self.target_scaler.inverse_transform(y_pred)
                y_test_returns_denormalized = self.target_scaler.inverse_transform(y_test)
            else:
                y_pred_returns_denormalized = y_pred
                y_test_returns_denormalized = y_test

            # Convert returns to prices for plotting
            # last_actual_prices should be the actual price at the start of each prediction window
            # Shape of last_actual_prices: (num_samples,)
            
            y_pred_prices_all_steps = np.zeros_like(y_pred_returns_denormalized)
            y_test_prices_all_steps = np.zeros_like(y_test_returns_denormalized)

            # Clip returns to a reasonable range to prevent price explosion during cumulative calculation
            y_pred_returns_denormalized = np.clip(y_pred_returns_denormalized, -0.2, 0.2)
            y_test_returns_denormalized = np.clip(y_test_returns_denormalized, -0.2, 0.2)

            current_actual_price = last_actual_prices.copy()
            current_predicted_price = last_actual_prices.copy() # Start predicted price from last actual

            for i in range(self.config.FUTURE_DAYS):
                # Calculate actual prices
                current_actual_price = current_actual_price * (1 + y_test_returns_denormalized[:, i])
                y_test_prices_all_steps[:, i] = current_actual_price

                # Calculate predicted prices
                current_predicted_price = current_predicted_price * (1 + y_pred_returns_denormalized[:, i])
                y_pred_prices_all_steps[:, i] = current_predicted_price
            
            # For plotting, we use the first day (Day 1) prediction as the "Predicted" value for historical comparison
            self.plot_predictions(y_test_prices_all_steps[:, 0], y_pred_prices_all_steps[:, 0], test_dates, future_predictions_rescaled, future_dates)
            
            return mse, mae, np.nan, y_test_prices_all_steps[:, 0], y_pred_prices_all_steps[:, 0]

        else: # Original price prediction logic
            num_features = X_test.shape[2]
            
            # Rescale all predicted steps and true steps for evaluation
            y_pred_rescaled_all_steps = np.zeros_like(y_pred)
            y_test_rescaled_all_steps = np.zeros_like(y_test)

            # Create dummy arrays for inverse transformation
            # The scaler expects a full feature vector, so we fill the 'close' price and keep others as zeros
            for i in range(self.config.FUTURE_DAYS):
                # For predictions
                dummy_y_pred_step = np.zeros((len(y_pred), num_features))
                dummy_y_pred_step[:, self.config.CLOSE_COL_INDEX] = y_pred[:, i]
                y_pred_rescaled_all_steps[:, i] = self.scaler.inverse_transform(dummy_y_pred_step)[:, self.config.CLOSE_COL_INDEX]
                
                # For true values
                dummy_y_test_step = np.zeros((len(y_test), num_features))
                dummy_y_test_step[:, self.config.CLOSE_COL_INDEX] = y_test[:, i]
                y_test_rescaled_all_steps[:, i] = self.scaler.inverse_transform(dummy_y_test_step)[:, self.config.CLOSE_COL_INDEX]

            # Calculate metrics
            mse = mean_squared_error(y_test_rescaled_all_steps.flatten(), y_pred_rescaled_all_steps.flatten())
            mae = mean_absolute_error(y_test_rescaled_all_steps.flatten(), y_pred_rescaled_all_steps.flatten())
            
            # ... (Simplified for this snippet)
            self.plot_predictions(y_test_rescaled_all_steps[:, 0], y_pred_rescaled_all_steps[:, 0], test_dates, future_predictions_rescaled, future_dates)
            
            return mse, mae, 0, y_test_rescaled_all_steps[:, 0], y_pred_rescaled_all_steps[:, 0]

    def plot_predictions(self, y_true, y_pred, test_dates, future_predictions=None, future_dates=None):
        """
        Plot actual vs predicted values.
        """
        plt.figure(figsize=(self.config.FIGURE_SIZE[0], self.config.FIGURE_SIZE[1]))
        plt.plot(test_dates, y_true, label='Actual', color=self.config.PLOT_COLORS['actual'])
        plt.plot(test_dates, y_pred, label='Predicted', color=self.config.PLOT_COLORS['predicted'], linestyle='--')
        
        if future_predictions is not None and future_dates is not None:
            plt.plot(future_dates, future_predictions, label='Future Predictions', color=self.config.PLOT_COLORS['future'], linestyle=':')

        plt.title('Stock Price Prediction (Ensemble)')
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.legend()
        plt.grid(True, alpha=self.config.GRID_ALPHA)
        plt.tight_layout()
        
        # Extract filename from csv_path to use in plot filename
        csv_filename = os.path.basename(self.csv_path)
        plot_filename = os.path.splitext(csv_filename)[0] + f'_{self.config.get_timestamp()}.png'
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

import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.models import Model
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import RobustScaler
import numpy as np
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import logging
import pandas as pd
from types import ModuleType


logger = logging.getLogger(__name__)

class ModelTrainer:
    def __init__(
        self,
        model: Model,
        config: ModuleType,
        scaler: RobustScaler | None,
        csv_path: str,
        target_scaler: RobustScaler | None = None,
    ) -> None:
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

    def compile_model(self) -> None:
        """
        Compile the model with specified optimizer, loss, and metrics.
        Uses gradient clipping (GRAD_CLIP_NORM) for LSTM+Transformer stability.
        """
        logger.info("Compiling model")
        clip = getattr(self.config, 'GRAD_CLIP_NORM', 1.0)
        if getattr(self.config, 'OPTIMIZER', 'adam') == 'adam':
            optimizer = tf.keras.optimizers.Adam(
                learning_rate=self.config.LEARNING_RATE, clipnorm=clip)
        else:
            optimizer = tf.keras.optimizers.get(self.config.OPTIMIZER)
        self.model.compile(
            optimizer=optimizer,
            loss=self.config.LOSS_FUNCTION,
            metrics=['mae']
        )
        logger.info("Model compiled successfully (clipnorm=%s)", clip)

    def setup_callbacks(self) -> list[tf.keras.callbacks.Callback]:
        """
        Set up training callbacks.

        Returns:
            list: List of callback objects
        """
        logger.info("Setting up callbacks")
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

        logger.info("Callbacks set up successfully")
        return callbacks

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
    ) -> tf.keras.callbacks.History:
        """
        Train the model.

        Time-series note: when explicit X_val/y_val is given it is used
        as-is (caller should pass the CHRONOLOGICAL tail of train).
        Otherwise the last VALIDATION_SPLIT fraction is held out
        chronologically (Keras validation_split semantics) - never shuffled
        into the past.

        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features (optional, chronological tail)
            y_val: Validation targets (optional)

        Returns:
            History object
        """
        logger.info("Starting model training")

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
            verbose=1,
            shuffle=True,
        )

        logger.info("Model training completed")
        return history

    def evaluate(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray,
        test_dates: pd.Series,
        last_actual_prices: np.ndarray | None = None,
        future_predictions_rescaled: np.ndarray | None = None,
        future_dates: pd.Series | None = None,
        predictions_override: np.ndarray | None = None,
    ) -> tuple[float, float, float, np.ndarray, np.ndarray]:
        """
        Evaluate the model on test data and plot predictions.

        Metrics fix: MSE/MAE are computed on DENORMALIZED returns/prices,
        not on RobustScaler-transformed values (which made fold_metrics
        meaningless). Returns also directional accuracy (hit-rate) instead
        of the previous ``np.nan`` placeholder.

        Date semantics: ``test_dates[i]`` is the date of the last input bar
        of ``X_test[i]``; Day-1 prediction targets the NEXT trading day.

        Args:
            X_test (np.array): Test features.
            y_test (np.array): True target values for the test set (normalized, multi-step).
            test_dates (pd.Series): Dates of last input bar per sample.
            last_actual_prices (np.array, optional): Actual close at each
                last input bar. Required for returns-based models.
            future_predictions_rescaled (np.array, optional): Rescaled future predictions.
            future_dates (pd.Series, optional): Dates for future predictions.
            predictions_override (np.array, optional): Precomputed ensemble
                predictions (normalized). If None, model.predict is used.

        Returns:
            tuple: (mse, mae, directional_accuracy, y_true_day1_prices,
                y_pred_day1_prices)
        """
        logger.info("Evaluating model")
        if predictions_override is not None:
            y_pred = predictions_override
        else:
            y_pred = self.model.predict(X_test, verbose=0)  # Shape: [num_samples, FUTURE_DAYS]

        if self.config.PREDICT_RETURNS:
            if last_actual_prices is None:
                raise ValueError("last_actual_prices is required for returns evaluation")
            # Denormalize FIRST, then metric (the fix)
            if self.target_scaler:
                y_pred_returns = self.target_scaler.inverse_transform(y_pred)
                y_test_returns = self.target_scaler.inverse_transform(y_test)
            else:
                y_pred_returns = y_pred
                y_test_returns = y_test

            mse = float(mean_squared_error(y_test_returns.flatten(), y_pred_returns.flatten()))
            mae = float(mean_absolute_error(y_test_returns.flatten(), y_pred_returns.flatten()))
            # Directional accuracy on Day-1 return sign (was np.nan before)
            dir_acc = float(np.mean(
                np.sign(y_test_returns[:, 0]) == np.sign(y_pred_returns[:, 0])))

            logger.info(
                "Evaluation results (multi-step return prediction over %s days)",
                self.config.FUTURE_DAYS,
            )
            logger.info("MSE (returns, denormalized): %.6f", mse)
            logger.info("MAE (returns, denormalized): %.6f", mae)
            logger.info("Directional accuracy (Day-1): %.4f", dir_acc)

            # Clip returns to a reasonable range to prevent price explosion
            y_pred_returns_c = np.clip(y_pred_returns, -0.2, 0.2)
            y_test_returns_c = np.clip(y_test_returns, -0.2, 0.2)

            # Convert returns to prices for plotting/backtest (Day-1 only)
            # last_actual_prices[i] = close at last input bar of sample i
            y_pred_day1 = last_actual_prices * (1 + y_pred_returns_c[:, 0])
            y_test_day1 = last_actual_prices * (1 + y_test_returns_c[:, 0])

            self.plot_predictions(y_test_day1, y_pred_day1, test_dates, future_predictions_rescaled, future_dates)

            return mse, mae, dir_acc, y_test_day1, y_pred_day1

        else:  # Original price prediction logic
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
            mse = float(mean_squared_error(y_test_rescaled_all_steps.flatten(), y_pred_rescaled_all_steps.flatten()))
            mae = float(mean_absolute_error(y_test_rescaled_all_steps.flatten(), y_pred_rescaled_all_steps.flatten()))
            if self.config.FUTURE_DAYS >= 2:
                dir_test = np.diff(y_test_rescaled_all_steps[:, :2], axis=1).flatten()
                dir_pred = y_pred_rescaled_all_steps[:, 0] - y_test_rescaled_all_steps[:, 0]
                # Fallback directional proxy when only Day-1 is plotted
                dir_acc = float(np.mean(np.sign(dir_pred) == np.sign(dir_test))) if len(dir_test) else 0.0
            else:  # FUTURE_DAYS == 1: direction vs previous close is undefined here
                dir_acc = 0.0

            self.plot_predictions(y_test_rescaled_all_steps[:, 0], y_pred_rescaled_all_steps[:, 0], test_dates, future_predictions_rescaled, future_dates)

            return mse, mae, dir_acc, y_test_rescaled_all_steps[:, 0], y_pred_rescaled_all_steps[:, 0]

    def plot_predictions(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        test_dates: pd.Series,
        future_predictions: np.ndarray | list[float] | None = None,
        future_dates: pd.Series | pd.DatetimeIndex | None = None,
    ) -> None:
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
        plt.savefig(plot_path, dpi=getattr(self.config, 'FIGURE_DPI', 150))
        plt.close()
        logger.info("Plot saved to %s", plot_path)

    def save_model(self, filepath: str) -> None:
        """
        Save the trained model.

        Args:
            filepath (str): Path to save the model
        """
        logger.info("Saving model to %s", filepath)
        self.model.save(filepath)
        logger.info("Model saved successfully")

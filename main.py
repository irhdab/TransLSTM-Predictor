
from config import config
from modules.data_loader import DataProcessor
from modules.model_builder import create_lstm_transformer_model as build_model
from modules.trainer import ModelTrainer
from modules.predictor import make_predictions
from modules.visualizer import plot_data
import os
import sys
import numpy as np
import pandas as pd

def main():
    """
    Main function to run the quantitative analysis pipeline.
    """
    # Ensure all necessary directories exist
    config.ensure_directories()

    # Get CSV path from command-line argument or terminate if not found
    if len(sys.argv) > 1:
        user_csv_path = sys.argv[1]
        if not os.path.exists(user_csv_path):
            print(f"Error: File not found at {user_csv_path}. Please provide a valid CSV file path.")
            sys.exit(1) # Terminate if file not found
    else:
        print("Error: No CSV path provided. Please provide a CSV file path as a command-line argument.")
        sys.exit(1) # Terminate if no CSV path provided
    
    # Initialize the data processor
    data_processor = DataProcessor(csv_path=user_csv_path, config=config)

    # Load and process the data
    raw_data = data_processor.load_raw_data()
    if data_processor.validate_data(raw_data):
        data = data_processor.parse_dates(raw_data)
        data = data_processor.handle_outliers(data)
        original_dates = data['date']
        features = data_processor.extract_features(data)
        normalized_features, scaler = data_processor.normalize_data(features)
        X_train, X_test, y_train, y_test, test_dates = data_processor.create_sequences(normalized_features, original_dates)

        # Build the model
        model = build_model(seq_length=config.SEQ_LENGTH, num_features=X_train.shape[2], config=config)

        # Train the model
        trainer = ModelTrainer(model, config, scaler, user_csv_path)
        trainer.compile_model()
        trainer.train(X_train, y_train)

        # Evaluate the model and plot predictions
        # Make future predictions
        last_sequence = normalized_features[-config.SEQ_LENGTH:]
        # Ensure last_sequence has the correct shape (1, SEQ_LENGTH, num_features)
        last_sequence = np.expand_dims(last_sequence, axis=0)

        future_predictions = []
        current_sequence_for_prediction = last_sequence.copy()

        for _ in range(config.FUTURE_DAYS):
            next_prediction_normalized = model.predict(current_sequence_for_prediction, verbose=0)[0, 0]
            future_predictions.append(next_prediction_normalized)
            
            # Update the sequence for the next prediction
            # Create a new feature row using the last known features and the new prediction for 'close'
            new_feature_row = current_sequence_for_prediction[0, -1, :].copy() # Get the last feature row
            new_feature_row[3] = next_prediction_normalized # Update only the close price (index 3)
            
            # Shift the sequence and add the new feature row
            current_sequence_for_prediction = np.roll(current_sequence_for_prediction, -1, axis=1)
            current_sequence_for_prediction[0, -1] = new_feature_row

        future_predictions_normalized = np.array(future_predictions).reshape(-1, 1)
        
        # Inverse transform future predictions
        # To inverse transform correctly, we need to provide a full feature set for each prediction.
        # We'll use the last known features from the normalized_features for the non-predicted values.
        
        # Get the last known feature row (normalized)
        last_known_normalized_features = normalized_features[-1, :].copy()

        future_predictions_rescaled = []
        for pred_normalized in future_predictions_normalized.flatten():
            # Create a full feature row for inverse transformation
            # Use the last known features, but replace the 'close' price with the predicted normalized close
            full_feature_row_normalized = last_known_normalized_features.copy()
            full_feature_row_normalized[3] = pred_normalized # Assuming close price is at index 3

            # Inverse transform this single row
            rescaled_row = scaler.inverse_transform(full_feature_row_normalized.reshape(1, -1))
            future_predictions_rescaled.append(rescaled_row[0, 3]) # Append only the rescaled close price

        future_predictions_rescaled = np.array(future_predictions_rescaled)

        # Generate future dates
        last_date = original_dates.iloc[-1]
        future_dates = pd.date_range(start=last_date, periods=config.FUTURE_DAYS + 1, freq='D')[1:]

        # Inverse transform y_test before passing to evaluate for consistent scaling in plot
        # Create a dummy array with the same number of features as original_features
        dummy_y_test = np.zeros((len(y_test), features.shape[1]))
        dummy_y_test[:, 3] = y_test.flatten() # Put y_test (normalized close prices) in the 'close' column
        y_test_rescaled = scaler.inverse_transform(dummy_y_test)[:, 3]

        print(f"\n--- Debugging main.py ---")
        print(f"y_test_rescaled sample (first 5): {y_test_rescaled[:5]}")
        print(f"test_dates sample (first 5): {test_dates[:5]}")
        print(f"future_predictions_rescaled sample (first 5): {future_predictions_rescaled[:5]}")
        print(f"future_dates sample (first 5): {future_dates[:5]}")
        print(f"--- End Debugging main.py ---\n")

        trainer.evaluate(X_test, y_test_rescaled, test_dates, future_predictions_rescaled, future_dates)

if __name__ == '__main__':
    main()

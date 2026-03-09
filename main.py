
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

        # Build and Train the model
        model = build_model(seq_length=config.SEQ_LENGTH, num_features=X_train.shape[2], config=config)
        trainer = ModelTrainer(model, config, scaler, user_csv_path)
        trainer.compile_model()
        trainer.train(X_train, y_train)

        # Multi-Step Future Prediction (One-shot)
        # Use the very last sequence from normalized_features
        last_sequence = normalized_features[-config.SEQ_LENGTH:]
        last_sequence = np.expand_dims(last_sequence, axis=0) # Shape: [1, SEQ_LENGTH, num_features]
        
        # Predict the next 30 days in one shot
        multi_step_prediction_normalized = model.predict(last_sequence, verbose=0) # Shape: [1, 30]
        
        # Inverse transform the predicted 'close' prices
        # We need to provide a full feature row for the scaler
        # We'll use the last known features and replace only the 'close' price
        last_known_normalized = normalized_features[-1, :].copy()
        
        future_predictions_rescaled = []
        for pred_norm in multi_step_prediction_normalized[0]:
            dummy_row = last_known_normalized.copy()
            dummy_row[config.CLOSE_COL_INDEX] = pred_norm
            rescaled_val = scaler.inverse_transform(dummy_row.reshape(1, -1))[0, config.CLOSE_COL_INDEX]
            future_predictions_rescaled.append(rescaled_val)
        
        future_predictions_rescaled = np.array(future_predictions_rescaled)
        
        # Generate future dates (Business Days - Excluding Weekends)
        last_date = original_dates.iloc[-1]
        future_dates = pd.date_range(start=last_date, periods=config.FUTURE_DAYS + 1, freq='B')[1:]

        print(f"\n--- Final Prediction Summary ---")
        print(f"Test data size: {len(X_test)}")
        print(f"Future prediction period: {config.FUTURE_DAYS} business days")
        print(f"First 5 future predictions: {future_predictions_rescaled[:5]}")
        print(f"Future dates: {future_dates[:5].strftime('%Y-%m-%d').tolist()}")
        print(f"--- End Summary ---\n")

        # Evaluate and plot
        # y_test here contains the actual future windows for each point in the test set
        trainer.evaluate(X_test, y_test, test_dates, future_predictions_rescaled, future_dates)

if __name__ == '__main__':
    main()

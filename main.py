import os
import sys
import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler
from config import config
from modules.data_loader import DataProcessor
from modules.model_builder import create_lstm_transformer_model as build_model
from modules.trainer import ModelTrainer
from modules.backtester import Backtester

def main():
    """
    Main function to run the high-accuracy quantitative pipeline.
    Includes Ensemble training and Walk-forward validation.
    """
    config.ensure_directories()

    if len(sys.argv) > 1:
        user_csv_path = sys.argv[1]
    else:
        print("Error: No CSV path provided.")
        sys.exit(1)
    
    data_processor = DataProcessor(csv_path=user_csv_path, config=config)
    raw_data = data_processor.load_raw_data()
    
    if not data_processor.validate_data(raw_data):
        sys.exit(1)

    # 1. Pipeline: Date parsing -> Outlier handling -> Feature extraction
    data = data_processor.parse_dates(raw_data)
    data = data_processor.handle_outliers(data)
    original_dates = data['date']
    features = data_processor.extract_features(data)
    
    # 2. Walk-forward Validation Setup
    print(f"\n--- Starting Walk-forward Validation ({config.WALK_FORWARD_FOLDS} folds) ---")
    
    # Calculate fold indices
    seq_length = config.SEQ_LENGTH
    future_days = config.FUTURE_DAYS
    total_samples = len(features) - seq_length - future_days + 1
    fold_size = total_samples // config.WALK_FORWARD_FOLDS
    
    all_fold_metrics = []
    
    # For future prediction, we'll use the models from the LAST fold
    final_ensemble_models = []
    final_scaler = None

    for fold in range(config.WALK_FORWARD_FOLDS):
        print(f"\n>> Processing Fold {fold + 1}/{config.WALK_FORWARD_FOLDS}")
        
        # Scaling (fit on train only)
        train_end_idx = int(total_samples * (0.6 + 0.1 * fold))
        test_end_idx = train_end_idx + (total_samples - train_end_idx) // (config.WALK_FORWARD_FOLDS - fold)
        
        # 1. Create Raw Sequences first to get targets for scaling
        X_all, y_all, dates_all = data_processor.create_sequences(features, original_dates)
        
        # 2. Dual Normalization (Explicitly prevent leakage for this fold)
        norm_features, norm_targets, scaler, target_scaler = data_processor.normalize_data(features, y_all, train_end=train_end_idx + seq_length)
        
        # Re-create sequences with normalized features and targets
        # We need to ensure we use the same sequence generation logic
        sequences_norm, _, _ = data_processor.create_sequences(norm_features, original_dates)
        
        fold_X_train = sequences_norm[:train_end_idx]
        fold_y_train = norm_targets[:train_end_idx]
        fold_X_test = sequences_norm[train_end_idx:test_end_idx]
        fold_y_test = norm_targets[train_end_idx:test_end_idx]
        fold_test_dates = dates_all[train_end_idx:test_end_idx].reset_index(drop=True)
        
        # last_actual_prices for conversion to prices
        last_prices_indices = np.arange(train_end_idx, test_end_idx) + seq_length - 1
        last_actual_prices = data['close'].iloc[last_prices_indices].values
        
        # 3. Ensemble Training
        fold_models = []
        fold_predictions_norm = []
        
        print(f"Training Ensemble (Size: {config.ENSEMBLE_SIZE}, Samples: {len(fold_X_train)})...")
        for m_idx in range(config.ENSEMBLE_SIZE):
            model = build_model(seq_length=seq_length, num_features=features.shape[1], config=config)
            trainer = ModelTrainer(model, config, scaler, user_csv_path, target_scaler=target_scaler)
            trainer.compile_model()
            trainer.train(fold_X_train, fold_y_train)
            fold_models.append(model)
            
            # Prediction for ensemble average
            pred_norm = model.predict(fold_X_test, verbose=0)
            fold_predictions_norm.append(pred_norm)
        
        # Average ensemble predictions (Normalized)
        avg_preds_norm = np.mean(fold_predictions_norm, axis=0)
        
        # Evaluate Ensemble performance by tricking ModelTrainer.evaluate
        # We replace the last model with a "dummy" or just use the logic
        # Actually, let's just use the trainer's evaluate method with our averaged predictions
        # Modification: trainer.evaluate needs to handle direct prediction input or we inject it
        
        # Let's perform a manual evaluation for the ensemble for better accuracy
        eval_trainer = ModelTrainer(fold_models[0], config, scaler, user_csv_path, target_scaler=target_scaler)
        # We "hack" the model's predict to return our average
        eval_trainer.model.predict = lambda x, **kwargs: avg_preds_norm
        
        res = eval_trainer.evaluate(fold_X_test, fold_y_test, fold_test_dates, last_actual_prices=last_actual_prices)
        fold_mse, fold_mae = res[0], res[1]
        test_prices_actual, test_prices_predicted = res[3], res[4]
        
        all_fold_metrics.append({'mse': fold_mse, 'mae': fold_mae})
        
        if fold == config.WALK_FORWARD_FOLDS - 1:
            final_ensemble_models = fold_models
            final_scaler = scaler
            final_target_scaler = target_scaler
            final_test_actual = test_prices_actual
            final_test_pred = test_prices_predicted
            final_test_dates = fold_test_dates

    # 4. Backtesting on Final Fold (Ensemble Results)
    backtester = Backtester(config)
    backtester.run(final_test_actual, final_test_pred, final_test_dates)

    # 5. Final Future One-Shot Prediction (using Ensemble)
    print("\n--- Generating Final Future Prediction ---")
    norm_features_final = final_scaler.transform(features)
    last_sequence = norm_features_final[-seq_length:]
    last_sequence = np.expand_dims(last_sequence, axis=0)
    
    ensemble_future_returns_norm = []
    for model in final_ensemble_models:
        row_pred = model.predict(last_sequence, verbose=0)
        ensemble_future_returns_norm.append(row_pred)
    
    avg_future_returns_norm = np.mean(ensemble_future_returns_norm, axis=0)
    
    # Denormalize returns
    future_returns_actual = final_target_scaler.inverse_transform(avg_future_returns_norm).flatten()
    
    # Convert returns to prices
    last_actual_price = data['close'].iloc[-1]
    future_prices = []
    curr_p = last_actual_price
    for r in future_returns_actual:
        curr_p = curr_p * (1 + r)
        future_prices.append(curr_p)
    
    # Future dates
    last_date = original_dates.iloc[-1]
    future_dates = pd.date_range(start=last_date, periods=config.FUTURE_DAYS + 1, freq='B')[1:]
    
    print(f"Final Future Prices (Next 5 days): {future_prices[:5]}")
    
    # Final Visualization
    final_trainer = ModelTrainer(final_ensemble_models[0], config, final_scaler, user_csv_path, target_scaler=final_target_scaler)
    final_trainer.plot_predictions(
        y_true=final_test_actual, 
        y_pred=final_test_pred, 
        test_dates=final_test_dates, 
        future_predictions=future_prices,
        future_dates=future_dates
    )

if __name__ == '__main__':
    main()

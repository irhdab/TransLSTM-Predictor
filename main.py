import os
import sys
import argparse
import random
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import RobustScaler
from config import config
from modules.data_loader import DataProcessor
from modules.model_builder import create_lstm_transformer_model as build_model
from modules.trainer import ModelTrainer
from modules.backtester import Backtester


def parse_args():
    """Parse command-line arguments with argparse. Overrides config values at runtime."""
    parser = argparse.ArgumentParser(
        description='TransLSTM-Predictor: CNN-BiLSTM-Transformer Hybrid Stock Prediction System'
    )
    parser.add_argument('csv_path', help='Path to the stock data CSV file (columns: date, open, high, low, close, volume)')
    parser.add_argument('--epochs', type=int, default=None, help=f'Max training epochs (default: {config.EPOCHS})')
    parser.add_argument('--ensemble-size', type=int, default=None, help=f'Number of ensemble models (default: {config.ENSEMBLE_SIZE})')
    parser.add_argument('--seq-length', type=int, default=None, help=f'Input sequence length (default: {config.SEQ_LENGTH})')
    parser.add_argument('--future-days', type=int, default=None, help=f'Future days to predict (default: {config.FUTURE_DAYS})')
    parser.add_argument('--folds', type=int, default=None, help=f'Walk-forward validation folds (default: {config.WALK_FORWARD_FOLDS})')
    parser.add_argument('--seed', type=int, default=None, help=f'Random seed (default: {config.RANDOM_SEED})')
    return parser.parse_args()


def apply_overrides(args):
    """Apply CLI argument overrides to config module."""
    if args.epochs is not None:
        config.EPOCHS = args.epochs
    if args.ensemble_size is not None:
        config.ENSEMBLE_SIZE = args.ensemble_size
    if args.seq_length is not None:
        config.SEQ_LENGTH = args.seq_length
    if args.future_days is not None:
        config.FUTURE_DAYS = args.future_days
    if args.folds is not None:
        config.WALK_FORWARD_FOLDS = args.folds
    if args.seed is not None:
        config.RANDOM_SEED = args.seed


def set_seed(seed):
    """Fix all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


def save_fold_metrics(all_fold_metrics, csv_path):
    """Save fold-level metrics to a CSV file in the logs directory."""
    df = pd.DataFrame(all_fold_metrics)
    df.index.name = 'fold'
    df.index = df.index + 1  # 1-indexed folds

    # Add summary row
    summary = df.mean()
    summary.name = 'MEAN'
    df = pd.concat([df, summary.to_frame().T])

    timestamp = config.get_timestamp()
    csv_name = os.path.splitext(os.path.basename(csv_path))[0]
    log_path = os.path.join(config.LOGS_PATH, f'fold_metrics_{csv_name}_{timestamp}.csv')
    df.to_csv(log_path)
    print(f"✓ Fold metrics saved to {log_path}")
    return df


def main():
    """
    Main function to run the high-accuracy quantitative pipeline.
    Includes Ensemble training and Walk-forward validation.
    """
    args = parse_args()
    apply_overrides(args)
    config.ensure_directories()

    # Fix random seeds
    set_seed(config.RANDOM_SEED)

    user_csv_path = args.csv_path

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
        
        # Evaluate using predictions_override (no more monkey patch)
        eval_trainer = ModelTrainer(fold_models[0], config, scaler, user_csv_path, target_scaler=target_scaler)
        res = eval_trainer.evaluate(
            fold_X_test, fold_y_test, fold_test_dates,
            last_actual_prices=last_actual_prices,
            predictions_override=avg_preds_norm
        )
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

    # Save fold metrics to CSV
    metrics_df = save_fold_metrics(all_fold_metrics, user_csv_path)
    print(f"\n--- Walk-forward Validation Summary ---")
    print(f"  Average MSE: {metrics_df.loc['MEAN', 'mse']:.6f}")
    print(f"  Average MAE: {metrics_df.loc['MEAN', 'mae']:.6f}")

    # Save ensemble models
    print("\n--- Saving Ensemble Models ---")
    csv_name = os.path.splitext(os.path.basename(user_csv_path))[0]
    timestamp = config.get_timestamp()
    for i, m in enumerate(final_ensemble_models):
        model_path = os.path.join(config.MODEL_SAVE_PATH, f'{csv_name}_ensemble_{i+1}_{timestamp}.keras')
        m.save(model_path)
        print(f"  ✓ Model {i+1} saved to {model_path}")

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

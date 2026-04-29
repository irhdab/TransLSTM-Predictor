import os
import sys
import argparse
import random
import numpy as np
import pandas as pd
import tensorflow as tf
from config import config
from modules.data_loader import DataProcessor
from modules.model_builder import create_lstm_transformer_model as build_model
from modules.trainer import ModelTrainer
from modules.backtester import Backtester


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='TransLSTM-Predictor: CNN-BiLSTM-Transformer Hybrid Stock Prediction System'
    )
    parser.add_argument('csv_path', help='Path to the stock data CSV file')
    parser.add_argument('--epochs', type=int, help=f'Max training epochs (default: {config.EPOCHS})')
    parser.add_argument('--ensemble-size', type=int, help=f'Number of ensemble models (default: {config.ENSEMBLE_SIZE})')
    parser.add_argument('--seq-length', type=int, help=f'Input sequence length (default: {config.SEQ_LENGTH})')
    parser.add_argument('--future-days', type=int, help=f'Future days to predict (default: {config.FUTURE_DAYS})')
    parser.add_argument('--folds', type=int, help=f'Validation folds (default: {config.WALK_FORWARD_FOLDS})')
    parser.add_argument('--seed', type=int, help=f'Random seed (default: {config.RANDOM_SEED})')
    return parser.parse_args()


def apply_config_overrides(args):
    """Apply CLI overrides to the global config."""
    overrides = {
        'EPOCHS': args.epochs,
        'ENSEMBLE_SIZE': args.ensemble_size,
        'SEQ_LENGTH': args.seq_length,
        'FUTURE_DAYS': args.future_days,
        'WALK_FORWARD_FOLDS': args.folds,
        'RANDOM_SEED': args.seed
    }
    for key, value in overrides.items():
        if value is not None:
            setattr(config, key, value)


def set_seed(seed):
    """Ensure reproducibility by fixing all random seeds."""
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


class StockPipeline:
    def __init__(self, csv_path):
        self.csv_path = csv_path
        self.data_processor = DataProcessor(csv_path=csv_path, config=config)
        self.raw_data = None
        self.processed_data = None
        self.features = None
        self.original_dates = None
        
        # State maintained across folds
        self.final_ensemble_models = []
        self.final_scalers = {}  # {scaler, target_scaler}
        self.last_fold_results = {} # {actual, pred, dates}

    def prepare_data(self):
        """Load and process the stock data."""
        self.raw_data = self.data_processor.load_raw_data()
        if not self.data_processor.validate_data(self.raw_data):
            print("✘ Data validation failed.")
            sys.exit(1)

        data = self.data_processor.parse_dates(self.raw_data)
        data = self.data_processor.handle_outliers(data)
        self.processed_data = data
        self.original_dates = data['date']
        self.features = self.data_processor.extract_features(data)
        print("✓ Data preparation complete.")

    def run_walk_forward_validation(self):
        """Execute the walk-forward validation loop."""
        print(f"\n--- Starting Walk-forward Validation ({config.WALK_FORWARD_FOLDS} folds) ---")
        
        seq_len = config.SEQ_LENGTH
        future_days = config.FUTURE_DAYS
        total_samples = len(self.features) - seq_len - future_days + 1

        all_fold_metrics = []

        for fold in range(config.WALK_FORWARD_FOLDS):
            print(f"\n>> Processing Fold {fold + 1}/{config.WALK_FORWARD_FOLDS}")
            
            # 1. Define split points (60% base + incremental folds)
            train_end_idx = int(total_samples * (0.6 + 0.1 * fold))
            test_end_idx = train_end_idx + (total_samples - train_end_idx) // (config.WALK_FORWARD_FOLDS - fold)
            
            # 2. Sequence creation and Dual Normalization (Leakage Prevention)
            X_all, y_all, dates_all = self.data_processor.create_sequences(self.features, self.original_dates)
            norm_features, norm_targets, scaler, target_scaler = self.data_processor.normalize_data(
                self.features, y_all, train_end=train_end_idx + seq_len
            )
            
            sequences_norm, _, _ = self.data_processor.create_sequences(norm_features, self.original_dates)
            
            # Slice fold data
            X_train = sequences_norm[:train_end_idx]
            y_train = norm_targets[:train_end_idx]
            X_test = sequences_norm[train_end_idx:test_end_idx]
            y_test = norm_targets[train_end_idx:test_end_idx]
            test_dates = dates_all[train_end_idx:test_end_idx].reset_index(drop=True)
            
            # Price conversion context
            price_indices = np.arange(train_end_idx, test_end_idx) + seq_len - 1
            last_actual_prices = self.processed_data['close'].iloc[price_indices].values
            
            # 3. Train Ensemble for this fold
            fold_models, avg_preds_norm = self._train_fold_ensemble(X_train, y_train, X_test)
            
            # 4. Evaluate Fold
            mse, mae, _, actual_prices, pred_prices = self._evaluate_fold(
                fold_models[0], X_test, y_test, test_dates, 
                scaler, target_scaler, last_actual_prices, avg_preds_norm
            )
            
            all_fold_metrics.append({'mse': mse, 'mae': mae})
            
            # Record last fold state for final predictions
            if fold == config.WALK_FORWARD_FOLDS - 1:
                self._update_final_state(fold_models, scaler, target_scaler, actual_prices, pred_prices, test_dates)

        self._save_summary_metrics(all_fold_metrics)

    def _train_fold_ensemble(self, X_train, y_train, X_test):
        """Internal helper to train an ensemble and return averaged predictions."""
        fold_models = []
        fold_predictions_norm = []
        
        print(f"Training Ensemble (Size: {config.ENSEMBLE_SIZE}, Samples: {len(X_train)})...")
        for i in range(config.ENSEMBLE_SIZE):
            model = build_model(seq_length=config.SEQ_LENGTH, num_features=self.features.shape[1], config=config)
            trainer = ModelTrainer(model, config, None, self.csv_path) # scalers passed during eval
            trainer.compile_model()
            trainer.train(X_train, y_train)
            fold_models.append(model)
            
            pred = model.predict(X_test, verbose=0)
            fold_predictions_norm.append(pred)
            
        avg_preds_norm = np.mean(fold_predictions_norm, axis=0)
        return fold_models, avg_preds_norm

    def _evaluate_fold(self, model, X_test, y_test, dates, scaler, target_scaler, last_prices, predictions):
        """Internal helper to evaluate model performance using rescaled prices."""
        eval_trainer = ModelTrainer(model, config, scaler, self.csv_path, target_scaler=target_scaler)
        return eval_trainer.evaluate(
            X_test, y_test, dates,
            last_actual_prices=last_prices,
            predictions_override=predictions
        )

    def _update_final_state(self, models, scaler, target_scaler, actual, pred, dates):
        self.final_ensemble_models = models
        self.final_scalers = {'scaler': scaler, 'target_scaler': target_scaler}
        self.last_fold_results = {'actual': actual, 'pred': pred, 'dates': dates}

    def _save_summary_metrics(self, all_fold_metrics):
        """Save and print overall metrics across all folds."""
        df = pd.DataFrame(all_fold_metrics)
        df.index.name = 'fold'
        df.index += 1
        
        summary = df.mean()
        summary.name = 'MEAN'
        df = pd.concat([df, summary.to_frame().T])
        
        timestamp = config.get_timestamp()
        csv_name = os.path.splitext(os.path.basename(self.csv_path))[0]
        log_path = os.path.join(config.LOGS_PATH, f'fold_metrics_{csv_name}_{timestamp}.csv')
        df.to_csv(log_path)
        
        print(f"\n--- Walk-forward Validation Summary ---")
        print(f"  Average MSE: {summary['mse']:.6f}")
        print(f"  Average MAE: {summary['mae']:.6f}")
        print(f"✓ Metrics saved to {log_path}")

    def save_models(self):
        """Save the final ensemble models."""
        print("\n--- Saving Ensemble Models ---")
        csv_name = os.path.splitext(os.path.basename(self.csv_path))[0]
        timestamp = config.get_timestamp()
        
        for i, model in enumerate(self.final_ensemble_models):
            fname = f'{csv_name}_ensemble_{i+1}_{timestamp}.keras'
            path = os.path.join(config.MODEL_SAVE_PATH, fname)
            model.save(path)
            print(f"  ✓ Model {i+1} saved to {path}")

    def run_backtest(self):
        """Run backtesting on the final fold results."""
        backtester = Backtester(config)
        backtester.run(
            self.last_fold_results['actual'], 
            self.last_fold_results['pred'], 
            self.last_fold_results['dates']
        )

    def generate_final_prediction(self):
        """Generate one-shot future prediction for the next N days."""
        print("\n--- Generating Final Future Prediction ---")
        scaler = self.final_scalers['scaler']
        target_scaler = self.final_scalers['target_scaler']
        
        # 1. Prepare input sequence (last window)
        norm_features = scaler.transform(self.features)
        last_seq = norm_features[-config.SEQ_LENGTH:]
        last_seq = np.expand_dims(last_seq, axis=0) # [1, seq_len, num_features]
        
        # 2. Ensemble prediction
        ensemble_preds = [m.predict(last_seq, verbose=0) for m in self.final_ensemble_models]
        avg_pred_norm = np.mean(ensemble_preds, axis=0)
        
        # 3. Denormalize and convert to prices
        future_returns = target_scaler.inverse_transform(avg_pred_norm).flatten()
        last_price = self.processed_data['close'].iloc[-1]
        
        future_prices = []
        curr_p = last_price
        for r in future_returns:
            curr_p *= (1 + r)
            future_prices.append(curr_p)
            
        # 4. Generate future dates
        last_date = self.original_dates.iloc[-1]
        future_dates = pd.date_range(start=last_date, periods=config.FUTURE_DAYS + 1, freq='B')[1:]
        
        print(f"Final Future Prices (Next {config.FUTURE_DAYS} days): {future_prices[:5]}")
        
        # 5. Visualize
        viz_trainer = ModelTrainer(self.final_ensemble_models[0], config, scaler, self.csv_path, target_scaler=target_scaler)
        viz_trainer.plot_predictions(
            y_true=self.last_fold_results['actual'],
            y_pred=self.last_fold_results['pred'],
            test_dates=self.last_fold_results['dates'],
            future_predictions=future_prices,
            future_dates=future_dates
        )


def main():
    """Main execution entry point."""
    args = parse_args()
    apply_config_overrides(args)
    config.ensure_directories()
    set_seed(config.RANDOM_SEED)

    # Initialize and execute pipeline
    pipeline = StockPipeline(args.csv_path)
    pipeline.prepare_data()
    pipeline.run_walk_forward_validation()
    pipeline.save_models()
    pipeline.run_backtest()
    pipeline.generate_final_prediction()


if __name__ == '__main__':
    main()

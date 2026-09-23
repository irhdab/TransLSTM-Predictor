import os
import sys
import argparse
import random
import logging
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model
from sklearn.preprocessing import RobustScaler
import config.config as config
from modules.data_loader import DataProcessor
from modules.model_builder import create_lstm_transformer_model as build_model
from modules.trainer import ModelTrainer
from modules.backtester import Backtester


logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='TransLSTM-Predictor: CNN-BiLSTM-Transformer Hybrid Stock Prediction System'
    )
    parser.add_argument('csv_path', help='Path to the stock data CSV file')
    parser.add_argument('--epochs', type=int, default=None,
                        help=f'Max training epochs (default: {config.EPOCHS})')
    parser.add_argument('--ensemble-size', type=int, default=None,
                        help=f'Number of ensemble models (default: {config.ENSEMBLE_SIZE})')
    parser.add_argument('--seq-length', type=int, default=None,
                        help=f'Input sequence length (default: {config.SEQ_LENGTH})')
    parser.add_argument('--future-days', type=int, default=None,
                        help=f'Future days to predict (default: {config.FUTURE_DAYS})')
    parser.add_argument('--folds', type=int, default=None,
                        help=f'Validation folds (default: {config.WALK_FORWARD_FOLDS})')
    parser.add_argument('--seed', type=int, default=None,
                        help=f'Random seed (default: {config.RANDOM_SEED})')
    parser.add_argument('--transaction-cost', type=float, default=None,
                        help=f'Transaction cost per turnover (default: {config.TRANSACTION_COST})')
    args = parser.parse_args()

    # Fail fast on invalid CLI values
    for name in ('epochs', 'ensemble_size', 'seq_length', 'future_days', 'folds'):
        v = getattr(args, name if name != 'ensemble_size' else 'ensemble_size')
        if v is not None and v < 1:
            parser.error(f"--{name.replace('_','-')} must be >= 1, got {v}")
    if args.seq_length is not None and args.seq_length < 5:
        parser.error(f"--seq-length must be >= 5, got {args.seq_length}")
    if args.transaction_cost is not None and not 0.0 <= args.transaction_cost < 0.1:
        parser.error("--transaction-cost must be in [0, 0.1)")
    if not os.path.isfile(args.csv_path):
        parser.error(f"csv_path not found: {args.csv_path}")
    return args


def apply_config_overrides(args: argparse.Namespace) -> None:
    """Apply CLI overrides to the global config."""
    overrides = {
        'EPOCHS': args.epochs,
        'ENSEMBLE_SIZE': args.ensemble_size,
        'SEQ_LENGTH': args.seq_length,
        'FUTURE_DAYS': args.future_days,
        'WALK_FORWARD_FOLDS': args.folds,
        'RANDOM_SEED': args.seed,
        'TRANSACTION_COST': getattr(args, 'transaction_cost', None),
    }
    for key, value in overrides.items():
        if value is not None:
            setattr(config, key, value)


def set_seed(seed: int) -> None:
    """Ensure reproducibility by fixing all random seeds.

    Note: PYTHONHASHSEED must be set BEFORE the Python process starts to
    have any effect - setting it here is a no-op for hashing. TF
    determinism is enabled via enable_op_determinism() (may cost speed).
    """
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)  # no-op at runtime, documented
    try:
        tf.config.experimental.enable_op_determinism()
    except Exception as e:  # TF version / GPU without deterministic kernels
        logging.getLogger(__name__).warning("TF determinism not enabled: %s", e)


class StockPipeline:
    def __init__(self, csv_path: str) -> None:
        if not os.path.isfile(csv_path):
            raise ValueError(f"CSV file not found: {csv_path}")
        self.csv_path = csv_path
        self.data_processor = DataProcessor(csv_path=csv_path, config=config)
        self.raw_data = None
        self.processed_data = None
        self.features = None
        self.aligned_dates: pd.Series | None = None
        self.aligned_close: np.ndarray | None = None

        # State maintained across folds
        self.final_ensemble_models: list[Model] = []
        self.final_scalers: dict[str, RobustScaler | None] = {}
        self.last_fold_results: dict[str, np.ndarray | pd.Series] = {}

    def prepare_data(self) -> None:
        """Load and process the stock data."""
        self.raw_data = self.data_processor.load_raw_data()
        if not self.data_processor.validate_data(self.raw_data):
            raise ValueError("Data validation failed")

        data = self.data_processor.parse_dates(self.raw_data)
        # Fit outlier bounds on the train prefix only (no future leakage).
        # Estimate: first 60% of parsed rows approximates the earliest fold's train.
        fit_end = max(30, int(len(data) * 0.6))
        data = self.data_processor.handle_outliers(data, fit_end_idx=fit_end)
        self.processed_data = data
        self.features = self.data_processor.extract_features(data)
        # Use POST-indicator aligned dates/prices (fixes 21-day misalignment)
        self.aligned_dates = self.data_processor.get_aligned_dates()
        self.aligned_close = self.data_processor.get_aligned_close()
        if len(self.features) < config.SEQ_LENGTH + config.FUTURE_DAYS + 1:
            raise ValueError(
                f"Not enough data after feature engineering: {len(self.features)} rows, "
                f"need > {config.SEQ_LENGTH + config.FUTURE_DAYS}"
            )
        logger.info("Data preparation complete")

    def run_walk_forward_validation(self) -> None:
        """Execute the walk-forward validation loop."""
        if config.WALK_FORWARD_FOLDS < 1:
            raise ValueError("WALK_FORWARD_FOLDS must be >= 1")
        logger.info("--- Starting Walk-forward Validation (%s folds) ---", config.WALK_FORWARD_FOLDS)

        seq_len = config.SEQ_LENGTH
        future_days = config.FUTURE_DAYS

        # Compute raw sequences ONCE (was recomputed 2x per fold before)
        X_all_raw, y_all_raw, dates_all = self.data_processor.create_sequences(
            self.features, self.aligned_dates
        )
        # Raw close aligned to sequences: close at last input bar of each sample
        # features[i] <-> aligned_close[i]; sequence i uses features[i:i+seq]
        # so last input bar index = i + seq - 1
        n_seq = len(X_all_raw)
        if n_seq < config.WALK_FORWARD_FOLDS + 1:
            raise ValueError(
                f"Not enough sequences ({n_seq}) for {config.WALK_FORWARD_FOLDS} folds"
            )
        seq_last_close = np.array([
            self.aligned_close[i + seq_len - 1] for i in range(n_seq)
        ])

        total_samples = n_seq
        all_fold_metrics = []

        for fold in range(config.WALK_FORWARD_FOLDS):
            logger.info(">> Processing Fold %s/%s", fold + 1, config.WALK_FORWARD_FOLDS)

            # 1. Expanding-window splits: first train = 60%, remainder split
            # evenly into F test chunks. Fixes old 0.6+0.1*fold formula
            # which left 0 test samples on the last fold when F>=5.
            n_folds = config.WALK_FORWARD_FOLDS
            initial_train = int(total_samples * 0.6)
            initial_train = max(1, min(initial_train, total_samples - n_folds))
            remaining = total_samples - initial_train
            test_size = remaining // n_folds
            if test_size < 1:
                raise ValueError(
                    f"Not enough sequences ({total_samples}) for {n_folds} folds"
                )
            train_end_idx = initial_train + fold * test_size
            if fold == n_folds - 1:
                test_end_idx = total_samples  # last fold consumes the tail
            else:
                test_end_idx = train_end_idx + test_size
            train_end_idx = max(1, min(train_end_idx, total_samples - 1))
            test_end_idx = max(train_end_idx + 1, min(test_end_idx, total_samples))

            # 2. Dual Normalization per fold (leakage prevention, fitted on train only)
            # Feature-row index of train end = train_end_idx + seq_len
            norm_features, norm_targets, scaler, target_scaler = self.data_processor.normalize_data(
                self.features, y_all_raw, train_end=train_end_idx + seq_len
            )
            # Rebuild sequences from NORMALIZED features (targets already normalized)
            seq_norm, _, _ = self.data_processor.create_sequences(
                norm_features, self.aligned_dates
            )

            # Slice fold data
            X_train = seq_norm[:train_end_idx]
            y_train = norm_targets[:train_end_idx]
            X_test = seq_norm[train_end_idx:test_end_idx]
            y_test = norm_targets[train_end_idx:test_end_idx]
            test_dates = dates_all[train_end_idx:test_end_idx].reset_index(drop=True)

            # Price conversion context: close at last input bar (aligned, no offset bug)
            last_actual_prices = seq_last_close[train_end_idx:test_end_idx]

            # 3. Train Ensemble for this fold
            fold_models, avg_preds_norm = self._train_fold_ensemble(X_train, y_train, X_test)

            # 4. Evaluate Fold
            mse, mae, dir_acc, actual_prices, pred_prices = self._evaluate_fold(
                fold_models[0], X_test, y_test, test_dates,
                scaler, target_scaler, last_actual_prices, avg_preds_norm
            )

            all_fold_metrics.append({'mse': mse, 'mae': mae, 'dir_acc': dir_acc})

            # Record last fold state for final predictions
            if fold == config.WALK_FORWARD_FOLDS - 1:
                self._update_final_state(fold_models, scaler, target_scaler, actual_prices, pred_prices, test_dates)

        self._save_summary_metrics(all_fold_metrics)

    def _train_fold_ensemble(self, X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray) -> tuple[list[Model], np.ndarray]:
        """Train ensemble with per-member seeds and chronological validation."""
        if len(X_train) < 3:
            raise ValueError(f"Not enough train samples ({len(X_train)}) for train/val split")
        fold_models = []
        fold_predictions_norm = []

        # Chronological validation: last VALIDATION_SPLIT of train (no shuffling into past)
        n_val = max(1, int(len(X_train) * config.VALIDATION_SPLIT))
        n_val = min(n_val, len(X_train) - 1)  # keep at least 1 train sample
        X_tr, y_tr = X_train[:-n_val], y_train[:-n_val]
        X_val, y_val = X_train[-n_val:], y_train[-n_val:]
        base_seed = int(config.RANDOM_SEED)

        logger.info("Training ensemble (size=%s, samples=%s, val=%s)",
                    config.ENSEMBLE_SIZE, len(X_tr), len(X_val))
        for i in range(config.ENSEMBLE_SIZE):
            # Per-member seed for real diversity (was single global seed)
            member_seed = base_seed + i * 100003
            random.seed(member_seed)
            np.random.seed(member_seed)
            tf.random.set_seed(member_seed)
            model = build_model(seq_length=config.SEQ_LENGTH, num_features=self.features.shape[1], config=config)
            trainer = ModelTrainer(model, config, None, self.csv_path)  # scalers passed during eval
            trainer.compile_model()
            trainer.train(X_tr, y_tr, X_val, y_val)
            fold_models.append(model)

            pred = model.predict(X_test, verbose=0)
            fold_predictions_norm.append(pred)

        avg_preds_norm = np.mean(fold_predictions_norm, axis=0)
        return fold_models, avg_preds_norm

    def _evaluate_fold(
        self,
        model: Model,
        X_test: np.ndarray,
        y_test: np.ndarray,
        dates: pd.Series,
        scaler: RobustScaler,
        target_scaler: RobustScaler | None,
        last_prices: np.ndarray,
        predictions: np.ndarray,
    ) -> tuple[float, float, float, np.ndarray, np.ndarray]:
        """Evaluate model performance using rescaled prices."""
        eval_trainer = ModelTrainer(model, config, scaler, self.csv_path, target_scaler=target_scaler)
        return eval_trainer.evaluate(
            X_test, y_test, dates,
            last_actual_prices=last_prices,
            predictions_override=predictions
        )

    def _update_final_state(
        self,
        models: list[Model],
        scaler: RobustScaler,
        target_scaler: RobustScaler | None,
        actual: np.ndarray,
        pred: np.ndarray,
        dates: pd.Series,
    ) -> None:
        self.final_ensemble_models = models
        self.final_scalers = {'scaler': scaler, 'target_scaler': target_scaler}
        self.last_fold_results = {'actual': actual, 'pred': pred, 'dates': dates}

    def _save_summary_metrics(self, all_fold_metrics: list[dict[str, float]]) -> None:
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

        logger.info("--- Walk-forward Validation Summary (denormalized) ---")
        logger.info("Average MSE: %.6f", summary['mse'])
        logger.info("Average MAE: %.6f", summary['mae'])
        if 'dir_acc' in summary:
            logger.info("Average Directional Accuracy: %.4f", summary['dir_acc'])
        logger.info("Metrics saved to %s", log_path)

    def save_models(self) -> None:
        """Save the final ensemble models."""
        if not self.final_ensemble_models:
            logger.warning("No models to save (walk-forward produced no folds)")
            return
        logger.info("--- Saving Ensemble Models ---")
        csv_name = os.path.splitext(os.path.basename(self.csv_path))[0]
        timestamp = config.get_timestamp()

        for i, model in enumerate(self.final_ensemble_models):
            fname = f'{csv_name}_ensemble_{i+1}_{timestamp}.keras'
            path = os.path.join(config.MODEL_SAVE_PATH, fname)
            model.save(path)
            logger.info("Model %s saved to %s", i + 1, path)

    def run_backtest(self) -> None:
        """Run backtesting on the final fold results."""
        if not self.last_fold_results:
            raise ValueError("No fold results for backtesting (check folds/data length)")
        backtester = Backtester(config)
        backtester.run(
            self.last_fold_results['actual'],
            self.last_fold_results['pred'],
            self.last_fold_results['dates']
        )

    def generate_final_prediction(self) -> None:
        """Generate one-shot future prediction for the next N days."""
        if not self.final_ensemble_models:
            raise ValueError("No trained models for future prediction")
        logger.info("--- Generating Final Future Prediction ---")
        scaler = self.final_scalers['scaler']
        target_scaler = self.final_scalers['target_scaler']

        # 1. Prepare input sequence (last window)
        norm_features = scaler.transform(self.features)
        last_seq = norm_features[-config.SEQ_LENGTH:]
        last_seq = np.expand_dims(last_seq, axis=0)  # [1, seq_len, num_features]

        # 2. Ensemble prediction
        ensemble_preds = [m.predict(last_seq, verbose=0) for m in self.final_ensemble_models]
        avg_pred_norm = np.mean(ensemble_preds, axis=0)

        # 3. Denormalize and convert to prices
        if target_scaler is None:
            raise ValueError("target_scaler missing - cannot denormalize returns")
        future_returns = target_scaler.inverse_transform(avg_pred_norm).flatten()
        future_returns = np.clip(future_returns, -0.2, 0.2)
        last_price = float(self.aligned_close[-1])

        future_prices = []
        curr_p = last_price
        for r in future_returns:
            curr_p *= (1 + float(r))
            future_prices.append(curr_p)

        # 4. Generate future dates
        last_date = self.aligned_dates.iloc[-1]
        future_dates = pd.date_range(start=last_date, periods=config.FUTURE_DAYS + 1, freq='B')[1:]

        logger.info("Final future prices (next %s days): %s", config.FUTURE_DAYS, future_prices[:5])

        # 5. Save predictions CSV + visualize
        csv_name = os.path.splitext(os.path.basename(self.csv_path))[0]
        timestamp = config.get_timestamp()
        pred_df = pd.DataFrame({'date': future_dates, 'predicted_close': future_prices})
        pred_path = os.path.join(config.PREDICTIONS_SAVE_PATH, f'{csv_name}_future_{timestamp}.csv')
        pred_df.to_csv(pred_path, index=False)
        logger.info("Future predictions saved to %s", pred_path)

        viz_trainer = ModelTrainer(self.final_ensemble_models[0], config, scaler, self.csv_path, target_scaler=target_scaler)
        viz_trainer.plot_predictions(
            y_true=self.last_fold_results['actual'],
            y_pred=self.last_fold_results['pred'],
            test_dates=self.last_fold_results['dates'],
            future_predictions=np.array(future_prices),
            future_dates=future_dates
        )


def main() -> None:
    """Main execution entry point."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    args = parse_args()
    apply_config_overrides(args)
    config.validate()
    config.ensure_directories()
    set_seed(config.RANDOM_SEED)

    try:
        pipeline = StockPipeline(args.csv_path)
        pipeline.prepare_data()
        pipeline.run_walk_forward_validation()
        pipeline.save_models()
        pipeline.run_backtest()
        pipeline.generate_final_prediction()
    except ValueError as err:
        logger.error("Pipeline aborted: %s", err)
        sys.exit(1)
    except Exception:
        logger.exception("Pipeline failed with an unexpected error")
        sys.exit(1)


if __name__ == '__main__':
    main()

# Configuration file for LSTM-Transformer Hybrid Stock Prediction System

import os
from datetime import datetime

# [1] Data Configuration
SEQ_LENGTH = 60                # Input sequence length
TEST_SPLIT_RATIO = 0.2         # Legacy fallback only: used when normalize_data()
                               # is called without explicit train_end.
                               # Walk-forward validation always passes an
                               # explicit train_end, so this is ignored there.
VALIDATION_SPLIT = 0.2         # Validation data ratio (chronological, last N% of train)
FUTURE_DAYS = 30               # Number of future days to predict
PREDICT_RETURNS = True         # If True, predict % returns instead of absolute prices
ENSEMBLE_SIZE = 3              # Number of models in the ensemble
WALK_FORWARD_FOLDS = 3         # Number of folds for walk-forward validation

RANDOM_SEED = 42              # Random seed for reproducibility

# Feature Columns Configuration
FEATURE_COLS = ['open', 'high', 'low', 'close', 'volume', 'ma_7', 'ma_21', 'rsi', 'macd', 'bollinger_h', 'bollinger_l', 'obv', 'atr']
CLOSE_COL_INDEX = 3            # Index of 'close' in FEATURE_COLS

# [2] Model Architecture Configuration
CONV_FILTERS = 64              # Filters for CNN layer
CONV_KERNEL_SIZE = 3           # Kernel size for CNN layer
TRANSFORMER_HEADS = 8          # Attention heads (key_dim = CONV_FILTERS // HEADS)
TRANSFORMER_FF_DIM = 256       # FF dimension
TRANSFORMER_LAYERS = 3         # Encoder layers
TRANSFORMER_MAX_LEN = 512      # Max positions for positional encoding (dynamic slice)
LSTM_UNITS_1 = 128
LSTM_UNITS_2 = 64
DENSE_UNITS = [256, 128]
DROPOUT_RATE = 0.2
ACTIVATION = 'relu'
GRAD_CLIP_NORM = 1.0           # Gradient clipping for LSTM+Transformer stability
L2_REG = 1e-5                  # L2 weight decay

# [3] Training Parameters
BATCH_SIZE = 32               # Minibatch size
EPOCHS = 100                 # Maximum number of epochs
LEARNING_RATE = 0.001         # Initial learning rate
OPTIMIZER = 'adam'            # Optimization algorithm
LOSS_FUNCTION = 'mse'         # Loss function (MSE)
EARLY_STOPPING_PATIENCE = 10   # Epochs to wait for early stopping
REDUCE_LR_PATIENCE = 5         # Epochs to wait for learning rate reduction
REDUCE_LR_FACTOR = 0.5        # Factor for learning rate reduction
MIN_LEARNING_RATE = 1e-5      # Minimum learning rate

# [3b] Backtest / Trading assumptions
TRANSACTION_COST = 0.001       # 0.1% per turnover (commission + slippage proxy)
RISK_FREE_RATE = 0.0           # Annualized risk-free rate for Sharpe

# [4] Path and File Configuration
DATA_PATH = './data/'          # Input data path
MODEL_SAVE_PATH = './results/models/'
PREDICTIONS_SAVE_PATH = './results/predictions/'
PLOTS_SAVE_PATH = './results/plots/'
LOGS_PATH = './logs/'
TIMESTAMP_FORMAT = '%Y%m%d_%H%M%S'  # Timestamp for filenames

# [5] Visualization Configuration
FIGURE_DPI = 150              # Graph resolution (300 is wasteful for iteration)
FIGURE_SIZE = (14, 6)         # Graph size
FONT_SIZE = 11                # Font size
PLOT_COLORS = {
    'actual': '#1f77b4',          # Actual value (blue)
    'predicted': '#ff7f0e',       # Predicted value (orange)
    'future': '#d62728'           # Future value (red)
}
GRID_ALPHA = 0.3              # Grid transparency

# Utility function to get current timestamp
def get_timestamp() -> str:
    """Return a timestamp string based on TIMESTAMP_FORMAT."""
    return datetime.now().strftime(TIMESTAMP_FORMAT)

# Utility function to ensure directories exist
def ensure_directories() -> None:
    """Create configured output directories if they do not exist."""
    os.makedirs(DATA_PATH, exist_ok=True)
    os.makedirs(MODEL_SAVE_PATH, exist_ok=True)
    os.makedirs(PREDICTIONS_SAVE_PATH, exist_ok=True)
    os.makedirs(PLOTS_SAVE_PATH, exist_ok=True)
    os.makedirs(LOGS_PATH, exist_ok=True)


def validate() -> None:
    """Validate configuration values. Raises ValueError on invalid config."""
    if SEQ_LENGTH < 5:
        raise ValueError(f"SEQ_LENGTH must be >= 5, got {SEQ_LENGTH}")
    if FUTURE_DAYS < 1:
        raise ValueError(f"FUTURE_DAYS must be >= 1, got {FUTURE_DAYS}")
    if ENSEMBLE_SIZE < 1:
        raise ValueError(f"ENSEMBLE_SIZE must be >= 1, got {ENSEMBLE_SIZE}")
    if WALK_FORWARD_FOLDS < 1:
        raise ValueError(f"WALK_FORWARD_FOLDS must be >= 1, got {WALK_FORWARD_FOLDS}")
    if not 0.0 < VALIDATION_SPLIT < 1.0:
        raise ValueError(f"VALIDATION_SPLIT must be in (0,1), got {VALIDATION_SPLIT}")
    if not 0.0 < TEST_SPLIT_RATIO < 1.0:
        raise ValueError(f"TEST_SPLIT_RATIO must be in (0,1), got {TEST_SPLIT_RATIO}")
    if BATCH_SIZE < 1:
        raise ValueError(f"BATCH_SIZE must be >= 1, got {BATCH_SIZE}")
    if EPOCHS < 1:
        raise ValueError(f"EPOCHS must be >= 1, got {EPOCHS}")
    if CONV_FILTERS % TRANSFORMER_HEADS != 0:
        raise ValueError(
            f"CONV_FILTERS ({CONV_FILTERS}) must be divisible by "
            f"TRANSFORMER_HEADS ({TRANSFORMER_HEADS})"
        )
    if not 0.0 <= TRANSACTION_COST < 0.1:
        raise ValueError(f"TRANSACTION_COST must be in [0, 0.1), got {TRANSACTION_COST}")
    # CLOSE_COL_INDEX must track FEATURE_COLS order (was a silent magic number)
    if 'close' in FEATURE_COLS and FEATURE_COLS.index('close') != CLOSE_COL_INDEX:
        raise ValueError(
            f"CLOSE_COL_INDEX ({CLOSE_COL_INDEX}) mismatches "
            f"FEATURE_COLS.index('close') ({FEATURE_COLS.index('close')}). "
            f"Fix config to keep them in sync."
        )

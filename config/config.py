# Configuration file for LSTM-Transformer Hybrid Stock Prediction System

import os
from datetime import datetime

# [1] Data Configuration
SEQ_LENGTH = 60                # Input sequence length
TEST_SPLIT_RATIO = 0.2         # Test data ratio
VALIDATION_SPLIT = 0.2         # Validation data ratio
FUTURE_DAYS = 30               # Number of future days to predict
PREDICT_RETURNS = True         # If True, predict % returns instead of absolute prices
ENSEMBLE_SIZE = 3              # Number of models in the ensemble
WALK_FORWARD_FOLDS = 3         # Number of folds for walk-forward validation
NORMALIZE_METHOD = 'minmax'    # Normalization method
RANDOM_SEED = 42              # Random seed for reproducibility

# Feature Columns Configuration
FEATURE_COLS = ['open', 'high', 'low', 'close', 'volume', 'ma_7', 'ma_21', 'rsi', 'macd', 'bollinger_h', 'bollinger_l', 'obv', 'atr']
CLOSE_COL_INDEX = 3            # Index of 'close' in FEATURE_COLS

# [2] Model Architecture Configuration
CONV_FILTERS = 64              # Filters for CNN layer
CONV_KERNEL_SIZE = 3           # Kernel size for CNN layer
TRANSFORMER_HEADS = 8          # Increased heads for better attention
TRANSFORMER_FF_DIM = 256       # Increased FF dimension
TRANSFORMER_LAYERS = 3         # More layers
LSTM_UNITS_1 = 128            # Increased units
LSTM_UNITS_2 = 64
DENSE_UNITS = [256, 128]
DROPOUT_RATE = 0.2            # Slightly higher dropout for regularization
ACTIVATION = 'relu'

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

# [4] Path and File Configuration
DATA_PATH = './data/'          # Input data path
MODEL_SAVE_PATH = './results/models/'
PREDICTIONS_SAVE_PATH = './results/predictions/'
PLOTS_SAVE_PATH = './results/plots/'
LOGS_PATH = './logs/'
TIMESTAMP_FORMAT = '%Y%m%d_%H%M%S'  # Timestamp for filenames

# [5] Visualization Configuration
FIGURE_DPI = 300              # Graph resolution
FIGURE_SIZE = (14, 6)         # Graph size
FONT_SIZE = 11                # Font size
PLOT_COLORS = {
    'actual': '#1f77b4',          # Actual value (blue)
    'predicted': '#ff7f0e',       # Predicted value (orange)
    'future': '#d62728'           # Future value (red)
}
GRID_ALPHA = 0.3              # Grid transparency

# Utility function to get current timestamp
def get_timestamp():
    return datetime.now().strftime(TIMESTAMP_FORMAT)

# Utility function to ensure directories exist
def ensure_directories():
    os.makedirs(DATA_PATH, exist_ok=True)
    os.makedirs(MODEL_SAVE_PATH, exist_ok=True)
    os.makedirs(PREDICTIONS_SAVE_PATH, exist_ok=True)
    os.makedirs(PLOTS_SAVE_PATH, exist_ok=True)
    os.makedirs(LOGS_PATH, exist_ok=True)
    

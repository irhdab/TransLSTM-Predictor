# Configuration file for LSTM-Transformer Hybrid Stock Prediction System

import os
from datetime import datetime

# [1] Data Configuration
SEQ_LENGTH = 60                # Input sequence length (use 60 days to predict the next day)
TEST_SPLIT_RATIO = 0.2         # Test data ratio (80% for training, 20% for testing)
VALIDATION_SPLIT = 0.2         # Validation data ratio (20% of training data)
FUTURE_DAYS = 30               # Number of future days to predict
NORMALIZE_METHOD = 'minmax'    # Normalization method ('minmax' or 'standard')
RANDOM_SEED = 42              # Random seed for reproducibility

# [2] Model Architecture Configuration
TRANSFORMER_HEADS = 4          # Number of Multi-head Attention heads
TRANSFORMER_FF_DIM = 128       # Transformer Feed-Forward dimension
TRANSFORMER_LAYERS = 2         # Number of Transformer encoder layers
LSTM_UNITS_1 = 64             # Hidden units in the first LSTM layer
LSTM_UNITS_2 = 32             # Hidden units in the second LSTM layer
DENSE_UNITS = [128, 64]       # Number of units in the fully connected layers (list)
DROPOUT_RATE = 0.1            # Dropout rate
ACTIVATION = 'relu'           # Activation function

# [3] Training Parameters
BATCH_SIZE = 32               # Minibatch size
EPOCHS = 100                  # Maximum number of epochs
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
    

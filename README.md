# LSTM-Transformer Hybrid Stock Prediction System

## Overview

This project is a stock price prediction system that uses a hybrid model of LSTM (Long Short-Term Memory) and Transformer networks. It is designed to predict future stock prices based on historical data.

## Features

- **Hybrid Model:** Combines LSTM and Transformer to capture both long-term dependencies and complex patterns in time-series data.
- **Data Preprocessing:** Includes data loading, normalization, and sequence creation.
- **Model Training:** Trains the hybrid model with the processed data.
- **Prediction:** Predicts future stock prices.
- **Visualization:** Plots the actual vs. predicted stock prices.
- **Configurable:** All major parameters can be configured in `config/config.py`.

## Requirements

The required Python packages are listed in `requirements.txt`. You can install them using pip:

```bash
pip install -r requirements.txt
```

## How to Use

1.  **Place your data:** Put your stock data CSV files in the `data/` directory. The CSV file should have at least 'date' and 'close' columns.
2.  **Run the script:** Execute the `main.py` script with the path to your CSV file as a command-line argument.

```bash
python main.py data/YOUR_STOCK_DATA.csv
```

If you don't provide a path, it will use `data/GOOGL.csv` by default.

The prediction results, including plots, will be saved in the `results/` directory.

## Configuration

You can customize the model and training parameters by editing the `config/config.py` file. The configurable parameters include:

- **Data Configuration:** Sequence length, test split ratio, etc.
- **Model Architecture:** Number of Transformer heads, LSTM units, dropout rate, etc.
- **Training Parameters:** Batch size, epochs, learning rate, etc.
- **Paths:** Paths for data, models, and results.
- **Visualization:** Figure size, colors, etc.

# TransLSTM-Predictor: State-of-the-art Stock Prediction System

## Overview

**TransLSTM-Predictor** is a high-performance quantitative trading model that combines **CNN, Bi-LSTM, and Transformer** architectures to achieve state-of-the-art (SOTA) accuracy in stock movement forecasting.

Unlike traditional price predictors, this system focuses on **Percentage Returns Prediction**, utilizing advanced ensemble methods and rigorous validation strategies to provide reliable trading signals.

## Advanced Features

- **Return-Based Prediction (SOTA Strategy)**: Predicts % daily returns instead of absolute prices, significantly improving model stability and generalizability across different price scales.
- **CNN-LSTM-Transformer Hybrid**:
  - **CNN**: Extracts local spatial features (price patterns).
  - **Bi-LSTM**: Captures long-term temporal dependencies.
  - **Transformer**: Multi-head attention mechanism for complex global relationships.
- **Quantitative Validation Suite**:
  - **Walk-forward Validation**: Multi-fold time-series cross-validation to prevent overfitting to specific market regimes.
  - **Ensemble Learning**: Averages predictions from multiple independently trained models to reduce variance and improve robustness.
- **Feature Engineering**: Includes high-impact technical indicators:
  - **Trend**: MA(7, 21), MACD.
  - **Volatility**: Bollinger Bands, ATR (Average True Range).
  - **Volume/Momentum**: RSI, OBV (On-Balance Volume).
- **Financial Backtesting**: Integrated simulator to evaluate the economic performance of the model (Total Return, Sharpe Ratio, MDD, Win Rate).
- **Dual Scaler System**: Separate normalization logic for features and targets to eliminate data leakage and price explosion issues during reconstruction.

## Requirements

```bash
pip install -r requirements.txt
```

## How to Use

1.  **Place your data**: Put your stock data CSV files in the `data/` directory (Columns: `date`, `open`, `high`, `low`, `close`, `volume`).
2.  **Run the pipeline**:
    ```bash
    python main.py data/YOUR_STOCK_DATA.csv
    ```

The system will orchestrate:

1. Data loading & Feature Extraction
2. Walk-forward Validation (Multi-fold training)
3. Ensemble Prediction
4. **Backtesting analysis**
5. 30-day Future Forecasting & Plotting

## Configuration (`config/config.py`)

- **PREDICT_RETURNS**: Toggle between price/return prediction modes.
- **ENSEMBLE_SIZE**: Number of parallel models to train.
- **WALK_FORWARD_FOLDS**: Number of folds for rigorous validation.
- **Model Hyperparameters**: Adjust Transformer heads, layers, and LSTM units.

## 📊 Output

- **Model Files**: `./results/models/`
- **Visualization Plots**: `./results/plots/` (Includes Prediction vs Actual & Backtest Equity Curve)
- **Predictions**: `./results/predictions/`

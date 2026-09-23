# TransLSTM-Predictor: CNN-BiLSTM-Transformer Stock Prediction System

## Overview

**TransLSTM-Predictor** is a quantitative trading research model that combines **CNN, Bi-LSTM, and Transformer** architectures for stock movement forecasting.

Unlike traditional price predictors, this system focuses on **Percentage Returns Prediction**, utilizing ensemble methods and walk-forward validation to provide trading signals. Past backtest performance does not guarantee future returns — use for research, not live trading without further validation.

## Features

- **Return-Based Prediction**: Predicts % daily returns instead of absolute prices, improving stability across price scales.
- **CNN-BiLSTM-Transformer Hybrid**:
  - **CNN**: Extracts local patterns.
  - **Bi-LSTM**: Captures temporal dependencies within the input window (both directions inside the lookback, not future data).
  - **Transformer**: Multi-head attention with dropout, gradient clipping and L2 regularization.
- **Quantitative Validation Suite**:
  - **Walk-forward Validation**: Multi-fold time-series cross-validation with per-fold scalers (no future leakage).
  - **Ensemble Learning**: Averages predictions from independently seeded models to reduce variance.
- **Feature Engineering**: Includes technical indicators:
  - **Trend**: MA(7, 21), MACD.
  - **Volatility**: Bollinger Bands, ATR (Average True Range).
  - **Volume/Momentum**: RSI (Wilder's smoothing), OBV (On-Balance Volume).
- **Financial Backtesting**: Simulator with transaction costs (Total Return, Sharpe Ratio, MDD, Win Rate, Turnover, Cost Drag).
- **Dual Scaler System**: Separate normalization for features and targets, fitted on train prefix only.
- **Reproducibility**: Python/NumPy/TF seeds fixed + `enable_op_determinism()` (best-effort; GPU kernels may still vary).

## Requirements

Python >= 3.10 required.

```bash
pip install -r requirements.txt
```

## How to Use

### Basic Usage

```bash
python main.py data/YOUR_STOCK_DATA.csv
```

### CLI Options

```bash
python main.py data/stock.csv --epochs 50 --ensemble-size 5 --folds 5
```

| Option | Description | Default |
|---|---|---|
| `csv_path` | Path to stock data CSV (required) | — |
| `--epochs` | Max training epochs | 100 |
| `--ensemble-size` | Number of ensemble models | 3 |
| `--seq-length` | Input sequence length (>=5) | 60 |
| `--future-days` | Future days to predict | 30 |
| `--folds` | Walk-forward validation folds | 3 |
| `--seed` | Random seed | 42 |
| `--transaction-cost` | Cost per turnover, [0, 0.1) | 0.001 |

Run `python main.py --help` for full details.

### Input CSV Format

CSV must contain columns: `date`, `open`, `high`, `low`, `close`, `volume`.
Extra `adj close` column is ignored (uses `close`). Minimum rows: `SEQ + FUTURE + 30`.

### Tests

```bash
python -m unittest discover -s tests -v
# or
pytest tests/ -v
```

## Pipeline

The system will orchestrate:

1. Data loading & Feature Extraction
2. Walk-forward Validation (Multi-fold training)
3. Ensemble Prediction
4. **Backtesting analysis**
5. 30-day Future Forecasting & Plotting (+ `results/predictions/*_future_*.csv`)

## Configuration (`config/config.py`)

- **PREDICT_RETURNS**: Toggle between price/return prediction modes.
- **ENSEMBLE_SIZE**: Number of parallel models to train.
- **WALK_FORWARD_FOLDS**: Number of folds for rigorous validation.
- **TRANSACTION_COST / RISK_FREE_RATE**: Backtest assumptions.
- **GRAD_CLIP_NORM / L2_REG**: Training stability.
- **Model Hyperparameters**: Adjust Transformer heads, layers, and LSTM units.

All config values can be overridden via CLI arguments at runtime. `config.validate()` fails fast on bad values.

## Output

| Output | Path |
|---|---|
| Trained Models (`.keras`) | `./results/models/` |
| Prediction Plots | `./results/plots/` |
| Backtest Equity Curve | `./results/plots/backtest_results.png` |
| Fold Metrics (CSV, denormalized MSE/MAE + dir_acc) | `./logs/fold_metrics_*.csv` |
| Future Predictions (CSV) | `./results/predictions/` |

## Limitations

- No comparison to baselines (ARIMA/naive) included — "SOTA" not claimed.
- Backtest uses Day-1 forecasts only; ignores market impact, gaps, and taxes.
- ~1M parameters vs a few thousand bars — overfitting risk; use early stopping + walk-forward means.
- `PYTHONHASHSEED` must be set before process start for full determinism.

## Google Colab

Open `TransLSTM_Predictor.ipynb` to run the full pipeline on Google Colab with GPU acceleration — no local setup required.

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

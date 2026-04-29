# TransLSTM-Predictor: State-of-the-art Stock Prediction System

## 🚀 Overview

**TransLSTM-Predictor** is a high-performance quantitative trading model that combines **CNN, Bi-LSTM, and Transformer** architectures to achieve state-of-the-art (SOTA) accuracy in stock movement forecasting.

Unlike traditional price predictors, this system focuses on **Percentage Returns Prediction**, utilizing advanced ensemble methods and rigorous validation strategies to provide reliable trading signals.

## ✨ Advanced Features

- **Return-Based Prediction (SOTA Strategy)**: Predicts % daily returns instead of absolute prices, significantly improving model stability and generalizability across different price scales.
- **CNN-BiLSTM-Transformer Hybrid**:
  - **CNN**: Extracts local spatial features (price patterns).
  - **Bi-LSTM**: Bidirectional LSTM captures both past and future temporal dependencies.
  - **Transformer**: Multi-head attention mechanism with dropout regularization for complex global relationships.
- **Quantitative Validation Suite**:
  - **Walk-forward Validation**: Multi-fold time-series cross-validation to prevent overfitting to specific market regimes.
  - **Ensemble Learning**: Averages predictions from multiple independently trained models to reduce variance and improve robustness.
- **Feature Engineering**: Includes high-impact technical indicators:
  - **Trend**: MA(7, 21), MACD.
  - **Volatility**: Bollinger Bands, ATR (Average True Range).
  - **Volume/Momentum**: RSI, OBV (On-Balance Volume).
- **Financial Backtesting**: Integrated simulator to evaluate the economic performance of the model (Total Return, Sharpe Ratio, MDD, Win Rate).
- **Dual Scaler System**: Separate normalization logic for features and targets to eliminate data leakage and price explosion issues during reconstruction.
- **Full Reproducibility**: All random seeds (Python, NumPy, TensorFlow) are fixed for deterministic results.

## 🛠️ Requirements

```bash
pip install -r requirements.txt
```

## 📈 How to Use

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
| `--seq-length` | Input sequence length | 60 |
| `--future-days` | Future days to predict | 30 |
| `--folds` | Walk-forward validation folds | 3 |
| `--seed` | Random seed | 42 |

Run `python main.py --help` for full details.

### Input CSV Format

CSV must contain columns: `date`, `open`, `high`, `low`, `close`, `volume`.

## ⚙️ Pipeline

The system will orchestrate:

1. Data loading & Feature Extraction
2. Walk-forward Validation (Multi-fold training)
3. Ensemble Prediction
4. **Backtesting analysis**
5. 30-day Future Forecasting & Plotting

## ⚙️ Configuration (`config/config.py`)

- **PREDICT_RETURNS**: Toggle between price/return prediction modes.
- **ENSEMBLE_SIZE**: Number of parallel models to train.
- **WALK_FORWARD_FOLDS**: Number of folds for rigorous validation.
- **Model Hyperparameters**: Adjust Transformer heads, layers, and LSTM units.

All config values can be overridden via CLI arguments at runtime.

## 📊 Output

| Output | Path |
|---|---|
| Trained Models (`.keras`) | `./results/models/` |
| Prediction Plots | `./results/plots/` |
| Backtest Equity Curve | `./results/plots/backtest_results.png` |
| Fold Metrics (CSV) | `./logs/fold_metrics_*.csv` |
| Predictions | `./results/predictions/` |

## 🔗 Google Colab

Open `TransLSTM_Predictor.ipynb` to run the full pipeline on Google Colab with GPU acceleration — no local setup required.

## 📝 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

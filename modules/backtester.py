import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import logging
from types import ModuleType


logger = logging.getLogger(__name__)

class Backtester:
    def __init__(self, config: ModuleType) -> None:
        self.config = config

    def run(self, actual_prices: np.ndarray, predicted_prices: np.ndarray, dates: pd.Series) -> pd.DataFrame:
        """
        Run a simple trading simulation.

        Convention (no lookahead):
          - ``actual[i]``   = actual close on ``dates[i]``
          - ``predicted[i]`` = model prediction FOR ``dates[i]``,
            made using data up to ``dates[i-1]`` (Day-1 forecast).
          - On close ``dates[i-1]`` we compare ``predicted[i]`` vs
            ``actual[i-1]``; if higher we hold through ``dates[i]``.
          - Implemented as ``signal[i-1] = (predicted[i] > actual[i-1])``
            i.e. ``signal = (predicted.shift(-1) > actual)`` evaluated at
            ``i-1``... rewritten explicitly below to avoid confusion.
          - ``strategy_return[i] = position[i-1] * market_return[i] - cost*turnover``

        Costs: ``config.TRANSACTION_COST`` applied on every position change
        (was 0 before, overstating returns).

        Args:
            actual_prices (np.array): Daily actual prices (Close).
            predicted_prices (np.array): Daily predicted prices (Day-1 forecast).
            dates (pd.Series): Corresponding dates.
        """
        logger.info("--- Running Backtesting Simulation ---")
        cost = getattr(self.config, 'TRANSACTION_COST', 0.0)

        # Create a signals dataframe
        results = pd.DataFrame({
            'date': pd.Series(dates).reset_index(drop=True),
            'actual': np.asarray(actual_prices, dtype=float),
            'predicted': np.asarray(predicted_prices, dtype=float),
        })
        if len(results) < 3:
            raise ValueError("Not enough samples for backtesting")

        # Position for NEXT day: 1 if predicted next close > current close
        # position[i] applies to return[i+1]. No lookahead: predicted[i+1]
        # is known at close of day i (Day-1 forecast).
        results['position'] = (
            results['predicted'].shift(-1) > results['actual']
        ).astype(int)
        # Last row has no next-day forecast -> flat
        results.loc[results.index[-1], 'position'] = 0

        # Market returns (Buy & Hold)
        results['market_return'] = results['actual'].pct_change()

        # Position held during day i is position[i-1]
        results['held'] = results['position'].shift(1).fillna(0)
        # Turnover incurs cost
        turnover = results['held'].diff().abs().fillna(results['held'])
        results['strategy_return'] = (
            results['held'] * results['market_return'] - turnover * cost
        )

        # Cumulative returns
        results['cum_market'] = (1 + results['market_return'].fillna(0)).cumprod()
        results['cum_strategy'] = (1 + results['strategy_return'].fillna(0)).cumprod()

        # Calculate Metrics
        total_return = results['cum_strategy'].iloc[-1] - 1
        market_total = results['cum_market'].iloc[-1] - 1

        # Sharpe Ratio (252 trading days, risk-free from config)
        rf_annual = getattr(self.config, 'RISK_FREE_RATE', 0.0)
        rf_daily = (1 + rf_annual) ** (1 / 252) - 1
        excess = results['strategy_return'].dropna() - rf_daily
        sharpe = float(np.sqrt(252) * excess.mean() / excess.std()) if excess.std() != 0 else 0.0

        # MDD (Maximum Drawdown)
        peak = results['cum_strategy'].cummax()
        drawdown = (results['cum_strategy'] - peak) / peak
        mdd = float(drawdown.min())

        # Win Rate (days held with positive market return)
        trades = results[results['held'] != 0]
        win_rate = float((trades['market_return'] > 0).mean()) if len(trades) > 0 else 0.0
        # Turnover / cost drag diagnostics
        total_turnover = float(turnover.sum())
        cost_drag = float((turnover * cost).sum())

        self.print_report(total_return, market_total, sharpe, mdd, win_rate,
                           cost=cost, turnover=total_turnover, cost_drag=cost_drag)
        self.plot_performance(results)

        return results

    def print_report(self, total: float, market: float, sharpe: float, mdd: float,
                     win_rate: float, cost: float = 0.0, turnover: float = 0.0,
                     cost_drag: float = 0.0) -> None:
        logger.info("%-20s | %-15s | %-15s", "Metric", "Strategy", "Market (B&H)")
        logger.info("%s", "-" * 55)
        logger.info("%-20s | %14.2f%% | %14.2f%%", "Total Return", total * 100, market * 100)
        logger.info("%-20s | %14.2f | N/A", "Sharpe Ratio", sharpe)
        logger.info("%-20s | %14.2f%% | N/A", "Max Drawdown", mdd * 100)
        logger.info("%-20s | %14.2f%% | N/A", "Win Rate", win_rate * 100)
        logger.info("%-20s | %14.4f | N/A", "Trans. Cost", cost)
        logger.info("%-20s | %14.2f | N/A", "Turnover", turnover)
        logger.info("%-20s | %13.2f%% | N/A", "Cost Drag", cost_drag * 100)
        logger.info("%s", "-" * 55)

    def plot_performance(self, results: pd.DataFrame) -> None:
        plt.figure(figsize=self.config.FIGURE_SIZE)
        plt.plot(results['date'], results['cum_market'], label='Market (Buy & Hold)', color='gray', alpha=0.6)
        plt.plot(results['date'], results['cum_strategy'], label='Model Strategy', color='green', linewidth=2)

        plt.title('Backtesting Performance: Strategy vs Market')
        plt.xlabel('Date')
        plt.ylabel('Cumulative Return (1.0 = 100%)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        plot_path = os.path.join(self.config.PLOTS_SAVE_PATH, 'backtest_results.png')
        plt.savefig(plot_path, dpi=getattr(self.config, 'FIGURE_DPI', 150))
        plt.close()
        logger.info("Backtest plot saved to %s", plot_path)

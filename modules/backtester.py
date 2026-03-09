import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

class Backtester:
    def __init__(self, config):
        self.config = config

    def run(self, actual_prices, predicted_prices, dates):
        """
        Run a simple trading simulation.
        Strategy: If Pred[T+1] > Actual[T], Buy/Hold. Otherwise, Sell/Cash.
        
        Args:
            actual_prices (np.array): Daily actual prices (Close).
            predicted_prices (np.array): Daily predicted prices (1st step of multi-step).
            dates (pd.Series): Corresponding dates.
        """
        print("\n--- Running Backtesting Simulation ---")
        
        # Create a signals dataframe
        results = pd.DataFrame({
            'date': dates,
            'actual': actual_prices,
            'predicted': predicted_prices
        })
        
        # Strategy: Buy if tomorrow's predicted price is higher than today's actual
        # Signal = 1 (Long), 0 (Cash)
        results['signal'] = (results['predicted'].shift(-1) > results['actual']).astype(int)
        
        # Market returns (Buy & Hold)
        results['market_return'] = results['actual'].pct_change()
        
        # Strategy returns (Signal is for tomorrow's return)
        results['strategy_return'] = results['signal'].shift(1) * results['market_return']
        
        # Cumulative returns
        results['cum_market'] = (1 + results['market_return'].fillna(0)).cumprod()
        results['cum_strategy'] = (1 + results['strategy_return'].fillna(0)).cumprod()
        
        # Calculate Metrics
        total_return = results['cum_strategy'].iloc[-1] - 1
        market_total = results['cum_market'].iloc[-1] - 1
        
        # Sharpe Ratio (Assuming 252 trading days, simplified risk-free rate = 0)
        sharpe = np.sqrt(252) * results['strategy_return'].mean() / results['strategy_return'].std() if results['strategy_return'].std() != 0 else 0
        
        # MDD (Maximum Drawdown)
        peak = results['cum_strategy'].cummax()
        drawdown = (results['cum_strategy'] - peak) / peak
        mdd = drawdown.min()
        
        # Win Rate (of the signals)
        trades = results[results['signal'].shift(1) != 0]
        win_rate = (trades['market_return'] > 0).mean() if len(trades) > 0 else 0

        self.print_report(total_return, market_total, sharpe, mdd, win_rate)
        self.plot_performance(results)
        
        return results

    def print_report(self, total, market, sharpe, mdd, win_rate):
        print(f"{'Metric':<20} | {'Strategy':<15} | {'Market (B&H)':<15}")
        print("-" * 55)
        print(f"{'Total Return':<20} | {total*100:14.2f}% | {market*100:14.2f}%")
        print(f"{'Sharpe Ratio':<20} | {sharpe:14.2f} | N/A")
        print(f"{'Max Drawdown':<20} | {mdd*100:14.2f}% | N/A")
        print(f"{'Win Rate':<20} | {win_rate*100:14.2f}% | N/A")
        print("-" * 55)

    def plot_performance(self, results):
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
        plt.savefig(plot_path)
        print(f"Backtest plot saved to {plot_path}")

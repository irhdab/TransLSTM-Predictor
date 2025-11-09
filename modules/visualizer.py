
import matplotlib.pyplot as plt
import pandas as pd
import os

def plot_data(data, predictions=None, config=None, test_dates=None):
    """
    Plot the given data and optionally the predictions.

    Args:
        data: The data to plot.
        predictions: The predictions to plot.
        config: The config object.
        test_dates: Dates for the test set.
    """
    plt.figure(figsize=(config.FIGURE_SIZE[0], config.FIGURE_SIZE[1]))
    plt.plot(test_dates, data, label='Actual', color=config.PLOT_COLORS['actual'])
    if predictions is not None:
        plt.plot(test_dates, predictions, label='Predicted', color=config.PLOT_COLORS['predicted'], linestyle='--')
    plt.title('Actual vs Predicted Stock Prices')
    plt.xlabel('Date')
    plt.ylabel('Normalized Price')
    plt.legend()
    plt.grid(True, alpha=config.GRID_ALPHA)
    plt.tight_layout()
    
    if config:
        plot_path = os.path.join(config.PLOTS_SAVE_PATH, 'predictions.png')
        plt.savefig(plot_path)
        print(f"Plot saved to {plot_path}")
    else:
        plt.show()

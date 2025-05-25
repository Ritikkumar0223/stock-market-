import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import warnings

warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8')

def step7_visualize_initial_findings(transformed_data):
    symbols = ['AAPL', 'GOOGL', 'MSFT', 'TSLA']

    for symbol in symbols:
        data = transformed_data[symbol]

        # Plot closing price
        plt.figure(figsize=(14, 4))
        plt.plot(data.index, data['Close'], label='Close Price', color='blue')
        plt.title(f'{symbol} - Closing Price Over Time')
        plt.xlabel('Date')
        plt.ylabel('Price ($)')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.show()

        # Plot original vs capped daily return
        plt.figure(figsize=(14, 4))
        plt.plot(data.index, data['Daily_Return'], label='Original Return', alpha=0.6)
        plt.plot(data.index, data['Capped_Return'], label='Capped Return', alpha=0.6)
        plt.title(f'{symbol} - Daily Returns (Original vs Capped)')
        plt.xlabel('Date')
        plt.ylabel('Daily Return')
        plt.axhline(0, color='black', linestyle='--', linewidth=0.5)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()

        # Histogram of capped returns
        plt.figure(figsize=(8, 4))
        sns.histplot(data['Capped_Return'].dropna(), bins=50, kde=True, color='green')
        plt.title(f'{symbol} - Capped Return Distribution')
        plt.xlabel('Capped Return')
        plt.ylabel('Frequency')
        plt.grid(True, alpha=0.3)
        plt.show()

if __name__ == "__main__":
    # Assuming you already have transformed_data from step6
    transformed_data, outlier_info = step6_handle_outliers_transformations()
    step7_visualize_initial_findings(transformed_data)
    print("Step 7 completed: Initial visualizations generated.")

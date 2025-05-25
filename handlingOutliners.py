import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
from scipy import stats
import warnings

warnings.filterwarnings('ignore')

def step6_handle_outliers_transformations():
    symbols = ['AAPL', 'GOOGL', 'MSFT', 'TSLA']
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365*2)

    outlier_info = {}
    transformed_data = {}

    for symbol in symbols:
        ticker = yf.Ticker(symbol)
        data = ticker.history(start=start_date, end=end_date)
        data = data.fillna(method='ffill').fillna(method='bfill')

        # Calculate daily return
        data['Daily_Return'] = data['Close'].pct_change()

        # Drop NA values for return analysis
        returns = data['Daily_Return'].dropna()

        # Z-score calculation
        z_scores = np.abs(stats.zscore(returns))
        outliers = returns[(z_scores > 3)]

        # Winsorize (cap) the outliers
        capped_returns = returns.copy()
        capped_returns[z_scores > 3] = returns[(z_scores <= 3)].max()
        capped_returns[z_scores < -3] = returns[(z_scores >= -3)].min()

        # Save back into the dataframe
        data['Capped_Return'] = capped_returns.reindex_like(data['Daily_Return'])

        outlier_info[symbol] = {
            'total': len(returns),
            'outliers': len(outliers),
            'outlier_pct': (len(outliers) / len(returns)) * 100
        }

        transformed_data[symbol] = data

        print(f"{symbol} - Total Returns: {len(returns)}, Outliers: {len(outliers)}, Outlier %: {outlier_info[symbol]['outlier_pct']:.2f}%")

    return transformed_data, outlier_info

if __name__ == "__main__":
    transformed_data, outlier_info = step6_handle_outliers_transformations()
    print("Step 6 completed: Outliers handled and data transformed.")

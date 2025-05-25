import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings

warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8')

def step1_data_cleaning():
    symbols = ['AAPL', 'GOOGL', 'MSFT', 'TSLA']
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365*2)
    
    raw_data = {}
    cleaned_data = {}
    
    for symbol in symbols:
        ticker = yf.Ticker(symbol)
        data = ticker.history(start=start_date, end=end_date)
        raw_data[symbol] = data
        
        total_rows = len(data)
        missing_counts = data.isnull().sum()
        missing_percentages = (missing_counts / total_rows) * 100
        
        print(f"{symbol}:")
        print(f"  Total rows: {total_rows}")
        print(f"  Missing values: {missing_counts.sum()}")
        for col in data.columns:
            if missing_counts[col] > 0:
                print(f"    {col}: {missing_counts[col]} ({missing_percentages[col]:.2f}%)")
        
        cleaned = data.fillna(method='ffill').fillna(method='bfill')
        duplicates = cleaned.duplicated().sum()
        if duplicates > 0:
            cleaned = cleaned.drop_duplicates()
        
        cleaned_data[symbol] = cleaned
        print(f"  Cleaned shape: {cleaned.shape}")
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle('Missing Data Heatmap')
    axes = axes.flatten()
    
    for i, symbol in enumerate(symbols):
        missing_data = raw_data[symbol].isnull()
        if missing_data.any().any():
            sns.heatmap(missing_data, yticklabels=False, cbar=True, ax=axes[i])
        else:
            axes[i].text(0.5, 0.5, 'No Missing Data', transform=axes[i].transAxes, ha='center')
        axes[i].set_title(symbol)
    
    plt.tight_layout()
    plt.show()
    
    return raw_data, cleaned_data

if __name__ == "__main__":
    raw_data, cleaned_data = step1_data_cleaning()
    print("Step 1 completed successfully!")
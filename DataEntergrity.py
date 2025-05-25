import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings

warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8')

def step3_data_integrity():
    symbols = ['AAPL', 'GOOGL', 'MSFT', 'TSLA']
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365*2)
    
    cleaned_data = {}
    integrity_report = {}
    
    for symbol in symbols:
        ticker = yf.Ticker(symbol)
        data = ticker.history(start=start_date, end=end_date)
        data = data.fillna(method='ffill').fillna(method='bfill')
        cleaned_data[symbol] = data
    
    for symbol, data in cleaned_data.items():
        print(f"\n{symbol} - Data Integrity Checks:")
        print("="*40)
        
        report = {}
        
        total_rows = len(data)
        report['total_rows'] = total_rows
        
        missing_values = data.isnull().sum().sum()
        report['missing_values'] = missing_values
        print(f"Missing values: {missing_values}")
        
        duplicate_rows = data.duplicated().sum()
        report['duplicate_rows'] = duplicate_rows
        print(f"Duplicate rows: {duplicate_rows}")
        
        price_consistency = (data['High'] >= data['Low']).all()
        report['price_high_low_consistent'] = price_consistency
        print(f"High >= Low consistency: {price_consistency}")
        
        price_range_check = ((data['Close'] >= data['Low']) & (data['Close'] <= data['High'])).all()
        report['price_range_consistent'] = price_range_check
        print(f"Close within High-Low range: {price_range_check}")
        
        open_range_check = ((data['Open'] >= data['Low']) & (data['Open'] <= data['High'])).all()
        report['open_range_consistent'] = open_range_check
        print(f"Open within High-Low range: {open_range_check}")
        
        negative_prices = (data[['Open', 'High', 'Low', 'Close']] < 0).any().any()
        report['negative_prices'] = negative_prices
        print(f"Negative prices found: {negative_prices}")
        
        negative_volume = (data['Volume'] < 0).any()
        report['negative_volume'] = negative_volume
        print(f"Negative volume found: {negative_volume}")
        
        zero_volume = (data['Volume'] == 0).sum()
        report['zero_volume_days'] = zero_volume
        print(f"Zero volume days: {zero_volume}")
        
        extreme_returns = data['Close'].pct_change().abs() > 0.5
        extreme_return_count = extreme_returns.sum()
        report['extreme_returns'] = extreme_return_count
        print(f"Extreme daily returns (>50%): {extreme_return_count}")
        
        date_gaps = pd.date_range(start=data.index.min(), end=data.index.max(), freq='D')
        business_days = pd.bdate_range(start=data.index.min(), end=data.index.max())
        missing_business_days = len(business_days) - len(data)
        report['missing_business_days'] = missing_business_days
        print(f"Missing business days: {missing_business_days}")
        
        sorted_check = data.index.is_monotonic_increasing
        report['chronological_order'] = sorted_check
        print(f"Data in chronological order: {sorted_check}")
        
        data_types_correct = all([
            data['Open'].dtype in ['float64', 'int64'],
            data['High'].dtype in ['float64', 'int64'],
            data['Low'].dtype in ['float64', 'int64'],
            data['Close'].dtype in ['float64', 'int64'],
            data['Volume'].dtype in ['float64', 'int64']
        ])
        report['correct_data_types'] = data_types_correct
        print(f"Correct data types: {data_types_correct}")
        
        reasonable_volume = (data['Volume'] > 0).mean()
        report['reasonable_volume_pct'] = reasonable_volume
        print(f"Days with positive volume: {reasonable_volume:.2%}")
        
        price_volatility = data['Close'].pct_change().std()
        report['daily_volatility'] = price_volatility
        print(f"Daily volatility (std): {price_volatility:.4f}")
        
        integrity_report[symbol] = report
    
    consistency_checks = []
    for symbol in symbols:
        data = cleaned_data[symbol]
        consistency_checks.append({
            'Symbol': symbol,
            'Data_Points': len(data),
            'Date_Range_Days': (data.index.max() - data.index.min()).days,
            'Avg_Volume': data['Volume'].mean(),
            'Avg_Price': data['Close'].mean(),
            'Price_Volatility': data['Close'].pct_change().std()
        })
    
    consistency_df = pd.DataFrame(consistency_checks)
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Data Integrity and Consistency Analysis', fontsize=16)
    
    axes[0,0].bar(consistency_df['Symbol'], consistency_df['Data_Points'])
    axes[0,0].set_title('Data Points per Symbol')
    axes[0,0].set_ylabel('Number of Records')
    
    axes[0,1].bar(consistency_df['Symbol'], consistency_df['Date_Range_Days'])
    axes[0,1].set_title('Date Range Coverage (Days)')
    axes[0,1].set_ylabel('Days')
    
    axes[1,0].bar(consistency_df['Symbol'], consistency_df['Avg_Volume'])
    axes[1,0].set_title('Average Trading Volume')
    axes[1,0].set_ylabel('Volume')
    axes[1,0].ticklabel_format(style='scientific', axis='y', scilimits=(0,0))
    
    axes[1,1].bar(consistency_df['Symbol'], consistency_df['Price_Volatility'])
    axes[1,1].set_title('Price Volatility (Daily Returns Std)')
    axes[1,1].set_ylabel('Volatility')
    
    plt.tight_layout()
    plt.show()
    
    integrity_summary = pd.DataFrame(integrity_report).T
    
    plt.figure(figsize=(12, 8))
    
    integrity_metrics = ['missing_values', 'duplicate_rows', 'extreme_returns', 
                        'zero_volume_days', 'missing_business_days']
    
    integrity_data = integrity_summary[integrity_metrics]
    
    sns.heatmap(integrity_data, annot=True, fmt='g', cmap='Reds', 
                cbar_kws={'label': 'Count'})
    plt.title('Data Integrity Issues Heatmap')
    plt.xlabel('Integrity Metrics')
    plt.ylabel('Stocks')
    plt.tight_layout()
    plt.show()
    
    print("\n" + "="*60)
    print("DATA INTEGRITY SUMMARY")
    print("="*60)
    
    for symbol in symbols:
        report = integrity_report[symbol]
        issues = 0
        
        if report['missing_values'] > 0: issues += 1
        if report['duplicate_rows'] > 0: issues += 1
        if not report['price_high_low_consistent']: issues += 1
        if not report['price_range_consistent']: issues += 1
        if report['negative_prices']: issues += 1
        if report['negative_volume']: issues += 1
        if report['extreme_returns'] > 5: issues += 1
        
        quality_score = max(0, 100 - (issues * 10))
        print(f"{symbol}: Quality Score = {quality_score}%")
    
    return cleaned_data, integrity_report, consistency_df

if __name__ == "__main__":
    cleaned_data, integrity_report, consistency_df = step3_data_integrity()
    print("Step 3 completed successfully!")
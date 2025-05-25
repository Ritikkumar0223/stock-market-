import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
from scipy import stats
from scipy.signal import find_peaks
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8')

def step5_patterns_trends_anomalies():
    symbols = ['AAPL', 'GOOGL', 'MSFT', 'TSLA']
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365*2)
    
    cleaned_data = {}
    patterns = {}
    
    for symbol in symbols:
        ticker = yf.Ticker(symbol)
        data = ticker.history(start=start_date, end=end_date)
        data = data.fillna(method='ffill').fillna(method='bfill')
        data['Daily_Return'] = data['Close'].pct_change()
        data['SMA_20'] = data['Close'].rolling(window=20).mean()
        data['SMA_50'] = data['Close'].rolling(window=50).mean()
        data['RSI'] = calculate_rsi(data['Close'])
        cleaned_data[symbol] = data
    
    for symbol, data in cleaned_data.items():
        print(f"\n{symbol} - Pattern Analysis:")
        print("="*40)
        
        pattern_dict = {}
        
        trend_direction = 'Upward' if data['Close'].iloc[-1] > data['Close'].iloc[0] else 'Downward'
        pattern_dict['overall_trend'] = trend_direction
        
        price_change = (data['Close'].iloc[-1] / data['Close'].iloc[0] - 1) * 100
        pattern_dict['price_change_pct'] = price_change
        
        sma_crossovers = []
        for i in range(1, len(data)):
            if data['SMA_20'].iloc[i-1] <= data['SMA_50'].iloc[i-1] and data['SMA_20'].iloc[i] > data['SMA_50'].iloc[i]:
                sma_crossovers.append(('Golden Cross', data.index[i]))
            elif data['SMA_20'].iloc[i-1] >= data['SMA_50'].iloc[i-1] and data['SMA_20'].iloc[i] < data['SMA_50'].iloc[i]:
                sma_crossovers.append(('Death Cross', data.index[i]))
        
        pattern_dict['sma_crossovers'] = len(sma_crossovers)
        
        highs, _ = find_peaks(data['Close'].values, distance=20)
        lows, _ = find_peaks(-data['Close'].values, distance=20)
        
        pattern_dict['local_highs'] = len(highs)
        pattern_dict['local_lows'] = len(lows)
        
        returns = data['Daily_Return'].dropna()
        extreme_up = (returns > returns.quantile(0.95)).sum()
        extreme_down = (returns < returns.quantile(0.05)).sum()
        
        pattern_dict['extreme_up_days'] = extreme_up
        pattern_dict['extreme_down_days'] = extreme_down
        
        consecutive_up = 0
        consecutive_down = 0
        max_consecutive_up = 0
        max_consecutive_down = 0
        
        for ret in returns:
            if ret > 0:
                consecutive_up += 1
                consecutive_down = 0
                max_consecutive_up = max(max_consecutive_up, consecutive_up)
            elif ret < 0:
                consecutive_down += 1
                consecutive_up = 0
                max_consecutive_down = max(max_consecutive_down, consecutive_down)
            else:
                consecutive_up = 0
                consecutive_down = 0
        
        pattern_dict['max_consecutive_gains'] = max_consecutive_up
        pattern_dict['max_consecutive_losses'] = max_consecutive_down
        
        volume_spikes = (data['Volume'] > data['Volume'].quantile(0.95)).sum()
        pattern_dict['volume_spikes'] = volume_spikes
        
        gap_ups = ((data['Open'] - data['Close'].shift(1)) / data['Close'].shift(1) > 0.02).sum()
        gap_downs = ((data['Open'] - data['Close'].shift(1)) / data['Close'].shift(1) < -0.02).sum()
        
        pattern_dict['gap_ups'] = gap_ups
        pattern_dict['gap_downs'] = gap_downs
        
        data['Month'] = data.index.month
        data['DayOfWeek'] = data.index.dayofweek
        
        monthly_returns = data.groupby('Month')['Daily_Return'].mean()
        best_month = monthly_returns.idxmax()
        worst_month = monthly_returns.idxmin()
        
        pattern_dict['best_month'] = best_month
        pattern_dict['worst_month'] = worst_month
        
        weekly_returns = data.groupby('DayOfWeek')['Daily_Return'].mean()
        best_day = weekly_returns.idxmax()
        worst_day = weekly_returns.idxmin()
        
        pattern_dict['best_day_of_week'] = best_day
        pattern_dict['worst_day_of_week'] = worst_day
        
        patterns[symbol] = pattern_dict
        
        print(f"Overall Trend: {trend_direction} ({price_change:.1f}%)")
        print(f"SMA Crossovers: {len(sma_crossovers)}")
        print(f"Local Highs/Lows: {len(highs)}/{len(lows)}")
        print(f"Extreme Days: {extreme_up} up, {extreme_down} down")
        print(f"Max Consecutive: {max_consecutive_up} gains, {max_consecutive_down} losses")
        print(f"Volume Spikes: {volume_spikes}")
        print(f"Price Gaps: {gap_ups} up, {gap_downs} down")
        print(f"Best/Worst Month: {best_month}/{worst_month}")
        print(f"Best/Worst Day: {best_day}/{worst_day}")
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Price Patterns and Trends Analysis', fontsize=16)
    
    for i, symbol in enumerate(symbols):
        ax = axes[i//2, i%2]
        data = cleaned_data[symbol]
        
        ax.plot(data.index, data['Close'], label='Close Price', linewidth=1.5)
        ax.plot(data.index, data['SMA_20'], label='SMA 20', alpha=0.7)
        ax.plot(data.index, data['SMA_50'], label='SMA 50', alpha=0.7)
        
        highs, _ = find_peaks(data['Close'].values, distance=20)
        lows, _ = find_peaks(-data['Close'].values, distance=20)
        
        ax.scatter(data.index[highs], data['Close'].iloc[highs], 
                  color='red', marker='v', s=50, label='Local Highs')
        ax.scatter(data.index[lows], data['Close'].iloc[lows], 
                  color='green', marker='^', s=50, label='Local Lows')
        
        ax.set_title(f'{symbol} - Price Trends & Patterns')
        ax.set_ylabel('Price ($)')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    anomaly_data = {}
    
    for symbol, data in cleaned_data.items():
        returns = data['Daily_Return'].dropna()
        
        z_scores = np.abs(stats.zscore(returns))
        anomalies_zscore = (z_scores > 3).sum()
        
        Q1 = returns.quantile(0.25)
        Q3 = returns.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        anomalies_iqr = ((returns < lower_bound) | (returns > upper_bound)).sum()
        
        volume = data['Volume']
        volume_z = np.abs(stats.zscore(volume))
        volume_anomalies = (volume_z > 3).sum()
        
        anomaly_data[symbol] = {
            'return_anomalies_zscore': anomalies_zscore,
            'return_anomalies_iqr': anomalies_iqr,
            'volume_anomalies': volume_anomalies
        }
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Anomaly Detection Analysis', fontsize=16)
    
    for i, symbol in enumerate(symbols):
        ax = axes[i//2, i%2]
        data = cleaned_data[symbol]
        returns = data['Daily_Return'].dropna()
        
        z_scores = stats.zscore(returns)
        anomaly_threshold = 3
        
        ax.scatter(range(len(z_scores)), z_scores, alpha=0.6, s=10)
        ax.axhline(y=anomaly_threshold, color='red', linestyle='--', label=f'Threshold (+{anomaly_threshold})')
        ax.axhline(y=-anomaly_threshold, color='red', linestyle='--', label=f'Threshold (-{anomaly_threshold})')
        
        anomaly_indices = np.where(np.abs(z_scores) > anomaly_threshold)[0]
        ax.scatter(anomaly_indices, z_scores[anomaly_indices], 
                  color='red', s=30, label=f'Anomalies ({len(anomaly_indices)})')
        
        ax.set_title(f'{symbol} - Return Anomalies (Z-Score)')
        ax.set_xlabel('Time Index')
        ax.set_ylabel('Z-Score')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    features_for_clustering = []
    stock_labels = []
    
    for symbol, data in cleaned_data.items():
        returns = data['Daily_Return'].dropna()
        features = [
            returns.mean(),
            returns.std(),
            returns.skew(),
            returns.kurtosis(),
            patterns[symbol]['max_consecutive_gains'],
            patterns[symbol]['max_consecutive_losses'],
            patterns[symbol]['volume_spikes'],
            patterns[symbol]['local_highs']
        ]
        features_for_clustering.append(features)
        stock_labels.append(symbol)
    
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features_for_clustering)
    
    kmeans = KMeans(n_clusters=2, random_state=42)
    clusters = kmeans.fit_predict(scaled_features)
    
    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(scaled_features[:, 0], scaled_features[:, 1], 
                         c=clusters, cmap='viridis', s=100)
    
    for i, label in enumerate(stock_labels):
        plt.annotate(label, (scaled_features[i, 0], scaled_features[i, 1]), 
                    xytext=(5, 5), textcoords='offset points')
    
    plt.xlabel('Mean Return (Scaled)')
    plt.ylabel('Volatility (Scaled)')
    plt.title('Stock Clustering Based on Pattern Features')
    plt.colorbar(scatter, label='Cluster')
    plt.grid(True, alpha=0.3)
    plt.show()
    
    seasonal_analysis = {}
    for symbol, data in cleaned_data.items():
        monthly_perf = data.groupby(data.index.month)['Daily_Return'].agg(['mean', 'std', 'count'])
        seasonal_analysis[symbol] = monthly_perf
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Seasonal Patterns Analysis', fontsize=16)
    
    months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
              'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    
    for i, symbol in enumerate(symbols):
        ax = axes[i//2, i%2]
        monthly_data = seasonal_analysis[symbol]
        
        ax.bar(range(1, 13), monthly_data['mean'] * 100, alpha=0.7)
        ax.set_title(f'{symbol} - Average Monthly Returns')
        ax.set_xlabel('Month')
        ax.set_ylabel('Average Return (%)')
        ax.set_xticks(range(1, 13))
        ax.set_xticklabels(months, rotation=45)
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='red', linestyle='-', alpha=0.5)
    
    plt.tight_layout()
    plt.show()
    
    print("\n" + "="*60)
    print("PATTERN ANALYSIS SUMMARY")
    print("="*60)
    
    for symbol in symbols:
        pattern = patterns[symbol]
        anomaly = anomaly_data[symbol]
        print(f"\n{symbol}:")
        print(f"  Trend: {pattern['overall_trend']} ({pattern['price_change_pct']:.1f}%)")
        print(f"  Pattern Signals: {pattern['sma_crossovers']} crossovers, {pattern['local_highs']} peaks")
        print(f"  Anomalies: {anomaly['return_anomalies_zscore']} return, {anomaly['volume_anomalies']} volume")
        print(f"  Seasonality: Best month {pattern['best_month']}, Worst month {pattern['worst_month']}")
    
    return cleaned_data, patterns, anomaly_data, seasonal_analysis

def calculate_rsi(prices, window=14):
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

if __name__ == "__main__":
    cleaned_data, patterns, anomaly_data, seasonal_analysis = step5_patterns_trends_anomalies()
    print("Step 5 completed successfully!")
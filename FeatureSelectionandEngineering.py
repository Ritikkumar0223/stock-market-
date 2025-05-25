import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8')

def step2_feature_engineering():
    symbols = ['AAPL', 'GOOGL', 'MSFT', 'TSLA']
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365*2)
    
    cleaned_data = {}
    features = {}
    
    for symbol in symbols:
        ticker = yf.Ticker(symbol)
        data = ticker.history(start=start_date, end=end_date)
        data = data.fillna(method='ffill').fillna(method='bfill')
        cleaned_data[symbol] = data
    
    for symbol, data in cleaned_data.items():
        df = data.copy()
        
        df['Daily_Return'] = df['Close'].pct_change()
        df['Log_Return'] = np.log(df['Close'] / df['Close'].shift(1))
        
        df['Price_Range'] = df['High'] - df['Low']
        df['Price_Change'] = df['Close'] - df['Open']
        df['Price_Change_Pct'] = (df['Close'] - df['Open']) / df['Open'] * 100
        
        df['Volume_MA_10'] = df['Volume'].rolling(window=10).mean()
        df['Volume_Ratio'] = df['Volume'] / df['Volume_MA_10']
        
        df['SMA_5'] = df['Close'].rolling(window=5).mean()
        df['SMA_10'] = df['Close'].rolling(window=10).mean()
        df['SMA_20'] = df['Close'].rolling(window=20).mean()
        df['SMA_50'] = df['Close'].rolling(window=50).mean()
        
        df['EMA_5'] = df['Close'].ewm(span=5).mean()
        df['EMA_10'] = df['Close'].ewm(span=10).mean()
        df['EMA_20'] = df['Close'].ewm(span=20).mean()
        
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        df['EMA_12'] = df['Close'].ewm(span=12).mean()
        df['EMA_26'] = df['Close'].ewm(span=26).mean()
        df['MACD'] = df['EMA_12'] - df['EMA_26']
        df['MACD_Signal'] = df['MACD'].ewm(span=9).mean()
        df['MACD_Histogram'] = df['MACD'] - df['MACD_Signal']
        
        df['BB_Middle'] = df['Close'].rolling(window=20).mean()
        bb_std = df['Close'].rolling(window=20).std()
        df['BB_Upper'] = df['BB_Middle'] + (bb_std * 2)
        df['BB_Lower'] = df['BB_Middle'] - (bb_std * 2)
        df['BB_Width'] = df['BB_Upper'] - df['BB_Lower']
        df['BB_Position'] = (df['Close'] - df['BB_Lower']) / (df['BB_Upper'] - df['BB_Lower'])
        
        high_low = df['High'] - df['Low']
        high_close = np.abs(df['High'] - df['Close'].shift())
        low_close = np.abs(df['Low'] - df['Close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = np.max(ranges, axis=1)
        df['ATR'] = true_range.rolling(14).mean()
        
        df['Volatility_5'] = df['Daily_Return'].rolling(window=5).std()
        df['Volatility_20'] = df['Daily_Return'].rolling(window=20).std()
        
        df['High_Low_Pct'] = (df['High'] - df['Low']) / df['Close'] * 100
        df['Open_Close_Pct'] = (df['Close'] - df['Open']) / df['Open'] * 100
        
        df['Price_Position'] = (df['Close'] - df['Low'].rolling(window=14).min()) / (df['High'].rolling(window=14).max() - df['Low'].rolling(window=14).min())
        
        df['Momentum_5'] = df['Close'] / df['Close'].shift(5) - 1
        df['Momentum_10'] = df['Close'] / df['Close'].shift(10) - 1
        
        features[symbol] = df.dropna()
        
        print(f"{symbol} Features Created:")
        print(f"  Original columns: {len(data.columns)}")
        print(f"  Total features: {len(df.columns)}")
        print(f"  New features: {len(df.columns) - len(data.columns)}")
    
    selected_features = ['Daily_Return', 'Log_Return', 'Price_Range', 'Volume_Ratio', 
                        'RSI', 'MACD', 'BB_Position', 'ATR', 'Volatility_20']
    
    feature_matrix = {}
    for symbol in symbols:
        feature_data = features[symbol][selected_features].dropna()
        feature_matrix[symbol] = feature_data
        
        corr_matrix = feature_data.corr()
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, 
                   square=True, linewidths=0.5)
        plt.title(f'{symbol} - Feature Correlation Matrix')
        plt.tight_layout()
        plt.show()
    
    all_features = pd.concat([feature_matrix[symbol].assign(Symbol=symbol) 
                             for symbol in symbols], ignore_index=True)
    
    numeric_features = all_features.select_dtypes(include=[np.number])
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(numeric_features)
    
    pca = PCA()
    pca_features = pca.fit_transform(scaled_features)
    
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(np.cumsum(pca.explained_variance_ratio_))
    plt.xlabel('Number of Components')
    plt.ylabel('Cumulative Explained Variance')
    plt.title('PCA - Explained Variance')
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    feature_importance = np.abs(pca.components_[0])
    feature_names = numeric_features.columns
    indices = np.argsort(feature_importance)[::-1]
    
    plt.bar(range(len(feature_importance)), feature_importance[indices])
    plt.xticks(range(len(feature_importance)), [feature_names[i] for i in indices], rotation=45)
    plt.title('Feature Importance (First Principal Component)')
    
    plt.tight_layout()
    plt.show()
    
    print(f"\nFeature Engineering Summary:")
    print(f"Selected features: {len(selected_features)}")
    print(f"Total samples: {len(all_features)}")
    print(f"PCA - First 3 components explain {sum(pca.explained_variance_ratio_[:3]):.2%} of variance")
    
    return features, feature_matrix, selected_features

if __name__ == "__main__":
    features, feature_matrix, selected_features = step2_feature_engineering()
    print("Step 2 completed successfully!")

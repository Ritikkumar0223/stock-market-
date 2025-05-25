import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
from scipy import stats

warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8')

def step4_summary_statistics():
    symbols = ['AAPL', 'GOOGL', 'MSFT', 'TSLA']
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365*2)
    
    cleaned_data = {}
    summary_stats = {}
    
    for symbol in symbols:
        ticker = yf.Ticker(symbol)
        data = ticker.history(start=start_date, end=end_date)
        data = data.fillna(method='ffill').fillna(method='bfill')
        data['Daily_Return'] = data['Close'].pct_change()
        data['Log_Return'] = np.log(data['Close'] / data['Close'].shift(1))
        cleaned_data[symbol] = data
    
    for symbol, data in cleaned_data.items():
        print(f"\n{symbol} - Summary Statistics:")
        print("="*50)
        
        stats_dict = {}
        
        price_stats = data['Close'].describe()
        volume_stats = data['Volume'].describe()
        return_stats = data['Daily_Return'].describe()
        
        stats_dict['price_mean'] = price_stats['mean']
        stats_dict['price_median'] = price_stats['50%']
        stats_dict['price_std'] = price_stats['std']
        stats_dict['price_min'] = price_stats['min']
        stats_dict['price_max'] = price_stats['max']
        stats_dict['price_range'] = price_stats['max'] - price_stats['min']
        
        stats_dict['volume_mean'] = volume_stats['mean']
        stats_dict['volume_median'] = volume_stats['50%']
        stats_dict['volume_std'] = volume_stats['std']
        
        stats_dict['return_mean'] = return_stats['mean']
        stats_dict['return_median'] = return_stats['50%']
        stats_dict['return_std'] = return_stats['std']
        stats_dict['return_min'] = return_stats['min']
        stats_dict['return_max'] = return_stats['max']
        
        stats_dict['volatility_annualized'] = return_stats['std'] * np.sqrt(252)
        
        stats_dict['skewness'] = stats.skew(data['Daily_Return'].dropna())
        stats_dict['kurtosis'] = stats.kurtosis(data['Daily_Return'].dropna())
        
        positive_returns = (data['Daily_Return'] > 0).sum()
        total_returns = len(data['Daily_Return'].dropna())
        stats_dict['win_rate'] = positive_returns / total_returns
        
        stats_dict['sharpe_ratio'] = stats_dict['return_mean'] / stats_dict['return_std'] * np.sqrt(252)
        
        drawdowns = []
        peak = data['Close'].iloc[0]
        for price in data['Close']:
            if price > peak:
                peak = price
            drawdown = (price - peak) / peak
            drawdowns.append(drawdown)
        
        stats_dict['max_drawdown'] = min(drawdowns)
        
        cumulative_return = (data['Close'].iloc[-1] / data['Close'].iloc[0]) - 1
        stats_dict['total_return'] = cumulative_return
        
        days_held = len(data)
        annualized_return = (1 + cumulative_return) ** (252 / days_held) - 1
        stats_dict['annualized_return'] = annualized_return
        
        stats_dict['var_95'] = np.percentile(data['Daily_Return'].dropna(), 5)
        stats_dict['cvar_95'] = data['Daily_Return'][data['Daily_Return'] <= stats_dict['var_95']].mean()
        
        high_low_range = data['High'] - data['Low']
        stats_dict['avg_daily_range'] = high_low_range.mean()
        stats_dict['avg_daily_range_pct'] = (high_low_range / data['Close']).mean() * 100
        
        summary_stats[symbol] = stats_dict
        
        print(f"Price Statistics:")
        print(f"  Mean: ${stats_dict['price_mean']:.2f}")
        print(f"  Median: ${stats_dict['price_median']:.2f}")
        print(f"  Std Dev: ${stats_dict['price_std']:.2f}")
        print(f"  Range: ${stats_dict['price_range']:.2f}")
        
        print(f"Return Statistics:")
        print(f"  Mean Daily Return: {stats_dict['return_mean']:.4f} ({stats_dict['return_mean']*100:.2f}%)")
        print(f"  Volatility (Annualized): {stats_dict['volatility_annualized']:.4f} ({stats_dict['volatility_annualized']*100:.1f}%)")
        print(f"  Sharpe Ratio: {stats_dict['sharpe_ratio']:.2f}")
        print(f"  Win Rate: {stats_dict['win_rate']:.2%}")
        
        print(f"Risk Metrics:")
        print(f"  Max Drawdown: {stats_dict['max_drawdown']:.2%}")
        print(f"  VaR (95%): {stats_dict['var_95']:.4f}")
        print(f"  Skewness: {stats_dict['skewness']:.2f}")
        print(f"  Kurtosis: {stats_dict['kurtosis']:.2f}")
        
        print(f"Performance:")
        print(f"  Total Return: {stats_dict['total_return']:.2%}")
        print(f"  Annualized Return: {stats_dict['annualized_return']:.2%}")
    
    stats_df = pd.DataFrame(summary_stats).T
    
    fig, axes = plt.subplots(3, 2, figsize=(15, 12))
    fig.suptitle('Summary Statistics Dashboard', fontsize=16)
    
    axes[0,0].bar(stats_df.index, stats_df['return_mean'] * 100)
    axes[0,0].set_title('Average Daily Returns (%)')
    axes[0,0].set_ylabel('Return %')
    
    axes[0,1].bar(stats_df.index, stats_df['volatility_annualized'] * 100)
    axes[0,1].set_title('Annualized Volatility (%)')
    axes[0,1].set_ylabel('Volatility %')
    
    axes[1,0].bar(stats_df.index, stats_df['sharpe_ratio'])
    axes[1,0].set_title('Sharpe Ratio')
    axes[1,0].set_ylabel('Sharpe Ratio')
    axes[1,0].axhline(y=0, color='red', linestyle='--', alpha=0.5)
    
    axes[1,1].bar(stats_df.index, stats_df['max_drawdown'] * 100)
    axes[1,1].set_title('Maximum Drawdown (%)')
    axes[1,1].set_ylabel('Drawdown %')
    
    axes[2,0].bar(stats_df.index, stats_df['win_rate'] * 100)
    axes[2,0].set_title('Win Rate (%)')
    axes[2,0].set_ylabel('Win Rate %')
    axes[2,0].axhline(y=50, color='red', linestyle='--', alpha=0.5)
    
    axes[2,1].bar(stats_df.index, stats_df['total_return'] * 100)
    axes[2,1].set_title('Total Return (%)')
    axes[2,1].set_ylabel('Return %')
    
    plt.tight_layout()
    plt.show()
    
    correlation_matrix = pd.DataFrame()
    for symbol in symbols:
        correlation_matrix[symbol] = cleaned_data[symbol]['Daily_Return']
    
    corr = correlation_matrix.corr()
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, annot=True, cmap='coolwarm', center=0, 
                square=True, linewidths=0.5, fmt='.3f')
    plt.title('Daily Returns Correlation Matrix')
    plt.tight_layout()
    plt.show()
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Return Distribution Analysis', fontsize=16)
    
    for i, symbol in enumerate(symbols):
        ax = axes[i//2, i%2]
        returns = cleaned_data[symbol]['Daily_Return'].dropna()
        
        ax.hist(returns, bins=50, alpha=0.7, density=True, label='Actual')
        
        mu, sigma = stats.norm.fit(returns)
        x = np.linspace(returns.min(), returns.max(), 100)
        ax.plot(x, stats.norm.pdf(x, mu, sigma), 'r-', linewidth=2, label=f'Normal(μ={mu:.4f}, σ={sigma:.4f})')
        
        ax.set_title(f'{symbol} - Return Distribution')
        ax.set_xlabel('Daily Returns')
        ax.set_ylabel('Density')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    key_insights = {
        'highest_return': stats_df['annualized_return'].idxmax(),
        'lowest_volatility': stats_df['volatility_annualized'].idxmin(),
        'best_sharpe': stats_df['sharpe_ratio'].idxmax(),
        'lowest_drawdown': stats_df['max_drawdown'].idxmax(),
        'highest_correlation': corr.abs().unstack().sort_values(ascending=False).drop_duplicates().iloc[1]
    }
    
    print("\n" + "="*60)
    print("KEY INSIGHTS")
    print("="*60)
    print(f"Best Performer (Annualized Return): {key_insights['highest_return']}")
    print(f"Lowest Risk (Volatility): {key_insights['lowest_volatility']}")
    print(f"Best Risk-Adjusted Return (Sharpe): {key_insights['best_sharpe']}")
    print(f"Smallest Drawdown: {key_insights['lowest_drawdown']}")
    print(f"Market Correlation Range: {corr.min().min():.3f} to {corr.max().max():.3f}")
    
    return cleaned_data, summary_stats, stats_df

if __name__ == "__main__":
    cleaned_data, summary_stats, stats_df = step4_summary_statistics()
    print("Step 4 completed successfully!")
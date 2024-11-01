from typing import Dict, List, Tuple
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from concurrent.futures import ProcessPoolExecutor
import itertools
import backtrader as bt

# Update data download to get 1h data for last 730 days
end_date = datetime.now()
start_date = end_date - timedelta(days=730)  # Using 365 days to be safe
eth = yf.download('BTC-USD', 
                  start=start_date, 
                  end=end_date,
                  interval='1h')

# Save data with index (datetime)
eth.reset_index(inplace=True)  # This converts the index to a column named 'Datetime'
eth.to_csv('StrategyTest/BTCUSD_1h.csv', index=False)

class TrendFollowingStrategy:
    def __init__(self, 
                 fast_ma: int = 5,        # Very fast MA for quick signals
                 slow_ma: int = 13,       # Shorter slow MA
                 atr_period: int = 10,    
                 risk_pct: float = 0.02,
                 rsi_period: int = 7,     # Faster RSI
                 rsi_overbought: int = 75,
                 rsi_oversold: int = 25,
                 volume_ma: int = 10,     # Shorter volume MA
                 take_profit: float = 0.015): # 1.5% take profit
        self.fast_ma = fast_ma
        self.slow_ma = slow_ma
        self.atr_period = atr_period
        self.risk_pct = risk_pct
        self.rsi_period = rsi_period
        self.rsi_overbought = rsi_overbought
        self.rsi_oversold = rsi_oversold
        self.volume_ma = volume_ma
        self.take_profit = take_profit
        self.position = 0
        self.entry_price = 0

    def calculate_atr(self, high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return tr.rolling(window=self.atr_period).mean()

    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        # Use EMA for faster response
        data['fast_ma'] = data['close'].ewm(span=self.fast_ma, adjust=False).mean()
        data['slow_ma'] = data['close'].ewm(span=self.slow_ma, adjust=False).mean()
        
        # Calculate ATR
        data['atr'] = self.calculate_atr(data['high'], data['low'], data['close'])
        
        # Calculate RSI with Wilder's smoothing
        delta = data['close'].diff()
        gain = (delta.where(delta > 0, 0)).ewm(span=self.rsi_period, adjust=False).mean()
        loss = (-delta.where(delta < 0, 0)).ewm(span=self.rsi_period, adjust=False).mean()
        rs = gain / loss
        data['rsi'] = 100 - (100 / (1 + rs))
        
        # Enhanced volume indicators
        data['volume_ma'] = data['volume'].ewm(span=self.volume_ma, adjust=False).mean()
        data['volume_ratio'] = data['volume'] / data['volume_ma']
        
        # Trend strength and volatility
        data['trend_strength'] = abs(data['fast_ma'] - data['slow_ma']) / data['atr']
        data['volatility'] = data['atr'] / data['close'] * 100
        
        # Price momentum and acceleration
        data['momentum'] = data['close'].pct_change(periods=self.fast_ma)
        data['momentum_ma'] = data['momentum'].ewm(span=self.fast_ma, adjust=False).mean()
        
        return data

    def generate_signals(self, data: pd.DataFrame) -> Dict[str, List]:
        signals = {
            'entry_long': [],
            'entry_short': [],
            'exit_long': [],
            'exit_short': []
        }
        
        data = self.calculate_indicators(data)
        
        for i in range(self.slow_ma, len(data)):
            current_price = data['close'].iloc[i]
            current_atr = data['atr'].iloc[i]
            current_rsi = data['rsi'].iloc[i]
            volume_ratio = data['volume_ratio'].iloc[i]
            trend_strength = data['trend_strength'].iloc[i]
            volatility = data['volatility'].iloc[i]
            
            # No position
            if self.position == 0:
                # Long entry conditions
                if (data['fast_ma'].iloc[i] > data['slow_ma'].iloc[i] and 
                    data['momentum'].iloc[i] > 0 and
                    data['momentum_ma'].iloc[i] > 0 and
                    current_rsi > 40 and current_rsi < 60 and  # Middle RSI range
                    volume_ratio > 1.1 and  # Slightly above average volume
                    trend_strength > 1.2 and  # Moderate trend
                    volatility < 5):  # Low volatility for safer entry
                    
                    signals['entry_long'].append(i)
                    self.position = 1
                    self.entry_price = current_price
                    self.stop_loss = current_price - 1.5 * current_atr  # Tighter stop
                
                # Short entry conditions
                elif (data['fast_ma'].iloc[i] < data['slow_ma'].iloc[i] and 
                      data['momentum'].iloc[i] < 0 and
                      data['momentum_ma'].iloc[i] < 0 and
                      current_rsi > 40 and current_rsi < 60 and  # Middle RSI range
                      volume_ratio > 1.1 and
                      trend_strength > 1.2 and
                      volatility < 5):
                    
                    signals['entry_short'].append(i)
                    self.position = -1
                    self.entry_price = current_price
                    self.stop_loss = current_price + 1.5 * current_atr
            
            # Long position
            elif self.position == 1:
                # Exit conditions - more sensitive
                if (data['fast_ma'].iloc[i] < data['slow_ma'].iloc[i] or 
                    current_price < self.stop_loss or
                    current_rsi > self.rsi_overbought or
                    (current_price - self.entry_price) / self.entry_price > self.take_profit or
                    data['momentum'].iloc[i] < 0):  # Exit on momentum change
                    
                    signals['exit_long'].append(i)
                    self.position = 0
                # Trail the stop loss
                else:
                    new_stop = current_price - 1.5 * current_atr
                    self.stop_loss = max(self.stop_loss, new_stop)
            
            # Short position
            elif self.position == -1:
                # Exit conditions - more sensitive
                if (data['fast_ma'].iloc[i] > data['slow_ma'].iloc[i] or 
                    current_price > self.stop_loss or
                    current_rsi < self.rsi_oversold or
                    (self.entry_price - current_price) / self.entry_price > self.take_profit or
                    data['momentum'].iloc[i] > 0):  # Exit on momentum change
                    
                    signals['exit_short'].append(i)
                    self.position = 0
                # Trail the stop loss
                else:
                    new_stop = current_price + 1.5 * current_atr
                    self.stop_loss = min(self.stop_loss, new_stop)

        return signals

def backtest_strategy(data: pd.DataFrame, 
                     strategy: TrendFollowingStrategy,
                     initial_capital: float = 100,
                     position_size_pct: float = 0.95) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Backtest the strategy and return trades and performance metrics
    """
    trades = []
    position = 0
    capital = initial_capital
    equity_curve = [initial_capital]
    
    signals = strategy.generate_signals(data)
    
    for i in range(len(data)):
        if i in signals['entry_long'] and position == 0:
            entry_price = data['close'].iloc[i]
            position = 1
            trade = {
                'entry_date': data.index[i],
                'type': 'long',
                'entry_price': entry_price
            }
            
        elif i in signals['entry_short'] and position == 0:
            entry_price = data['close'].iloc[i]
            position = -1
            trade = {
                'entry_date': data.index[i],
                'type': 'short',
                'entry_price': entry_price
            }
            
        elif i in signals['exit_long'] and position > 0:
            exit_price = data['close'].iloc[i]
            pnl = position * (exit_price - entry_price)
            capital += pnl
            
            trade['exit_date'] = data.index[i]
            trade['exit_price'] = exit_price
            trade['pnl'] = pnl
            trade['return_pct'] = (pnl / trade['entry_price']) * 100
            trades.append(trade)
            position = 0
            
        elif i in signals['exit_short'] and position < 0:
            exit_price = data['close'].iloc[i]
            pnl = position * (entry_price - exit_price)
            capital += pnl
            
            trade['exit_date'] = data.index[i]
            trade['exit_price'] = exit_price
            trade['pnl'] = pnl
            trade['return_pct'] = (pnl / trade['entry_price']) * 100
            trades.append(trade)
            position = 0
            
        equity_curve.append(capital)
    
    trades_df = pd.DataFrame(trades)
    return trades_df, calculate_performance_metrics(trades_df, equity_curve)

def calculate_max_drawdown(equity_curve: pd.Series) -> float:
    """Calculate the maximum drawdown from peak to trough"""
    rolling_max = equity_curve.expanding().max()
    drawdowns = equity_curve / rolling_max - 1
    return abs(drawdowns.min())

def plot_equity_curve(equity_curve: pd.Series, trades_df: pd.DataFrame):
    """Plot equity curve with entry/exit points"""
    plt.figure(figsize=(15, 7))
    plt.plot(equity_curve.index, equity_curve.values, label='Equity Curve')
    
    # Plot long entries and exits
    long_trades = trades_df[trades_df['type'] == 'long']
    plt.scatter(long_trades['entry_date'], long_trades['entry_price'], 
               marker='^', color='g', label='Long Entry')
    plt.scatter(long_trades['exit_date'], long_trades['exit_price'], 
               marker='v', color='r', label='Long Exit')
    
    # Plot short entries and exits
    short_trades = trades_df[trades_df['type'] == 'short']
    plt.scatter(short_trades['entry_date'], short_trades['entry_price'], 
               marker='v', color='r', label='Short Entry')
    plt.scatter(short_trades['exit_date'], short_trades['exit_price'], 
               marker='^', color='g', label='Short Exit')
    
    plt.title('Strategy Equity Curve')
    plt.xlabel('Date')
    plt.ylabel('Portfolio Value')
    plt.legend()
    plt.grid(True)
    plt.show()

def format_metrics(trades_df: pd.DataFrame, performance: pd.Series, initial_capital: float):
    """Format metrics in an ASCII table format"""
    
    if trades_df.empty:
        print("\nNo trades were executed during the backtest period.")
        return
    
    try:
        # Ensure trades_df has the required columns
        required_columns = ['entry_date', 'exit_date', 'type', 'pnl', 'return_pct']
        missing_columns = [col for col in required_columns if col not in trades_df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns in trades_df: {missing_columns}")
        
        # Calculate monthly breakdown
        trades_df['month'] = pd.to_datetime(trades_df['exit_date']).dt.strftime('%Y-%m')
        monthly_stats = trades_df.groupby('month').agg({
            'pnl': 'sum',
            'type': 'count',
            'return_pct': lambda x: (x > 0).sum()  # wins
        }).reset_index()
        
        # Print monthly breakdown
        print("\n" + "=" * 20 + " MONTH BREAKDOWN " + "=" * 20)
        print("| Month      | Tot Profit USDT |  Wins  | Draws | Losses |")
        print("|------------|----------------|--------|-------|---------|")
        
        for _, row in monthly_stats.iterrows():
            total_trades = row['type']
            wins = row['return_pct']
            losses = total_trades - wins
            print(f"| {row['month']} | {row['pnl']:12.3f} | {wins:6d} | {0:5d} | {losses:7d} |")
        
        # Calculate additional metrics
        total_trades = len(trades_df)
        winning_trades = len(trades_df[trades_df['pnl'] > 0])
        losing_trades = len(trades_df[trades_df['pnl'] < 0])
        
        # Long/Short breakdown
        long_trades = trades_df[trades_df['type'] == 'long']
        short_trades = trades_df[trades_df['type'] == 'short']
        
        print("\n" + "=" * 20 + " SUMMARY METRICS " + "=" * 20)
        print("| Metric                    | Value                    |")
        print("|---------------------------|--------------------------|")
        
        metrics = {
            'Backtesting Period': f"{trades_df['entry_date'].min().strftime('%Y-%m-%d')} to {trades_df['exit_date'].max().strftime('%Y-%m-%d')}",
            'Total Trades': total_trades,
            'Win Rate': f"{(winning_trades/total_trades*100):.2f}%",
            'Total Profit': f"{trades_df['pnl'].sum():.2f} USDT",
            'Average Trade': f"{trades_df['pnl'].mean():.2f} USDT",
            'Best Trade': f"{trades_df['pnl'].max():.2f} USDT",
            'Worst Trade': f"{trades_df['pnl'].min():.2f} USDT",
            'Profit Factor': f"{abs(trades_df[trades_df['pnl'] > 0]['pnl'].sum() / trades_df[trades_df['pnl'] < 0]['pnl'].sum()):.2f}",
            'Average Win': f"{trades_df[trades_df['pnl'] > 0]['pnl'].mean():.2f} USDT",
            'Average Loss': f"{trades_df[trades_df['pnl'] < 0]['pnl'].mean():.2f} USDT",
            'Long/Short Ratio': f"{len(long_trades)}/{len(short_trades)}",
            'Long Win Rate': f"{(len(long_trades[long_trades['pnl'] > 0])/len(long_trades)*100):.2f}%" if len(long_trades) > 0 else "N/A",
            'Short Win Rate': f"{(len(short_trades[short_trades['pnl'] > 0])/len(short_trades)*100):.2f}%" if len(short_trades) > 0 else "N/A",
            'Max Drawdown': f"{performance['Max Drawdown (%)']:.2f}%",
            'Avg Trade Duration': f"{(trades_df['exit_date'] - trades_df['entry_date']).mean().total_seconds()/3600:.1f}h"
        }
        
        for metric, value in metrics.items():
            print(f"| {metric:<25} | {value:>24} |")
            
    except Exception as e:
        print(f"\nError formatting metrics: {str(e)}")
        print("Raw trades data:")
        print(trades_df.head())
        print("\nColumns:", trades_df.columns.tolist())

def calculate_max_consecutive(series):
    """Calculate maximum consecutive True values in a series"""
    max_consecutive = 0
    current_consecutive = 0
    
    for value in series:
        if value:
            current_consecutive += 1
            max_consecutive = max(max_consecutive, current_consecutive)
        else:
            current_consecutive = 0
            
    return max_consecutive

# Move evaluate_params outside of optimize_strategy
def evaluate_params(params: Dict, data: pd.DataFrame, initial_capital: float = 100) -> Dict:
    """Evaluate a single parameter set"""
    strategy = TrendFollowingStrategy(**params)
    trades, performance = backtest_strategy(
        data=data,
        strategy=strategy,
        initial_capital=initial_capital,
        position_size_pct=0.95
    )
    
    # Calculate average trade duration
    if len(trades) > 0:
        avg_duration = (trades['exit_date'] - trades['entry_date']).mean().total_seconds() / 3600
    else:
        avg_duration = float('inf')
        
    # Calculate score based on multiple metrics
    score = calculate_strategy_score(trades, performance, avg_duration)
    
    return {
        'params': params,
        'trades': len(trades),
        'win_rate': performance['Win Rate (%)'],
        'profit_factor': performance['Profit Factor'],
        'avg_duration': avg_duration,
        'total_return': performance['Total Returns (%)'],
        'max_drawdown': performance['Max Drawdown (%)'],
        'score': score
    }

def optimize_strategy(data: pd.DataFrame, 
                     param_ranges: Dict,
                     initial_capital: float = 100) -> Tuple[Dict, pd.DataFrame]:
    """
    Optimize strategy parameters using grid search
    Parameters:
        data: DataFrame with OHLCV data
        param_ranges: Dictionary of parameter ranges to test
        initial_capital: Initial capital for backtesting
    Returns:
        Tuple of (best parameters, results DataFrame)
    """
    param_combinations = [dict(zip(param_ranges.keys(), v)) 
                        for v in itertools.product(*param_ranges.values())]
    
    total_combinations = len(param_combinations)
    print(f"\nOptimization Started")
    print(f"Total parameter combinations to test: {total_combinations}")
    print("\nOptimization Progress:")
    print("=" * 80)
    
    results = []
    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(evaluate_params, params, data, initial_capital) 
                  for params in param_combinations]
        
        for i, future in enumerate(futures, 1):
            result = future.result()
            results.append(result)
            
            # Progress display
            progress = (i / total_combinations) * 100
            print(f"\rProgress: [{('=' * int(progress/2)):50}] {progress:.1f}%", end='')
            
            # Print detailed metrics every 10%
            if i % max(1, total_combinations // 10) == 0:
                print(f"\n\nInterim Results (Combination {i}/{total_combinations}):")
                print(f"Parameters: {result['params']}")
                print(f"Score: {result['score']:.3f}")
                print(f"Win Rate: {result['win_rate']:.2f}%")
                print(f"Profit Factor: {result['profit_factor']:.2f}")
                print(f"Avg Duration: {result['avg_duration']:.2f}h")
                print(f"Total Return: {result['total_return']:.2f}%")
                print(f"Max Drawdown: {result['max_drawdown']:.2f}%")
                print("-" * 80)
    
    print("\n\nOptimization Completed!")
    
    # Create results DataFrame and sort by score
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('score', ascending=False)
    
    # Display top 5 parameter combinations
    print("\nTop 5 Parameter Combinations:")
    print("=" * 80)
    for i, row in results_df.head().iterrows():
        print(f"\nRank {i+1} (Score: {row['score']:.3f}):")
        print(f"Parameters: {row['params']}")
        print(f"Win Rate: {row['win_rate']:.2f}%")
        print(f"Profit Factor: {row['profit_factor']:.2f}")
        print(f"Avg Duration: {row['avg_duration']:.2f}h")
        print(f"Total Return: {row['total_return']:.2f}%")
        print(f"Max Drawdown: {row['max_drawdown']:.2f}%")
        print("-" * 80)
    
    best_result = results_df.iloc[0]
    
    # Save detailed results
    results_df.to_csv('optimization_results_detailed.csv', index=False)
    
    return best_result['params'], results_df

def calculate_strategy_score(trades: pd.DataFrame, 
                           performance: pd.Series, 
                           avg_duration: float) -> float:
    """
    Calculate overall strategy score
    """
    if len(trades) < 10 or avg_duration > 1.5:
        return -float('inf')
    
    weights = {
        'win_rate': 0.35,
        'profit_factor': 0.20,
        'duration': 0.15,
        'total_return': 0.20,
        'drawdown': 0.10
    }
    
    win_rate_score = min(performance['Win Rate (%)'] / 100, 1.0)
    profit_factor_score = min(performance['Profit Factor'] / 3, 1.0)
    duration_score = max(0, 1 - (avg_duration / 1.5))
    return_score = min(performance['Total Returns (%)'] / 200, 1.0)
    drawdown_score = max(0, 1 - (performance['Max Drawdown (%)'] / 50))
    
    total_score = (
        weights['win_rate'] * win_rate_score +
        weights['profit_factor'] * profit_factor_score +
        weights['duration'] * duration_score +
        weights['total_return'] * return_score +
        weights['drawdown'] * drawdown_score
    )
    
    return total_score

def calculate_performance_metrics(trades_df: pd.DataFrame, equity_curve: list) -> pd.Series:
    """Calculate key performance metrics from trades and equity curve"""
    equity_series = pd.Series(equity_curve)
    
    # Calculate metrics
    total_return = ((equity_series.iloc[-1] / equity_series.iloc[0]) - 1) * 100
    max_drawdown = calculate_max_drawdown(pd.Series(equity_curve)) * 100
    
    if len(trades_df) > 0:
        win_rate = len(trades_df[trades_df['pnl'] > 0]) / len(trades_df) * 100
        profit_factor = abs(trades_df[trades_df['pnl'] > 0]['pnl'].sum() / 
                          trades_df[trades_df['pnl'] < 0]['pnl'].sum()) if len(trades_df[trades_df['pnl'] < 0]) > 0 else float('inf')
    else:
        win_rate = 0
        profit_factor = 0
    
    return pd.Series({
        'Total Returns (%)': total_return,
        'Win Rate (%)': win_rate,
        'Profit Factor': profit_factor,
        'Max Drawdown (%)': max_drawdown
    })

# Example usage:
if __name__ == "__main__":
    try:
        print("\n=== Algorithmic Trading Strategy Optimizer ===\n")
        
        # Phase 1: Data Download and Processing
        print("Phase 1: Data Download and Processing")
        print("-" * 40)
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=30)
        
        print(f"Downloading ETH-USD data...")
        print(f"Period: {start_date.date()} to {end_date.date()}")
        print(f"Interval: 1 hour")
        
        eth = yf.download(
            tickers='ETH-USD',
            start=start_date,
            end=end_date,
            interval='1h',
            progress=False
        )
        
        if eth.empty:
            raise ValueError("Failed to download data from Yahoo Finance")
        
        print("Data downloaded, processing...")
        
        # Handle multi-level columns
        if isinstance(eth.columns, pd.MultiIndex):
            eth.columns = eth.columns.get_level_values(0)
        
        # Reset index to make datetime a column
        eth = eth.reset_index()
        
        # Rename columns to lowercase
        eth.columns = [col.lower().replace(' ', '_') for col in eth.columns]
        
        # Convert numeric columns
        numeric_columns = ['open', 'high', 'low', 'close', 'adj_close', 'volume']
        for col in numeric_columns:
            if col in eth.columns:
                eth[col] = pd.to_numeric(eth[col], errors='coerce')
        
        # Set datetime index
        eth['datetime'] = pd.to_datetime(eth['datetime'])
        eth.set_index('datetime', inplace=True)
        
        # Drop any rows with NaN values
        eth = eth.dropna()
        
        # Assign processed data to data variable
        data = eth.copy()
        
        print(f"Data processing completed. Shape: {data.shape}")
        print(f"Date range: {data.index.min()} to {data.index.max()}")
        
        # Phase 2: Strategy Optimization
        print("\nPhase 2: Strategy Optimization")
        print("-" * 40)
        
        # Parameter ranges centered around optimal values
        param_ranges = {
            'fast_ma': range(9, 13, 1),         # Center around 11
            'slow_ma': range(22, 26, 1),        # Center around 24
            'atr_period': range(18, 22, 1),     # Center around 20
            'rsi_period': range(13, 15, 1),     # Center around 14
            'rsi_overbought': range(78, 82, 1), # Center around 80
            'rsi_oversold': range(28, 32, 1),   # Center around 30
            'volume_ma': range(28, 32, 1),      # Center around 30
            'take_profit': [0.015, 0.02, 0.025] # Center around 0.02
        }
        
        print("\nParameter Ranges to Test:")
        for param, range_val in param_ranges.items():
            if isinstance(range_val, range):
                print(f"{param:15}: {min(range_val)} to {max(range_val)}")
            else:
                print(f"{param:15}: {range_val}")
        
        best_params, results_df = optimize_strategy(data, param_ranges)
        
        # Phase 3: Final Backtest
        print("\nPhase 3: Final Backtest with Best Parameters")
        print("-" * 40)
        
        strategy = TrendFollowingStrategy(**best_params)
        trades, performance = backtest_strategy(
            data=data,
            strategy=strategy,
            initial_capital=100,
            position_size_pct=0.95
        )
        
        # Display results
        format_metrics(trades, performance, 100)
        
        # Create equity curve
        equity_curve = pd.Series(index=data.index, dtype=float)
        equity_curve.iloc[0] = 100
        for trade in trades.itertuples():
            if trade.type == 'long':
                equity_curve[trade.exit_date] = equity_curve[trade.entry_date] * (1 + trade.return_pct/100)
            else:
                equity_curve[trade.exit_date] = equity_curve[trade.entry_date] * (1 - trade.return_pct/100)
        equity_curve.ffill(inplace=True)
        
        # Plot results
        plot_equity_curve(equity_curve, trades)
        
        print("\nBacktest completed successfully!")
        print(f"Results saved to: optimization_results_detailed.csv")
        
    except Exception as e:
        print(f"\n❌ An error occurred: {e}")
        print("\nDetailed error trace:")
        print("-" * 40)
        import traceback
        print(traceback.format_exc())

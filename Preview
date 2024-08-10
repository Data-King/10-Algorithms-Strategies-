import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
import backtrader as bt
# 1. Moving Average Crossover
"""" Concept:
A simple strategy where a short-term moving averge crosses above or beloew a long-term
moving average to signal buy or sell opportunities
 
"""
# Sample data
data = pd.DataFrame({
    'Date': pd.date_range(start='2023-01-01', periods=100),
    'Close': pd.Series(range(100)) + pd.Series(range(100)).apply(lambda x: x*0.5).cumsum()
})
data.set_index('Date', inplace=True)

# Calculate moving averages
data['Short_MA'] = data['Close'].rolling(window=5).mean()
data['Long_MA'] = data['Close'].rolling(window=20).mean()

# Generate signals
data['Signal'] = 0
data['Signal'][data['Short_MA'] > data['Long_MA']] = 1
data['Signal'][data['Short_MA'] < data['Long_MA']] = -1

# Plotting
plt.figure(figsize=(12,6))
plt.plot(data['Close'], label='Close Price')
plt.plot(data['Short_MA'], label='Short-term MA')
plt.plot(data['Long_MA'], label='Long-term MA')
plt.title('Moving Average Crossover')
plt.legend()
plt.show()

# 2. Relative Strength Index(RSI)
""""Concept: 
RSI measures the speed and change of price movements 
and indicates overbought or oversold conditions. """

# Sample data
data = pd.DataFrame({
    'Date': pd.date_range(start='2023-01-01', periods=100),
    'Close': np.random.randn(100).cumsum() + 100
})
data.set_index('Date', inplace=True)

# Calculate RSI
delta = data['Close'].diff()
gain = delta.where(delta > 0, 0).rolling(window=14).mean()
loss = -delta.where(delta < 0, 0).rolling(window=14).mean()
rs = gain / loss
data['RSI'] = 100 - (100 / (1 + rs))

# Plotting
plt.figure(figsize=(12,6))
plt.plot(data['Close'], label='Close Price')
plt.plot(data['RSI'], label='RSI')
plt.axhline(70, color='r', linestyle='--', label='Overbought')
plt.axhline(30, color='g', linestyle='--', label='Oversold')
plt.title('RSI')
plt.legend()
plt.show()


# 3. Bollinger Bands
"""" Concept: 
Bollinger Bands use a moving average and standard deviation to identify
overbought or oversold conditions.

"""

# Sample data
data = pd.DataFrame({
    'Date': pd.date_range(start='2023-01-01', periods=100),
    'Close': pd.Series(range(100)) + pd.Series(range(100)).apply(lambda x: x*0.5).cumsum()
})
data.set_index('Date', inplace=True)

# Calculate Bollinger Bands
data['MA'] = data['Close'].rolling(window=20).mean()
data['STD'] = data['Close'].rolling(window=20).std()
data['Upper'] = data['MA'] + (data['STD'] * 2)
data['Lower'] = data['MA'] - (data['STD'] * 2)

# Plotting
plt.figure(figsize=(12,6))
plt.plot(data['Close'], label='Close Price')
plt.plot(data['MA'], label='Moving Average')
plt.plot(data['Upper'], label='Upper Band', linestyle='--')
plt.plot(data['Lower'], label='Lower Band', linestyle='--')
plt.fill_between(data.index, data['Lower'], data['Upper'], color='gray', alpha=0.1)
plt.title('Bollinger Bands')
plt.legend()
plt.show()


# 4. Momentum Trading
"""" Concept:
Momentum strategies involve buying assets that are trending
upwards and selling those that are trending downwards.
"""

# Sample data
data = pd.DataFrame({
    'Date': pd.date_range(start='2023-01-01', periods=100),
    'Close': pd.Series(range(100)) + pd.Series(range(100)).apply(lambda x: x*0.5).cumsum()
})
data.set_index('Date', inplace=True)

# Calculate momentum
data['Momentum'] = data['Close'].diff(5)

# Plotting
plt.figure(figsize=(12,6))
plt.plot(data['Close'], label='Close Price')
plt.plot(data['Momentum'], label='Momentum')
plt.title('Momentum Trading')
plt.legend()
plt.show()

# 5. Mean Reversion
"""" Concept:
Mean reversion strategies assume that prices will revert to their mean or average level.
"""
# Sample data
data = pd.DataFrame({
    'Date': pd.date_range(start='2023-01-01', periods=100),
    'Close': np.sin(np.linspace(0, 10, 100)) * 20 + 100
})
data.set_index('Date', inplace=True)

# Calculate mean and standard deviation
data['Mean'] = data['Close'].rolling(window=20).mean()
data['Std'] = data['Close'].rolling(window=20).std()
data['Upper'] = data['Mean'] + (data['Std'] * 2)
data['Lower'] = data['Mean'] - (data['Std'] * 2)

# Plotting
plt.figure(figsize=(12,6))
plt.plot(data['Close'], label='Close Price')
plt.plot(data['Mean'], label='Mean')
plt.plot(data['Upper'], label='Upper Band', linestyle='--')
plt.plot(data['Lower'], label='Lower Band', linestyle='--')
plt.title('Mean Reversion')
plt.legend()
plt.show()




# 6. Moving Average Convergence Divergence(MACD)
"""" 
Concept: 
MACD is a trend-following momentum indicator that shows the relationship 
between two moving averages of a security’s price.
"""
# Sample data
data = pd.DataFrame({
    'Date': pd.date_range(start='2023-01-01', periods=100),
    'Close': pd.Series(range(100)) + pd.Series(range(100)).apply(lambda x: x*0.5).cumsum()
})
data.set_index('Date', inplace=True)

# Calculate MACD
data['EMA_12'] = data['Close'].ewm(span=12, adjust=False).mean()
data['EMA_26'] = data['Close'].ewm(span=26, adjust=False).mean()
data['MACD'] = data['EMA_12'] - data['EMA_26']
data['Signal'] = data['MACD'].ewm(span=9, adjust=False).mean()

# Plotting
plt.figure(figsize=(12,6))
plt.plot(data['MACD'], label='MACD')
plt.plot(data['Signal'], label='Signal Line')
plt.title('MACD')
plt.legend()
plt.show()




# 7. Pair trading
"""" Concept:
Pair trading involves taking long and short positions 
in two correlated assets to profit from the relative movements between them.
"""
# Sample data for two correlated assets
np.random.seed(0)
dates = pd.date_range(start='2023-01-01', periods=100)
data = pd.DataFrame({
    'Asset1': np.random.randn(100).cumsum() + 50,
    'Asset2': np.random.randn(100).cumsum() + 50
}, index=dates)

# Calculate spread and z-score
data['Spread'] = data['Asset1'] - data['Asset2']
data['Mean'] = data['Spread'].rolling(window=20).mean()
data['Std'] = data['Spread'].rolling(window=20).std()
data['Z-Score'] = (data['Spread'] - data['Mean']) / data['Std']

# Plotting
plt.figure(figsize=(12,6))
plt.plot(data['Z-Score'], label='Z-Score')
plt.axhline(1, color='r', linestyle='--', label='Entry Signal')
plt.axhline(-1, color='g', linestyle='--', label='Exit Signal')
plt.title('Pair Trading')
plt.legend()
plt.show()






# 8. Artirage
"""Concept: 
Arbitrage strategies involve exploiting price differences 
between similar or identical financial instruments to make a profit.
"""


# Sample data for arbitrage opportunity
data = pd.DataFrame({
    'Asset1': np.random.randn(100).cumsum() + 100,
    'Asset2': np.random.randn(100).cumsum() + 100
})

# Assume a simple arbitrage opportunity if asset1 is significantly higher than asset2
data['Spread'] = data['Asset1'] - data['Asset2']
data['Arbitrage'] = np.where(data['Spread'] > data['Spread'].mean() + 2*data['Spread'].std(), 'Sell Asset1', 
                             np.where(data['Spread'] < data['Spread'].mean() - 2*data['Spread'].std(), 'Buy Asset1', 'Hold'))

# Plotting
plt.figure(figsize=(12,6))
plt.plot(data['Spread'], label='Spread')
plt.axhline(data['Spread'].mean() + 2*data['Spread'].std(), color='r', linestyle='--', label='Sell Signal')
plt.axhline(data['Spread'].mean() - 2*data['Spread'].std(), color='g', linestyle='--', label='Buy Signal')
plt.title('Arbitrage Opportunity')
plt.legend()
plt.show()




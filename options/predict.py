import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import datetime as dt

# Load your CSV file
df = pd.read_csv('INDEX_processed.csv')
df['Date'] = pd.to_datetime(df['Date'])
df.sort_values('Date', inplace=True)

# Convert date to ordinal
df['Date_Ordinal'] = df['Date'].map(dt.datetime.toordinal)

# Calculate RSI
def calculate_rsi(data, window=14):
    delta = data['Close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window).mean()
    avg_loss = loss.rolling(window).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

df['RSI'] = calculate_rsi(df)

# Add MACD and Bollinger Bands
def add_indicators(df):
    df['EMA12'] = df['Close'].ewm(span=12, adjust=False).mean()
    df['EMA26'] = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = df['EMA12'] - df['EMA26']
    df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()
    rolling_mean = df['Close'].rolling(window=20).mean()
    rolling_std = df['Close'].rolling(window=20).std()
    df['Upper_Band'] = rolling_mean + 2 * rolling_std
    df['Lower_Band'] = rolling_mean - 2 * rolling_std
    return df

df = add_indicators(df)

# Identify trends
df['Trend'] = df['Close'].diff().apply(lambda x: 'UP' if x > 0 else 'DOWN')
df['Trend_Change'] = df['Trend'] != df['Trend'].shift(1)

trend_data = []
start_idx = 0

for i in range(1, len(df)):
    if df['Trend_Change'].iloc[i]:
        end_idx = i - 1
        trend_data.append({
            'Start Date': df['Date'].iloc[start_idx],
            'End Date': df['Date'].iloc[end_idx],
            'Trend': df['Trend'].iloc[start_idx],
            'RSI Start': df['RSI'].iloc[start_idx]
        })
        start_idx = i

trend_df = pd.DataFrame(trend_data)

# Label RSI signal correctness
def rsi_signal_followed(row):
    if row['RSI Start'] < 30 and row['Trend'] == 'UP':
        return True
    elif row['RSI Start'] > 70 and row['Trend'] == 'DOWN':
        return True
    return False

trend_df['RSI Signal Followed'] = trend_df.apply(rsi_signal_followed, axis=1)

# Trade simulation
cash = 10000
trade_log = []

for i, row in trend_df.iterrows():
    signal = None
    if row['RSI Start'] < 30 and row['Trend'] == 'UP' and row['RSI Signal Followed']:
        signal = 'CALL'
    elif row['RSI Start'] > 70 and row['Trend'] == 'DOWN' and row['RSI Signal Followed']:
        signal = 'PUT'

    if signal:
        entry_price = df[df['Date'] == row['Start Date']]['Close'].values[0]
        exit_price = df[df['Date'] == row['End Date']]['Close'].values[0]
        price_change = exit_price - entry_price if signal == 'CALL' else entry_price - exit_price
        profit = price_change * 10
        cash += profit
        trade_log.append({
            'Signal': signal,
            'Start Date': row['Start Date'],
            'End Date': row['End Date'],
            'Entry': round(entry_price, 2),
            'Exit': round(exit_price, 2),
            'Profit': round(profit, 2),
            'Cash': round(cash, 2)
        })

trade_df = pd.DataFrame(trade_log)

# Predict tomorrow's signal with relaxed rules and confidence score
latest = df.iloc[-1]
score = 0

if latest['RSI'] < 35:
    score += 1
if latest['MACD'] > latest['Signal_Line']:
    score += 1
if latest['Close'] < latest['Lower_Band']:
    score += 1

if latest['RSI'] > 65:
    score -= 1
if latest['MACD'] < latest['Signal_Line']:
    score -= 1
if latest['Close'] > latest['Upper_Band']:
    score -= 1

if score >= 2:
    tomorrow_signal = 'Likely UP (CALL)'
elif score <= -2:
    tomorrow_signal = 'Likely DOWN (PUT)'
else:
    tomorrow_signal = 'Uncertain'

print(f"Tomorrow Prediction: {tomorrow_signal} (Confidence Score: {score})")

print('\nLast 5 Trades:\n', trade_df.tail())

# Plot latest day indicators for debugging
plt.figure(figsize=(14, 7))
plt.plot(df['Date'], df['Close'], label='Close Price')
plt.plot(df['Date'], df['Upper_Band'], label='Upper Bollinger Band', linestyle='--', alpha=0.5)
plt.plot(df['Date'], df['Lower_Band'], label='Lower Bollinger Band', linestyle='--', alpha=0.5)
plt.scatter(latest['Date'], latest['Close'], color='red', label='Latest Close')

plt.twinx()
plt.plot(df['Date'], df['RSI'], label='RSI', color='orange')
plt.axhline(30, color='orange', linestyle='--', alpha=0.3)
plt.axhline(70, color='orange', linestyle='--', alpha=0.3)

plt.legend(loc='upper left')
plt.title('Latest Close Price and RSI with Bollinger Bands')
plt.show()

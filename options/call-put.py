import pandas as pd
import numpy as np

# Load and prepare data
df = pd.read_csv('INDEX_processed.csv')
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values('Date').reset_index(drop=True)

# Calculate RSI (optional, can skip if not needed here)
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

# Determine direction of price movement each day
df['Direction'] = df['Close'].diff().apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0))

# Identify trend cycles (UP or DOWN)
cycles = []
start_idx = 0

for i in range(1, len(df)):
    if df.loc[i, 'Direction'] != df.loc[i - 1, 'Direction'] and df.loc[i, 'Direction'] != 0:
        direction_val = df.loc[start_idx, 'Direction']
        trend = 'UP' if direction_val == 1 else 'DOWN'
        start_date = df.loc[start_idx, 'Date']
        end_date = df.loc[i - 1, 'Date']
        duration = (end_date - start_date).days + 1
        rsi_start = df.loc[start_idx, 'RSI']
        rsi_end = df.loc[i - 1, 'RSI']

        cycles.append({
            'Trend': trend,
            'Start Date': start_date,
            'End Date': end_date,
            'Duration (days)': duration,
            'RSI Start': round(rsi_start, 2) if not np.isnan(rsi_start) else None,
            'RSI End': round(rsi_end, 2) if not np.isnan(rsi_end) else None
        })
        start_idx = i - 1

# Add last trend cycle
if start_idx < len(df) - 1:
    direction_val = df.loc[start_idx, 'Direction']
    trend = 'UP' if direction_val == 1 else 'DOWN'
    start_date = df.loc[start_idx, 'Date']
    end_date = df.loc[len(df) - 1, 'Date']
    duration = (end_date - start_date).days + 1
    rsi_start = df.loc[start_idx, 'RSI']
    rsi_end = df.loc[len(df) - 1, 'RSI']

    cycles.append({
        'Trend': trend,
        'Start Date': start_date,
        'End Date': end_date,
        'Duration (days)': duration,
        'RSI Start': round(rsi_start, 2) if not np.isnan(rsi_start) else None,
        'RSI End': round(rsi_end, 2) if not np.isnan(rsi_end) else None
    })

trend_df = pd.DataFrame(cycles)

# Function to get closing price for a given date
def get_close_price(date):
    row = df.loc[df['Date'] == date]
    if not row.empty:
        return row['Close'].values[0]
    else:
        return None

# Calculate profit/loss for each trend cycle
profits = []
for _, row in trend_df.iterrows():
    start_price = get_close_price(row['Start Date'])
    end_price = get_close_price(row['End Date'])
    if start_price is None or end_price is None:
        continue

    if row['Trend'] == 'UP':  # "Call" means buy at start, sell at end (profit = end - start)
        profit = end_price - start_price
    else:  # "Put" means profit if price falls, so profit = start - end
        profit = start_price - end_price

    profits.append({
        'Trend': row['Trend'],
        'Start Date': row['Start Date'],
        'End Date': row['End Date'],
        'Profit': profit
    })

profit_df = pd.DataFrame(profits)

# Summarize total profits by trend type
total_up_profit = profit_df[profit_df['Trend'] == 'UP']['Profit'].sum()
total_down_profit = profit_df[profit_df['Trend'] == 'DOWN']['Profit'].sum()

print(f"Total profit from CALL (UP) trends: {total_up_profit:.2f}")
print(f"Total profit from PUT (DOWN) trends: {total_down_profit:.2f}")

if total_up_profit > total_down_profit:
    print("CALL (UP) trends made more money overall.")
elif total_down_profit > total_up_profit:
    print("PUT (DOWN) trends made more money overall.")
else:
    print("Both CALL (UP) and PUT (DOWN) trends made about the same money.")

# Optional: save detailed profit info to CSV
profit_df.to_csv('trend_cycle_profits.csv', index=False)

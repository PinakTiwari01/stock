import pandas as pd
import numpy as np

# Load data
df = pd.read_csv('INDEX_processed.csv')
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values('Date').reset_index(drop=True)

# Calculate RSI (14-day)
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

# Add trend direction: 1 if up from previous day, -1 if down, 0 if no change
df['Direction'] = df['Close'].diff().apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0))

# Identify trend cycles and record RSI info
cycles = []
start_idx = 0

for i in range(1, len(df)):
    # If direction changes and current is non-zero, it's a new trend
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
            'Start Date': start_date.date(),
            'End Date': end_date.date(),
            'Duration (days)': duration,
            'RSI Start': round(rsi_start, 2) if not np.isnan(rsi_start) else None,
            'RSI End': round(rsi_end, 2) if not np.isnan(rsi_end) else None
        })
        start_idx = i - 1  # New trend starts here

# Add final trend cycle
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
        'Start Date': start_date.date(),
        'End Date': end_date.date(),
        'Duration (days)': duration,
        'RSI Start': round(rsi_start, 2) if not np.isnan(rsi_start) else None,
        'RSI End': round(rsi_end, 2) if not np.isnan(rsi_end) else None
    })

# Create DataFrame of trend cycles
trend_df = pd.DataFrame(cycles)

print("\n📈 Price Trend Cycles with RSI:")
print(trend_df)

# Calculate average durations
avg_up = trend_df[trend_df['Trend'] == 'UP']['Duration (days)'].mean()
avg_down = trend_df[trend_df['Trend'] == 'DOWN']['Duration (days)'].mean()

print(f"\nAverage UP trend duration: {avg_up:.2f} days")
print(f"Average DOWN trend duration: {avg_down:.2f} days")

# Add 'Next Start Date' to calculate gap between cycles
trend_df['Next Start Date'] = trend_df['Start Date'].shift(-1)

# Convert date columns to datetime for difference calculation
trend_df['Start Date'] = pd.to_datetime(trend_df['Start Date'])
trend_df['End Date'] = pd.to_datetime(trend_df['End Date'])
trend_df['Next Start Date'] = pd.to_datetime(trend_df['Next Start Date'])

# Calculate gap days between current trend end and next trend start
trend_df['Gap Days'] = (trend_df['Next Start Date'] - trend_df['End Date']).dt.days

# Analyze pattern of trend changes after certain durations or gaps
# For example, count how many times after X days of DOWN trend, an UP trend follows
def pattern_after_trend(trend_df, current_trend, days, next_trend):
    subset = trend_df[(trend_df['Trend'] == current_trend) & (trend_df['Duration (days)'] == days)]
    count = 0
    follow_up = 0
    for idx in subset.index:
        if idx + 1 < len(trend_df):
            count += 1
            if trend_df.loc[idx + 1, 'Trend'] == next_trend:
                follow_up += 1
    if count == 0:
        return 0
    return follow_up / count

# Example: Probability that after 4 days DOWN, next trend is UP
prob_after_4down = pattern_after_trend(trend_df, 'DOWN', 4, 'UP')
print(f"\nProbability that after 4 days DOWN trend, the next trend is UP: {prob_after_4down:.2f}")

# Predict tomorrow's trend based on last trend and these probabilities
last_trend = trend_df.iloc[-1]['Trend']
last_duration = trend_df.iloc[-1]['Duration (days)']

print(f"\nLast trend: {last_trend} lasting {last_duration} days")

# Simple heuristic: if last trend was DOWN for ~4 days, predict UP next with probability prob_after_4down
if last_trend == 'DOWN' and round(last_duration) == 4:
    prediction = 'UP (Call option likely profitable)'
elif last_trend == 'UP':
    prediction = 'DOWN (Put option likely profitable)'
else:
    prediction = 'Trend uncertain'

print(f"\nPrediction for tomorrow's trend: {prediction}")

# Optional: Save trend_df to CSV
trend_df.to_csv("trend_cycles_with_rsi_and_gaps.csv", index=False)

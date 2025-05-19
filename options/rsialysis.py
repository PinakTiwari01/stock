import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

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

# RSI thresholds
oversold_threshold = 30
overbought_threshold = 70
max_lookahead = 10

# Track time to upward/downward movement
days_to_up_after_oversold = []
days_to_down_after_overbought = []

for i in range(len(df)):
    rsi = df.loc[i, 'RSI']
    price_today = df.loc[i, 'Close']
    if np.isnan(rsi):
        continue

    # Oversold (potential call)
    if rsi < oversold_threshold:
        for j in range(1, max_lookahead + 1):
            if i + j >= len(df):
                break
            if df.loc[i + j, 'Close'] > price_today:
                days_to_up_after_oversold.append(j)
                break

    # Overbought (potential put)
    elif rsi > overbought_threshold:
        for j in range(1, max_lookahead + 1):
            if i + j >= len(df):
                break
            if df.loc[i + j, 'Close'] < price_today:
                days_to_down_after_overbought.append(j)
                break

# Averages
avg_up = np.mean(days_to_up_after_oversold) if days_to_up_after_oversold else None
avg_down = np.mean(days_to_down_after_overbought) if days_to_down_after_overbought else None

print(f"📈 Avg days to go UP after RSI < {oversold_threshold}: {avg_up:.2f}" if avg_up else "No data for RSI < oversold")
print(f"📉 Avg days to go DOWN after RSI > {overbought_threshold}: {avg_down:.2f}" if avg_down else "No data for RSI > overbought")

# RSI distribution plots
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.hist(days_to_up_after_oversold, bins=range(1, max_lookahead+2), align='left', color='green', edgecolor='black')
plt.title('Days to Price UP after RSI < 30')
plt.xlabel('Days')
plt.ylabel('Frequency')

plt.subplot(1, 2, 2)
plt.hist(days_to_down_after_overbought, bins=range(1, max_lookahead+2), align='left', color='red', edgecolor='black')
plt.title('Days to Price DOWN after RSI > 70')
plt.xlabel('Days')
plt.ylabel('Frequency')

plt.tight_layout()
plt.show()

# 🔮 Predict Tomorrow's Movement
today_rsi = df['RSI'].iloc[-1]
today_close = df['Close'].iloc[-1]
today_date = df['Date'].iloc[-1]

print(f"\n📅 Last Date: {today_date.date()}")
print(f"📊 RSI Today: {today_rsi:.2f}")

if today_rsi < oversold_threshold:
    print(f"✅ Call signal detected! RSI < {oversold_threshold}")
    print(f"🔁 Based on history, price typically goes UP in {avg_up:.2f} days")
elif today_rsi > overbought_threshold:
    print(f"⚠️ Put signal detected! RSI > {overbought_threshold}")
    print(f"🔁 Based on history, price typically goes DOWN in {avg_down:.2f} days")
else:
    print("➖ RSI is in neutral range. No strong signal for Call or Put.")


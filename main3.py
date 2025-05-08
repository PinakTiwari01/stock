import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from datetime import timedelta

# Load processed data
df = pd.read_csv('TTPW_processed.csv')
df['Date'] = pd.to_datetime(df['Date'])

# ----- MACD & Moving Averages -----
df['EMA12'] = df['Price'].ewm(span=12, adjust=False).mean()
df['EMA26'] = df['Price'].ewm(span=26, adjust=False).mean()
df['MACD'] = df['EMA12'] - df['EMA26']
df['Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()

# Moving Averages
df['SMA_5'] = df['Price'].rolling(window=5).mean()
df['SMA_10'] = df['Price'].rolling(window=10).mean()

# Drop rows with any NaNs (from rolling windows)
df.dropna(inplace=True)

# Features and target
features = ['Price', 'RSI', 'Vol.', 'MACD', 'SMA_5', 'SMA_10']
X = df[features]
y = df['Target']

# Split data (no shuffle to keep time order)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Train model
model = LogisticRegression()
model.fit(X_train, y_train)

# Accuracy on test set
accuracy = accuracy_score(y_test, model.predict(X_test)) * 100

# Get latest row to predict tomorrow
latest = df.iloc[-1]
latest_features = latest[features].values.reshape(1, -1)
prediction = model.predict(latest_features)[0]

# Calculate tomorrow's date
tomorrow = (latest['Date'] + timedelta(days=1)).strftime('%d/%m/%Y')

# Interpret prediction
movement = "📈 UP" if prediction == 1 else "📉 DOWN"

# Reasoning
rsi = latest['RSI']
vol = latest['Vol.']
macd = latest['MACD']
signal = latest['Signal']
sma5 = latest['SMA_5']
sma10 = latest['SMA_10']
price = latest['Price']

reasons = []

# RSI Reason
if rsi < 30:
    reasons.append("RSI is below 30 — the stock might be **oversold**, suggesting potential **upward reversal**.")
elif rsi > 70:
    reasons.append("RSI is above 70 — the stock might be **overbought**, indicating possible **downward correction**.")
else:
    reasons.append("RSI is in the neutral zone — no strong momentum signal.")

# Volume
if vol > df['Vol.'].rolling(10).mean().iloc[-1]:
    reasons.append("Volume is **above average**, indicating **strong interest** from traders.")
else:
    reasons.append("Volume is **below average**, suggesting **low market activity**.")

# MACD
if macd > signal:
    reasons.append("MACD is above the signal line — this is a **bullish** signal.")
else:
    reasons.append("MACD is below the signal line — this is a **bearish** signal.")

# Moving Averages
if sma5 > sma10:
    reasons.append("Short-term trend (5-day MA) is **above** long-term (10-day MA) — indicating **positive momentum**.")
else:
    reasons.append("Short-term trend (5-day MA) is **below** long-term (10-day MA) — indicating **weak momentum**.")

# Final output
print(f"📅 Prediction for: {tomorrow}")
print(f"📊 Predicted Movement: {movement}")
print(f"✅ Model Accuracy on Test Data: {accuracy:.2f}%\n")
print("🔍 Explanation:")
for r in reasons:
    print(f"• {r}")
print("\n🔍 Note: This is a simplified model. Always consider multiple factors before making trading decisions.")
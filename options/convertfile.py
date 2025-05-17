import pandas as pd

# Load the CSV file (change path/filename as needed)
df = pd.read_csv("options\SENSEX_01012010_16052025.csv")

# Step 1: Rename columns (strip whitespace)
df.columns = [col.strip() for col in df.columns]

# Step 2: Convert 'Date' to datetime and sort oldest to newest
df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
df = df.dropna(subset=['Date'])  # Remove invalid dates
df = df.sort_values('Date').reset_index(drop=True)

# Function to convert values with K, M, B suffixes to float
def convert_to_numeric(value):
    if isinstance(value, str):
        value = value.replace(',', '')  # Remove commas
        if 'K' in value:
            return float(value.replace('K', '')) * 1e3
        elif 'M' in value:
            return float(value.replace('M', '')) * 1e6
        elif 'B' in value:
            return float(value.replace('B', '')) * 1e9
    return float(value)

# Step 3: Apply conversion on numeric columns - no 'Vol.' column now
for col in ['Open', 'High', 'Low', 'Close']:
    df[col] = df[col].apply(convert_to_numeric)

# Step 4: Calculate RSI on 'Close' prices
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

# Step 5: Create Target column: 1 if next day's Close > today's Close else 0
df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)

# Step 6: Drop rows with NaN from RSI or shift
df = df.dropna().reset_index(drop=True)

# Step 7: Save the processed dataframe
df.to_csv("INDEX_processed.csv", index=False)
print("✅ Data processing complete. Saved as INDEX_processed.csv")

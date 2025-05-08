# yyyy/mm/dd
# 2023/10/01
import pandas as pd

# Load the CSV file (change this name if needed)
df = pd.read_csv("download.csv")

# Step 1: Rename columns if needed (standardize)
df.columns = [col.strip() for col in df.columns]

# Step 2: Convert 'Date' to datetime and sort oldest to newest
df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
df = df.dropna(subset=['Date'])  # Remove rows with invalid dates
df = df.sort_values('Date')

# Function to handle conversion of numbers with suffixes (K, M, etc.)
def convert_to_numeric(value):
    if isinstance(value, str):
        value = value.replace(',', '')  # Remove commas
        if 'K' in value:
            return float(value.replace('K', '')) * 1e3  # Convert 'K' to thousands
        elif 'M' in value:
            return float(value.replace('M', '')) * 1e6  # Convert 'M' to millions
        elif 'B' in value:
            return float(value.replace('B', '')) * 1e9  # Convert 'B' to billions
    return float(value)

# Step 3: Apply the conversion to relevant columns
for col in ['Open', 'High', 'Low', 'Price', 'Vol.']:
    df[col] = df[col].apply(convert_to_numeric)

# Step 4: Calculate RSI
def calculate_rsi(data, window=14):
    delta = data['Price'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window).mean()
    avg_loss = loss.rolling(window).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

df['RSI'] = calculate_rsi(df)

# Step 5: Calculate price movement direction (Target column)
df['Target'] = (df['Price'].shift(-1) > df['Price']).astype(int)

# Step 6: Drop rows with NaN (from RSI calculation or shift)
df = df.dropna()

# Step 7: Save processed data
df.to_csv("TTPW_processed.csv", index=False)
print("✅ Data processing complete. Saved as TTPW_processed.csv")

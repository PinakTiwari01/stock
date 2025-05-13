import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Load the CSV data
df = pd.read_csv('TTPW_processed.csv')

# Data Preprocessing
# Convert columns to string type and remove commas, then convert to numeric
df['Vol.'] = df['Vol.'].astype(str).str.replace(',', '').astype(float)
df['Price'] = df['Price'].astype(str).str.replace(',', '').astype(float)
df['Open'] = df['Open'].astype(str).str.replace(',', '').astype(float)
df['High'] = df['High'].astype(str).str.replace(',', '').astype(float)
df['Low'] = df['Low'].astype(str).str.replace(',', '').astype(float)

# Handle NaN values (choose your preferred approach)
df.dropna(inplace=True)  # Drops rows with NaN values
# Alternatively, you can use:
# df.fillna(df.mean(), inplace=True)  # Fill NaN with mean

# Calculate RSI (Relative Strength Index)
def calculate_rsi(data, period=14):
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period, min_periods=1).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period, min_periods=1).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

# Apply the RSI function
df['RSI'] = calculate_rsi(df['Price'], period=14)

# Define the target column - stock movement (up or down)
df['Target'] = (df['Price'].shift(-1) > df['Price']).astype(int)

# Feature Engineering (use relevant columns for prediction)
X = df[['Price', 'RSI', 'Vol.']]  # Features: Price, RSI, and Volume
y = df['Target']  # Target: Price movement (1 for up, 0 for down)

# Handle missing values in X data (if any NaN values exist in the features)
X.dropna(inplace=True)  # This will drop any row in X that has NaN values
y = y[X.index]  # Ensure that y has the same index as X after dropping rows

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Initialize and train the Logistic Regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy * 100:.2f}%')

# Confusion Matrix and Classification Report
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("Classification Report:")
print(classification_report(y_test, y_pred))

# Visualization: Plot Price and RSI
plt.figure(figsize=(12, 6))

# Plot Price
plt.subplot(2, 1, 1)
plt.plot(df['Date'], df['Price'], label='Price')
plt.title('Stock Price')
plt.legend()

# Plot RSI
plt.subplot(2, 1, 2)
plt.plot(df['Date'], df['RSI'], label='RSI', color='orange')
plt.title('RSI')
plt.legend()

plt.tight_layout()
plt.show()

# Save the processed data to a new CSV file
df.to_csv('TTPW_processed_with_rsi.csv', index=False)
print("✅ Data processing complete. Saved as TTPW_processed_with_rsi.csv")

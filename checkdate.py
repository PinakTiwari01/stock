import pandas as pd

# Load the CSV file
df = pd.read_csv("download.csv")

# Clean and print column names
df.columns = [col.strip() for col in df.columns]
print("🧾 Available Columns:", df.columns.tolist())

# Now try accessing the Date column
if 'Date' in df.columns:
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    print("✅ Date column parsed successfully.")
    print("📅 Start Date:", df['Date'].min().strftime('%d %b %Y'))
    print("📅 End Date:", df['Date'].max().strftime('%d %b %Y'))
else:
    print("❌ 'Date' column not found. Check the column headers in your CSV.")

# =========================
# 1. Install
# =========================
!pip install seaborn --quiet

# =========================
# 2. Import
# =========================
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# =========================
# 3. Load Data
# =========================
sentiment = pd.read_csv('/content/fear_greed_index.csv')
trades = pd.read_csv('/content/historical_data.csv')

# =========================
# 4. Clean Columns
# =========================
sentiment.columns = sentiment.columns.str.strip().str.lower()
trades.columns = trades.columns.str.strip().str.lower()

print("Sentiment Columns:", sentiment.columns)
print("Trades Columns:", trades.columns)

# =========================
# 5. Fix Column Names (IMPORTANT)
# =========================
trades.rename(columns={
    'execution price': 'price',
    'size tokens': 'size_tokens',
    'size usd': 'size_usd',
    'closed pnl': 'closed_pnl',
    'timestamp ist': 'time'
}, inplace=True)

# =========================
# 6. Convert Dates Properly
# =========================
sentiment['date'] = pd.to_datetime(sentiment['date'], errors='coerce')
trades['time'] = pd.to_datetime(trades['time'], errors='coerce')

sentiment = sentiment.dropna(subset=['date'])
trades = trades.dropna(subset=['time'])

# =========================
# 7. Merge (FIXED PROPERLY)
# =========================
sentiment = sentiment.sort_values('date')
trades = trades.sort_values('time')

merged = pd.merge_asof(
    trades,
    sentiment[['date', 'classification']],
    left_on='time',
    right_on='date',
    direction='nearest'
)

# Remove missing sentiment
merged = merged.dropna(subset=['classification'])

print("Merged Shape:", merged.shape)

# =========================
# 8. Convert Numeric Columns
# =========================
merged['closed_pnl'] = pd.to_numeric(merged['closed_pnl'], errors='coerce')
merged['size_usd'] = pd.to_numeric(merged['size_usd'], errors='coerce')

merged = merged.dropna(subset=['closed_pnl'])

# =========================
# 9. Feature Engineering
# =========================
merged['win'] = merged['closed_pnl'] > 0

# =========================
# 10. Analysis
# =========================
print("\n===== SENTIMENT COUNT =====")
print(merged['classification'].value_counts())

print("\n===== AVG PnL =====")
print(merged.groupby('classification')['closed_pnl'].mean())

print("\n===== WIN RATE =====")
print(merged.groupby('classification')['win'].mean())

print("\n===== TRADE SIZE =====")
print(merged.groupby('classification')['size_usd'].mean())

print("\n===== SIDE vs SENTIMENT =====")
print(pd.crosstab(merged['classification'], merged['side']))

# =========================
# 11. Visualization
# =========================
plt.figure()
sns.boxplot(x='classification', y='closed_pnl', data=merged)
plt.title("PnL Distribution by Sentiment")
plt.show()

plt.figure()
merged.groupby('classification')['closed_pnl'].mean().plot(kind='bar')
plt.title("Average PnL by Sentiment")
plt.show()

plt.figure()
merged.groupby('classification')['win'].mean().plot(kind='bar')
plt.title("Win Rate by Sentiment")
plt.show()

# =========================
# 12. Top Traders
# =========================
print("\n===== TOP TRADERS =====")
print(merged.groupby('account')['closed_pnl'].sum().sort_values(ascending=False).head(10))

# =========================
# 13. Save Output
# =========================
merged.to_csv("final_output.csv", index=False)

print("\n✅ SUCCESS: Code ran perfectly!")

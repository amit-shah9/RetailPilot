import pandas as pd
import os

# 📁 Set base directories
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR = os.path.join(BASE_DIR, 'data', 'raw')
PROCESSED_DIR = os.path.join(BASE_DIR, 'data', 'processed')

# 📁 Ensure processed data folder exists
os.makedirs(PROCESSED_DIR, exist_ok=True)

# 📥 Load raw CSVs
print("🔄 Loading raw data from:", RAW_DIR)
train = pd.read_csv(os.path.join(RAW_DIR, 'train.csv'))
oil = pd.read_csv(os.path.join(RAW_DIR, 'oil.csv'))
holidays = pd.read_csv(os.path.join(RAW_DIR, 'holidays_events.csv'))
stores = pd.read_csv(os.path.join(RAW_DIR, 'stores.csv'))

# 🧪 Parse dates
train['date'] = pd.to_datetime(train['date'])
oil['date'] = pd.to_datetime(oil['date'])
holidays['date'] = pd.to_datetime(holidays['date'])

# ✂️ Filter last year of data (adjust to control size)
train = train[train['date'] >= '2015-07-01']

# 🔗 Merge oil prices
data = pd.merge(train, oil, on='date', how='left')

# 🔗 Merge holidays (only national)
holidays = holidays[holidays['locale'] == 'National']
holidays = holidays[['date', 'type']].rename(columns={'type': 'holiday_type'})
data = pd.merge(data, holidays, on='date', how='left')

# 🧹 Fill missing values
data['dcoilwtico'] = data['dcoilwtico'].ffill()
data['holiday_type'] = data['holiday_type'].fillna('None')

# 💾 Save the trimmed dataset
output_path = os.path.join(PROCESSED_DIR, 'retail_processed.csv')
data.to_csv(output_path, index=False)

print("✅ ETL complete. Processed file saved at:", output_path)

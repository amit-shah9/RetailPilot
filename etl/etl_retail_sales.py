import pandas as pd
import os

# Get absolute paths for raw and processed data folders
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR = os.path.join(BASE_DIR, 'data', 'raw')
PROCESSED_DIR = os.path.join(BASE_DIR, 'data', 'processed')

# Ensure output directory exists
os.makedirs(PROCESSED_DIR, exist_ok=True)

# Load CSVs
print("Loading data from:", RAW_DIR)
train = pd.read_csv(os.path.join(RAW_DIR, 'train.csv'))
oil = pd.read_csv(os.path.join(RAW_DIR, 'oil.csv'))
holidays = pd.read_csv(os.path.join(RAW_DIR, 'holidays_events.csv'))
stores = pd.read_csv(os.path.join(RAW_DIR, 'stores.csv'))

# Merge oil prices
train['date'] = pd.to_datetime(train['date'])
oil['date'] = pd.to_datetime(oil['date'])
data = pd.merge(train, oil, on='date', how='left')

# Merge holidays
holidays['date'] = pd.to_datetime(holidays['date'])
holidays = holidays[holidays['locale'] == 'National']
holidays = holidays[['date', 'type']]
holidays = holidays.rename(columns={'type': 'holiday_type'})
data = pd.merge(data, holidays, on='date', how='left')

# Fill missing values
data['dcoilwtico'] = data['dcoilwtico'].ffill()
data['holiday_type'] = data['holiday_type'].fillna('None')

# Save processed data
output_path = os.path.join(PROCESSED_DIR, 'retail_processed.csv')
data.to_csv(output_path, index=False)
print("✅ ETL complete. Processed file saved at:", output_path)

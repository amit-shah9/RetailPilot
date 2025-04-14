import os
import gdown
import pandas as pd

# Define base and raw data directories
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR = os.path.join(BASE_DIR, 'data', 'raw')
PROCESSED_DIR = os.path.join(BASE_DIR, 'data', 'processed')
os.makedirs(RAW_DIR, exist_ok=True)

# Path where the test file should be saved
test_csv_path = os.path.join(RAW_DIR, 'test.csv')

# Google Drive file ID and direct download URL
file_id = '1czC5vOsE7YpFRBN7HdzvNRuQ6KPhLXwl'
download_url = f'https://drive.google.com/uc?id={file_id}'

# Check if file exists; download if missing
if not os.path.exists(test_csv_path):
    print("test.csv not found locally. Downloading from Google Drive...")
    try:
        gdown.download(download_url, test_csv_path, quiet=False)
        print("Download complete.")
    except Exception as e:
        print(f"Download failed: {e}")
else:
    print("test.csv already exists. Skipping download.")

# Load datasets
print("Loading raw data...")
train = pd.read_csv(os.path.join(RAW_DIR, 'train.csv'), parse_dates=['date'])
oil = pd.read_csv(os.path.join(RAW_DIR, 'oil.csv'), parse_dates=['date'])
holidays = pd.read_csv(os.path.join(RAW_DIR, 'holidays_events.csv'), parse_dates=['date'])
stores = pd.read_csv(os.path.join(RAW_DIR, 'stores.csv'))

# Merge datasets
data = pd.merge(train, oil, on='date', how='left')
holidays = holidays[holidays['locale'] == 'National']
holidays = holidays[['date', 'type']].rename(columns={'type': 'holiday_type'})
data = pd.merge(data, holidays, on='date', how='left')

# Handle missing values
data['dcoilwtico'] = data['dcoilwtico'].ffill()
data['holiday_type'] = data['holiday_type'].fillna('None')

# Add date-based features
data['dayofweek'] = data['date'].dt.dayofweek
data['month'] = data['date'].dt.month
data['year'] = data['date'].dt.year

# Sort to prepare for lag/rolling features
data.sort_values(['store_nbr', 'family', 'date'], inplace=True)

# Add lag and rolling features
data['lag_7'] = data.groupby(['store_nbr', 'family'])['sales'].shift(7)
data['lag_14'] = data.groupby(['store_nbr', 'family'])['sales'].shift(14)
data['rolling_mean_7'] = (
    data.groupby(['store_nbr', 'family'])['sales']
    .shift(1)
    .rolling(7)
    .mean()
    .reset_index(0, drop=True)
)
data['promo_last_week'] = data.groupby(['store_nbr', 'family'])['onpromotion'].shift(7)

# Filter to exactly 15 top stores and 25 top families
top_stores = (
    data.groupby('store_nbr')
    .size()
    .sort_values(ascending=False)
    .head(15)
    .index
)

top_families = (
    data.groupby('family')
    .size()
    .sort_values(ascending=False)
    .head(25)
    .index
)

data = data[
    (data['store_nbr'].isin(top_stores)) &
    (data['family'].isin(top_families))
]

# Filter by date
data = data[data['date'] >= '2013-01-01']

# Drop rows with missing values from lag/rolling features
data.dropna(inplace=True)

# Optional: display how much data is retained
combo_count = data.groupby(['store_nbr', 'family']).size().shape[0]
print(f"Filtered to {data['store_nbr'].nunique()} stores, {data['family'].nunique()} families.")
print(f"Total combinations used: {combo_count}")

# Save the processed dataset
output_path = os.path.join(PROCESSED_DIR, 'retail_processed.csv')
data.to_csv(output_path, index=False)
print(f"ETL complete. File saved at: {output_path}")

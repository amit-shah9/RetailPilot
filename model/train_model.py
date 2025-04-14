import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from math import sqrt
import joblib
import os

# Set paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, 'data', 'processed', 'retail_processed.csv')
MODEL_PATH = os.path.join(BASE_DIR, 'model', 'sales_model.pkl')

# Load the cleaned dataset
data = pd.read_csv(DATA_PATH, low_memory=False)
print("Columns in dataset:", data.columns.tolist())

# Convert date column to datetime
data['date'] = pd.to_datetime(data['date'])

# Extract date-based features
data['dayofweek'] = data['date'].dt.dayofweek
data['month'] = data['date'].dt.month
data['year'] = data['date'].dt.year

# Fill missing oil prices using forward fill
data['dcoilwtico'] = data['dcoilwtico'].ffill()

# Clean and convert 'onpromotion'
data['onpromotion'] = data['onpromotion'].fillna(0).astype(int)

# Encode categorical columns
data['holiday_type'] = data['holiday_type'].astype('category').cat.codes
data['family'] = data['family'].astype('category').cat.codes

# Define input features and target column
features = ['store_nbr', 'family', 'dcoilwtico', 'onpromotion', 'dayofweek', 'month', 'year', 'holiday_type']
target = 'sales'

# Drop rows with missing values in feature columns (just in case)
data = data.dropna(subset=features)

# Split features and target
X = data[features]
y = data[target]

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
rmse = sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("\nModel trained successfully.")
print(f"RMSE: {rmse:.2f}")
print(f"R² Score: {r2:.2f}")

# Save the trained model
joblib.dump(model, MODEL_PATH)
print(f"Model saved to: {MODEL_PATH}")

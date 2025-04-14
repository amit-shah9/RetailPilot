import pandas as pd
import numpy as np
import os
from catboost import CatBoostRegressor, Pool
from sklearn.metrics import mean_squared_log_error
import joblib

# 📁 Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, 'data', 'processed', 'retail_processed.csv')
MODEL_PATH = os.path.join(BASE_DIR, 'model', 'catboost_model.pkl')

# 📥 Load data
print("Loading processed data...")
df = pd.read_csv(DATA_PATH)

# 🔍 Feature engineering
df['date'] = pd.to_datetime(df['date'])
df['dayofweek'] = df['date'].dt.dayofweek
df['month'] = df['date'].dt.month
df['year'] = df['date'].dt.year

# 🧼 Fill NAs
df['dcoilwtico'].fillna(method='ffill', inplace=True)

# 🎯 Target
y = df['sales']

# 🔣 Categorical features
categorical = ['store_nbr', 'family', 'holiday_type']
df[categorical] = df[categorical].astype(str)

# 🧪 Features
features = ['store_nbr', 'family', 'onpromotion', 'dcoilwtico',
            'holiday_type', 'dayofweek', 'month', 'year']

X = df[features]

# 🪄 CatBoost handles categorical internally
cat_features_idx = [features.index(col) for col in categorical]

# 🔁 Train-test split
from sklearn.model_selection import train_test_split
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, shuffle=True, random_state=42)

# 🧠 CatBoostRegressor
model = CatBoostRegressor(
    iterations=1000,
    learning_rate=0.05,
    depth=6,
    cat_features=cat_features_idx,
    loss_function='RMSE',
    verbose=100
)

# 🏋️‍♂️ Train
print("Training CatBoost model...")
model.fit(X_train, y_train, eval_set=(X_valid, y_valid), use_best_model=True)

# 🧪 Evaluate
y_pred = model.predict(X_valid)
rmsle = np.sqrt(mean_squared_log_error(y_valid, np.maximum(y_pred, 0)))

print(f"✅ RMSLE: {rmsle:.4f}")

# 💾 Save model
joblib.dump(model, MODEL_PATH)
print("📦 Model saved to:", MODEL_PATH)
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# 🖼 Page config MUST be first Streamlit call
st.set_page_config(page_title="Retail Sales Forecast", layout="wide")

# 📁 Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, 'data', 'processed', 'retail_processed.csv')
MODEL_PATH = os.path.join(BASE_DIR, 'model', 'catboost_model.pkl')

# 🚀 Load model + data
@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)

model = load_model()
data = load_data()

# 🌟 Title + intro
st.title("🛍️ Retail Sales Forecasting App")
st.markdown("""
Welcome!  
This app uses a machine learning model trained on real retail data to **forecast daily sales** for a given product and store.  
We take into account past sales, promotions, oil prices, and seasonal trends to help guide predictions.
""")

# 📊 Extract dropdown values
stores = sorted(data['store_nbr'].unique())
families = sorted(data['family'].unique())
years = sorted(data['year'].unique())
months = sorted(data['month'].unique())

# 📥 User inputs
with st.sidebar:
    st.header("🔧 Prediction Settings")
    store = st.selectbox("Store Number", stores)
    family = st.selectbox("Product Family", families)
    year = st.selectbox("Year", years)
    month = st.selectbox("Month", months)
    dayofweek = st.slider("Day of Week (0 = Monday)", 0, 6)
    onpromotion = st.checkbox("Is this product on promotion?", value=False)

    avg_oil = float(data['dcoilwtico'].mean())
    min_oil = float(data['dcoilwtico'].min())
    max_oil = float(data['dcoilwtico'].max())

    dcoilwtico = st.slider(
        "🛢️ Oil Price (WTI - USD/barrel)",
        min_value=round(min_oil, 2),
        max_value=round(max_oil, 2),
        value=round(avg_oil, 2),
        help="Oil price influences economy-wide demand."
    )

    # 👇 Move button here
    predict = st.button("📈 Predict Sales")


# 📘 Info note
st.markdown("🔍 Based on your selections, we pull historical sales trends to improve the prediction.")

# 🧠 Estimate lag/rolling features from similar past records
filtered = data[
    (data['store_nbr'] == store) &
    (data['family'] == family) &
    (data['year'] == year) &
    (data['month'] == month) &
    (data['dayofweek'] == dayofweek)
]

if not filtered.empty:
    lag_7 = float(filtered['lag_7'].mean())
    lag_14 = float(filtered['lag_14'].mean())
    rolling_mean_7 = float(filtered['rolling_mean_7'].mean())
    promo_last_week = int(filtered['promo_last_week'].mean())
    st.success("✅ Historical patterns loaded successfully.")
else:
    recent = data[(data['store_nbr'] == store) & (data['family'] == family)].sort_values('date', ascending=False)

    if not recent.empty:
        st.warning("⚠️ No exact historical match. Using most recent sales info for this store/product.")
        lag_7 = float(recent.iloc[0]['lag_7'])
        lag_14 = float(recent.iloc[0]['lag_14'])
        rolling_mean_7 = float(recent.iloc[0]['rolling_mean_7'])
        promo_last_week = int(recent.iloc[0]['promo_last_week'])
    else:
        st.warning("⚠️ No historical data at all for this store/family. Using global averages.")
        lag_7 = lag_14 = rolling_mean_7 = data['sales'].mean()
        promo_last_week = 0


# 📦 Build input DataFrame for model
input_df = pd.DataFrame([{
    'store_nbr': str(store),
    'family': str(family),
    'onpromotion': int(onpromotion),
    'dcoilwtico': dcoilwtico,
    'holiday_type': 'None',
    'dayofweek': dayofweek,
    'month': month,
    'year': year,
    'lag_7': lag_7,
    'lag_14': lag_14,
    'rolling_mean_7': rolling_mean_7,
    'promo_last_week': promo_last_week
}])

# 🧠 Prediction logic
if predict:
    prediction = model.predict(input_df)[0]
    prediction = int(round(max(prediction, 0)))  # ensure non-negative

    st.subheader("🔮 Predicted Sales")
    st.success(f"🛒 Expected daily sales: **{prediction} units**")
    st.caption("Prediction is for **one day**, based on selected date, past trends, and promotion settings.")

    with st.expander("📚 Learn how this was predicted"):
        st.markdown("""
        This ML model uses:
        - Recent sales (lags)
        - Weekly moving averages
        - Promotion trends (current + past)
        - Oil prices as an economic proxy
        - Seasonality via month/year/day features
        """)

# 📈 Show historical sales trend
st.markdown("### 📉 Past Sales Trends")
history = data[(data['store_nbr'] == store) & (data['family'] == family)]
if history.empty:
    st.info("No past data available for this selection.")
else:
    chart = history[['date', 'sales']].copy()
    chart['date'] = pd.to_datetime(chart['date'])
    chart.set_index('date', inplace=True)
    st.line_chart(chart['sales'])
    st.caption("Historical sales for the selected store and product family.")

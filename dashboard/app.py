import streamlit as st
import pandas as pd
import joblib
import os
from datetime import datetime

# -------------------------------
# 💅 Page layout and config
# -------------------------------
st.set_page_config(page_title="Retail Sales Predictor", layout="wide")
st.markdown("<h1 style='color:#00B8E4'>🛍️ Retail Sales Predictor</h1>", unsafe_allow_html=True)
st.caption("Use this app to predict **daily sales** for a specific store and product family using past trends, promotions, and seasonality.")

# -------------------------------
# 📦 Load model and data
# -------------------------------
model_path = os.path.join("model", "sales_model.pkl")
data_path = os.path.join("data", "processed", "retail_processed.csv")

model = joblib.load(model_path)
data = pd.read_csv(data_path)

# -------------------------------
# 🧼 Clean and prepare data
# -------------------------------
data['date'] = pd.to_datetime(data['date'])
data['family'] = data['family'].astype('category')
data['holiday_type'] = data['holiday_type'].astype('category')

# Create encoders
family_map = {cat: code for code, cat in enumerate(data['family'].cat.categories)}
holiday_map = {cat: code for code, cat in enumerate(data['holiday_type'].cat.categories)}
family_rev = {v: k for k, v in family_map.items()}

# Extract time features
data['dayofweek'] = data['date'].dt.dayofweek
data['month'] = data['date'].dt.month
data['year'] = data['date'].dt.year

# -------------------------------
# 🧠 Sidebar filters
# -------------------------------
st.sidebar.header("📊 Filters")

# Year filter
year_filter = st.sidebar.multiselect(
    "Filter by Year", options=sorted(data['year'].unique()), default=sorted(data['year'].unique())
)
data = data[data['year'].isin(year_filter)]

# Month filter
month_filter = st.sidebar.multiselect(
    "Filter by Month", options=list(range(1, 13)), default=list(range(1, 13))
)
data = data[data['month'].isin(month_filter)]

# -------------------------------
# 🧾 User input section
# -------------------------------
st.sidebar.header("🧾 Prediction Inputs")

# Store selector
store = st.sidebar.selectbox("Select Store", sorted(data['store_nbr'].unique()))

# Filter available families for selected store
available_families = data[data['store_nbr'] == store]['family'].unique()
available_family_names = [cat for cat in data['family'].cat.categories if cat in available_families]

if not available_family_names:
    st.sidebar.warning("⚠️ No product families available for this store.")
    st.stop()

# Product family selector
family_label = st.sidebar.selectbox("Select Product Family", sorted(available_family_names))

# Promotion
on_promo = st.sidebar.checkbox("On Promotion?", value=False)

# Prediction date
date = st.sidebar.date_input("Prediction Date", value=datetime(2016, 8, 1))

# -------------------------------
# 🧮 Prepare input data for model
# -------------------------------
dayofweek = date.weekday()
month = date.month
year = date.year
oil_price = data['dcoilwtico'].ffill().iloc[-1]

input_data = pd.DataFrame([{
    'store_nbr': store,
    'family': family_map[family_label],
    'dcoilwtico': oil_price,
    'onpromotion': int(on_promo),
    'dayofweek': dayofweek,
    'month': month,
    'year': year,
    'holiday_type': holiday_map.get('None', 0)
}])

# -------------------------------
# 📈 Prediction section
# -------------------------------
st.markdown("### 🔮 Predicted Sales (for a single day)")
st.info("This number represents the **expected sales (in units)** for **one specific day**, based on the selected store, product family, date, and promotion status.")

# Show prediction
prediction = model.predict(input_data)[0]
col1, col2 = st.columns(2)
col1.metric("Predicted Units", f"{prediction:.2f}")
col2.markdown(f"""
**Selected Inputs:**  
- 🏬 Store: `{store}`  
- 🏷️ Product Family: `{family_label}`  
- 📅 Date: `{date.strftime('%Y-%m-%d')}`  
- 🔖 Promotion: `{on_promo}`  
""")

# -------------------------------
# 📊 Historical sales trend
# -------------------------------
st.markdown("---")
st.markdown("### 📊 Historical Sales Trend")
st.caption("Below is the **daily sales history** for the selected store and product family from the available data.")

trend_data = data[(data['store_nbr'] == store) & (data['family'] == family_label)]

if trend_data.empty:
    st.warning("⚠️ No historical sales data found for this combination.")
else:
    trend_summary = trend_data.groupby('date')['sales'].sum().reset_index()
    st.line_chart(trend_summary.set_index('date'))
    st.caption(f"Showing {len(trend_data)} days of historical records.")

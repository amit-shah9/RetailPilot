# 🚀 Retail Sales Forecasting App

A modern web app that predicts **daily retail sales** using real-world data, advanced feature engineering, and a powerful machine learning model.

> 📈 Built to forecast. Designed to impress.

---

### 🌟 Why This Project?

Retail demand forecasting is at the core of inventory planning, supply chain optimization, and business strategy. This app solves it — using a real Kaggle dataset and best practices in **data science**, **MLOps**, and **UX**.

If you're a hiring manager, engineer, or data enthusiast — **explore the live dashboard and see it in action**.

---

### 🔗 [👉 Try the Live App](https://retailpilot.onrender.com)  
> ⚠️ First load may take 30–60s (free Render tier). Worth the wait.

---

### 🧠 What It Does

You select:

- ✅ A **store**
- ✅ A **product family**
- ✅ A **date**, promotion status, oil price, etc.

And it instantly predicts:

> 🛍️ **How many units will sell that day?**

Plus, it shows:

- 📉 Historical sales trends  
- 🧠 Model insights  
- 🎯 Forecast explanation

---

### ⚙️ Tech Stack

| Layer            | Tools / Frameworks                            |
|------------------|-----------------------------------------------|
| **Frontend**     | Streamlit                                     |
| **Model**        | CatBoostRegressor (w/ lag, trend, rolling)    |
| **ETL**          | pandas, numpy, custom lag pipelines           |
| **Deployment**   | Docker, Render, CI/CD via GitHub              |
| **Extras**       | Model fallback logic, oil slider, UX polish   |

---

### 📊 Powered by Real Data

- **Source:** [Kaggle - Corporación Favorita Grocery Sales Forecasting](https://www.kaggle.com/competitions/store-sales-time-series-forecasting/)
- 125M+ rows of daily sales, oil prices, holidays, promotions
- Smart sampling + feature engineering to fit real-world deployment

---

### ✨ Features

- 🔮 Forecasts sales using:
  - Store, product, promotions, oil prices, holidays
  - Rolling averages & lagged sales
- 📉 Visualizes historical patterns
- 🧠 Uses fallback logic if exact match isn’t found
- 🐳 Fully Dockerized for reproducibility
- 🔄 Auto-deploys with every push

---

### 👀 See It in Action

| Trends |
|--------|
| ![Sales Trend](https://github.com/user-attachments/assets/3f2b61b0-dd90-4eb7-b265-76831e889967) |

---

### 👨‍💻 Author

**Amit Shah**  
Data Science & MLOps Enthusiast  
🔗 [Connect on LinkedIn](https://www.linkedin.com/in/amit-shah-296099237/)  
💼 Open to ML Engineering / AI Product Roles

---

### ⭐ Don’t Just Scroll — Try It

This isn’t just another notebook. It’s a full-stack, production-ready machine learning app.

👉 [Click here to try it](https://retailpilot.onrender.com)

---

### 📦 Run Locally

```bash
git clone https://github.com/amit-shah9/RetailPilot.git
cd RetailPilot
docker build -t retail-pilot .
docker run -p 8501:8501 retail-pilot




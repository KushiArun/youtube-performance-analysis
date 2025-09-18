# ==========================================
# Unlocking YouTube Channel Performance Secrets
# Streamlit App (app.py)
# ==========================================

import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# -----------------------------
# Load Dataset
# -----------------------------
@st.cache_data
def load_data():
    data = pd.read_csv("youtube_channel_real_performance_analytics.csv")
    # Convert publish time if exists
    if "Video Publish Time" in data.columns:
        data["Video Publish Time"] = pd.to_datetime(data["Video Publish Time"])
    # Drop missing
    data = data.dropna()
    return data

data = load_data()
st.title("📊 YouTube Channel Performance Analysis")
st.write("Dataset Loaded Successfully ✅")
st.write(data.head())

# -----------------------------
# Visualizations
# -----------------------------
st.subheader("Revenue Distribution")
fig1, ax1 = plt.subplots()
sns.histplot(data["Estimated Revenue (USD)"], bins=40, kde=True, ax=ax1, color="green")
st.pyplot(fig1)

st.subheader("Revenue vs Views")
fig2, ax2 = plt.subplots()
sns.scatterplot(x=data["Views"], y=data["Estimated Revenue (USD)"], alpha=0.7, ax=ax2)
st.pyplot(fig2)

st.subheader("Correlation Heatmap")
fig3, ax3 = plt.subplots(figsize=(10, 6))
sns.heatmap(data.select_dtypes(include=[np.number]).corr(), cmap="coolwarm", ax=ax3)
st.pyplot(fig3)

# -----------------------------
# Train Model
# -----------------------------
st.subheader("Train Revenue Prediction Model")

possible_features = ['Views', 'Subscribers', 'Likes', 'Shares']
features = [f for f in possible_features if f in data.columns]
target = 'Estimated Revenue (USD)'

X = data[features]
y = data[target]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

st.write(f"📈 **Model Evaluation:** MSE={mse:.2f}, R²={r2:.2f}")

# -----------------------------
# Feature Importance
# -----------------------------
importances = model.feature_importances_
feature_df = pd.DataFrame({"Feature": features, "Importance": importances})
fig4, ax4 = plt.subplots()
sns.barplot(x="Importance", y="Feature", data=feature_df.sort_values("Importance", ascending=False), ax=ax4)
st.subheader("Feature Importance")
st.pyplot(fig4)

# -----------------------------
# Prediction Form
# -----------------------------
st.subheader("🔮 Predict Revenue for a New Video")

views = st.number_input("Views", min_value=0, value=1000)
subs = st.number_input("Subscribers", min_value=0, value=100)
likes = st.number_input("Likes", min_value=0, value=50)
shares = st.number_input("Shares", min_value=0, value=10)

if st.button("Predict Revenue"):
    input_data = pd.DataFrame([[views, subs, likes, shares]], columns=features)
    prediction = model.predict(input_data)[0]
    st.success(f"💵 Predicted Revenue: **${prediction:.2f} USD**")

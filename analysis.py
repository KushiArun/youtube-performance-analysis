# ==========================================
# Unlocking YouTube Channel Performance Secrets
# analysis.py (Final Clean Version)
# ==========================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib

# ------------------------------------------
# 1. Load Dataset
# ------------------------------------------
data = pd.read_csv("youtube_channel_real_performance_analytics.csv")
print("✅ Data Loaded Successfully!")
print("Shape of dataset:", data.shape)
print(data.head())

# ------------------------------------------
# 2. Data Cleaning
# ------------------------------------------
# If duration is already numeric, skip conversion
if pd.api.types.is_numeric_dtype(data['Video Duration']):
    print("⏱️ Video Duration already numeric, skipping conversion.")
else:
    try:
        import isodate
        data['Video Duration'] = data['Video Duration'].apply(
            lambda x: isodate.parse_duration(x).total_seconds()
        )
        print("✅ Converted Video Duration to seconds")
    except Exception as e:
        print("⚠️ Duration conversion skipped:", e)

# Convert publish time
if "Video Publish Time" in data.columns:
    data['Video Publish Time'] = pd.to_datetime(data['Video Publish Time'])

# Drop missing values (basic handling)
data = data.dropna()
print("✅ Cleaned Data, remaining shape:", data.shape)

# ------------------------------------------
# 3. Feature Engineering
# ------------------------------------------
if "Estimated Revenue (USD)" in data.columns and "Views" in data.columns:
    data['Revenue per View'] = data['Estimated Revenue (USD)'] / data['Views']
else:
    print("⚠️ Revenue/Views columns missing!")

# Engagement Rate (only if all columns exist)
if all(col in data.columns for col in ["Likes", "Shares", "Comments", "Views"]):
    data['Engagement Rate'] = (
        (data['Likes'] + data['Shares'] + data['Comments']) / data['Views'] * 100
    )
    print("✅ Engagement Rate column created.")
else:
    print("⚠️ Engagement columns missing, skipping Engagement Rate.")

# ------------------------------------------
# 4. Exploratory Data Analysis (EDA)
# ------------------------------------------
plt.figure(figsize=(10, 6))
sns.histplot(data['Estimated Revenue (USD)'], bins=50, kde=True, color="green")
plt.title("Distribution of Estimated Revenue")
plt.xlabel("Revenue (USD)")
plt.ylabel("Frequency")
plt.savefig("revenue_distribution.png")
plt.close()
print("📊 Saved: revenue_distribution.png")

plt.figure(figsize=(10, 6))
sns.scatterplot(x=data['Views'], y=data['Estimated Revenue (USD)'], alpha=0.7)
plt.title("Revenue vs Views")
plt.xlabel("Views")
plt.ylabel("Revenue (USD)")
plt.savefig("revenue_vs_views.png")
plt.close()
print("📊 Saved: revenue_vs_views.png")

# Correlation Heatmap (numeric only)
plt.figure(figsize=(12, 8))
numeric_data = data.select_dtypes(include=[np.number])
sns.heatmap(numeric_data.corr(numeric_only=True), annot=False, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.savefig("correlation_heatmap.png")
plt.close()
print("📊 Saved: correlation_heatmap.png")

# ------------------------------------------
# 5. Machine Learning Model
# ------------------------------------------
possible_features = ['Views', 'Subscribers', 'Likes', 'Shares', 'Comments', 'Engagement Rate']
features = [f for f in possible_features if f in data.columns]  # only keep existing ones
target = 'Estimated Revenue (USD)'

if not features:
    raise ValueError("❌ No valid features found in dataset!")

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
print(f"📈 Model Evaluation: MSE={mse:.2f}, R2={r2:.2f}")

# ------------------------------------------
# 6. Feature Importance
# ------------------------------------------
importances = model.feature_importances_
feature_importance_df = pd.DataFrame(
    {"Feature": features, "Importance": importances}
).sort_values(by="Importance", ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(x="Importance", y="Feature", data=feature_importance_df)
plt.title("Feature Importance")
plt.savefig("feature_importance.png")
plt.close()
print("📊 Saved: feature_importance.png")

# ------------------------------------------
# 7. Save Model
# ------------------------------------------
joblib.dump(model, "youtube_revenue_predictor.pkl")
print("💾 Model saved as youtube_revenue_predictor.pkl")

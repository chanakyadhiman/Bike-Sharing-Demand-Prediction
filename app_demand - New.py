import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

st.set_page_config(page_title="Bike Demand Prediction")
st.title("🚲 Bike Sharing Demand Prediction")

# =============================
# 1. Load Data
# =============================
df = pd.read_csv("Dataset.csv")

# =============================
# 2. Define features & target
# =============================
features = [
    "season", "yr", "mnth", "hr",
    "holiday", "workingday", "weathersit",
    "temp", "atemp", "hum", "windspeed"
]

target = "cnt"

# =============================
# 3. Validate columns
# =============================
missing_cols = [c for c in features + [target] if c not in df.columns]
if missing_cols:
    st.error(f"❌ Missing columns in dataset: {missing_cols}")
    st.stop()

# =============================
# 4. Keep only required columns
# =============================
df = df[features + [target]]

# Force numeric conversion (VERY IMPORTANT)
df = df.apply(pd.to_numeric, errors="coerce")

# Drop rows with NaN
df = df.dropna()

# =============================
# 5. Split data
# =============================
X = df[features]
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# =============================
# 6. Train model
# =============================
model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)
model.fit(X_train, y_train)

# =============================
# 7. User Inputs (Sliders & Dropdowns)
# =============================
st.subheader("🔧 Input Conditions")

season = st.selectbox("Season", [1, 2, 3, 4])
yr = st.selectbox("Year", [0, 1])
mnth = st.slider("Month", 1, 12, 6)
hr = st.slider("Hour", 0, 23, 12)

holiday = st.selectbox("Holiday", [0, 1])
workingday = st.selectbox("Working Day", [0, 1])
weathersit = st.selectbox("Weather Situation", [1, 2, 3, 4])

temp = st.slider("Temperature", 0.0, 1.0, 0.5)
atemp = st.slider("Feels Like Temp", 0.0, 1.0, 0.5)
hum = st.slider("Humidity", 0.0, 1.0, 0.5)
windspeed = st.slider("Windspeed", 0.0, 1.0, 0.3)

input_df = pd.DataFrame([{
    "season": season,
    "yr": yr,
    "mnth": mnth,
    "hr": hr,
    "holiday": holiday,
    "workingday": workingday,
    "weathersit": weathersit,
    "temp": temp,
    "atemp": atemp,
    "hum": hum,
    "windspeed": windspeed
}])

# =============================
# 8. Prediction
# =============================
if st.button("🚀 Predict Bike Demand"):
    prediction = model.predict(input_df)
    st.success(f"🚴 Predicted Bike Demand: **{int(prediction[0])} bikes**")

# =============================
# 9. Feature Importance
# =============================
st.subheader("📊 Feature Importance")

importances = model.feature_importances_
plt.figure()
plt.barh(features, importances)
plt.xlabel("Importance")
plt.ylabel("Feature")
st.pyplot(plt)

# -------------------------------------------------
# Footer
# -------------------------------------------------
st.markdown("---")
st.caption("Created by Group -1 | CHANAKYA, KRISHNA et al. | Built with Streamlit & Random Forest Regressor")
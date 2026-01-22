import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

st.set_page_config(page_title="Bike Demand Prediction", layout="centered")
st.title("🚲 Bike Sharing Demand Prediction")

# =============================
# 1. Load Data
# =============================
df = pd.read_csv("Dataset.csv")

# Selected features & target
features = [
    "season",
    "yr",
    "mnth",
    "hr",
    "holiday",
    "workingday",
    "weathersit",
    "temp",
    "atemp",
    "hum",
    "windspeed"
]

target = "cnt"

df = df[features + [target]].dropna()

X = df[features]
y = df[target]

# =============================
# 2. Train Model
# =============================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# =============================
# 3. User Inputs (UI)
# =============================
st.subheader("🔧 Input Conditions")

season = st.selectbox(
    "Season",
    options=[1, 2, 3, 4],
    format_func=lambda x: ["Spring", "Summer", "Fall", "Winter"][x-1]
)

yr = st.selectbox("Year", [0, 1], format_func=lambda x: "2011" if x == 0 else "2012")

mnth = st.slider("Month", 1, 12, 6)
hr = st.slider("Hour", 0, 23, 12)

holiday = st.selectbox("Holiday", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
workingday = st.selectbox("Working Day", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")

weathersit = st.selectbox(
    "Weather Situation",
    [1, 2, 3, 4],
    format_func=lambda x: ["Clear", "Mist/Cloudy", "Light Snow/Rain", "Heavy Rain/Snow"][x-1]
)

temp = st.slider("Temperature (Normalized)", 0.0, 1.0, 0.5)
atemp = st.slider("Feels Like Temperature", 0.0, 1.0, 0.5)
hum = st.slider("Humidity", 0.0, 1.0, 0.5)
windspeed = st.slider("Windspeed", 0.0, 1.0, 0.3)

# =============================
# 4. Prediction
# =============================
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

if st.button("🚀 Predict Bike Demand"):
    prediction = model.predict(input_df)
    st.success(f"🚴 Predicted Bike Demand: **{int(prediction[0])} bikes**")

# =============================
# 5. Feature Importance Graph
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
st.caption("Created by Group -1 | CHANAKYA, KRISHNA et al.| Built with Streamlit & Random Forest Regressor")
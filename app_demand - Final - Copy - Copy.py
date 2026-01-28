import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

st.set_page_config(page_title="Bike Demand Prediction", layout="wide")
st.title("🚲 Bike Sharing Demand Prediction System")

# =============================
# Load Dataset
# =============================
df = pd.read_csv("Dataset.csv")

# =============================
# Force numeric for key columns FIRST
# =============================
for col in ["temp", "atemp", "windspeed", "hum"]:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

# =============================
# Convert normalized values to real units (only if needed)
# =============================
if "temp" in df.columns and df["temp"].max() <= 1:
    df["temp"] = df["temp"] * 41

if "atemp" in df.columns and df["atemp"].max() <= 1:
    df["atemp"] = df["atemp"] * 50

if "windspeed" in df.columns and df["windspeed"].max() <= 1:
    df["windspeed"] = df["windspeed"] * 67

# =============================
# Mapping dictionaries
# =============================
month_map = {
    1: "January", 2: "February", 3: "March", 4: "April",
    5: "May", 6: "June", 7: "July", 8: "August",
    9: "September", 10: "October", 11: "November", 12: "December"
}

season_map = {
    1: "Spring",
    2: "Summer",
    3: "Fall",
    4: "Winter"
}

weekday_map = {
    0: "Sunday",
    1: "Monday",
    2: "Tuesday",
    3: "Wednesday",
    4: "Thursday",
    5: "Friday",
    6: "Saturday"
}

# =============================
# Create DISPLAY columns (do not affect model)
# =============================
df["month_name"] = df["mnth"].map(month_map)
df["season_name"] = df["season"].map(season_map)
df["weekday_name"] = df["weekday"].map(weekday_map)

# =============================
# Feature selection (numeric for model)
# =============================
target = "cnt"

features = [
    "season", "yr", "mnth", "weekday", "hr",
    "holiday", "workingday", "weathersit",
    "temp", "atemp", "hum", "windspeed"
]

df = df[features + [target, "month_name", "season_name", "weekday_name"]]

df = df.dropna()

# =============================
# Safety check
# =============================
if df.shape[0] < 50:
    st.error("❌ Dataset too small after cleaning. Check Dataset.csv format.")
    st.stop()

X = df[features]
y = df[target]

# =============================
# Train model
# =============================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# =============================
# USER INPUTS
# =============================
st.subheader("🔧 Input Conditions")

month_order = list(month_map.values())
season_order = list(season_map.values())
weekday_order = list(weekday_map.values())

season_selected = st.selectbox("Season", season_order)
season_val = list(season_map.keys())[season_order.index(season_selected)]

month_selected = st.selectbox("Month", month_order)
month_val = list(month_map.keys())[month_order.index(month_selected)]

weekday_selected = st.selectbox("Weekday", weekday_order)
weekday_val = list(weekday_map.keys())[weekday_order.index(weekday_selected)]

yr_val = st.selectbox("Year (0 = 2011, 1 = 2012)", [0, 1])
holiday_val = st.selectbox("Holiday (0 = No, 1 = Yes)", [0, 1])
workingday_val = st.selectbox("Working Day (0 = No, 1 = Yes)", [0, 1])
weathersit_val = st.selectbox("Weather Situation (1–4)", [1, 2, 3, 4])
hr_val = st.slider("Hour", 0, 23, 12)
temp_val = st.slider("Temperature (°C)", 0.0, 50.0, 25.0)
atemp_val = st.slider("Feels Like Temperature (°C)", 0.0, 50.0, 25.0)
hum_val = st.slider("Humidity", 0.0, 1.0, 0.5)
windspeed_val = st.slider("Windspeed (km/h)", 0.0, 70.0, 10.0)

input_df = pd.DataFrame([{
    "season": season_val,
    "yr": yr_val,
    "mnth": month_val,
    "weekday": weekday_val,
    "hr": hr_val,
    "holiday": holiday_val,
    "workingday": workingday_val,
    "weathersit": weathersit_val,
    "temp": temp_val,
    "atemp": atemp_val,
    "hum": hum_val,
    "windspeed": windspeed_val
}])

# =============================
# Prediction
# =============================
if st.button("🚀 Predict Bike Demand"):
    prediction = model.predict(input_df)
    st.success(f"🚴 Predicted Bike Demand: **{int(prediction[0])} bikes**")

# =============================
# GRAPHS
# =============================
st.subheader("📊 Data Visualizations")

fig1, ax1 = plt.subplots()
df.groupby("month_name")["cnt"].mean().reindex(month_order).plot(kind="bar", ax=ax1)
ax1.set_title("Average Bike Demand per Month")
st.pyplot(fig1)

fig2, ax2 = plt.subplots()
df.groupby("weekday_name")["cnt"].mean().reindex(weekday_order).plot(kind="bar", ax=ax2)
ax2.set_title("Average Bike Demand per Weekday")
st.pyplot(fig2)

fig3, ax3 = plt.subplots()
ax3.scatter(df["temp"], df["cnt"])
ax3.set_title("Temperature vs Bike Demand")
ax3.set_xlabel("Temperature (°C)")
ax3.set_ylabel("Bike Demand")
st.pyplot(fig3)

st.caption("Bike Demand Prediction System | Streamlit + Random Forest")


st.markdown("---")
st.caption("Bike Demand Prediction System | Streamlit + Random Forest")




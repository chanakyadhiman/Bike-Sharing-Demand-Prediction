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

st.write("Dataset columns:", df.columns.tolist())

# =============================
# Convert normalized values to real units (ONLY if needed)
# =============================
if df["temp"].max() <= 1:
    df["temp"] = df["temp"] * 41

if df["atemp"].max() <= 1:
    df["atemp"] = df["atemp"] * 50

if df["windspeed"].max() <= 1:
    df["windspeed"] = df["windspeed"] * 67

# =============================
# Create DISPLAY columns (DO NOT affect model)
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

df["month_name"] = df["mnth"].map(month_map)
df["season_name"] = df["season"].map(season_map)
df["weekday_name"] = df["weekday"].map(weekday_map)

# =============================
# Feature selection (NUMERIC ONLY)
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

input_data = {}

input_data["season"] = season_map.keys().__iter__().__next__()

season_selected = st.selectbox("Season", season_order)
input_data["season"] = list(season_map.keys())[season_order.index(season_selected)]

month_selected = st.selectbox("Month", month_order)
input_data["mnth"] = list(month_map.keys())[month_order.index(month_selected)]

weekday_selected = st.selectbox("Weekday", weekday_order)
input_data["weekday"] = list(weekday_map.keys())[weekday_order.index(weekday_selected)]

input_data["yr"] = st.selectbox("Year (0=2011, 1=2012)", [0, 1])
input_data["holiday"] = st.selectbox("Holiday (0=No,1=Yes)", [0, 1])
input_data["workingday"] = st.selectbox("Working Day (0=No,1=Yes)", [0, 1])
input_data["weathersit"] = st.selectbox("Weather Situation (1-4)", [1, 2, 3, 4])
input_data["hr"] = st.slider("Hour", 0, 23, 12)
input_data["temp"] = st.slider("Temperature (°C)", 0.0, 50.0, 25.0)
input_data["atemp"] = st.slider("Feels Like Temp (°C)", 0.0, 50.0, 25.0)
input_data["hum"] = st.slider("Humidity", 0.0, 1.0, 0.5)
input_data["windspeed"] = st.slider("Windspeed (km/h)", 0.0, 70.0, 10.0)

input_df = pd.DataFrame([input_data])

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



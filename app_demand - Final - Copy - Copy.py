import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

st.set_page_config(page_title="Bike Demand Prediction", layout="wide")
st.title("🚴 Bike Sharing Demand Prediction System")

# =========================
# Load Dataset
# =========================
df = pd.read_csv("Dataset.csv")

# =========================
# Column mappings
# =========================
season_map = {1: "Spring", 2: "Summer", 3: "Fall", 4: "Winter"}
month_map = {
    1: "January", 2: "February", 3: "March", 4: "April",
    5: "May", 6: "June", 7: "July", 8: "August",
    9: "September", 10: "October", 11: "November", 12: "December"
}
weekday_map = {
    0: "Sunday", 1: "Monday", 2: "Tuesday",
    3: "Wednesday", 4: "Thursday",
    5: "Friday", 6: "Saturday"
}

# =========================
# Convert columns
# =========================
df["season"] = df["season"].map(season_map)
df["mnth"] = df["mnth"].map(month_map)
df["weekday"] = df["weekday"].map(weekday_map)

# Convert temp to °C if normalized
if df["temp"].max() <= 1:
    df["temp"] = df["temp"] * 41

if df["atemp"].max() <= 1:
    df["atemp"] = df["atemp"] * 50

# Convert windspeed to km/h if normalized
if df["windspeed"].max() <= 1:
    df["windspeed"] = df["windspeed"] * 67

# Humidity %
if df["hum"].max() <= 1:
    df["hum"] = df["hum"] * 100

# =========================
# Select columns
# =========================
features = ["season", "mnth", "weekday", "temp", "atemp", "hum", "windspeed"]
target = "cnt"

df = df[features + [target]].dropna()

# =========================
# Safety Check
# =========================
if df.shape[0] < 20:
    st.error("❌ Dataset too small after cleaning. Check Dataset.csv format.")
    st.stop()

# =========================
# Encode categories
# =========================
df_encoded = pd.get_dummies(df, columns=["season", "mnth", "weekday"], drop_first=True)

X = df_encoded.drop(target, axis=1)
y = df_encoded[target]

# =========================
# Train Model
# =========================
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(random_state=42)
model.fit(X_train, y_train)

# =========================
# Sidebar Inputs
# =========================
st.sidebar.header("Input Conditions")

season_input = st.sidebar.selectbox(
    "Season",
    sorted(df["season"].dropna().unique())
)

month_input = st.sidebar.selectbox(
    "Month",
    list(month_map.values())
)

weekday_input = st.sidebar.selectbox(
    "Weekday",
    list(weekday_map.values())
)

temp_input = st.sidebar.slider("Temperature (°C)", 0.0, 50.0, float(df["temp"].mean()))
atemp_input = st.sidebar.slider("Feels Like (°C)", 0.0, 50.0, float(df["atemp"].mean()))
hum_input = st.sidebar.slider("Humidity (%)", 0.0, 100.0, float(df["hum"].mean()))
wind_input = st.sidebar.slider("Windspeed (km/h)", 0.0, 70.0, float(df["windspeed"].mean()))

# =========================
# Create input dataframe
# =========================
input_dict = {
    "temp": temp_input,
    "atemp": atemp_input,
    "hum": hum_input,
    "windspeed": wind_input
}

for col in X.columns:
    if col.startswith("season_"):
        input_dict[col] = 1 if col == f"season_{season_input}" else 0
    elif col.startswith("mnth_"):
        input_dict[col] = 1 if col == f"mnth_{month_input}" else 0
    elif col.startswith("weekday_"):
        input_dict[col] = 1 if col == f"weekday_{weekday_input}" else 0

input_df = pd.DataFrame([input_dict])

# =========================
# Prediction
# =========================
if st.button("🚲 Predict Bike Demand"):
    prediction = model.predict(input_df)[0]
    st.success(f"Estimated Bike Demand: **{int(prediction)} bikes**")

# =========================
# Interactive Graph
# =========================
st.subheader("📊 Demand vs Temperature")

fig, ax = plt.subplots()
ax.scatter(df["temp"], df["cnt"])
ax.set_xlabel("Temperature (°C)")
ax.set_ylabel("Bike Demand")
ax.set_title("Bike Demand vs Temperature")
st.pyplot(fig)

# =========================
# Footer
# =========================
st.caption("Project - Bike Demand Prediction System | Group-1: Chanakya, Krishna et al. | Streamlit + Random Forest")

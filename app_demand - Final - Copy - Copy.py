import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

st.set_page_config(page_title="Bike Demand Prediction", layout="wide")
st.title("🚲 Bike Sharing Demand Prediction & Analytics")

# =============================
# 1. Load Dataset
# =============================
df = pd.read_csv("Dataset.csv")

# -----------------------------
# Convert Normalized Values to Real Units
# -----------------------------
if "temp" in df.columns:
    df["temp"] = df["temp"] * 41

if "atemp" in df.columns:
    df["atemp"] = df["atemp"] * 50

if "windspeed" in df.columns:
    df["windspeed"] = df["windspeed"] * 67

# -----------------------------
# Convert Month Number to Month Name
# -----------------------------
if "mnth" in df.columns:
    df["mnth"] = df["mnth"].astype(int)
    month_map = {
        1: "January", 2: "February", 3: "March", 4: "April",
        5: "May", 6: "June", 7: "July", 8: "August",
        9: "September", 10: "October", 11: "November", 12: "December"
    }
    df["mnth"] = df["mnth"].map(month_map)

# -----------------------------
# Convert Season Number to Season Name
# -----------------------------
if "season" in df.columns:
    df["season"] = df["season"].astype(int)
    season_map = {
        1: "Spring",
        2: "Summer",
        3: "Fall",
        4: "Winter"
    }
    df["season"] = df["season"].map(season_map)

# -----------------------------
# FORCE NUMERIC COLUMNS TO FLOAT
# -----------------------------
numeric_cols = ["temp", "atemp", "hum", "windspeed"]
for col in numeric_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

st.write("📄 Dataset Columns:")
st.write(list(df.columns))

# =============================
# 2. Target & Feature Selection
# =============================
target = "cnt"

features = [
    "season", "yr", "mnth", "hr",
    "holiday", "workingday", "weathersit",
    "temp", "atemp", "hum", "windspeed"
]

features = [c for c in features if c in df.columns]

df = df[features + [target]]

# =============================
# 3. Encode ONLY Categorical Columns
# =============================
categorical_cols = ["season", "mnth", "holiday", "workingday", "weathersit"]

label_encoders = {}

for col in categorical_cols:
    if col in df.columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        label_encoders[col] = le

# =============================
# 4. Handle Missing Values
# =============================
df = df.dropna()

X = df[features]
y = df[target]

# =============================
# 5. Train-Test Split
# =============================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# =============================
# 6. Train Model
# =============================
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# =============================
# 7. User Inputs (CORRECT SLIDERS)
# =============================
st.subheader("🔧 Input Conditions")
input_data = {}

for col in features:
    display_name = col

    if col == "temp":
        display_name = "Temperature (°C)"
    elif col == "atemp":
        display_name = "Feels Like Temperature (°C)"
    elif col == "windspeed":
        display_name = "Windspeed (km/h)"
    elif col == "hum":
        display_name = "Humidity (%)"

    if col in label_encoders:
        le = label_encoders[col]
        selected = st.selectbox(display_name, le.classes_, key=col)
        input_data[col] = le.transform([selected])[0]
    else:
        input_data[col] = st.slider(
            display_name,
            float(X[col].min()),
            float(X[col].max()),
            float(X[col].mean()),
            key=col
        )

input_df = pd.DataFrame([input_data])

# =============================
# 8. Prediction
# =============================
if st.button("🚀 Predict Bike Demand"):
    prediction = model.predict(input_df)
    st.success(f"🚴 Predicted Bike Demand: **{int(prediction[0])} bikes**")

# =============================
# Footer
# =============================
st.markdown("---")
st.caption("Created by Group -1 | CHANAKYA, KRISHNA et al. | Built with Streamlit & Random Forest Regressor")
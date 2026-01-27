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
    df["temp"] = pd.to_numeric(df["temp"], errors="coerce") * 41  # °C

if "atemp" in df.columns:
    df["atemp"] = pd.to_numeric(df["atemp"], errors="coerce") * 50  # °C

if "windspeed" in df.columns:
    df["windspeed"] = pd.to_numeric(df["windspeed"], errors="coerce") * 67  # km/h

# -----------------------------
# Safe Month Conversion
# -----------------------------
if "mnth" in df.columns:
    month_map = {
        1: "January", 2: "February", 3: "March", 4: "April",
        5: "May", 6: "June", 7: "July", 8: "August",
        9: "September", 10: "October", 11: "November", 12: "December"
    }
    df["mnth"] = pd.to_numeric(df["mnth"], errors="coerce")
    df["mnth"] = df["mnth"].map(month_map)

# -----------------------------
# Safe Season Conversion
# -----------------------------
if "season" in df.columns:
    season_map = {
        1: "Spring",
        2: "Summer",
        3: "Fall",
        4: "Winter"
    }
    df["season"] = pd.to_numeric(df["season"], errors="coerce")
    df["season"] = df["season"].map(season_map)

# -----------------------------
# Force Numeric Columns to Float
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

if target not in df.columns or len(features) == 0:
    st.error("❌ Required columns not found in dataset")
    st.stop()

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
# 7. User Inputs (Correct Widgets & Units)
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
# 9. Feature Importance
# =============================
st.subheader("📊 Feature Importance")
plt.figure(figsize=(8, 6))
plt.barh(features, model.feature_importances_)
plt.xlabel("Importance")
st.pyplot(plt)

# -------------------------------------------------
# Footer
# -------------------------------------------------
st.markdown("---")
st.caption("Created by Group -1 | CHANAKYA, KRISHNA et al. | Built with Streamlit & Random Forest Regressor")
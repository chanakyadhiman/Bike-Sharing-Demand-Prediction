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
# (UCI Bike Sharing Dataset formula)
# -----------------------------
if "temp" in df.columns:
    df["temp"] = df["temp"] * 41        # Celsius

if "atemp" in df.columns:
    df["atemp"] = df["atemp"] * 50      # Celsius

if "windspeed" in df.columns:
    df["windspeed"] = df["windspeed"] * 67  # km/h

# -----------------------------
# Convert Month Number to Month Name
# -----------------------------
if "mnth" in df.columns:
    try:
        df["mnth"] = df["mnth"].astype(int)
        month_map = {
            1: "January", 2: "February", 3: "March", 4: "April",
            5: "May", 6: "June", 7: "July", 8: "August",
            9: "September", 10: "October", 11: "November", 12: "December"
        }
        df["mnth"] = df["mnth"].map(month_map)
    except:
        pass

# -----------------------------
# Convert Season Number to Season Name
# -----------------------------
if "season" in df.columns:
    try:
        df["season"] = df["season"].astype(int)
        season_map = {
            1: "Spring",
            2: "Summer",
            3: "Fall",
            4: "Winter"
        }
        df["season"] = df["season"].map(season_map)
    except:
        pass

st.write("📄 Dataset Columns:")
st.write(list(df.columns))

# =============================
# 2. Target & Feature Selection
# =============================
target = "cnt"

candidate_features = [
    "season", "yr", "mnth", "hr",
    "holiday", "workingday", "weathersit",
    "temp", "atemp", "hum", "windspeed"
]

features = [c for c in candidate_features if c in df.columns]

if target not in df.columns or len(features) == 0:
    st.error("❌ Required columns not found in dataset")
    st.stop()

df = df[features + [target]]

# =============================
# 3. Encode Categorical Columns
# =============================
label_encoders = {}

for col in df.columns:
    if df[col].dtype == "object":
        df[col] = df[col].astype(str)
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
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
# 7. User Inputs for Prediction (WITH REAL UNITS)
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
        selected = st.selectbox(display_name, le.classes_)
        input_data[col] = le.transform([selected])[0]
    else:
        input_data[col] = st.slider(
            display_name,
            float(X[col].min()),
            float(X[col].max()),
            float(X[col].mean())
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

# =============================
# Footer
# =============================
st.markdown("---")
st.caption("Created by Group -1 | CHANAKYA, KRISHNA et al. | Built with Streamlit & Random Forest Regressor")
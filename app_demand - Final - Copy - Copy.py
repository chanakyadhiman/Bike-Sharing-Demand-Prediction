import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

st.set_page_config(page_title="Bike Sharing Demand Prediction", layout="wide")
st.title("🚲 Bike Sharing Demand Prediction System")

# =============================
# Load dataset
# =============================
try:
    df = pd.read_csv("Dataset.csv")
    st.write("📄 Raw Dataset Preview")
    st.dataframe(df.head())
except Exception as e:
    st.error(f"❌ Failed to load dataset: {e}")
    st.stop()

# =============================
# Replace ? with NaN
# =============================
df.replace("?", pd.NA, inplace=True)

# =============================
# Convert numeric columns safely
# =============================
numeric_cols = ["temp", "atemp", "hum", "windspeed", "cnt", "hr", "mnth", "yr"]
for col in numeric_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

# =============================
# Convert normalized values
# =============================
def safe_scale(col, factor):
    if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
        max_val = df[col].dropna().max()
        if pd.notna(max_val) and max_val <= 1:
            df[col] = df[col] * factor

safe_scale("temp", 41)
safe_scale("atemp", 50)
safe_scale("windspeed", 67)

# =============================
# Encode categorical columns
# =============================
categorical_cols = ["season", "holiday", "workingday", "weathersit"]
label_encoders = {}

for col in categorical_cols:
    if col in df.columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        label_encoders[col] = le

# =============================
# Feature selection
# =============================
features = [
    "season", "yr", "mnth", "hr",
    "holiday", "workingday", "weathersit",
    "temp", "atemp", "hum", "windspeed"
]
target = "cnt"

df = df[features + [target]].dropna()

# =============================
# Safety check
# =============================
if len(df) < 20:
    st.error("❌ Dataset too small after cleaning. Check Dataset.csv format.")
    st.stop()

# =============================
# Train model
# =============================
X = df[features]
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestRegressor(n_estimators=150, random_state=42)
model.fit(X_train, y_train)

st.success("✅ Model trained successfully")

# =============================
# User input UI
# =============================
st.subheader("🔧 Enter Input Conditions")
input_data = {}

symbol_map = {
    "temp": "🌡️ Temp (°C)",
    "atemp": "🤖 Atemp (°C)",
    "hum": "💧 Humidity (%)",
    "windspeed": "🌬️ Windspeed (km/h)"
}

for col in features:
    if col in categorical_cols:
        le = label_encoders[col]
        # remove <NA> or None values from dropdown
        options = [cls for cls in le.classes_ if pd.notna(cls) and str(cls).lower() != "<na>"]
        selected = st.selectbox(col.capitalize(), options)
        input_data[col] = le.transform([selected])[0]
    else:
        min_val = float(X[col].min())
        max_val = float(X[col].max())
        mean_val = float(X[col].mean())
        label = symbol_map.get(col, col.capitalize())
        input_data[col] = st.slider(label, min_val, max_val, mean_val)

input_df = pd.DataFrame([input_data])

# =============================
# Prediction
# =============================
if st.button("🚀 Predict Bike Demand"):
    prediction = model.predict(input_df)
    st.success(f"🚴 Predicted Bike Demand: **{int(prediction[0])} bikes**")

# =============================
# Graphs
# =============================
st.subheader("📊 Visualizations")

fig1, ax1 = plt.subplots()
df.groupby("mnth")["cnt"].mean().plot(kind="bar", ax=ax1)
ax1.set_title("Average Demand per Month")
st.pyplot(fig1)

fig2, ax2 = plt.subplots()
df.groupby("hr")["cnt"].mean().plot(kind="line", ax=ax2)
ax2.set_title("Average Demand per Hour")
st.pyplot(fig2)

fig3, ax3 = plt.subplots()
ax3.scatter(df["temp"], df["cnt"])
ax3.set_xlabel("Temperature")
ax3.set_ylabel("Bike Demand")
ax3.set_title("Temperature vs Bike Demand")
st.pyplot(fig3)

st.caption("Bike Sharing Demand Prediction System | Random Forest + Streamlit")

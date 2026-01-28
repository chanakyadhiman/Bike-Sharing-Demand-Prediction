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
numeric_cols = ["temp", "atemp", "hum", "windspeed", "cnt", "hr", "mnth", "yr", "weekday"]
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
categorical_cols = ["season", "holiday", "workingday", "weathersit", "weekday"]
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
    "weekday", "temp", "atemp", "hum", "windspeed"
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

month_names = [
    "January","February","March","April","May","June",
    "July","August","September","October","November","December"
]
weekday_names = [
    "Sunday","Monday","Tuesday","Wednesday","Thursday","Friday","Saturday"
]

for col in features:
    if col in categorical_cols and col != "weekday":
        le = label_encoders[col]
        options = [cls for cls in le.classes_ if pd.notna(cls) and str(cls).lower() != "<na>"]
        selected = st.selectbox(col.capitalize(), options)
        input_data[col] = le.transform([selected])[0]

    elif col == "yr":
        # Checkbox for year
        year_selected = st.checkbox("Year 2012 (checked) / 2011 (unchecked)", value=True)
        input_data[col] = 1 if year_selected else 0

    elif col == "mnth":
        # Dropdown for month names
        selected_month = st.selectbox("📅 Month", month_names)
        input_data[col] = month_names.index(selected_month) + 1

    elif col == "weekday":
        # Dropdown for weekdays
        selected_day = st.selectbox("🗓️ Weekday", weekday_names)
        input_data[col] = weekday_names.index(selected_day)

    else:
        # Sliders for numeric features
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
df.groupby("weekday")["cnt"].mean().plot(kind="bar", ax=ax3)
ax3.set_title("Average Demand per Weekday")
st.pyplot(fig3)

fig4, ax4 = plt.subplots()
ax4.scatter(df["temp"], df["cnt"])
ax4.set_xlabel("Temperature")
ax4.set_ylabel("Bike Demand")
ax4.set_title("Temperature vs Bike Demand")
st.pyplot(fig4)

st.caption("Bike Sharing Demand Prediction System | Random Forest + Streamlit")

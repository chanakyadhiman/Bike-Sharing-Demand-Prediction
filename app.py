import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

# -------------------------------------------------
# Streamlit Config
# -------------------------------------------------
st.set_page_config(page_title="Bike Demand Prediction", layout="centered")
st.title("🚲 Bike Sharing Demand Prediction")

# -------------------------------------------------
# Load Dataset
# -------------------------------------------------
df = pd.read_csv("Dataset.csv")

# Normalize column names
df.columns = df.columns.str.strip().str.lower()

# Rename columns to simpler names
COLUMN_MAP = {
    'hour': 'hr',
    'temperature(°c)': 'temp',
    'humidity(%)': 'hum',
    'wind speed (m/s)': 'windspeed',
    'seasons': 'season',
    'holiday': 'holiday',
    'functioning day': 'workingday',
    'rented bike count': 'cnt'
}
df = df.rename(columns=COLUMN_MAP)

# Convert workingday to numeric
if 'workingday' in df.columns:
    df['workingday'] = df['workingday'].map({'yes': 1, 'no': 0})

# Keep only required columns
features = ['hr', 'temp', 'hum', 'windspeed', 'season', 'holiday', 'workingday']
target = 'cnt'
df = df[features + [target]]

# Convert to numeric and drop NAs
df = df.apply(pd.to_numeric, errors='coerce').dropna()

# -------------------------------------------------
# Train Model
# -------------------------------------------------
X = df[features]
y = df[target]

if "model" not in st.session_state:
    model = RandomForestRegressor(
        n_estimators=300,
        max_depth=10,
        random_state=42
    )
    model.fit(X, y)
    st.session_state.model = model

model = st.session_state.model

# -------------------------------------------------
# Sidebar Inputs
# -------------------------------------------------
st.sidebar.header("🔧 Input Features")

hr = st.sidebar.slider("Hour of Day", 0, 23, 12)
temp = st.sidebar.slider("Temperature (°C)", 0.0, 40.0, 20.0)
hum = st.sidebar.slider("Humidity (%)", 0, 100, 50)
windspeed = st.sidebar.slider("Windspeed", 0.0, 50.0, 10.0)
season = st.sidebar.selectbox("Season", [1, 2, 3, 4])
holiday = st.sidebar.selectbox("Holiday", [0, 1])
workingday = st.sidebar.selectbox("Working Day", [0, 1])

input_df = pd.DataFrame({
    'hr': [hr],
    'temp': [temp],
    'hum': [hum],
    'windspeed': [windspeed],
    'season': [season],
    'holiday': [holiday],
    'workingday': [workingday]
})

st.subheader("📊 Input Data")
st.write(input_df)

# -------------------------------------------------
# Prediction
# -------------------------------------------------
if st.button("Predict Demand"):
    prediction = model.predict(input_df)
    st.success(f"🚴 Predicted Bike Demand: **{int(prediction[0])} bikes**")

# -------------------------------------------------
# Footer
# -------------------------------------------------
st.markdown("---")
st.caption("Created by Group -1 | Built with Streamlit & Random Forest Regressor")
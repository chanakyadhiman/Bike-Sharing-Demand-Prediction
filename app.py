# ---------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------
# Import important Libraries
# ---------------------------------------------------------------------------------------------

import streamlit as st        # Helps to load Streamlit libraries that converts Python script to interactive web apps
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

# ---------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------
# Streamlit Page
# ---------------------------------------------------------------------------------------------

st.set_page_config(page_title="Bike Demand Prediction", layout="centered")

st.title("🚲 Bike Sharing Demand Prediction")
st.markdown("Predict hourly bike rental demand based on weather & time factors")

# ---------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------
# Load dataset
# ---------------------------------------------------------------------------------------------

df = pd.read_csv("Dataset.csv")

# --------------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------
# Feature Selection
# --------------------------------------------------------------------------------------------

df = df[['hr', 'temp', 'hum', 'windspeed', 'season', 'holiday', 'workingday', 'cnt']]
df = df.apply(pd.to_numeric, errors='coerce')

# Drop rows with missing values

df = df.dropna()

X = df[['hr', 'temp', 'hum', 'windspeed', 'season', 'holiday', 'workingday']]
y = df['cnt']

# --------------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------
# Train Model
# --------------------------------------------------------------------------------------------

if "model" not in st.session_state:
    model = RandomForestRegressor(
        n_estimators=300,
        max_depth=10,
        random_state=42
    )

    model.fit(X, y)

    st.session_state.model = model

model = st.session_state.model

# -------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------
# Sidebar Inputs
# -------------------------------------------------------------------------------------------

st.sidebar.header("🔧 Input Features")

hr = st.sidebar.slider("Hour of Day", 0, 23, 12)
temp = st.sidebar.slider("Temperature (°C)", 0.0, 40.0, 20.0)
hum = st.sidebar.slider("Humidity (%)", 0, 100, 50)
windspeed = st.sidebar.slider("Windspeed", 0.0, 50.0, 10.0)
season = st.sidebar.selectbox("Season", [1, 2, 3, 4])
holiday = st.sidebar.selectbox("Holiday", [0, 1])
workingday = st.sidebar.selectbox("Working Day", [0, 1])

# -------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------
# Input Dataframe
# -------------------------------------------------------------------------------------------

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

# -------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------
# Prediction
# -------------------------------------------------------------------------------------------

if st.button("Predict Demand"):
    prediction = model.predict(input_df)
    st.success(f"🚴 Predicted Bike Demand: **{int(prediction[0])} bikes**")

# -------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------
# Footer
# -------------------------------------------------------------------------------------------

st.markdown("---")
st.caption("Created by (Group -1) | Chanakya Dhiman, Krishna Mohith, et al. | Built with Streamlit & Random Forest Regressor")
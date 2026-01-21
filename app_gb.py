
# Import important Libraries


import streamlit as st        # Helps to load Streamlit libraries that converts Python script to interactive web apps
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split


# Streamlit App

st.set_page_config(page_title="Bike Demand Prediction", layout="centered")

st.title("🚲 Bike Sharing Demand Prediction")
st.markdown('Predict hourly bike rental demand based on weather & time factors')

# Load dataset

df = pd.read_csv("Dataset.csv")

# Features & target
X = df[['hr', 'temp', 'hum', 'windspeed', 'season', 'holiday', 'workingday']]
y = df['cnt']

# Train model (cached to avoid retraining every click)
@st.cache_resource
def train_model():
    model = GradientBoostingRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=4,
        random_state=42
    )
    model.fit(X, y)
    return model

model = train_model()

# Sidebar inputs
st.sidebar.header("Input Features")

hour = st.sidebar.slider("Hour of Day", 0, 23, 12)
temp = st.sidebar.slider("Temperature (°C)", 0.0, 40.0, 20.0)
humidity = st.sidebar.slider("Humidity (%)", 0, 100, 50)
windspeed = st.sidebar.slider("Windspeed", 0.0, 50.0, 10.0)
season = st.sidebar.selectbox("Season", [1, 2, 3, 4])
holiday = st.sidebar.selectbox("Holiday", [0, 1])
workingday = st.sidebar.selectbox("Working Day", [0, 1])

# Input dataframe
input_df = pd.DataFrame({
    "hour": [hour],
    "temp": [temp],
    "humidity": [humidity],
    "windspeed": [windspeed],
    "season": [season],
    "holiday": [holiday],
    "workingday": [workingday]
})

st.subheader("Input Data")
st.write(input_df)

# Prediction
if st.button("Predict Demand"):
    prediction = model.predict(input_df)
    st.success(f"🚴 Predicted Bike Demand: **{int(prediction[0])} bikes**")

st.markdown("---")
st.caption("Created by (Group -1) Chanakya Dhiman, Krishna Mohith, Dabbara CHindu, Ankita Bairagi, Abhinav EV, Adhwaith S, Taruni | Built with Streamlit & Gradient Boosting Regressor")
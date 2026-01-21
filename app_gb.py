
# Import important Libraries


import streamlit as st        # Helps to load Streamlit libraries that converts Python script to interactive web apps
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor



# Streamlit App

st.set_page_config(page_title='Bike Demand Prediction', page_icon='random', layout='centered', initial_sidebar_state='auto')

st.title('Bike Sharing Demand Prediction')
st.markdown('Predict hourly bike rental demand based on weather & time factors')

# Sidebar Inputs

st.sidebar.header('Input Parameters')

# User inputs
hour = st.sidebar.slider("Hour of Day", 0, 23, 12)
temp = st.sidebar.slider("Temperature (°C)", 0.0, 40.0, 20.0)
humidity = st.sidebar.slider("Humidity (%)", 0, 100, 50)
windspeed = st.sidebar.slider("Windspeed", 0.0, 50.0, 10.0)
season = st.sidebar.selectbox("Season", [1, 2, 3, 4])
holiday = st.sidebar.selectbox("Holiday", [0, 1])
workingday = st.sidebar.selectbox("Working Day", [0, 1])

# Input DataFrame
input_data = pd.DataFrame({
    "hour": [hour],
    "temp": [temp],
    "humidity": [humidity],
    "windspeed": [windspeed],
    "season": [season],
    "holiday": [holiday],
    "workingday": [workingday]
})

st.subheader("Input Data")
st.write(input_data)

# Prediction
if st.button("Predict Bike Demand"):
    prediction = model.predict(input_data)
    st.success(f"Estimated Bike Demand: **{int(prediction[0])} bikes**")


st.markdown("---")
st.caption("Created by Chanakya Dhiman | Built with Streamlit & Logistic Regression")
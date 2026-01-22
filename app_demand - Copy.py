import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

st.title("🚲 Bike Sharing Demand Prediction - RF Regressor")

# 1. Load data
df = pd.read_csv("Dataset.csv")

# 2. Select required columns
features = [
    "temp",        # temperature
    "atemp",       # feels like temperature
    "hum",         # humidity
    "windspeed",   # wind speed
    "season",
    "yr",
    "mnth",
    "hr",
    "holiday",
    "workingday",
    "weathersit"
]

target = "cnt"

# Keep only selected columns
df = df[features + [target]].dropna()

# 3. Split features & target
X = df[features]
y = df[target]

st.write("### Dataset Preview")
st.dataframe(df.head())

# 4. Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 5. Train model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 6. User input
st.write("### Predict Bike Demand")

input_data = {}
for col in features:
    input_data[col] = st.number_input(col, float(X[col].mean()))

input_df = pd.DataFrame([input_data])

if st.button("Predict Demand"):
    prediction = model.predict(input_df)
    st.success(f"🚴 Predicted Bike Demand: **{int(prediction[0])} bikes**")

# 7. Feature Importance
st.write("### Feature Importance")

importances = model.feature_importances_
plt.figure()
plt.barh(features, importances)
plt.xlabel("Importance")
plt.ylabel("Feature")
st.pyplot(plt)

# -------------------------------------------------
# Footer
# -------------------------------------------------
st.markdown("---")
st.caption("Created by Group -1 | Chanakya, Krishna et al. | Built with Streamlit & Random Forest Regressor")


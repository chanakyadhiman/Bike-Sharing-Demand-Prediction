import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

st.title("🌳 Bike Rental Demand - Random Forest Regression App")

# 1. Load data
df = pd.read_csv("Dataset.csv")

# 2. Keep only numeric data & clean
df = df.select_dtypes(include=np.number).dropna()

# 3. Select features & target
X = df.iloc[:, :-1]   # features
y = df.iloc[:, -1]    # target


# 4. Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 5. Train model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 6. Prediction inputs
st.write("### Make a Prediction")

input_data = {}
for col in X.columns:
    input_data[col] = st.number_input(col, float(X[col].mean()))

input_df = pd.DataFrame([input_data])

if st.button("Predict"):
    prediction = model.predict(input_df)
    st.success(f"Predicted Value: {prediction[0]:.2f}")

# 7. Feature Importance Graph
st.write("### Feature Importance")

importances = model.feature_importances_
plt.figure()
plt.barh(X.columns, importances)
plt.xlabel("Importance")
plt.ylabel("Feature")
st.pyplot(plt)

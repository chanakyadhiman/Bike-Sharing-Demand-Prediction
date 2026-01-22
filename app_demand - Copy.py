import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

st.title("🚴 Bike Demand Prediction App (Random Forest)")

# Load dataset directly (Dataset.csv must be in the repo)
df = pd.read_csv("Dataset.csv")

st.write("Preview of dataset:", df.head())

# --- Define target and features ---

target = "cnt"  

# Features = all other columns except target

features = [c for c in df.columns if c != target]

st.write("Target column:", target)
st.write("Feature columns:", features)

X = df[features]
y = df[target]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Random Forest model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Evaluation
mse = mean_squared_error(y_test, y_pred)
st.write(f"📊 Mean Squared Error: {mse:.2f}")

# Prediction section
st.subheader("Make a Prediction")
input_data = {}
for col in features:
    val = st.number_input(
        f"Enter value for {col}",
        float(df[col].min()),
        float(df[col].max()),
        float(df[col].mean())
    )
    input_data[col] = val

if st.button("Predict Demand"):
    input_df = pd.DataFrame([input_data])
    prediction = model.predict(input_df)[0]
    st.success(f"🚴 Predicted Bike Demand: {prediction:.0f}")
# -------------------------------------------------
# Footer
# -------------------------------------------------
st.markdown("---")
st.caption("Created by Group -1 | Built with Streamlit & Random Forest Regressor")








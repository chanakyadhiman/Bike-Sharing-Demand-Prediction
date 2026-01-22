import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

st.title("🚴 Bike Demand Prediction App (Random Forest)")

# Upload dataset
uploaded_file = st.file_uploader("Upload Bike Demand CSV", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("Preview of dataset:", df.head())

    # Basic EDA
    st.subheader("Data Overview")
    st.write(df.describe())

    # Plot demand distribution
    if "demand" in df.columns:
        fig, ax = plt.subplots()
        sns.histplot(df["demand"], bins=30, kde=True, ax=ax)
        st.pyplot(fig)

    # Feature selection
    all_columns = df.columns.tolist()
    target = st.selectbox("Select target column (e.g., demand)", all_columns)
    features = st.multiselect("Select feature columns", [c for c in all_columns if c != target])

    if features and target:
        X = df[features]
        y = df[target]

        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Random Forest Model
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # Evaluation
        mse = mean_squared_error(y_test, y_pred)
        st.write(f"📊 Mean Squared Error: {mse:.2f}")

        # Prediction
        st.subheader("Make a Prediction")
        input_data = {}
        for col in features:
            val = st.number_input(f"Enter value for {col}", float(df[col].min()), float(df[col].max()), float(df[col].mean()))
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








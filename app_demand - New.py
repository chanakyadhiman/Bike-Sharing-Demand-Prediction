import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

st.set_page_config(page_title="Bike Demand Prediction")
st.title("🚲 Bike Sharing Demand Prediction")

# =============================
# 1. Load data
# =============================
df = pd.read_csv("Dataset.csv")

st.write("📄 Columns in dataset:")
st.write(list(df.columns))

# =============================
# 2. Target
# =============================
target = "cnt"

if target not in df.columns:
    st.error("❌ Target column 'cnt' not found in dataset")
    st.stop()

# =============================
# 3. Candidate features (safe list)
# =============================
candidate_features = [
    "season", "yr", "mnth", "hr",
    "holiday", "workingday", "weathersit",
    "temp", "atemp", "hum", "windspeed"
]

# Keep only features that exist
features = [c for c in candidate_features if c in df.columns]

if len(features) == 0:
    st.error("❌ No valid feature columns found")
    st.stop()

st.success(f"✅ Using features: {features}")

# =============================
# 4. Prepare data
# =============================
df = df[features + [target]]

# Convert to numeric safely
df = df.apply(pd.to_numeric, errors="coerce")

# Drop missing values
df = df.dropna()

# 🚨 CRITICAL CHECK
if len(df) < 5:
    st.error("❌ Not enough valid rows after cleaning")
    st.write(df)
    st.stop()

X = df[features]
y = df[target]

# =============================
# 5. Train-test split
# =============================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# =============================
# 6. Train model
# =============================
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# =============================
# 7. Inputs
# =============================
st.subheader("🔧 Input Values")

input_data = {}
for col in features:
    if df[col].nunique() <= 5:
        input_data[col] = st.selectbox(col, sorted(df[col].unique()))
    else:
        input_data[col] = st.slider(
            col,
            float(df[col].min()),
            float(df[col].max()),
            float(df[col].mean())
        )

input_df = pd.DataFrame([input_data])

# =============================
# 8. Prediction
# =============================
if st.button("🚀 Predict Bike Demand"):
    prediction = model.predict(input_df)
    st.success(f"🚴 Predicted Bike Demand: **{int(prediction[0])} bikes**")

# =============================
# 9. Feature importance
# =============================
st.subheader("📊 Feature Importance")

plt.figure()
plt.barh(features, model.feature_importances_)
plt.xlabel("Importance")
st.pyplot(plt)



# -------------------------------------------------
# Footer
# -------------------------------------------------
st.markdown("---")
st.caption("Created by Group -1 | CHANAKYA, KRISHNA et al. | Built with Streamlit & Random Forest Regressor")
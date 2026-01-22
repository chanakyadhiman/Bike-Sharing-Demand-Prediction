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

# =============================
# 2. Define target & features
# =============================
target = "cnt"

features = [
    "season", "yr", "mnth", "hr",
    "holiday", "workingday", "weathersit",
    "temp", "atemp", "hum", "windspeed"
]

# Keep only columns that exist
features = [c for c in features if c in df.columns]

if target not in df.columns or len(features) == 0:
    st.error("❌ Required columns not found in dataset")
    st.stop()

# =============================
# 3. Prepare data (SAFE)
# =============================
df = df[features + [target]]
df = df.dropna()   # ← ONLY this, no coercion

if df.shape[0] < 10:
    st.error("❌ Dataset too small after cleaning")
    st.stop()

X = df[features]
y = df[target]

# =============================
# 4. Train-test split
# =============================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# =============================
# 5. Train model
# =============================
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# =============================
# 6. Inputs (Auto UI)
# =============================
st.subheader("🔧 Input Conditions")

input_data = {}
for col in features:
    if X[col].nunique() <= 6:
        input_data[col] = st.selectbox(col, sorted(X[col].unique()))
    else:
        input_data[col] = st.slider(
            col,
            float(X[col].min()),
            float(X[col].max()),
            float(X[col].mean())
        )

input_df = pd.DataFrame([input_data])

# =============================
# 7. Prediction
# =============================
if st.button("🚀 Predict Bike Demand"):
    prediction = model.predict(input_df)
    st.success(f"🚴 Predicted Bike Demand: **{int(prediction[0])} bikes**")

# =============================
# 8. Feature importance
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
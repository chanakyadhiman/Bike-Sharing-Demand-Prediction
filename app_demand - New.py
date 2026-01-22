import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

st.set_page_config(page_title="Bike Demand Prediction")
st.title("🚲 Bike Sharing Demand Prediction")

# =============================
# 1. Load Dataset
# =============================
df = pd.read_csv("Dataset.csv")


# =============================
# 2. Target & Feature Selection
# =============================
target = "cnt"

candidate_features = [
    "season", "yr", "mnth", "hr",
    "holiday", "workingday", "weathersit",
    "temp", "atemp", "hum", "windspeed"
]

# Keep only columns that exist
features = [c for c in candidate_features if c in df.columns]

if target not in df.columns or len(features) == 0:
    st.error("❌ Required columns not found in dataset")
    st.stop()

df = df[features + [target]]

# =============================
# 3. Clean Categorical Columns & Encode
# =============================
label_encoders = {}

for col in df.columns:
    if df[col].dtype == "object":
        # Remove invalid rows
        df = df[df[col].notnull()]  # remove NaN
        df = df[df[col] != "?"]     # remove "?"
        df[col] = df[col].astype(str)

        # Encode
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le

# =============================
# 4. Handle Missing Values
# =============================
df = df.dropna()

if df.shape[0] < 10:
    st.error("❌ Not enough valid rows after cleaning")
    st.stop()

X = df[features]
y = df[target]

# =============================
# 5. Train-Test Split
# =============================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# =============================
# 6. Train Model
# =============================
model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)
model.fit(X_train, y_train)

# =============================
# 7. User Inputs (Sliders & Dropdowns)
# =============================
st.subheader("🔧 Input Conditions")
input_data = {}

for col in features:
    if col in label_encoders:
        # Dropdown for categorical columns (show original labels)
        le = label_encoders[col]
        classes = le.classes_
        selected = st.selectbox(col, classes)
        input_data[col] = le.transform([selected])[0]
    else:
        # Slider for numerical columns
        input_data[col] = st.slider(
            col,
            float(X[col].min()),
            float(X[col].max()),
            float(X[col].mean())
        )

input_df = pd.DataFrame([input_data])

# =============================
# 8. Prediction
# =============================
if st.button("🚀 Predict Bike Demand"):
    prediction = model.predict(input_df)
    st.success(f"🚴 Predicted Bike Demand: **{int(prediction[0])} bikes**")

# =============================
# 9. Feature Importance
# =============================
st.subheader("📊 Feature Importance")

plt.figure(figsize=(8,6))
plt.barh(features, model.feature_importances_)
plt.xlabel("Importance")
st.pyplot(plt)





# -------------------------------------------------
# Footer
# -------------------------------------------------
st.markdown("---")

st.caption("Created by Group -1 | CHANAKYA, KRISHNA et al. | Built with Streamlit & Random Forest Regressor")

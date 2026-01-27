import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

st.set_page_config(page_title="Bike Demand Prediction", layout="wide")
st.title("🚲 Bike Sharing Demand Prediction & Analytics")

# =============================
# 1. Load Dataset
# =============================
df = pd.read_csv("Dataset.csv")

# -----------------------------
# Convert Units for Better Interpretation
# -----------------------------

# Convert normalized temperature to Celsius
if "temp" in df.columns:
    df["temp"] = df["temp"] * 41

if "atemp" in df.columns:
    df["atemp"] = df["atemp"] * 50

# Convert normalized windspeed to km/h
if "windspeed" in df.columns:
    df["windspeed"] = df["windspeed"] * 67

# Convert month number to month name
if "mnth" in df.columns:
    month_map = {
        1: "January", 2: "February", 3: "March", 4: "April",
        5: "May", 6: "June", 7: "July", 8: "August",
        9: "September", 10: "October", 11: "November", 12: "December"
    }
    df["mnth"] = df["mnth"].map(month_map)

st.write("📄 Dataset Columns:")
st.write(list(df.columns))

# =============================
# 2. Target & Feature Selection
# =============================
target = "cnt"

candidate_features = [
    "season", "yr", "mnth", "hr",
    "holiday", "workingday", "weathersit",
    "temp", "atemp", "hum", "windspeed"
]

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
        df = df[df[col].notnull()]
        df = df[df[col] != "?"]
        df[col] = df[col].astype(str)

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
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# =============================
# 7. User Inputs for Prediction
# =============================
st.subheader("🔧 Input Conditions")
input_data = {}

for col in features:
    if col in label_encoders:
        le = label_encoders[col]
        classes = le.classes_
        selected = st.selectbox(col, classes)
        input_data[col] = le.transform([selected])[0]
    else:
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

# =============================
# 10. Interactive Column Bar Graph
# =============================
st.subheader("📈 Explore Data with Bar Graph")
column_to_plot = st.selectbox("Select a column to visualize", df.columns)

if column_to_plot in df.columns:
    if column_to_plot in label_encoders:
        df_plot = df.copy()
        le = label_encoders[column_to_plot]
        df_plot[column_to_plot] = le.inverse_transform(df_plot[column_to_plot])
        counts = df_plot[column_to_plot].value_counts()
        st.bar_chart(counts)
    else:
        st.bar_chart(df[column_to_plot].value_counts(bins=20, sort=False))

# =============================
# 11. Bike Demand Graphs
# =============================
st.subheader("🚲 Bike Demand Analysis")

st.write("Histogram of Bike Demand")
st.bar_chart(df['cnt'].value_counts(bins=20, sort=False))

cat_cols = [col for col in df.columns if col in label_encoders]
selected_cat = st.selectbox("Show average demand per category", ["None"] + cat_cols)

if selected_cat != "None":
    df_plot = df.copy()
    le = label_encoders[selected_cat]
    df_plot[selected_cat] = le.inverse_transform(df_plot[selected_cat])
    avg_demand = df_plot.groupby(selected_cat)['cnt'].mean()
    st.bar_chart(avg_demand)

# -------------------------------------------------
# Footer
# -------------------------------------------------
st.markdown("---")
st.caption("Created by Group -1 | CHANAKYA, KRISHNA et al. | Built with Streamlit & Random Forest Regressor")

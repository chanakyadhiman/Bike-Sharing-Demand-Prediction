import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

st.set_page_config(page_title="Bike Sharing Demand Prediction System", layout="wide")
st.title("🚲 Bike Sharing Demand Prediction System")

# =============================
# Load Dataset
# =============================
df = pd.read_csv("Dataset.csv")

# =============================
# Clean dataset
# =============================
df.replace("?", pd.NA, inplace=True)

# Convert numeric columns safely
for col in ["temp", "atemp", "hum", "windspeed", "cnt", "hr", "mnth"]:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

# =============================
# Convert normalized values
# =============================
if df["temp"].max() <= 1:
    df["temp"] = df["temp"] * 41

if df["atemp"].max() <= 1:
    df["atemp"] = df["atemp"] * 50

if df["windspeed"].max() <= 1:
    df["windspeed"] = df["windspeed"] * 67

# =============================
# Encode categorical columns
# =============================
categorical_cols = ["season", "holiday", "workingday", "weathersit"]

label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    label_encoders[col] = le

# =============================
# Feature selection
# =============================
target = "cnt"

features = [
    "season", "yr", "mnth", "hr",
    "holiday", "workingday", "weathersit",
    "temp", "atemp", "hum", "windspeed"
]

df = df[features + [target]]
df = df.dropna()

# =============================
# Train model
# =============================
X = df[features]
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestRegressor(n_estimators=150, random_state=42)
model.fit(X_train, y_train)

# =============================
# USER INPUTS
# =============================
st.subheader("🔧 Enter Conditions")

input_data = {}

for col in features:

    label = col
    if col == "temp":
        label = "Temperature (°C)"
    elif col == "atemp":
        label = "Feels Like Temperature (°C)"
    elif col == "windspeed":
        label = "Windspeed (km/h)"
    elif col == "hum":
        label = "Humidity"
    elif col == "yr":
        label = "Year"
    elif col == "mnth":
        label = "Month (1-12)"
    elif col == "hr":
        label = "Hour (0-23)"

    if col in categorical_cols:
        le = label_encoders[col]
        selected = st.selectbox(label, le.classes_)
        input_data[col] = le.transform([selected])[0]
    else:
        input_data[col] = st.slider(
            label,
            float(X[col].min()),
            float(X[col].max()),
            float(X[col].mean())
        )

input_df = pd.DataFrame([input_data])

# =============================
# Prediction
# =============================
if st.button("🚀 Predict Bike Demand"):
    prediction = model.predict(input_df)
    st.success(f"🚴 Predicted Bike Demand: **{int(prediction[0])} bikes**")

# =============================
# GRAPHS
# =============================
st.subheader("📊 Visualizations")

fig1, ax1 = plt.subplots()
df.groupby("mnth")["cnt"].mean().plot(kind="bar", ax=ax1)
ax1.set_title("Average Bike Demand per Month")
st.pyplot(fig1)

fig2, ax2 = plt.subplots()
df.groupby("hr")["cnt"].mean().plot(kind="line", ax=ax2)
ax2.set_title("Average Bike Demand by Hour")
st.pyplot(fig2)

fig3, ax3 = plt.subplots()
ax3.scatter(df["temp"], df["cnt"])
ax3.set_title("Temperature vs Bike Demand")
ax3.set_xlabel("Temperature (°C)")
ax3.set_ylabel("Bike Demand")
st.pyplot(fig3)


st.markdown("---")
st.caption(Project - "Bike Demand Prediction System | Group-1: Chanakya, Krishna et al. | Streamlit + Random Forest")





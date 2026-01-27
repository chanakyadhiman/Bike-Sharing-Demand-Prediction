import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

st.set_page_config(page_title="Bike Demand Prediction", layout="wide")
st.title("🚲 Bike Sharing Demand Prediction System")

# =============================
# Load Dataset
# =============================
df = pd.read_csv("Dataset.csv")

# =============================
# Convert normalized values to real units
# =============================
if "temp" in df.columns:
    df["temp"] = pd.to_numeric(df["temp"], errors="coerce") * 41

if "atemp" in df.columns:
    df["atemp"] = pd.to_numeric(df["atemp"], errors="coerce") * 50

if "windspeed" in df.columns:
    df["windspeed"] = pd.to_numeric(df["windspeed"], errors="coerce") * 67

# =============================
# Month mapping
# =============================
month_map = {
    1: "January", 2: "February", 3: "March", 4: "April",
    5: "May", 6: "June", 7: "July", 8: "August",
    9: "September", 10: "October", 11: "November", 12: "December"
}

if "mnth" in df.columns:
    df["mnth"] = pd.to_numeric(df["mnth"], errors="coerce")
    df["mnth"] = df["mnth"].map(month_map)

# =============================
# Season mapping
# =============================
season_map = {
    1: "Spring",
    2: "Summer",
    3: "Fall",
    4: "Winter"
}

if "season" in df.columns:
    df["season"] = pd.to_numeric(df["season"], errors="coerce")
    df["season"] = df["season"].map(season_map)

# =============================
# Weekday mapping
# =============================
weekday_map = {
    0: "Sunday",
    1: "Monday",
    2: "Tuesday",
    3: "Wednesday",
    4: "Thursday",
    5: "Friday",
    6: "Saturday"
}

if "weekday" in df.columns:
    df["weekday"] = pd.to_numeric(df["weekday"], errors="coerce")
    df["weekday"] = df["weekday"].map(weekday_map)

# =============================
# Feature selection
# =============================
target = "cnt"

features = [
    "season", "yr", "mnth", "weekday", "hr",
    "holiday", "workingday", "weathersit",
    "temp", "atemp", "hum", "windspeed"
]

features = [f for f in features if f in df.columns]
df = df[features + [target]]

# =============================
# Drop NaNs BEFORE encoding
# =============================
df = df.dropna()

# =============================
# Encode categorical columns
# =============================
categorical_cols = ["season", "mnth", "weekday", "holiday", "workingday", "weathersit"]
label_encoders = {}

for col in categorical_cols:
    if col in df.columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le

# =============================
# Force numeric
# =============================
for col in df.columns:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df = df.dropna()

X = df[features]
y = df[target]

# =============================
# Train model
# =============================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# =============================
# USER INPUTS
# =============================
st.subheader("🔧 Input Conditions")
input_data = {}

month_order = list(month_map.values())
season_order = list(season_map.values())
weekday_order = list(weekday_map.values())

for col in features:

    label = col
    if col == "temp":
        label = "Temperature (°C)"
    elif col == "atemp":
        label = "Feels Like Temperature (°C)"
    elif col == "windspeed":
        label = "Windspeed (km/h)"
    elif col == "hum":
        label = "Humidity (%)"

    if col in label_encoders:
        le = label_encoders[col]

        if col == "mnth":
            options = [m for m in month_order if m in le.classes_]
        elif col == "season":
            options = [s for s in season_order if s in le.classes_]
        elif col == "weekday":
            options = [d for d in weekday_order if d in le.classes_]
        else:
            options = list(le.classes_)

        selected = st.selectbox(label, options)
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
# INTERACTIVE GRAPHS
# =============================
st.subheader("📊 Interactive Data Visualizations")

col1, col2 = st.columns(2)

with col1:
    fig1 = px.bar(df, x="mnth", y="cnt", title="Average Bike Demand per Month")
    st.plotly_chart(fig1, use_container_width=True)

with col2:
    fig2 = px.bar(df, x="weekday", y="cnt", title="Average Bike Demand per Weekday")
    st.plotly_chart(fig2, use_container_width=True)

fig3 = px.scatter(df, x="temp", y="cnt", title="Temperature vs Bike Demand")
st.plotly_chart(fig3, use_container_width=True)

st.markdown("---")
st.caption("Bike Demand Prediction System | Streamlit + Random Forest + Plotly")

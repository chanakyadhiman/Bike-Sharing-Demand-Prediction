import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# -------------------------------------------------
# Streamlit Config
# -------------------------------------------------
st.set_page_config(page_title="Bike Demand Prediction", layout="centered")
st.title("🚲 Bike Sharing Demand Prediction")
st.markdown("Predict hourly bike rental demand based on weather & time factors")

# -------------------------------------------------
# Helpers
# -------------------------------------------------
def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    # Lowercase, strip, collapse spaces, remove common punctuation
    df = df.copy()
    df.columns = (
        df.columns.str.strip()
        .str.lower()
        .str.replace(r"\s+", " ", regex=True)
        .str.replace("[()%-/]", "", regex=True)
    )
    return df

def map_columns(df: pd.DataFrame) -> pd.DataFrame:
    # Try multiple aliases for each feature across common bike datasets
    aliases = {
        "hr": ["hr", "hour", "hours"],
        "temp": ["temp", "temperature c", "temperature", "atemp", "feelslike"],
        "hum": ["hum", "humidity", "humidity percent"],
        "windspeed": ["windspeed", "wind speed ms", "wind speed", "windspeed ms"],
        "season": ["season", "seasons"],
        "holiday": ["holiday", "is holiday"],
        "workingday": ["workingday", "functioning day", "working day"],
        "cnt": ["cnt", "rented bike count", "count", "total count"]
    }
    col_map = {}
    for canonical, candidates in aliases.items():
        for c in candidates:
            if c in df.columns:
                col_map[c] = canonical
                break

    df = df.rename(columns=col_map)
    return df

def coerce_binary(series: pd.Series) -> pd.Series:
    # Map yes/no, true/false, y/n, strings/numbers to 0/1
    s = series.astype(str).str.strip().str.lower()
    mapping = {
        "yes": 1, "no": 0,
        "true": 1, "false": 0,
        "y": 1, "n": 0,
        "1": 1, "0": 0
    }
    return s.map(mapping).astype(float)

def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = normalize_columns(df)
    df = map_columns(df)

    # Required columns
    required = ["hr", "temp", "hum", "windspeed", "season", "holiday", "workingday", "cnt"]
    present = [c for c in required if c in df.columns]
    missing = [c for c in required if c not in df.columns]

    # Show inspector
    with st.expander("🔎 Column inspector"):
        st.write("**Normalized columns:**", list(df.columns))
        st.write("**Present (mapped):**", present)
        st.write("**Missing (expected):**", missing)

    # If target missing, we can still allow inference later if user uploads a trained model,
    # but for this app we require 'cnt' to train.
    if "cnt" not in df.columns:
        st.error("❌ Target column 'cnt' (rented bike count) not found after mapping.")
        return pd.DataFrame()

 # Binary conversions for holiday & workingday if present
    if "holiday" in df.columns:
        df["holiday"] = coerce_binary(df["holiday"])
    if "workingday" in df.columns:
        df["workingday"] = coerce_binary(df["workingday"])

    # Season: try to coerce categorical strings to numeric 1–4 if needed
    if "season" in df.columns:
        s = df["season"].astype(str).str.strip().str.lower()
        season_map = {
            "spring": 1, "summer": 2, "autumn": 3, "fall": 3, "winter": 4
        }
        df["season"] = np.where(
            s.isin(season_map.keys()),
            s.map(season_map),
            pd.to_numeric(df["season"], errors="coerce")
        )

    # Keep only the columns we need (drop extras safely)
    keep = ["hr", "temp", "hum", "windspeed", "season", "holiday", "workingday", "cnt"]
    df = df[[c for c in keep if c in df.columns]]

    # Convert numerics
    for c in ["hr", "temp", "hum", "windspeed", "season", "holiday", "workingday", "cnt"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # Drop impossible hours or seasons if present
    if "hr" in df.columns:
        df = df[(df["hr"] >= 0) & (df["hr"] <= 23)]
    if "season" in df.columns:
        df = df[(df["season"] >= 1) & (df["season"] <= 4)]

    # Final NA drop
    df = df.dropna(subset=["hr", "temp", "hum", "windspeed", "season", "holiday", "workingday", "cnt"])

    return df

@st.cache_data(show_spinner=False)
def load_data(uploaded_file) -> pd.DataFrame:
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
    else:
        # Fallback to local file if present in repo
        try:
            df = pd.read_csv("Dataset.csv")
        except Exception:
            st.warning("No uploaded file and local 'Dataset.csv' not found.")
            return pd.DataFrame()
    return df

@st.cache_resource(show_spinner=False)
def train_model(df: pd.DataFrame):
    features = ["hr", "temp", "hum", "windspeed", "season", "holiday", "workingday"]
    X = df[features]
    y = df["cnt"]

    model = RandomForestRegressor(
        n_estimators=300,
        max_depth=10,
        random_state=42,
        n_jobs=-1
    )
    # Optional: quick split to validate training sanity
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    model.fit(X_train, y_train)
    return model

# -------------------------------------------------
# Data Ingestion
# -------------------------------------------------
st.sidebar.header("📥 Data")
uploaded = st.sidebar.file_uploader("Upload Dataset.csv", type=["csv"])

raw_df = load_data(uploaded)
if raw_df.empty:
    st.stop()

df = clean_dataframe(raw_df)
if df.empty:
    st.stop()

# -------------------------------------------------
# Train Model (ONCE PER SESSION)
# -------------------------------------------------
if "model" not in st.session_state:
    try:
        st.session_state.model = train_model(df)
    except Exception as e:
        st.error(f"❌ Model training failed: {e}")
        st.stop()

model = st.session_state.model

# -------------------------------------------------
# Sidebar Inputs
# -------------------------------------------------
st.sidebar.header("🔧 Input Features")

hr = st.sidebar.slider("Hour of Day", 0, 23, 12)
temp = st.sidebar.slider("Temperature (°C)", 0.0, 45.0, 20.0)
hum = st.sidebar.slider("Humidity (%)", 0, 100, 50)
windspeed = st.sidebar.slider("Windspeed (m/s)", 0.0, 60.0, 10.0)
season = st.sidebar.selectbox("Season (1:Spring, 2:Summer, 3:Autumn, 4:Winter)", [1, 2, 3, 4], index=0)
holiday = st.sidebar.selectbox("Holiday", [0, 1], index=0)
workingday = st.sidebar.selectbox("Working Day", [0, 1], index=1)

input_df = pd.DataFrame({
    "hr": [hr],
    "temp": [temp],
    "hum": [hum],
    "windspeed": [windspeed],
    "season": [season],
    "holiday": [holiday],
    "workingday": [workingday]
})

st.subheader("📊 Input Data")
st.write(input_df)

# -------------------------------------------------
# Prediction
# -------------------------------------------------
if st.button("Predict Demand"):
    try:
        prediction = model.predict(input_df)
        st.success(f"🚴 Predicted Bike Demand: **{int(round(prediction[0]))} bikes**")
    except Exception as e:
        st.error(f"❌ Prediction failed: {e}")

# -------------------------------------------------
# Footer
# -------------------------------------------------
st.markdown("---")
st.caption("Created by (Group -1) | Chanakya Dhiman, Krishna Mohith, et al. | Built with Streamlit & Random Forest Regressor")

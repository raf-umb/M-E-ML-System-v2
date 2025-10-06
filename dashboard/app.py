import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

st.set_page_config(page_title="M&E ML Dashboard", layout="wide")
st.title("📊 M&E Predictive Dashboard with Map")

processed = Path("data/processed/mne_data.csv")
if not processed.exists():
    st.warning("Processed data not found. Run `python src/preprocess.py` first.")
else:
    df = pd.read_csv(processed)

    # Load model if available
    model_path = Path("models/mne_success_model.pkl")
    if model_path.exists():
        model = joblib.load(model_path)
    else:
        model = None

    st.sidebar.header("Manual Prediction (one-off)")
    budget = st.sidebar.number_input("Project Budget (₱)", min_value=0, value=100000)
    staff = st.sidebar.number_input("Staff Count", min_value=0, value=5)
    risk = st.sidebar.slider("Region Risk Index", 0.0, 10.0, 3.0)
    completion = st.sidebar.slider("Activity Completion (%)", 0, 100, 75)

    if st.sidebar.button("Predict Success") and model is not None:
        X_new = pd.DataFrame([[budget, staff, risk, completion]],
                             columns=["budget", "staff_count", "region_risk_index", "activity_completion_rate"])
        pred = model.predict_proba(X_new)[0, 1]
        st.sidebar.metric("Predicted Success Probability", f"{pred*100:.2f}%")
    elif st.sidebar.button("Predict Success") and model is None:
        st.sidebar.error("Model not found. Run `python src/train_model.py` to train the demo model.")

    st.subheader("📍 Project Site Locations")
    if {"latitude","longitude"}.issubset(set(df.columns)):
        st.map(df[["latitude","longitude"]])
    else:
        st.write("No latitude/longitude columns found in the data.")

    st.subheader("Project Data")
    st.dataframe(df)

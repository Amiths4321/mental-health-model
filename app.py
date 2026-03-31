import streamlit as st
import joblib
import numpy as np
import pandas as pd

st.title("🧠 Mental Health Prediction")

pipeline = joblib.load(
    "models/mental_health_pipeline.pkl"
)

df = pd.read_csv(
    "data/processed/mental_health_processed.csv"
)

feature_columns = df.drop(
    "treatment",
    axis=1
).columns

st.header("Enter Input Values")

input_data = []

for col in feature_columns:

    value = st.number_input(
        col,
        value=0.0
    )

    input_data.append(value)

if st.button("Predict"):

    data = np.array(
        input_data
    ).reshape(1, -1)

    prediction = pipeline.predict(
        data
    )[0]

    probability = pipeline.predict_proba(
        data
    )[0][1]

    if prediction == 1:

        st.error(
            f"⚠️ Risk Detected ({probability:.2f})"
        )

    else:

        st.success(
            f"✅ Low Risk ({probability:.2f})"
        )
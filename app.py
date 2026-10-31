
import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

# ---------------- PAGE CONFIGURATION ----------------
st.set_page_config(
    page_title="Heart Disease Prediction",
    page_icon="❤️",
    layout="wide"
)

# ---------------- LOAD MODEL ----------------
MODEL_PATH = (
    Path(__file__).parent
    / "saved_models"
    / "heart_disease_xgboost.pkl"
)

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

model = load_model()

# ---------------- HEADER ----------------
st.title("❤️ Heart Disease Prediction")
st.markdown(
    "### Machine Learning Based Risk Prediction"
)
st.write(
    "Enter the clinical parameters below to generate "
    "a prediction using the trained XGBoost model."
)

st.divider()

# ---------------- INPUT FORM ----------------
st.subheader("Patient Information")

with st.form("heart_prediction_form"):

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("#### Personal Details")

        age = st.number_input(
            "Age",
            min_value=1,
            max_value=120,
            value=45
        )

        sex = st.selectbox(
            "Sex (dataset code)",
            [0, 1]
        )

        chest_pain_type = st.selectbox(
            "Chest Pain Type (dataset code)",
            [1.0, 2.0, 3.0, 4.0]
        )

        resting_blood_pressure = st.number_input(
            "Resting Blood Pressure",
            min_value=50,
            max_value=250,
            value=120
        )

        cholesterol = st.number_input(
            "Cholesterol",
            min_value=50,
            max_value=600,
            value=200
        )

    with col2:
        st.markdown("#### Clinical Measurements")

        fasting_blood_sugar_gt_120 = st.selectbox(
            "Fasting Blood Sugar > 120 (code)",
            [0, 1]
        )

        resting_ecg = st.selectbox(
            "Resting ECG (dataset code)",
            [0, 1, 2]
        )

        max_heart_rate = st.number_input(
            "Maximum Heart Rate",
            min_value=40,
            max_value=250,
            value=150
        )

        exercise_induced_angina = st.selectbox(
            "Exercise Induced Angina (code)",
            [0, 1]
        )

    with col3:
        st.markdown("#### Additional Parameters")

        oldpeak = st.number_input(
            "Oldpeak",
            min_value=0.0,
            max_value=10.0,
            value=1.0,
            step=0.1
        )

        st_slope = st.selectbox(
            "ST Slope (dataset code)",
            [1, 2, 3]
        )

        num_major_vessels = st.selectbox(
            "Number of Major Vessels",
            [0, 1, 2, 3, 4]
        )

        thalassemia = st.selectbox(
            "Thalassemia (dataset code)",
            [3.0, 6.0, 7.0]
        )

    st.divider()

    submitted = st.form_submit_button(
        "Predict Heart Disease",
        use_container_width=True
    )

# ---------------- PREDICTION ----------------
if submitted:

    input_data = pd.DataFrame([{
        "age": age,
        "sex": sex,
        "chest_pain_type": chest_pain_type,
        "resting_blood_pressure": resting_blood_pressure,
        "cholesterol": cholesterol,
        "fasting_blood_sugar_gt_120": fasting_blood_sugar_gt_120,
        "resting_ecg": resting_ecg,
        "max_heart_rate": max_heart_rate,
        "exercise_induced_angina": exercise_induced_angina,
        "oldpeak": oldpeak,
        "st_slope": st_slope,
        "num_major_vessels": num_major_vessels,
        "thalassemia": thalassemia
    }])

    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    st.divider()
    st.subheader("Prediction Results")

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "Estimated probability of class 1",
            f"{probability * 100:.2f}%"
        )

    with col2:
        if prediction == 1:
            st.error("Model prediction: Class 1")
        else:
            st.success("Model prediction: Class 0")

    st.progress(float(probability))

    st.caption(
        "The probability represents the model's estimated "
        "probability for class 1, not a clinically validated "
        "measure of an individual's medical risk."
    )

    with st.expander("View submitted patient data"):
        st.dataframe(input_data, use_container_width=True)

# ---------------- DISCLAIMER ----------------
st.divider()

st.warning(
    "Educational demonstration only. This project uses a "
    "synthetic dataset and is not clinically validated. "
    "It must not be used for medical diagnosis or treatment."
)

st.caption("Built using Python, XGBoost and Streamlit.")

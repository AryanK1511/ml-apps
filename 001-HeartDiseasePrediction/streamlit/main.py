import pickle

import pandas as pd
import streamlit as st

with open("model/classification_model.pkl", "rb") as f:
    model = pickle.load(f)

st.title("Heart Disease Prediction")
st.write(
    "Enter the patient data below to predict whether heart disease is likely or unlikely."
)

feature_names = [
    "age",
    "sex",
    "cp",
    "trestbps",
    "chol",
    "fbs",
    "restecg",
    "thalach",
    "exang",
    "oldpeak",
    "slope",
    "ca",
    "thal",
]

with st.form(key="heart_disease_form"):
    age = float(
        st.number_input("Age (in years)", min_value=1, max_value=120, value=63, step=1)
    )
    sex = st.radio("Sex", options=["Female", "Male"], index=1)
    sex = 1.0 if sex == "Male" else 0.0

    cp = st.selectbox(
        "Chest Pain Type",
        options=[
            "Typical angina",
            "Atypical angina",
            "Non-anginal pain",
            "Asymptomatic",
        ],
    )
    cp_map = {
        "Typical angina": 0.0,
        "Atypical angina": 1.0,
        "Non-anginal pain": 2.0,
        "Asymptomatic": 3.0,
    }
    cp = float(cp_map[cp])

    trestbps = float(
        st.number_input(
            "Resting Blood Pressure (in mm Hg)",
            min_value=0.0,
            max_value=200.0,
            value=145.0,
        )
    )
    chol = float(
        st.number_input(
            "Serum Cholesterol (in mg/dl)", min_value=0.0, max_value=600.0, value=233.0
        )
    )
    fbs = st.radio("Fasting Blood Sugar > 120 mg/dl", options=["No", "Yes"])
    fbs = 1.0 if fbs == "Yes" else 0.0

    restecg = st.selectbox(
        "Resting Electrocardiographic Results",
        options=[
            "Nothing to note",
            "ST-T Wave abnormality",
            "Left ventricular hypertrophy",
        ],
    )
    restecg_map = {
        "Nothing to note": 0.0,
        "ST-T Wave abnormality": 1.0,
        "Left ventricular hypertrophy": 2.0,
    }
    restecg = float(restecg_map[restecg])

    thalach = float(
        st.number_input(
            "Maximum Heart Rate Achieved", min_value=0.0, max_value=250.0, value=150.0
        )
    )
    exang = st.radio("Exercise Induced Angina", options=["No", "Yes"])
    exang = 1.0 if exang == "Yes" else 0.0

    oldpeak = float(
        st.number_input(
            "ST Depression Induced by Exercise",
            min_value=0.0,
            max_value=10.0,
            value=2.3,
        )
    )
    slope = st.selectbox(
        "Slope of the Peak Exercise ST Segment",
        options=["Upsloping", "Flat", "Downsloping"],
    )
    slope_map = {"Upsloping": 0.0, "Flat": 1.0, "Downsloping": 2.0}
    slope = float(slope_map[slope])

    ca = float(
        st.selectbox(
            "Number of Major Vessels Colored by Fluoroscopy",
            options=[0.0, 1.0, 2.0, 3.0],
        )
    )

    thal = st.selectbox(
        "Thalium Stress Result", options=["Normal", "Fixed defect", "Reversible defect"]
    )
    thal_map = {"Normal": 1.0, "Fixed defect": 6.0, "Reversible defect": 7.0}
    thal = float(thal_map[thal])

    submit_button = st.form_submit_button("Predict")

if submit_button:
    input_data = pd.DataFrame(
        [
            [
                age,
                sex,
                cp,
                trestbps,
                chol,
                fbs,
                restecg,
                thalach,
                exang,
                oldpeak,
                slope,
                ca,
                thal,
            ]
        ],
        columns=feature_names,
    )

    prediction = model.predict(input_data)

    if prediction == 1:
        st.write("Heart Disease Likely")
    else:
        st.write("Heart Disease Unlikely")

    st.write("Input Data:")
    st.dataframe(input_data)

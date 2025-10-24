import streamlit as st
import requests

st.set_page_config(page_title="Diabetes Prediction", layout="centered")
st.title("🩺 Diabetes Prediction App")

st.write("Enter patient details to predict diabetes risk:")

# Input fields
Pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20, step=1)
Glucose = st.number_input("Glucose Level", min_value=0)
BloodPressure = st.number_input("Blood Pressure", min_value=0)
SkinThickness = st.number_input("Skin Thickness", min_value=0)
Insulin = st.number_input("Insulin", min_value=0)
BMI = st.number_input("BMI", min_value=0.0)
DiabetesPedigreeFunction = st.number_input("Diabetes Pedigree Function", min_value=0.0)
Age = st.number_input("Age", min_value=1, max_value=120, step=1)

if st.button("Predict"):
    payload = {
        "Pregnancies": Pregnancies,
        "Glucose": Glucose,
        "BloodPressure": BloodPressure,
        "SkinThickness": SkinThickness,
        "Insulin": Insulin,
        "BMI": BMI,
        "DiabetesPedigreeFunction": DiabetesPedigreeFunction,
        "Age": Age
    }
    try:
        response = requests.post("https://diabetesapp-n440.onrender.com/predict", json=payload)
        if response.status_code == 200:
            result = response.json()
            if result["diabetes_risk"]:
                st.error(f"⚠️ Likely Diabetic (Probability: {result['probability']})")
            else:
                st.success(f"✅ Non-Diabetic (Probability: {result['probability']})")
        else:
            st.warning(f"Error: {response.json()['detail']}")
    except requests.exceptions.RequestException:
        st.error("❌ Could not connect to API. Is FastAPI running?")

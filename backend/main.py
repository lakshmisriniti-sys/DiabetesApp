# backend/main.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import joblib
import numpy as np
import os

app = FastAPI(title="Diabetes Prediction API")

# Initialize model and scaler
model = None
scaler = None

@app.on_event("startup")
def load_model():
    global model, scaler
    model_path = os.path.join(os.path.dirname(__file__), "diabetes_model.pkl")
    scaler_path = os.path.join(os.path.dirname(__file__), "scaler.pkl")

    if os.path.exists(model_path) and os.path.exists(scaler_path):
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        print("✅ Model and scaler loaded successfully.")
    else:
        raise FileNotFoundError("Model or scaler not found in backend folder.")

# Input schema with validation
class PatientData(BaseModel):
    Pregnancies: int = Field(..., ge=0, le=20)
    Glucose: float = Field(..., gt=0)
    BloodPressure: float = Field(..., gt=0)
    SkinThickness: float = Field(..., ge=0)
    Insulin: float = Field(..., ge=0)
    BMI: float = Field(..., gt=0)
    DiabetesPedigreeFunction: float = Field(..., ge=0)
    Age: int = Field(..., ge=1, le=120)

# Prediction endpoint
@app.post("/predict")
def predict_diabetes(data: PatientData):
    if model is None or scaler is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    features = np.array([[data.Pregnancies, data.Glucose, data.BloodPressure,
                          data.SkinThickness, data.Insulin, data.BMI,
                          data.DiabetesPedigreeFunction, data.Age]])
    try:
        features_scaled = scaler.transform(features)
        prediction = model.predict(features_scaled)[0]
        probability = model.predict_proba(features_scaled)[0][1]

        return {
            "diabetes_risk": bool(prediction),
            "probability": round(float(probability), 3)
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing input: {e}")

# Root endpoint
@app.get("/")
def root():
    return {"message": "Diabetes Prediction API is running."}

# Auto-open Swagger UI when running directly
if __name__ == "__main__":
    import uvicorn, webbrowser, threading, time

    url = "http://127.0.0.1:8000/docs"
    def open_browser():
        time.sleep(1)
        webbrowser.open(url)

    threading.Thread(target=open_browser).start()
    uvicorn.run("backend.main:app", host="127.0.0.1", port=8000, reload=True)
if __name__ == "__main__":
    import uvicorn
    import webbrowser
    import threading
    import time

    url = "http://127.0.0.1:8000/docs"

    def open_browser():
        time.sleep(1)  # wait for server to start
        webbrowser.open(url)

    threading.Thread(target=open_browser).start()

    uvicorn.run("backend.main:app", host="127.0.0.1", port=8000, reload=True)

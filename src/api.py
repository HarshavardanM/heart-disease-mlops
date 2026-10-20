
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "artifacts" / "heart_disease_pipeline.joblib"

model = joblib.load(MODEL_PATH)

app = FastAPI(
    title="Heart Disease Prediction API",
    description=(
        "Educational demonstration using the UCI Cleveland "
        "Heart Disease dataset. Not for clinical diagnosis."
    ),
    version="1.0.0",
)


class PatientFeatures(BaseModel):
    age: float = Field(ge=0, le=120)
    sex: int = Field(ge=0, le=1)
    cp: int = Field(ge=1, le=4)
    trestbps: float = Field(gt=0)
    chol: float = Field(ge=0)
    fbs: int = Field(ge=0, le=1)
    restecg: int = Field(ge=0, le=2)
    thalach: float = Field(gt=0)
    exang: int = Field(ge=0, le=1)
    oldpeak: float
    slope: int = Field(ge=1, le=3)
    ca: float = Field(ge=0, le=3)
    thal: float = Field(ge=3, le=7)


@app.get("/")
def root():
    return {
        "message": "Heart Disease Prediction API",
        "docs": "/docs",
        "warning": "Educational use only; not a clinical diagnostic tool.",
    }


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None}


@app.post("/predict")
def predict(patient: PatientFeatures):
    try:
        features = pd.DataFrame([patient.model_dump()])
        prediction = int(model.predict(features)[0])
        probabilities = model.predict_proba(features)[0]

        return {
            "prediction": prediction,
            "label": (
                "disease_present" if prediction == 1 else "no_disease"
            ),
            "probability_no_disease": round(float(probabilities[0]), 4),
            "probability_disease_present": round(float(probabilities[1]), 4),
            "notice": "Educational demonstration only; not medical advice.",
        }
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail="Prediction failed. Check the application logs.",
        ) from exc
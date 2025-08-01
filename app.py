from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import json

# === Load model and metadata ===
model = joblib.load("model.pkl")
labels = joblib.load("labels.pkl")

with open("symptom_to_index.json") as f:
    symptom_to_index = json.load(f)

# === Create app ===
app = FastAPI()

# Enable frontend access (CORS)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change to your frontend domain in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define request format
class SymptomRequest(BaseModel):
    symptom_names: list[str]

# === Predict endpoint ===
@app.post("/predict")
def predict(request: SymptomRequest):
    input_vector = [0] * len(symptom_to_index)

    for name in request.symptom_names:
        key = name.strip().lower()
        if key in symptom_to_index:
            input_vector[symptom_to_index[key]] = 1

    if sum(input_vector) == 0:
        return {"error": "No valid symptoms provided."}

    proba = model.predict_proba([input_vector])[0]

    ranked = sorted(
        [
            {"disease": labels[i], "confidence": round(prob, 4)}
            for i, prob in enumerate(proba)
            if prob > 0
        ],
        key=lambda x: x["confidence"],
        reverse=True
    )

    return {
        "top_prediction": ranked[0]["disease"],
        "confidence": ranked[0]["confidence"],
        "ranked_predictions": ranked
    }

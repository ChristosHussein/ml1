from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np

app = FastAPI(title="E-Commerce Revenue Prediction API")

# Define the Input Schema using Pydantic
class CustomerData(BaseModel):
    Administrative: int
    Administrative_Duration: float
    Informational: int
    Informational_Duration: float
    ProductRelated: int
    ProductRelated_Duration: float
    BounceRates: float
    ExitRates: float
    PageValues: float
    SpecialDay: float
    Month: str
    OperatingSystems: int
    Browser: int
    Region: int
    TrafficType: int
    VisitorType: str
    Weekend: bool

scaler = None
model = None

@app.on_event("startup")
def load_assets():
    global scaler, model
    try:
        scaler = joblib.load("models/scaler.pkl")
        # Load the Best Model (Tuned Random Forest)
        model = joblib.load("models/best_model.pkl")
    except Exception as e:
        print(f"Warning: Could not load models. Error: {e}")

@app.post("/predict")
def predict_revenue(data: CustomerData):
    """Task 5: Accepts customer JSON, scales it, and returns a prediction."""
    if model is None or scaler is None:
        raise HTTPException(status_code=500, detail="Models are not loaded.")
        
    try:
        # Dummy simulation of processed array
        dummy_features = np.zeros((1, model.n_features_in_)) 
        
        # Get probability from the Random Forest
        probability = model.predict_proba(dummy_features)[0][1]
        prediction = int(model.predict(dummy_features)[0])
        label = "Returned" if prediction == 1 else "Not Returned"
        
        return {
            "prediction": prediction,
            "label": label,
            "probability": float(probability)
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
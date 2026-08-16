import joblib
import pandas as pd

from fastapi import FastAPI
from pydantic import BaseModel


# =========================
# 1. LOAD MODEL
# =========================
MODEL_PATH = "./models/best_model.pkl"
model = joblib.load(MODEL_PATH)


# =========================
# 2. CREATE API
# =========================
app = FastAPI(
    title="House Price Prediction API",
    description="API dự đoán giá nhà sử dụng RandomForestRegressor",
    version="1.0.0"
)


# =========================
# 3. INPUT SCHEMA
# =========================
class HouseFeatures(BaseModel):
    MedInc: float
    HouseAge: float
    AveRooms: float
    AveBedrms: float
    Population: float
    AveOccup: float
    Latitude: float
    Longitude: float


# =========================
# 4. HEALTH CHECK
# =========================
@app.get("/")
def root():
    return {
        "message": "House Price Prediction API is running"
    }


# =========================
# 5. PREDICTION
# =========================
@app.post("/predict")
def predict(data: HouseFeatures):

    input_data = pd.DataFrame([data.model_dump()])

    prediction = model.predict(input_data)

    return {
        "predicted_price": float(prediction[0])
    }
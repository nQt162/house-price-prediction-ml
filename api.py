# api.py

from fastapi import FastAPI
import joblib

app = FastAPI()

# Load model khi API khởi động
model = joblib.load("./models/best_model.pkl")


@app.get("/")
def root():
    return {"message": "API is running"}


@app.post("/predict")
def predict(data: list[float]):
    prediction = model.predict([data])

    return {
        "prediction": prediction[0]
    }
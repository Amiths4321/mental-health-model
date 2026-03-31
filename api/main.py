from fastapi import FastAPI
import joblib
import numpy as np

app = FastAPI()

model = joblib.load(
    "models/mental_health_pipeline.pkl"
)


@app.get("/")
def home():

    return {"message": "Mental Health API Running"}


@app.post("/predict")
def predict(data: list):

    data = np.array(data).reshape(1, -1)

    prediction = model.predict(data)[0]

    probability = model.predict_proba(data)[0][1]

    return {

        "prediction": int(prediction),

        "probability": float(probability)

    }
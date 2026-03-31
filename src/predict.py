import joblib
import numpy as np

model = joblib.load(
    "models/mental_health_pipeline.pkl"
)


def predict(data):

    data = np.array(data).reshape(1, -1)

    prediction = model.predict(data)[0]

    probability = model.predict_proba(data)[0][1]

    return prediction, probability
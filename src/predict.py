import joblib
import numpy as np

model = joblib.load(
    "models/mental_health_model.pkl"
)

selector = joblib.load(
    "models/feature_selector.pkl"
)


def predict(data):

    data = np.array(data).reshape(1, -1)

    data_selected = selector.transform(data)

    prediction = model.predict(
        data_selected
    )

    probability = model.predict_proba(
        data_selected
    )

    return prediction[0], probability[0][1]
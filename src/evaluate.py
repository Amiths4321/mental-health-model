import pandas as pd
import joblib

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

TARGET_COLUMN = "treatment"


def evaluate():

    df = pd.read_csv(
        "data/processed/mental_health_processed.csv"
    )

    X = df.drop(TARGET_COLUMN, axis=1)

    y = df[TARGET_COLUMN]

    model = joblib.load(
        "models/mental_health_pipeline.pkl"
    )

    y_pred = model.predict(X)

    print("\nClassification Report")

    print(
        classification_report(y, y_pred)
    )

    print("\nConfusion Matrix")

    print(
        confusion_matrix(y, y_pred)
    )


if __name__ == "__main__":
    evaluate()
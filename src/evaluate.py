import pandas as pd
import joblib

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    classification_report,
    confusion_matrix
)

TARGET_COLUMN = "treatment"


def evaluate():

    print("Loading data...")

    df = pd.read_csv(
        "data/processed/mental_health_processed.csv"
    )

    X = df.drop(TARGET_COLUMN, axis=1)
    y = df[TARGET_COLUMN]

    # Load selector
    selector = joblib.load(
        "models/feature_selector.pkl"
    )

    X_selected = selector.transform(X)

    # Load model
    model = joblib.load(
        "models/mental_health_model.pkl"
    )

    y_pred = model.predict(X_selected)

    y_prob = model.predict_proba(X_selected)[:, 1]

    print("\nAccuracy:")
    print(accuracy_score(y, y_pred))

    print("\nROC-AUC:")
    print(roc_auc_score(y, y_prob))

    print("\nClassification Report:")
    print(classification_report(y, y_pred))

    print("\nConfusion Matrix:")
    print(confusion_matrix(y, y_pred))


if __name__ == "__main__":
    evaluate()
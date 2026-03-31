import pandas as pd
import joblib

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    roc_auc_score
)

from sklearn.model_selection import train_test_split

TARGET_COLUMN = "treatment"

def evaluate():

    df = pd.read_csv(
        "data/processed/mental_health_processed.csv"
    )

    X = df.drop(TARGET_COLUMN, axis=1)
    y = df[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    model = joblib.load(
        "models/mental_health_model.pkl"
    )

    y_pred = model.predict(X_test)

    y_prob = model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(
        y_test,
        y_pred
    )

    roc = roc_auc_score(
        y_test,
        y_prob
    )

    print("Accuracy:", acc)
    print("ROC-AUC:", roc)

    print("\nClassification Report:\n")

    print(
        classification_report(
            y_test,
            y_pred
        )
    )


if __name__ == "__main__":
    evaluate()
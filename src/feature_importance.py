import pandas as pd
import joblib
import matplotlib.pyplot as plt

TARGET_COLUMN = "treatment"


def plot_importance():

    df = pd.read_csv(
        "data/processed/mental_health_processed.csv"
    )

    X = df.drop(TARGET_COLUMN, axis=1)

    selector = joblib.load(
        "models/feature_selector.pkl"
    )

    model = joblib.load(
        "models/mental_health_model.pkl"
    )

    selected_features = X.columns[
        selector.get_support()
    ]

    importance = model.feature_importances_

    plt.figure()

    plt.barh(
        selected_features,
        importance
    )

    plt.title("Feature Importance")

    plt.tight_layout()

    plt.show()


if __name__ == "__main__":
    plot_importance()
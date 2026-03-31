import pandas as pd
import joblib
import shap

def explain():

    df = pd.read_csv(
        "data/processed/mental_health_processed.csv"
    )

    X = df.drop("treatment", axis=1)

    model = joblib.load(
        "models/mental_health_model.pkl"
    )

    explainer = shap.TreeExplainer(model)

    shap_values = explainer.shap_values(X)

    # Summary Plot
    shap.summary_plot(
        shap_values,
        X
    )


if __name__ == "__main__":
    explain()
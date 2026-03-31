import pandas as pd
import joblib

from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from sklearn.feature_selection import chi2

from xgboost import XGBClassifier

TARGET_COLUMN = "treatment"


def tune():

    print("Loading data...")

    df = pd.read_csv(
        "data/processed/mental_health_processed.csv"
    )

    # Split features and target
    X = df.drop(TARGET_COLUMN, axis=1)
    y = df[TARGET_COLUMN]

    print("Original Features:", X.shape[1])

    # -------------------------
    # Feature Selection
    # -------------------------

    selector = SelectKBest(
        score_func=f_classif,
        k=15
    )

    X_selected = selector.fit_transform(X, y)

    print("Selected Features:", X_selected.shape[1])

    # -------------------------
    # Hyperparameter Grid
    # -------------------------

    param_grid = {

        "n_estimators": [200, 300],

        "max_depth": [4, 6],

        "learning_rate": [0.05, 0.1]

    }

    model = XGBClassifier(
        random_state=42,
        eval_metric="logloss"
    )

    grid = GridSearchCV(
        model,
        param_grid,
        cv=3,
        scoring="roc_auc",
        verbose=1
    )

    print("Training Grid Search...")

    grid.fit(X_selected, y)

    print("\n✅ Best Parameters:")

    print(grid.best_params_)

    # Save model
    joblib.dump(
        grid.best_estimator_,
        "models/mental_health_model.pkl"
    )

    # Save selector (IMPORTANT)
    joblib.dump(
        selector,
        "models/feature_selector.pkl"
    )

    print("✅ Tuned model saved!")


if __name__ == "__main__":
    tune()
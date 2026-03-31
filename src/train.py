import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif

from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score

from xgboost import XGBClassifier

TARGET_COLUMN = "treatment"


def train():

    print("Loading data...")

    df = pd.read_csv(
        "data/processed/mental_health_processed.csv"
    )

    X = df.drop(TARGET_COLUMN, axis=1)
    y = df[TARGET_COLUMN]

    print("Dataset Shape:", df.shape)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # Pipeline
    pipeline = Pipeline([

        (
            "selector",
            SelectKBest(
                score_func=f_classif,
                k=15
            )
        ),

        (
            "model",
            XGBClassifier(
                n_estimators=300,
                max_depth=6,
                learning_rate=0.05,
                random_state=42,
                eval_metric="logloss"
            )
        )

    ])

    print("Training model...")

    pipeline.fit(
        X_train,
        y_train
    )

    # Evaluate
    y_pred = pipeline.predict(X_test)

    y_prob = pipeline.predict_proba(
        X_test
    )[:, 1]

    print("\nAccuracy:",
          accuracy_score(y_test, y_pred))

    print("ROC-AUC:",
          roc_auc_score(y_test, y_prob))

    # Save pipeline
    joblib.dump(
        pipeline,
        "models/mental_health_pipeline.pkl"
    )

    print("\n✅ Pipeline model saved!")


if __name__ == "__main__":
    train()
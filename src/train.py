import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from imblearn.over_sampling import SMOTE

from xgboost import XGBClassifier

TARGET_COLUMN = "treatment"


def train():

    df = pd.read_csv(
        "data/processed/mental_health_processed.csv"
    )

    X = df.drop(TARGET_COLUMN, axis=1)
    y = df[TARGET_COLUMN]

    print("Class Distribution:")
    print(y.value_counts())

    # Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # Apply SMOTE
    smote = SMOTE(random_state=42)

    X_train, y_train = smote.fit_resample(
        X_train,
        y_train
    )

    print("After SMOTE:")
    print(pd.Series(y_train).value_counts())

    # XGBoost Model
    model = XGBClassifier(
        n_estimators=400,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )

    model.fit(X_train, y_train)

    # Test Accuracy
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)

    print("Accuracy:", acc)

    # Save Model
    joblib.dump(
        model,
        "models/mental_health_model.pkl"
    )

    print("✅ Improved model saved!")


if __name__ == "__main__":
    train()
import pandas as pd
import joblib

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

TARGET_COLUMN = "treatment"


def preprocess():

    df = pd.read_csv(
        "data/raw/mental_health.csv"
    )

    print("Original Shape:", df.shape)

    # Drop unwanted columns
    df = df.drop(
        columns=["Timestamp", "comments"],
        errors="ignore"
    )

    # Remove missing values
    df = df.dropna()

    # Convert target
    df[TARGET_COLUMN] = df[TARGET_COLUMN].map({
        "Yes": 1,
        "No": 0
    })

    # Encode categorical
    encoders = {}

    categorical_cols = df.select_dtypes(
        include=["object"]
    ).columns

    for col in categorical_cols:

        le = LabelEncoder()

        df[col] = le.fit_transform(df[col])

        encoders[col] = le

    joblib.dump(
        encoders,
        "models/label_encoders.pkl"
    )

    # Split
    X = df.drop(TARGET_COLUMN, axis=1)

    y = df[TARGET_COLUMN]

    # Scale
    scaler = StandardScaler()

    X_scaled = scaler.fit_transform(X)

    joblib.dump(
        scaler,
        "models/scaler.pkl"
    )

    processed_df = pd.DataFrame(
        X_scaled,
        columns=X.columns
    )

    processed_df[TARGET_COLUMN] = y.reset_index(drop=True)

    processed_df.to_csv(
        "data/processed/mental_health_processed.csv",
        index=False
    )

    print("✅ Preprocessing Done")


if __name__ == "__main__":
    preprocess()
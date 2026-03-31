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

    # -------------------------
    # Drop unnecessary columns
    # -------------------------

    drop_cols = [
        "Timestamp",
        "comments"
    ]

    df = df.drop(
        columns=drop_cols,
        errors="ignore"
    )

    # -------------------------
    # Handle missing values
    # -------------------------

    df = df.dropna()

    # -------------------------
    # Convert target
    # -------------------------

    df[TARGET_COLUMN] = df[TARGET_COLUMN].map({
        "Yes": 1,
        "No": 0
    })

    # -------------------------
    # Encode categorical columns
    # -------------------------

    encoders = {}

    categorical_cols = df.select_dtypes(
        include=["object"]
    ).columns

    for col in categorical_cols:

        if col != TARGET_COLUMN:

            le = LabelEncoder()

            df[col] = le.fit_transform(
                df[col]
            )

            encoders[col] = le

    # Save encoders
    joblib.dump(
        encoders,
        "models/label_encoders.pkl"
    )

    print("✅ Encoders saved")

    # -------------------------
    # Split features and target
    # -------------------------

    X = df.drop(
        TARGET_COLUMN,
        axis=1
    )

    y = df[TARGET_COLUMN]

    # -------------------------
    # Scale features
    # -------------------------

    scaler = StandardScaler()

    X_scaled = scaler.fit_transform(X)

    # Save scaler
    joblib.dump(
        scaler,
        "models/scaler.pkl"
    )

    print("✅ Scaler saved")

    # -------------------------
    # Save processed dataset
    # -------------------------

    processed_df = pd.DataFrame(
        X_scaled,
        columns=X.columns
    )

    processed_df[TARGET_COLUMN] = y.reset_index(
        drop=True
    )

    processed_df.to_csv(
        "data/processed/mental_health_processed.csv",
        index=False
    )

    print("✅ Preprocessing completed!")
    print("Processed Shape:", processed_df.shape)


if __name__ == "__main__":
    preprocess()
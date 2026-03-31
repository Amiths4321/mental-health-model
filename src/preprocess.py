import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

TARGET_COLUMN = "treatment"

def preprocess():

    df = pd.read_csv(
        "data/raw/mental_health.csv"
    )

    print("Original Shape:", df.shape)

    # Drop unnecessary columns
    drop_cols = [
        "Timestamp",
        "comments"
    ]

    df = df.drop(
        columns=drop_cols,
        errors="ignore"
    )

    # Drop rows with missing values
    df = df.dropna()

    # Convert target Yes/No → 1/0
    df[TARGET_COLUMN] = df[TARGET_COLUMN].map({
        "Yes": 1,
        "No": 0
    })

    # Encode categorical columns
    for col in df.select_dtypes(include=["object"]).columns:

        if col != TARGET_COLUMN:

            le = LabelEncoder()

            df[col] = le.fit_transform(df[col])

    # Split features and target
    X = df.drop(TARGET_COLUMN, axis=1)

    y = df[TARGET_COLUMN]

    # Scale features
    scaler = StandardScaler()

    X_scaled = scaler.fit_transform(X)

    processed_df = pd.DataFrame(
        X_scaled,
        columns=X.columns
    )

    processed_df[TARGET_COLUMN] = y.reset_index(drop=True)

    # Save processed file
    processed_df.to_csv(
        "data/processed/mental_health_processed.csv",
        index=False
    )

    print("✅ Preprocessing completed!")
    print("Processed Shape:", processed_df.shape)


if __name__ == "__main__":
    preprocess()
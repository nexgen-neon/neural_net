import numpy as np

from rain_pred.data_loader import load_dataset
from rain_pred.data_audit import audit_dataset, audit_target
from rain_pred.cleaning import clean_dataset
from rain_pred.preprocessing import prepare_data


def main():

    print("=" * 60)
    print("RAIN PREDICTION")
    print("=" * 60)

    # --------------------------------------------------
    # 1. Load raw dataset
    # --------------------------------------------------

    print("\nLoading dataset...")

    file_path = "data/raw/weatherAUS.csv"

    df = load_dataset(file_path)

    print(f"Original shape: {df.shape}")

    # --------------------------------------------------
    # 2. Audit dataset
    # --------------------------------------------------

    print("\nRunning dataset audit...")

    audit_dataset(df)
    audit_target(df)

    # --------------------------------------------------
    # 3. Clean dataset
    # --------------------------------------------------

    print("\nCleaning dataset...")

    cleaned_df = clean_dataset(df)

    print(f"Cleaned shape: {cleaned_df.shape}")

    print("\nCleaned columns:")
    print(cleaned_df.columns.tolist())

    print("\nMissing values in cleaned dataset:")
    print(cleaned_df.isnull().sum())

    # --------------------------------------------------
    # 4. Save cleaned dataset
    # --------------------------------------------------

    cleaned_path = "data/processed/weather_cleaned.csv"

    cleaned_df.to_csv(
        cleaned_path,
        index=False,
    )

    print(
        f"\nCleaned dataset saved to: {cleaned_path}"
    )

    # --------------------------------------------------
    # 5. Train-test split + preprocessing
    # --------------------------------------------------

    print("\nPreparing data...")

    (
        X_train,
        X_test,
        y_train,
        y_test,
        imputer,
        scaler,
    ) = prepare_data(cleaned_df)

    # --------------------------------------------------
    # 6. Display shapes
    # --------------------------------------------------

    print("\nTrain-test split completed.")

    print("\nData shapes:")

    print("X_train :", X_train.shape)
    print("y_train :", y_train.shape)

    print("X_test  :", X_test.shape)
    print("y_test  :", y_test.shape)

    # --------------------------------------------------
    # 7. Check missing values after preprocessing
    # --------------------------------------------------

    print("\nMissing values after preprocessing:")

    print("NaN in X_train:", np.isnan(X_train).sum())
    print("NaN in X_test :", np.isnan(X_test).sum())

    # --------------------------------------------------
    # 8. Display first training sample
    # --------------------------------------------------

    print("\nFirst training sample:")
    print(X_train[0])

    print("\nFirst training target:")
    print(y_train[0])

    # --------------------------------------------------
    # 9. Display first test sample
    # --------------------------------------------------

    print("\nFirst test sample:")
    print(X_test[0])

    print("\nFirst test target:")
    print(y_test[0])


if __name__ == "__main__":
    main()
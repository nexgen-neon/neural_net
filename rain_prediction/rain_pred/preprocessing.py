import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler


FEATURE_COLUMNS = [
    "MinTemp",
    "MaxTemp",
    "Rainfall",
    "Humidity3pm",
    "Pressure3pm",
]

TARGET_COLUMN = "RainTomorrow"


def prepare_data(df: pd.DataFrame):

    # --------------------------------------------------
    # 1. Separate features and target
    # --------------------------------------------------

    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]

    # --------------------------------------------------
    # 2. 90% train / 10% test
    # --------------------------------------------------

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.10,
        random_state=42,
        stratify=y,
    )

    # --------------------------------------------------
    # 3. Create train/test DataFrames
    # --------------------------------------------------

    train_df = X_train.copy()
    train_df[TARGET_COLUMN] = y_train

    test_df = X_test.copy()
    test_df[TARGET_COLUMN] = y_test

    # --------------------------------------------------
    # 4. Save train/test datasets
    # --------------------------------------------------

    train_df.to_csv(
        "data/processed/train.csv",
        index=False,
    )

    test_df.to_csv(
        "data/processed/test.csv",
        index=False,
    )

    # --------------------------------------------------
    # 5. Impute missing feature values
    # --------------------------------------------------

    imputer = SimpleImputer(strategy="median")

    X_train = imputer.fit_transform(X_train)
    X_test = imputer.transform(X_test)

    # --------------------------------------------------
    # 6. Standardize features
    # --------------------------------------------------

    scaler = StandardScaler()

    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # --------------------------------------------------
    # 7. Convert targets to NumPy arrays
    # --------------------------------------------------

    y_train = y_train.to_numpy()
    y_test = y_test.to_numpy()

    return (
        X_train,
        X_test,
        y_train,
        y_test,
        imputer,
        scaler,
    )
def preprocess_input(inputs, preprocessing):

    imputer_statistics = preprocessing[
        "imputer_statistics"
    ]

    scaler_mean = preprocessing[
        "scaler_mean"
    ]

    scaler_scale = preprocessing[
        "scaler_scale"
    ]

    processed_inputs = []

    for i in range(len(inputs)):

        value = inputs[i]

        if value is None:
            value = imputer_statistics[i]

        value = (
            value - scaler_mean[i]
        ) / scaler_scale[i]

        processed_inputs.append(value)

    return processed_inputs
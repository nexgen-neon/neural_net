import pandas as pd


FEATURE_COLUMNS = [
    "MinTemp",
    "MaxTemp",
    "Rainfall",
    "Humidity3pm",
    "Pressure3pm",
]

TARGET_COLUMN = "RainTomorrow"


def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Select the features needed for the neural network
    and perform basic cleaning.
    """

    df = df.copy()

    # Keep only the columns needed
    df = df[FEATURE_COLUMNS + [TARGET_COLUMN]]

    # Remove rows where the target is missing
    df = df.dropna(subset=[TARGET_COLUMN])

    # Convert target:
    # No  -> 0
    # Yes -> 1
    df[TARGET_COLUMN] = df[TARGET_COLUMN].map({
        "No": 0,
        "Yes": 1,
    })

    # Convert feature columns to numeric
    for column in FEATURE_COLUMNS:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce",
        )

    return df
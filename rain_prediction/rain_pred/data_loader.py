import pandas as pd


def load_dataset(file_path: str) -> pd.DataFrame:
    """Load the weather dataset from a CSV file."""
    return pd.read_csv(file_path)
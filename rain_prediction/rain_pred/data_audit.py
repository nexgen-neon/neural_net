import pandas as pd


def audit_dataset(df: pd.DataFrame) -> None:
    """Print basic information about the dataset."""

    print("=" * 60)
    print("DATASET AUDIT")
    print("=" * 60)

    print("\n1. Shape")
    print("-" * 60)
    print(f"Rows    : {df.shape[0]}")
    print(f"Columns : {df.shape[1]}")

    print("\n2. Columns")
    print("-" * 60)

    for column in df.columns:
        print(column)

    print("\n3. Data Types")
    print("-" * 60)
    print(df.dtypes)

    print("\n4. Missing Values")
    print("-" * 60)

    missing = df.isnull().sum()

    missing_percentage = (
        df.isnull().mean() * 100
    )

    missing_report = pd.DataFrame({
        "missing_count": missing,
        "missing_percentage": missing_percentage
    })

    missing_report = missing_report.sort_values(
        "missing_percentage",
        ascending=False
    )

    print(missing_report)

    print("\n5. Duplicate Rows")
    print("-" * 60)
    print(f"Duplicate rows: {df.duplicated().sum()}")

    print("\n6. Numerical Summary")
    print("-" * 60)
    print(df.describe())

    print("\n7. Categorical Summary")
    print("-" * 60)
    print(df.describe(include="object"))

def audit_target(df: pd.DataFrame) -> None:
    """Inspect the target variable."""

    print("\n8. Target Variable")
    print("-" * 60)

    print(df["RainTomorrow"].value_counts(dropna=False))

    print("\nTarget percentages:")
    print(
        df["RainTomorrow"]
        .value_counts(normalize=True, dropna=False)
        .mul(100)
    )

def audit_missing_patterns(df: pd.DataFrame) -> None:
    """Investigate patterns in missing values."""

    print("\n9. Missing Values by Location")
    print("-" * 60)

    location_missing = (
        df.groupby("Location")
        [
            [
                "Sunshine",
                "Evaporation",
                "Cloud9am",
                "Cloud3pm",
            ]
        ]
        .apply(lambda x: x.isna().mean() * 100)
    )

    print(location_missing)

    print("\n10. Missing Values by Year")
    print("-" * 60)

    date = pd.to_datetime(df["Date"])

    year_missing = (
        df.assign(Year=date.dt.year)
        .groupby("Year")
        [
            [
                "Sunshine",
                "Evaporation",
                "Cloud9am",
                "Cloud3pm",
            ]
        ]
        .apply(lambda x: x.isna().mean() * 100)
    )

    print(year_missing)

def audit_missing_patterns_by_location_year(df: pd.DataFrame) -> None:
    """Investigate missing values by location and year."""

    print("\n11. Missing Values by Location and Year")
    print("-" * 60)

    date = pd.to_datetime(df["Date"])

    temp_df = df.assign(Year=date.dt.year)

    missing_by_location_year = (
        temp_df
        .groupby(["Location", "Year"])[
            [
                "Sunshine",
                "Evaporation",
                "Cloud9am",
                "Cloud3pm",
            ]
        ]
        .apply(lambda x: x.isna().mean() * 100)
    )

    print(missing_by_location_year)

def audit_combined_missing_values(df: pd.DataFrame) -> None:
    """Check how many important weather measurements are missing together."""

    print("\n12. Combined Missing Values")
    print("-" * 60)

    columns = [
        "Sunshine",
        "Evaporation",
        "Cloud9am",
        "Cloud3pm",
    ]

    missing_count = df[columns].isna().sum(axis=1)

    print("Number of rows by missing-column count:")
    print(missing_count.value_counts().sort_index())

    print("\nPercentage of rows:")
    print(
        missing_count
        .value_counts(normalize=True)
        .sort_index()
        .mul(100)
    )
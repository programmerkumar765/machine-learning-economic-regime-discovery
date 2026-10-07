from pathlib import Path
import pandas as pd


def load_gdp_data(file_path: str | Path) -> pd.DataFrame:
    """
    Load GDP data from CSV/TSV-like files.

    Expected format:
    - One 'Year' column
    - Remaining columns are country names
    """
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"Dataset not found: {file_path}")

    # sep=None lets pandas infer comma/tab separators.
    df = pd.read_csv(file_path, sep=None, engine="python")

    if "Year" not in df.columns:
        raise ValueError("Dataset must contain a 'Year' column.")

    df["Year"] = pd.to_numeric(df["Year"], errors="coerce")

    if df["Year"].isna().any():
        raise ValueError("The 'Year' column contains invalid values.")

    country_columns = [c for c in df.columns if c != "Year"]

    if not country_columns:
        raise ValueError("No country columns were found.")

    for column in country_columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    return df


def validate_dataset(df: pd.DataFrame) -> dict:
    """Return basic validation information about the dataset."""
    country_columns = [c for c in df.columns if c != "Year"]

    return {
        "rows": len(df),
        "countries": len(country_columns),
        "year_start": int(df["Year"].min()),
        "year_end": int(df["Year"].max()),
        "missing_values": int(df.isna().sum().sum()),
        "duplicate_years": int(df["Year"].duplicated().sum()),
    }

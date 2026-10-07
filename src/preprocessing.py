import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


def compute_growth_rates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert GDP levels into annual percentage growth rates.

    Output:
        Index   -> Year
        Columns -> Countries
    """
    growth = (
        df.set_index("Year")
        .sort_index()
        .pct_change(fill_method=None)
        .mul(100)
        .dropna(how="all")
    )

    return growth


def winsorize_growth_rates(
    growth_df: pd.DataFrame,
    lower_quantile: float = 0.01,
    upper_quantile: float = 0.99,
) -> pd.DataFrame:
    """
    Cap extreme values independently for each country's time series.

    This keeps every observation while reducing the influence of extreme
    sample values.
    """
    clipped = growth_df.copy()

    for column in clipped.columns:
        lower = clipped[column].quantile(lower_quantile)
        upper = clipped[column].quantile(upper_quantile)
        clipped[column] = clipped[column].clip(lower=lower, upper=upper)

    return clipped


def build_country_feature_matrix(
    growth_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Transform the dataset so each row is a country and each feature is a year.

    Returns:
        country_features: raw country x year matrix
        scaled_features: standardized country x year matrix
    """
    country_features = growth_df.T.copy()
    country_features.columns = country_features.columns.astype(str)

    scaler = StandardScaler()
    scaled = scaler.fit_transform(country_features)

    scaled_features = pd.DataFrame(
        scaled,
        index=country_features.index,
        columns=country_features.columns,
    )

    return country_features, scaled_features

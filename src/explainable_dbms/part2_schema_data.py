"""
Part 2: Data loading and feature engineering for user-provided data.
"""

from __future__ import annotations

from pathlib import Path
import pandas as pd
from sqlalchemy import text
from sqlalchemy.engine import Engine

from .config import PathsConfig, get_paths_config
from .io_utils import save_dataframe

def store_dataframe(engine: Engine, df: pd.DataFrame, table_name: str) -> None:
    """Persist the provided dataframe into MySQL, replacing existing data."""
    with engine.begin() as connection:
        connection.execute(text(f"DROP TABLE IF EXISTS {table_name}"))
    
    df.to_sql(table_name, con=engine, if_exists="replace", index=False, chunksize=500)


def load_user_data(engine: Engine, file_path: str, paths_config: PathsConfig | None = None) -> pd.DataFrame:
    """Load user-provided data from a CSV file and store it in the database."""
    table_name = Path(file_path).stem
    user_df = pd.read_csv(file_path)

    store_dataframe(engine, user_df, table_name)

    paths = paths_config or get_paths_config()
    save_dataframe(user_df, paths.output_dir / f"{table_name}_snapshot.csv")

    return user_df

def compute_aggregated_features(engine: Engine, table_name: str, paths_config: PathsConfig | None = None) -> pd.DataFrame:
    """Run the analytical SQL query and return a feature matrix for modelling."""
    with engine.connect() as connection:
        feature_df = pd.read_sql_table(table_name, con=connection)

    # The following feature engineering is specific to the old synthetic dataset.
    # This will be replaced with a more dynamic approach based on user's data and query.
    # For now, we will just return the dataframe.
    
    # feature_df["transaction_count"].fillna(0, inplace=True)
    # feature_df["total_spending"].fillna(0.0, inplace=True)
    # feature_df["avg_transaction_amount"].fillna(0.0, inplace=True)
    # feature_df["last_transaction_date"] = pd.to_datetime(feature_df["last_transaction_date"])
    # feature_df["unique_categories"].fillna(0, inplace=True)
    # feature_df["avg_discount"].fillna(0.0, inplace=True)

    # feature_df["days_since_last_transaction"] = (
    #     pd.Timestamp(date(2024, 12, 31)) - feature_df["last_transaction_date"]
    # ).dt.days.fillna(999)

    # feature_df["transaction_rate"] = feature_df["transaction_count"] / 730  # transactions per day
    # feature_df["spending_rate"] = feature_df["total_spending"] / (feature_df["days_since_last_transaction"] + 1)
    
    # rng = np.random.default_rng(42)
    # churn_probability = (
    #     0.3 * (feature_df["days_since_last_transaction"] > 120).astype(float) +
    #     0.25 * (feature_df["transaction_count"] < 5).astype(float) +
    #     0.2 * (feature_df["total_spending"] < 500).astype(float) +
    #     0.15 * (feature_df["credit_score"] < 600).astype(float) +
    #     0.1 * (feature_df["income"] < feature_df["income"].median()).astype(float) +
    #     rng.normal(0, 0.1, size=len(feature_df))  # Add noise for realism
    # )
    # churn_probability = np.clip(churn_probability, 0, 1)
    # feature_df["churn"] = (rng.random(len(feature_df)) < churn_probability).astype(int)

    paths = paths_config or get_paths_config()
    save_dataframe(feature_df, paths.output_dir / "aggregated_features.csv")

    return feature_df
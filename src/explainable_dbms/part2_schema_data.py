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

import featuretools as ft

def compute_aggregated_features(engine: Engine, table_name: str, paths_config: PathsConfig | None = None) -> pd.DataFrame:
    """Run the analytical SQL query and return a feature matrix for modelling."""
    with engine.connect() as connection:
        feature_df = pd.read_sql_table(table_name, con=connection)

    try:
        print("--- Starting automated feature engineering with featuretools ---")
        es = ft.EntitySet(id="user_data")
        es = es.add_dataframe(
            dataframe_name=table_name,
            dataframe=feature_df,
            index=feature_df.columns[0],
        )

        print("--- Running Deep Feature Synthesis (DFS) ---")
        feature_matrix, feature_defs = ft.dfs(
            entityset=es,
            target_dataframe_name=table_name,
            max_depth=2,
            verbose=1,
            n_jobs=-1,
        )

        feature_matrix = feature_matrix.reset_index()
        print("--- Automated feature engineering complete ---")

    except Exception as e:
        print(f"--- Error during automated feature engineering: {e} ---")
        print("--- Returning original dataframe ---")
        feature_matrix = feature_df

    paths = paths_config or get_paths_config()
    save_dataframe(feature_matrix, paths.output_dir / "aggregated_features.csv")

    return feature_matrix
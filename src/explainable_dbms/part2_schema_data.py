"""
Part 2: Synthetic data generation, ingestion, and feature engineering.
"""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from sqlalchemy import text
from sqlalchemy.engine import Engine

from .config import PathsConfig, get_paths_config
from .io_utils import save_dataframe


def generate_customers(num_customers: int = 1000, random_state: int = 42) -> pd.DataFrame:
    """Create a synthetic customer dataset."""
    rng = np.random.default_rng(random_state)
    customer_ids = np.arange(1, num_customers + 1)
    genders = rng.choice(["Male", "Female", "Non-binary"], size=num_customers, p=[0.49, 0.49, 0.02])
    locations = rng.choice(
        ["North", "South", "East", "West", "Central"],
        size=num_customers,
        p=[0.2, 0.2, 0.2, 0.2, 0.2],
    )
    customer_df = pd.DataFrame(
        {
            "customer_id": customer_ids,
            "name": [f"Customer_{i:04d}" for i in customer_ids],
            "age": rng.integers(18, 75, size=num_customers),
            "gender": genders,
            "location": locations,
            "income": rng.normal(75000, 20000, size=num_customers).clip(25000, 200000).round(2),
            "credit_score": rng.integers(300, 850, size=num_customers),
        }
    )
    return customer_df


def generate_products(num_products: int = 100, random_state: int = 42) -> pd.DataFrame:
    """Create a synthetic product catalog."""
    rng = np.random.default_rng(random_state + 1)
    categories = ["Electronics", "Home", "Beauty", "Sports", "Automotive", "Fashion", "Grocery", "Books"]
    product_ids = np.arange(1, num_products + 1)
    base_prices = rng.uniform(10, 500, size=num_products)
    costs = base_prices * rng.uniform(0.4, 0.8, size=num_products)
    product_df = pd.DataFrame(
        {
            "product_id": product_ids,
            "product_name": [f"Product_{i:03d}" for i in product_ids],
            "category": rng.choice(categories, size=num_products),
            "price": base_prices.round(2),
            "cost": costs.round(2),
            "supplier": rng.choice(["Supplier_A", "Supplier_B", "Supplier_C", "Supplier_D"], size=num_products),
        }
    )
    return product_df


def generate_transactions(
    customers: pd.DataFrame,
    products: pd.DataFrame,
    num_transactions: int = 5000,
    random_state: int = 42,
) -> pd.DataFrame:
    """Generate synthetic transactions referencing customer and product IDs."""
    rng = np.random.default_rng(random_state + 2)
    transaction_ids = np.arange(1, num_transactions + 1)
    transaction_dates = [
        date(2023, 1, 1) + timedelta(days=int(day))
        for day in rng.integers(0, 730, size=num_transactions)
    ]
    quantities = rng.integers(1, 5, size=num_transactions)
    chosen_products = products.sample(n=num_transactions, replace=True, random_state=random_state).reset_index(drop=True)
    discounts = rng.uniform(0, 0.3, size=num_transactions).round(3)
    payment_methods = rng.choice(["Credit Card", "Debit Card", "Cash", "Digital Wallet"], size=num_transactions)

    totals = (chosen_products["price"].values * quantities) * (1 - discounts)

    transactions_df = pd.DataFrame(
        {
            "transaction_id": transaction_ids,
            "customer_id": rng.choice(customers["customer_id"], size=num_transactions),
            "product_id": chosen_products["product_id"],
            "transaction_date": transaction_dates,
            "quantity": quantities,
            "total_amount": totals.round(2),
            "discount": discounts,
            "payment_method": payment_methods,
        }
    )
    return transactions_df


def store_dataframe(engine: Engine, df: pd.DataFrame, table_name: str) -> None:
    """Persist the provided dataframe into MySQL, replacing existing data."""
    df.to_sql(table_name, con=engine, if_exists="replace", index=False, chunksize=500)


def generate_and_store_all(engine: Engine, paths_config: PathsConfig | None = None) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Generate synthetic datasets and load them into the configured database."""
    customers = generate_customers()
    products = generate_products()
    transactions = generate_transactions(customers, products)

    store_dataframe(engine, customers, "customers")
    store_dataframe(engine, products, "products")
    store_dataframe(engine, transactions, "transactions")

    paths = paths_config or get_paths_config()
    save_dataframe(customers, paths.output_dir / "customers_snapshot.csv")
    save_dataframe(products, paths.output_dir / "products_snapshot.csv")
    save_dataframe(transactions, paths.output_dir / "transactions_snapshot.csv")

    return customers, products, transactions


AGGREGATED_FEATURE_QUERY = """
SELECT c.customer_id,
       c.age,
       c.income,
       c.credit_score,
       COUNT(t.transaction_id) AS transaction_count,
       SUM(t.total_amount) AS total_spending,
       AVG(t.total_amount) AS avg_transaction_amount,
       MAX(t.transaction_date) AS last_transaction_date,
       COUNT(DISTINCT p.category) AS unique_categories,
       AVG(t.discount) AS avg_discount
FROM customers c
LEFT JOIN transactions t ON c.customer_id = t.customer_id
LEFT JOIN products p ON t.product_id = p.product_id
GROUP BY c.customer_id, c.age, c.income, c.credit_score
"""


def compute_aggregated_features(engine: Engine, paths_config: PathsConfig | None = None) -> pd.DataFrame:
    """Run the analytical SQL query and return a feature matrix for modelling."""
    with engine.connect() as connection:
        feature_df = pd.read_sql(text(AGGREGATED_FEATURE_QUERY), con=connection)

    feature_df["transaction_count"].fillna(0, inplace=True)
    feature_df["total_spending"].fillna(0.0, inplace=True)
    feature_df["avg_transaction_amount"].fillna(0.0, inplace=True)
    feature_df["last_transaction_date"] = pd.to_datetime(feature_df["last_transaction_date"])
    feature_df["unique_categories"].fillna(0, inplace=True)
    feature_df["avg_discount"].fillna(0.0, inplace=True)

    feature_df["days_since_last_transaction"] = (
        pd.Timestamp(date(2024, 12, 31)) - feature_df["last_transaction_date"]
    ).dt.days.fillna(999)

    feature_df["churn"] = (feature_df["days_since_last_transaction"] > 120).astype(int)

    paths = paths_config or get_paths_config()
    save_dataframe(feature_df, paths.output_dir / "aggregated_features.csv")

    return feature_df


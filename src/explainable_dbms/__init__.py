"""
Explainable DBMS package.

This package provides a modular pipeline that covers:
1. Database setup and schema creation.
2. Synthetic data generation and ingestion.
3. Machine learning modelling and explainability.
4. Visualization dashboards.
5. System-level evaluation and benchmarking.

Each phase is implemented in a dedicated module (`part*_*.py`) so the
pipeline can be orchestrated end-to-end or run stepwise.
"""

__all__ = [
    "part1_setup",
    "part2_schema_data",
    "part3_ml_xai",
    "part4_visualization",
    "part5_evaluation",
    "config",
    "io_utils",
    "models",
]


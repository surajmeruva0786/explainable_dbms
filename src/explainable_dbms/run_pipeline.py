"""
Command-line entry point to run the full explainable DBMS pipeline.
"""

from __future__ import annotations

import logging
from typing import List

from .config import get_database_config, get_paths_config
from .part1_setup import initialize_database
from .part2_schema_data import compute_aggregated_features, generate_and_store_all
from .part3_ml_xai import ModelArtifact, train_models_and_explain
from .part4_visualization import generate_visualizations
from .part5_evaluation import compile_benchmark_report


logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def run_pipeline() -> None:
    """Execute the complete pipeline end-to-end."""
    logger.info("Initialising configuration.")
    db_config = get_database_config()
    paths_config = get_paths_config()

    logger.info("Setting up database and schema.")
    engine = initialize_database(db_config)

    logger.info("Generating and loading synthetic datasets.")
    generate_and_store_all(engine, paths_config)

    logger.info("Computing aggregated analytical features.")
    feature_df = compute_aggregated_features(engine, paths_config)

    logger.info("Training models and generating explanations.")
    artifacts: List[ModelArtifact] = train_models_and_explain(engine, feature_df, paths_config)

    logger.info("Creating visualization dashboards.")
    generate_visualizations(artifacts, paths_config)

    logger.info("Compiling evaluation and benchmark report.")
    report = compile_benchmark_report(engine, artifacts, paths_config)

    logger.info("Pipeline completed successfully.")
    logger.info("Benchmark report saved to %s", paths_config.metrics_path)
    logger.debug("Full report: %s", report)


if __name__ == "__main__":
    run_pipeline()

